"""Train EA-FT EEGNet on OpenBMI (secondary target) and export dashboard artifacts.

Source domain : all 9 BCI Competition IV 2a subjects (from results/demo/demo_bundle.npz)
Target domain : OpenBMI / Lee2019_MI sess01_subj01 (local .mat)

Writes / updates:
  results/demo/eaft_models.pt   (+ OB01 state dict)
  results/demo/demo_bundle.npz  (+ OB01_X / y / ea / ncal)
  results/demo/demo_summary.json (+ openbmi + OB01 accuracy)
  results/openbmi_results.json
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from eegnet_model import EEGNet  # noqa: E402

ART = ROOT / "results" / "demo"
OPENBMI_FILE = ROOT / "openBMI dataset" / "sess01_subj01_EEG_MI.mat"
OUT_JSON = ROOT / "results" / "openbmi_results.json"

SEED = 42
FS = 250
N_CHANNELS = 22
TRIAL_LEN = 1000
OPENBMI_SRC_FS = 1000
OPENBMI_MI_SEC = 4.0
N_CALIB_TRIALS = 36
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1e-3
FINETUNE_LR = 5e-5
FINETUNE_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]
FCZ_NEIGHBOURS = ["Fz", "FC1", "FC2", "Cz"]
TARGET_KEY = "OB01"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def artifact_rejection_threshold(X, threshold_std=5.0):
    ptp = X.max(axis=2) - X.min(axis=2)
    ptp_max = ptp.max(axis=1)
    mu, sigma = ptp_max.mean(), ptp_max.std()
    mask = ptp_max < (mu + threshold_std * sigma)
    return X[mask], mask


def bandpass_filter(X, low=8.0, high=30.0, fs=250, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, X, axis=2).astype(np.float32)


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    n, c, _ = X.shape
    r = np.zeros((c, c), dtype=np.float64)
    for i in range(n):
        xi = X[i].astype(np.float64)
        r += xi @ xi.T
    r /= n
    r += 1e-8 * np.eye(c)
    w, v = np.linalg.eigh(r)
    w = np.maximum(w, 1e-10)
    p = v @ np.diag(w ** -0.5) @ v.T
    return np.einsum("cd,ndt->nct", p, X.astype(np.float64)).astype(np.float32)


def ea_fit(X: np.ndarray) -> np.ndarray:
    n, c, _ = X.shape
    r = np.zeros((c, c), dtype=np.float64)
    for i in range(n):
        xi = X[i].astype(np.float64)
        r += xi @ xi.T
    r /= n
    r += 1e-8 * np.eye(c)
    w, v = np.linalg.eigh(r)
    w = np.maximum(w, 1e-10)
    return (v @ np.diag(w ** -0.5) @ v.T).astype(np.float32)


def ea_apply(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("cd,ndt->nct", p, x.astype(np.float64)).astype(np.float32)


def train_one_epoch(model, loader, opt, crit):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * len(yb)
        correct += int((logits.argmax(1) == yb).sum().item())
        n += len(yb)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        prob = F.softmax(logits, dim=1)[:, 1]
        total_loss += float(loss.item()) * len(yb)
        correct += int((logits.argmax(1) == yb).sum().item())
        n += len(yb)
        ys.append(yb.cpu().numpy())
        ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    acc = correct / max(n, 1)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5
    return total_loss / max(n, 1), acc, auc, y_true, y_prob


def pretrain_on_2a(dataset_ea: dict) -> EEGNet:
    ids = list(dataset_ea.keys())
    x = np.concatenate([dataset_ea[s]["X"] for s in ids], axis=0)
    y = np.concatenate([dataset_ea[s]["y"] for s in ids], axis=0)
    perm = np.random.permutation(len(y))
    x, y = x[perm], y[perm]
    n_val = max(int(0.2 * len(y)), 1)
    x_tr, y_tr = x[n_val:], y[n_val:]
    x_va, y_va = x[:n_val], y[:n_val]
    tr = DataLoader(EEGDataset(x_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va = DataLoader(EEGDataset(x_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    model = EEGNet(n_classes=2, n_channels=N_CHANNELS, n_samples=TRIAL_LEN).to(DEVICE)
    opt = Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for epoch in range(PRETRAIN_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, tr, opt, crit)
        _, va_acc, _, _, _ = evaluate(model, va, crit)
        sched.step()
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  pretrain epoch {epoch+1:02d}/{PRETRAIN_EPOCHS}  "
                  f"train={tr_acc*100:.1f}%  val={va_acc*100:.1f}%")
    model.load_state_dict(best_state)
    model.to(DEVICE)
    print(f"  best source val acc: {best_acc*100:.1f}%")
    return model


def _val_split_from_calib(y_cal, seed, val_frac=0.25):
    rng = np.random.RandomState(seed)
    n = len(y_cal)
    n_val = max(2, int(round(val_frac * n)))
    idx = rng.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    if len(np.unique(y_cal[val_idx])) < 2 or len(np.unique(y_cal[tr_idx])) < 2:
        order = np.argsort(y_cal)
        val_idx = np.concatenate([order[: n_val // 2], order[-(n_val - n_val // 2) :]])
        tr_idx = np.setdiff1d(np.arange(n), val_idx)
    return tr_idx, val_idx


def fine_tune(pretrained, x_cal, y_cal, x_test, y_test, seed=SEED):
    model = copy.deepcopy(pretrained).to(DEVICE)
    for p in model.block1.parameters():
        p.requires_grad = False
    for p in list(model.block2.parameters()) + list(model.classifier.parameters()):
        p.requires_grad = True

    tr_idx, val_idx = _val_split_from_calib(y_cal, seed)
    cal_loader = DataLoader(
        EEGDataset(x_cal[tr_idx], y_cal[tr_idx]),
        batch_size=min(16, len(tr_idx)),
        shuffle=True,
    )
    val_loader = DataLoader(
        EEGDataset(x_cal[val_idx], y_cal[val_idx]), batch_size=32, shuffle=False
    )
    test_loader = DataLoader(EEGDataset(x_test, y_test), batch_size=32, shuffle=False)
    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_score, best_state, stale = -1.0, None, 0
    for _ in range(FINETUNE_EPOCHS):
        train_one_epoch(model, cal_loader, opt, crit)
        _, val_acc, val_auc, _, _ = evaluate(model, val_loader, crit)
        score = val_acc + 1e-3 * val_auc
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.to(DEVICE)
    _, acc, auc, y_true, y_prob = evaluate(model, test_loader, crit)
    kappa = float(cohen_kappa_score(y_true, (y_prob >= 0.5).astype(int)))
    return model, float(acc), float(auc), kappa, y_true, y_prob


def load_openbmi_local(mat_path: Path):
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    run = mat["EEG_MI_train"]
    x = np.asarray(run.x, dtype=np.float64)
    onsets = np.ravel(run.t).astype(int)
    y_dec = np.ravel(run.y_dec).astype(int)
    src_ch = [str(np.squeeze(c)) for c in np.ravel(run.chan)]
    win_src = int(OPENBMI_MI_SEC * OPENBMI_SRC_FS)
    trials = [x[on : on + win_src, :].T for on in onsets if on + win_src <= x.shape[0]]
    x_ep = np.stack(trials, axis=0)
    x_ep = resample_poly(x_ep, FS, OPENBMI_SRC_FS, axis=2).astype(np.float32)[:, :, :TRIAL_LEN]
    y = (y_dec[: x_ep.shape[0]] == 1).astype(np.int64)  # right=1, left=0
    return x_ep, y, src_ch


def harmonise_to_2a(x_src, src_ch, target_ch=CH_NAMES):
    idx = {n: i for i, n in enumerate(src_ch)}
    out = np.zeros((x_src.shape[0], len(target_ch), x_src.shape[2]), dtype=np.float32)
    reconstructed = []
    for j, name in enumerate(target_ch):
        if name in idx:
            out[:, j, :] = x_src[:, idx[name], :]
        elif name == "FCz":
            av = [idx[n] for n in FCZ_NEIGHBOURS if n in idx]
            out[:, j, :] = x_src[:, av, :].mean(axis=1)
            reconstructed.append(name)
        else:
            raise ValueError(f"Channel {name} missing from OpenBMI")
    return out, reconstructed


def load_2a_source_from_bundle(bundle) -> dict:
    """Build EA-aligned binary source dataset from existing demo bundle."""
    keys = bundle.files if hasattr(bundle, "files") else list(bundle.keys())
    dataset = {}
    for key in keys:
        if not key.endswith("_X") or key.startswith("OB"):
            continue
        sid = key[:-2]  # e.g. S01
        x = np.asarray(bundle[key])
        y = np.asarray(bundle[f"{sid}_y"])
        dataset[sid] = {"X": euclidean_alignment(x), "y": y.astype(np.int64)}
        print(f"  source {sid}: {dataset[sid]['X'].shape}  "
              f"L={(y==0).sum()} R={(y==1).sum()}")
    return dataset


def main():
    set_seed(SEED)
    print("=" * 64)
    print("OpenBMI secondary-target EA-FT training")
    print(f"Device: {DEVICE}")
    print("=" * 64)

    if not OPENBMI_FILE.exists():
        raise FileNotFoundError(f"Missing OpenBMI file: {OPENBMI_FILE}")
    if not (ART / "demo_bundle.npz").exists() or not (ART / "eaft_models.pt").exists():
        raise FileNotFoundError("Missing results/demo artifacts. Export BCI 2a demo first.")

    # 1) Source = all BCI IV 2a subjects from demo bundle
    print("\n[1/5] Loading BCI IV 2a source from demo_bundle …")
    old_bundle = dict(np.load(ART / "demo_bundle.npz", allow_pickle=True))
    dataset_ea = load_2a_source_from_bundle(old_bundle)

    print("\n[2/5] Pre-training EEGNet on all 9 BCI 2a subjects …")
    source_model = pretrain_on_2a(dataset_ea)

    # 2) Target = OpenBMI subject
    print("\n[3/5] Loading / preprocessing OpenBMI target …")
    x62, y_ob, ch = load_openbmi_local(OPENBMI_FILE)
    print(f"  raw trials: {x62.shape[0]}  L={(y_ob==0).sum()} R={(y_ob==1).sum()}")
    x22, recon = harmonise_to_2a(x62, ch)
    print(f"  harmonised: {x22.shape}  reconstructed={recon or 'none'}")
    x_clean, mask = artifact_rejection_threshold(x22)
    y_clean = y_ob[mask]
    print(f"  artifact keep: {x_clean.shape[0]}/{x22.shape[0]}")
    x_filt = bandpass_filter(x_clean)
    print(f"  bandpass: {x_filt.shape}  L={(y_clean==0).sum()} R={(y_clean==1).sum()}")

    if len(y_clean) <= N_CALIB_TRIALS + 5:
        raise RuntimeError(
            f"Not enough OpenBMI trials after cleaning ({len(y_clean)}) for "
            f"{N_CALIB_TRIALS} calibration + test split."
        )

    rng = np.random.RandomState(SEED + 777)
    perm = rng.permutation(len(y_clean))
    xp, yp = x_filt[perm], y_clean[perm]
    # Keep original filtered order for dashboard replay, but train on shuffled split
    x_cal, y_cal = xp[:N_CALIB_TRIALS], yp[:N_CALIB_TRIALS]
    x_test, y_test = xp[N_CALIB_TRIALS:], yp[N_CALIB_TRIALS:]

    p_ob = ea_fit(x_cal)
    x_cal_a, x_test_a = ea_apply(p_ob, x_cal), ea_apply(p_ob, x_test)

    print("\n[4/5] Evaluating zero-shot + fine-tuning on OpenBMI …")
    crit = nn.CrossEntropyLoss()
    zs_loader = DataLoader(EEGDataset(x_test_a, y_test), batch_size=32, shuffle=False)
    _, zs_acc, zs_auc, _, _ = evaluate(source_model, zs_loader, crit)
    ft_model, cal_acc, cal_auc, cal_kap, y_true, y_prob = fine_tune(
        source_model, x_cal_a, y_cal, x_test_a, y_test, seed=SEED + 777
    )

    # Transductive oracle (full-session EA)
    x_all_a = euclidean_alignment(xp)
    _, tr_acc, tr_auc, tr_kap, _, _ = fine_tune(
        source_model,
        x_all_a[:N_CALIB_TRIALS],
        y_cal,
        x_all_a[N_CALIB_TRIALS:],
        y_test,
        seed=SEED + 777,
    )

    results = {
        "target": TARGET_KEY,
        "dataset": "OpenBMI / Lee2019_MI",
        "file": OPENBMI_FILE.name,
        "n_trials_clean": int(len(y_clean)),
        "n_calib": int(N_CALIB_TRIALS),
        "n_test": int(len(y_test)),
        "zero_shot_acc": float(zs_acc),
        "zero_shot_auc": float(zs_auc),
        "ea_ft_calib_acc": float(cal_acc),
        "ea_ft_calib_auc": float(cal_auc),
        "ea_ft_calib_kappa": float(cal_kap),
        "ea_ft_transductive_acc": float(tr_acc),
        "ea_ft_transductive_auc": float(tr_auc),
        "ea_ft_transductive_kappa": float(tr_kap),
    }
    print("\n" + "=" * 64)
    print("OpenBMI results")
    print("=" * 64)
    print(f"  Zero-shot              : {zs_acc*100:5.1f}%   AUC {zs_auc:.3f}")
    print(f"  EA-FT (calib-only)     : {cal_acc*100:5.1f}%   AUC {cal_auc:.3f}  kappa {cal_kap:.3f}  <- headline")
    print(f"  EA-FT (transductive)   : {tr_acc*100:5.1f}%   AUC {tr_auc:.3f}  kappa {tr_kap:.3f}")
    print("=" * 64)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    # 3) Update dashboard artifacts
    print("\n[5/5] Updating dashboard artifacts …")
    try:
        ckpt = torch.load(ART / "eaft_models.pt", map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ART / "eaft_models.pt", map_location="cpu")

    # Store filtered UN-aligned trials in shuffled train order used above,
    # so first ncal trials match the calibration reference P.
    old_bundle[f"{TARGET_KEY}_X"] = xp.astype(np.float32)
    old_bundle[f"{TARGET_KEY}_y"] = yp.astype(np.int64)
    old_bundle[f"{TARGET_KEY}_ea"] = p_ob.astype(np.float32)
    old_bundle[f"{TARGET_KEY}_ncal"] = np.array(N_CALIB_TRIALS)
    old_bundle["class_names"] = np.array(["Left Hand", "Right Hand"])
    np.savez_compressed(ART / "demo_bundle.npz", **old_bundle)

    ckpt.setdefault("model_states", {})
    ckpt["model_states"][TARGET_KEY] = {
        k: v.detach().cpu() for k, v in ft_model.state_dict().items()
    }
    if "arch" not in ckpt:
        ckpt["arch"] = dict(
            n_classes=2, n_channels=N_CHANNELS, n_samples=TRIAL_LEN, dropout_rate=0.5
        )
    torch.save(ckpt, ART / "eaft_models.pt")

    summary_path = ART / "demo_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.setdefault("per_subject_acc", {})
    summary["per_subject_acc"][TARGET_KEY] = float(cal_acc * 100.0)
    summary.setdefault("per_subject_detail", {})
    summary["per_subject_detail"][TARGET_KEY] = {
        "zero_shot": float(zs_acc * 100.0),
        "ft_acc": float(cal_acc * 100.0),
        "auc": float(cal_auc),
        "kappa": float(cal_kap),
    }
    summary.setdefault("calib_only_ea_ft", {})
    summary["calib_only_ea_ft"][TARGET_KEY] = float(cal_acc * 100.0)
    summary["subject_meta"] = summary.get("subject_meta", {})
    summary["subject_meta"][TARGET_KEY] = {
        "dataset": "OpenBMI / Lee2019_MI",
        "label": "OB01 (OpenBMI)",
        "file": OPENBMI_FILE.name,
    }
    for sid in [f"S{i:02d}" for i in range(1, 10)]:
        summary["subject_meta"].setdefault(
            sid, {"dataset": "BCI IV 2a", "label": sid}
        )
    summary["openbmi"] = results
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Updated {ART / 'demo_bundle.npz'}")
    print(f"Updated {ART / 'eaft_models.pt'}")
    print(f"Updated {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
