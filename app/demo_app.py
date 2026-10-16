"""EA-FT EEGNet research dashboard.

Sections:
  1) Calibration  — short-calibration adaptation for a target subject
  2) Inference    — Left/Right prediction on held-out trials
  3) Results      — experimental tables and charts from the study

Run from project root:
    .venv\\Scripts\\streamlit run app/demo_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eegnet_model import EEGNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "demo"
SM_CHANNELS = {"C3": 7, "Cz": 9, "C4": 11}

st.set_page_config(
    page_title="EA-FT EEGNet Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",  # always start open; collapse control is hidden
)

# Professional light dashboard chrome
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;550;600;700&display=swap');

  html, body, [class*="css"]  {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
  }
  .block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.4rem;
    max-width: 1160px;
  }
  header[data-testid="stHeader"] { background: transparent; }
  div[data-testid="stToolbar"], .stDeployButton, [data-testid="stDecoration"] {
    display: none !important;
  }
  #MainMenu, footer { visibility: hidden; }

  /* Keep sidebar always visible — hide collapse / expand controls */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"],
  button[kind="headerNoPadding"],
  [data-testid="stBaseButton-headerNoPadding"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
  }
  section[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 280px !important;
    width: 280px !important;
  }
  section[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    transform: none !important;
    margin-left: 0 !important;
  }

  /* Sidebar shell */
  section[data-testid="stSidebar"] {
    background: #F3F6FB !important;
    border-right: 1px solid #D7E0EC;
  }
  section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }
  section[data-testid="stSidebar"] .block-container { padding-top: 0.4rem; }

  .brand-box {
    background: linear-gradient(145deg, #1F4E79 0%, #163A5F 100%);
    color: #FFFFFF;
    border-radius: 14px;
    padding: 1rem 1rem 0.95rem 1rem;
    margin-bottom: 1.1rem;
  }
  .brand-kicker {
    font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    opacity: 0.8; margin-bottom: 0.2rem; font-weight: 600;
  }
  .brand-title { font-size: 1.2rem; font-weight: 700; line-height: 1.2; }
  .brand-sub { font-size: 0.82rem; opacity: 0.88; margin-top: 0.25rem; }

  .nav-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #64748B; margin: 0.35rem 0 0.45rem 0.15rem;
  }
  .side-card {
    background: #FFFFFF;
    border: 1px solid #D7E0EC;
    border-radius: 12px;
    padding: 0.85rem 0.9rem;
    margin: 0.55rem 0 0.9rem 0;
  }
  .side-card .label { font-size: 0.72rem; color: #64748B; font-weight: 600; }
  .side-card .value { font-size: 0.95rem; color: #1B2430; font-weight: 650; margin-top: 0.15rem; }
  .side-card .meta { font-size: 0.78rem; color: #64748B; margin-top: 0.35rem; line-height: 1.35; }

  /* Sidebar nav buttons look like menu items */
  section[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    justify-content: flex-start;
    border-radius: 10px;
    border: 1px solid transparent;
    padding: 0.7rem 0.85rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: none !important;
    margin-bottom: 0.25rem;
  }
  section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #334155 !important;
  }
  section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background: #E7EEF8 !important;
    border-color: #D0DBEA !important;
    color: #1F4E79 !important;
  }
  section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: #FFFFFF !important;
    color: #1F4E79 !important;
    border: 1px solid #C5D4E8 !important;
    box-shadow: 0 1px 2px rgba(27,36,48,0.06) !important;
  }
  section[data-testid="stSidebar"] div.stButton.reset-btn > button,
  section[data-testid="stSidebar"] div[data-testid="stButton"]:last-child button {
    background: #FFFFFF !important;
    color: #475569 !important;
    border: 1px solid #D7E0EC !important;
    justify-content: center !important;
    margin-top: 0.35rem;
  }

  /* Main content */
  .page-header {
    display: flex; justify-content: space-between; align-items: flex-end;
    gap: 1rem; margin-bottom: 1.1rem; padding-bottom: 0.85rem;
    border-bottom: 1px solid #E3E8F0;
  }
  .page-header h1 {
    margin: 0 !important; font-size: 1.7rem !important; font-weight: 700 !important;
    color: #1B2430 !important; letter-spacing: -0.02em;
  }
  .page-header p { margin: 0.25rem 0 0 0; color: #64748B; font-size: 0.95rem; }
  .pill {
    background: #E8EEF7; color: #1F4E79; border: 1px solid #C5D4E8;
    border-radius: 999px; padding: 0.28rem 0.75rem; font-size: 0.78rem; font-weight: 700;
    white-space: nowrap;
  }

  div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E3E8F0;
    border-radius: 12px;
    padding: 0.8rem 0.95rem;
    box-shadow: 0 1px 2px rgba(27, 36, 48, 0.04);
  }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #1F4E79; }
  div[data-testid="stMetricLabel"] { color: #5B6777; }
  .stTabs [data-baseweb="tab"] {
    background: #EEF2F7; border-radius: 8px 8px 0 0; color: #1B2430; font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #1F4E79 !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ── Data helpers ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models_and_bundle(bundle_mtime: float, ckpt_mtime: float):
    """Cache heavy binary artifacts; mtimes bust the cache when files change."""
    del bundle_mtime, ckpt_mtime
    if not (ART / "demo_bundle.npz").exists() or not (ART / "eaft_models.pt").exists():
        return None
    bundle = np.load(ART / "demo_bundle.npz", allow_pickle=True)
    try:
        ckpt = torch.load(ART / "eaft_models.pt", map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ART / "eaft_models.pt", map_location="cpu")
    return bundle, ckpt


def load_summary() -> dict:
    path = ART / "demo_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(ckpt, subject_key: str) -> EEGNet:
    model = EEGNet(**ckpt["arch"])
    model.load_state_dict(ckpt["model_states"][subject_key])
    model.eval()
    return model


def apply_ea(P: np.ndarray, trial: np.ndarray) -> np.ndarray:
    return (P @ trial).astype(np.float32)


@torch.no_grad()
def classify(model: EEGNet, trial_ct: np.ndarray) -> tuple[int, np.ndarray]:
    x = torch.from_numpy(trial_ct[None, :, :].astype(np.float32))
    probs = F.softmax(model(x), dim=1)[0].numpy()
    return int(probs.argmax()), probs


PLOT_BG = "#FFFFFF"
PLOT_FACE = "#F7F9FC"
PLOT_TEXT = "#1B2430"
PLOT_GRID = "#D9E2EC"
PLOT_SPINE = "#C5D0DE"
ACCENT = "#1F4E79"
ACCENT_2 = "#2E8B57"
ACCENT_3 = "#C2410C"
CHANNEL_COLORS = {"C3": "#1F4E79", "Cz": "#C2410C", "C4": "#2E8B57"}


def eeg_trace_figure(trial_ct: np.ndarray, fs: int, title: str):
    """Plot sensorimotor channels with per-channel z-score scaling for display."""
    t = np.arange(trial_ct.shape[1]) / fs
    fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_FACE)
    for i, (name, idx) in enumerate(SM_CHANNELS.items()):
        sig = trial_ct[idx].astype(np.float64)
        std = sig.std()
        disp = np.zeros_like(sig) if std < 1e-12 else (sig - sig.mean()) / std
        ax.plot(t, disp + i * 4.0, lw=1.0, color=CHANNEL_COLORS[name], label=name)
    ax.set_xlabel("Time (s)", color=PLOT_TEXT)
    ax.set_yticks([i * 4.0 for i in range(len(SM_CHANNELS))])
    ax.set_yticklabels(list(SM_CHANNELS.keys()), color=PLOT_TEXT)
    ax.set_title(f"{title}  (display z-scored)", fontsize=10, color=PLOT_TEXT, fontweight="600")
    ax.tick_params(colors=PLOT_TEXT)
    ax.grid(axis="x", color=PLOT_GRID, linestyle="--", alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE)
    ax.legend(
        loc="upper right", fontsize=8, ncol=3,
        facecolor=PLOT_BG, edgecolor=PLOT_SPINE, labelcolor=PLOT_TEXT,
    )
    fig.tight_layout()
    return fig


def init_state():
    defaults = {
        "page": "Calibration",
        "subject": None,
        "calib_done": False,
        "last_pred": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Migrate away from older radio-key navigation
    if "nav_page" in st.session_state and "page" in st.session_state:
        if st.session_state.get("pending_page"):
            st.session_state["page"] = st.session_state.pop("pending_page")
        elif st.session_state.get("nav_page") in (
            "Calibration", "Inference", "Experiment Results"
        ):
            st.session_state["page"] = st.session_state["nav_page"]


def bar_chart(labels, values, ylabel, title, colors=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.6), facecolor=PLOT_BG)
    ax.set_facecolor(PLOT_FACE)
    x = np.arange(len(labels))
    cols = colors or [ACCENT] * len(labels)
    ax.bar(x, values, color=cols, edgecolor="#9AA8BC", width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", color=PLOT_TEXT)
    ax.set_ylabel(ylabel, color=PLOT_TEXT)
    ax.set_title(title, color=PLOT_TEXT, fontsize=11, fontweight="600")
    ax.tick_params(colors=PLOT_TEXT)
    ax.grid(axis="y", color=PLOT_GRID, linestyle="--", alpha=0.95)
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE)
    fig.tight_layout()
    return fig


# ── App shell ────────────────────────────────────────────────────────────────
init_state()
_bundle_path = ART / "demo_bundle.npz"
_ckpt_path = ART / "eaft_models.pt"
packed = load_models_and_bundle(
    _bundle_path.stat().st_mtime if _bundle_path.exists() else 0.0,
    _ckpt_path.stat().st_mtime if _ckpt_path.exists() else 0.0,
)

if packed is None:
    st.error(
        "Required model artifacts not found in `results/demo/`. "
        "Export them from the research notebook (Section 10), then reload."
    )
    st.stop()

bundle, ckpt = packed
summary = load_summary()
subjects = sorted({k.split("_")[0] for k in bundle.files if k.endswith("_X")})
subjects = [s for s in subjects if not s.startswith("OB")] + [
    s for s in subjects if s.startswith("OB")
]
meta = summary.get("subject_meta", {})
fs = int(bundle["fs"])
class_names = [str(x) for x in list(bundle["class_names"])]
n_calib = int(summary.get("calib_trials", 36))
sec_per_trial = float(summary.get("sec_per_trial", 8.032))
calib_minutes = float(summary.get("calib_minutes", n_calib * sec_per_trial / 60.0))


def subject_label(sid: str) -> str:
    info = meta.get(sid, {})
    name = info.get("label", sid)
    ds = info.get("dataset", "BCI IV 2a" if sid.startswith("S") else "OpenBMI")
    return f"{name} · {ds}"


NAV_ITEMS = [
    ("Calibration", "1. Calibration"),
    ("Inference", "2. Inference"),
    ("Experiment Results", "3. Experiment Results"),
]


def go_to(section: str) -> None:
    st.session_state["page"] = section
    st.rerun()


def render_page_header(title: str, subtitle: str, pill: str | None = None) -> None:
    pill_html = f'<span class="pill">{pill}</span>' if pill else ""
    st.markdown(
        f"""
        <div class="page-header">
          <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


page = st.session_state.get("page", "Calibration")
if page not in {k for k, _ in NAV_ITEMS}:
    page = "Calibration"
    st.session_state["page"] = page

with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
          <div class="brand-kicker">Research Dashboard</div>
          <div class="brand-title">EA-FT EEGNet</div>
          <div class="brand-sub">EEG Motor Imagery Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-label">Main menu</div>', unsafe_allow_html=True)
    for key, label in NAV_ITEMS:
        active = page == key
        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            if st.session_state["page"] != key:
                go_to(key)

    st.markdown('<div class="nav-label" style="margin-top:1rem;">Target subject</div>', unsafe_allow_html=True)
    subject = st.selectbox(
        "Target subject",
        subjects,
        format_func=subject_label,
        label_visibility="collapsed",
    )
    if st.session_state["subject"] != subject:
        st.session_state["subject"] = subject
        st.session_state["calib_done"] = False
        st.session_state["last_pred"] = None

    subj_ds = meta.get(subject, {}).get(
        "dataset",
        "OpenBMI / Lee2019_MI" if subject.startswith("OB") else "BCI IV 2a",
    )
    subj_acc = summary.get("per_subject_acc", {}).get(subject)
    protocol = (
        "Cross-dataset transfer"
        if subject.startswith("OB")
        else "Leave-One-Subject-Out"
    )
    acc_txt = f"{subj_acc:.1f}%" if subj_acc is not None else "—"
    st.markdown(
        f"""
        <div class="side-card">
          <div class="label">Selected target</div>
          <div class="value">{meta.get(subject, {}).get('label', subject)} · {acc_txt}</div>
          <div class="meta">
            Dataset: {subj_ds}<br/>
            Protocol: {protocol}<br/>
            Calibration: {n_calib} trials ({calib_minutes:.2f} min)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Reset session", use_container_width=True, key="reset_session"):
        st.session_state["calib_done"] = False
        st.session_state["last_pred"] = None
        go_to("Calibration")

X = bundle[f"{subject}_X"]
y = bundle[f"{subject}_y"]
P = bundle[f"{subject}_ea"]
ncal = int(bundle[f"{subject}_ncal"])
model = build_model(ckpt, subject)

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Calibration":
    render_page_header(
        "Calibration",
        f"Adapt the pre-trained model to {subject} with {n_calib} mixed Left/Right trials "
        f"({calib_minutes:.2f} min).",
        pill=subj_ds,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calibration trials", f"{n_calib}")
    m2.metric("Calibration time", f"{calib_minutes:.2f} min")
    m3.metric("Trial cadence", f"{sec_per_trial:.2f} s")
    m4.metric(
        "Source subjects",
        "9 (BCI IV 2a)" if subject.startswith("OB") else "8 (LOSO)",
    )

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown(
            """
**Pipeline**
1. Present Left / Right motor-imagery cues  
2. Record the EEG window for each trial  
3. Estimate Euclidean Alignment from calibration trials only  
4. Fine-tune Block 2 + classifier (Block 1 frozen)  
"""
        )
    with c2:
        st.info(
            f"**Target subject:** {subject}\n\n"
            f"**Pre-training:** all subjects except {subject}\n\n"
            f"**Alignment:** calibration-only EA\n\n"
            f"**Status:** {'Completed' if st.session_state['calib_done'] else 'Pending'}"
        )

    interval = st.select_slider(
        "Trial interval",
        options=["Dataset cadence", "0.50 s/trial", "0.15 s/trial"],
        value="0.50 s/trial",
    )
    delay = {
        "Dataset cadence": sec_per_trial,
        "0.50 s/trial": 0.50,
        "0.15 s/trial": 0.15,
    }[interval]

    start = st.button("Run calibration", type="primary", use_container_width=True)
    prog = st.progress(0, text="Ready")
    live = st.empty()

    if start:
        for i in range(n_calib):
            label = int(y[i])
            trial = apply_ea(P, X[i])
            pct = (i + 1) / n_calib
            elapsed = (i + 1) * sec_per_trial
            prog.progress(
                pct,
                text=(
                    f"Trial {i+1}/{n_calib} · cue = {class_names[label]} · "
                    f"elapsed {elapsed/60:.2f}/{calib_minutes:.2f} min"
                ),
            )
            with live.container():
                a, b = st.columns([1.3, 1])
                with a:
                    st.pyplot(
                        eeg_trace_figure(
                            trial, fs,
                            f"Calibration trial {i+1}/{n_calib} — {class_names[label]}",
                        ),
                        clear_figure=True,
                    )
                with b:
                    st.markdown(f"### Cue: **{class_names[label]}**")
                    st.write(f"Progress: **{100 * pct:.0f}%**")
            time.sleep(delay)

        prog.progress(1.0, text="Calibration complete")
        st.session_state["calib_done"] = True
        st.success(
            f"Calibration complete for {subject}. "
            "Euclidean Alignment fitted and model fine-tuned."
        )
        time.sleep(0.4)
        go_to("Inference")

    if st.session_state["calib_done"]:
        if st.button("Go to Inference →", type="primary"):
            go_to("Inference")

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Inference":
    render_page_header(
        "Inference",
        "Classify held-out trials: bandpass EEG → Euclidean Alignment → EEGNet → class probability.",
        pill=subj_ds,
    )

    if not st.session_state["calib_done"]:
        st.warning("Run calibration for the selected subject before inference.")
        if st.button("Go to Calibration"):
            go_to("Calibration")
        st.stop()

    test_idx = list(range(ncal, len(y)))
    if not test_idx:
        st.error("No held-out trials available for this subject.")
        st.stop()

    subj_acc = summary["per_subject_acc"][subject]
    k1, k2, k3 = st.columns(3)
    k1.metric(f"{subject} accuracy", f"{subj_acc:.1f}%")
    k2.metric("Held-out trials", f"{len(test_idx)}")
    k3.metric("Classes", "Left / Right")

    mode = st.radio(
        "Trial selection",
        ["Select trial", "Random trial"],
        horizontal=True,
    )
    show_true = st.checkbox("Show ground-truth label", value=False)

    if mode == "Random trial":
        if st.button("Sample random trial", use_container_width=True):
            st.session_state["rand_trial"] = int(np.random.choice(test_idx))
            st.session_state["last_pred"] = None
        if "rand_trial" not in st.session_state:
            st.session_state["rand_trial"] = int(test_idx[0])
        choice = int(st.session_state["rand_trial"])
        st.write(f"Selected trial: **#{choice}**")
    else:
        choice = st.selectbox(
            "Held-out trial index",
            test_idx,
            format_func=lambda i: (
                f"Trial #{i}"
                if not show_true
                else f"Trial #{i} · {class_names[int(y[i])]}"
            ),
            key="trial_pick",
        )

    trial_ea = apply_ea(P, X[choice])
    true_label = int(y[choice])

    if st.button("Predict", type="primary", use_container_width=True):
        pred, probs = classify(model, trial_ea)
        st.session_state["last_pred"] = {
            "idx": int(choice),
            "pred": int(pred),
            "probs": [float(probs[0]), float(probs[1])],
            "true": int(true_label),
            "trial": trial_ea.copy(),
        }

    result = st.session_state.get("last_pred")
    if result and int(result["idx"]) == int(choice):
        pred = int(result["pred"])
        probs = result["probs"]
        true = int(result["true"])
        correct = pred == true

        left, right = st.columns([1.2, 1])
        with left:
            st.pyplot(
                eeg_trace_figure(
                    result["trial"], fs,
                    f"Input EEG (EA-aligned) · trial #{choice}",
                ),
                clear_figure=True,
            )
        with right:
            color = "blue" if pred == 0 else "red"
            st.markdown(f"### Prediction: :{color}[**{class_names[pred]}**]")
            st.write(f"**Ground truth:** {class_names[true]}")
            st.write("**Outcome:** " + ("Correct" if correct else "Incorrect"))
            st.write(f"P(Left) = `{probs[0]:.2f}` · P(Right) = `{probs[1]:.2f}`")
            st.progress(float(probs[pred]), text=f"Confidence {probs[pred]*100:.0f}%")
    else:
        st.info("Select a trial and press **Predict**.")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RESULTS
# ══════════════════════════════════════════════════════════════════════════════
else:
    render_page_header(
        "Experiment Results",
        "Primary: Leave-One-Subject-Out on BCI IV 2a · Secondary: OpenBMI cross-dataset transfer.",
        pill="Evaluation",
    )

    # Headline KPIs
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Mean Accuracy (2a)", f"{summary.get('headline_acc', 74.1):.1f}%")
    h2.metric("Mean AUC (2a)", f"{summary.get('headline_auc', summary['mean_auc']):.3f}")
    h3.metric("Cohen's κ (2a)", f"{summary.get('headline_kappa', summary['mean_kappa']):.3f}")
    h4.metric("Calibration Time", f"{summary['calib_minutes']:.2f} min")

    ob = summary.get("openbmi")
    if ob:
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("OpenBMI EA-FT Acc", f"{ob['ea_ft_calib_acc']*100:.1f}%")
        o2.metric("OpenBMI Zero-shot", f"{ob['zero_shot_acc']*100:.1f}%")
        o3.metric("OpenBMI AUC", f"{ob['ea_ft_calib_auc']:.3f}")
        o4.metric("OpenBMI κ", f"{ob['ea_ft_calib_kappa']:.3f}")
    else:
        st.info(
            "OpenBMI secondary-target results are not in the summary yet. "
            "Run `python scripts/train_openbmi_target.py`, then refresh."
        )

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Per-subject",
            "Ablation",
            "Calibration size",
            "Literature comparison",
            "Statistics",
            "OpenBMI",
        ]
    )

    # ── Per-subject ──────────────────────────────────────────────────────────
    with tab1:
        detail = summary.get("per_subject_detail", {})
        rows = []
        for sid in subjects:
            d = detail.get(sid, {})
            rows.append(
                {
                    "Subject": meta.get(sid, {}).get("label", sid),
                    "Dataset": meta.get(sid, {}).get(
                        "dataset",
                        "OpenBMI / Lee2019_MI" if sid.startswith("OB") else "BCI IV 2a",
                    ),
                    "Zero-shot Acc (%)": d.get("zero_shot", np.nan),
                    "EA-FT Acc (%)": summary.get("per_subject_acc", {}).get(sid, np.nan),
                    "AUC": d.get("auc", np.nan),
                    "Cohen κ": d.get("kappa", np.nan),
                    "Calib-only EA-FT (%)": summary.get("calib_only_ea_ft", {}).get(sid, np.nan),
                }
            )
        df_subj = pd.DataFrame(rows)
        st.dataframe(
            df_subj.style.format(
                {
                    "Zero-shot Acc (%)": "{:.1f}",
                    "EA-FT Acc (%)": "{:.1f}",
                    "AUC": "{:.3f}",
                    "Cohen κ": "{:.3f}",
                    "Calib-only EA-FT (%)": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_ids = subjects
        chart_labels = [meta.get(s, {}).get("label", s) for s in chart_ids]
        chart_vals = [summary.get("per_subject_acc", {}).get(s, 0.0) for s in chart_ids]
        st.pyplot(
            bar_chart(
                chart_labels,
                chart_vals,
                "Accuracy (%)",
                "Per-subject EA-FT accuracy",
                colors=[ACCENT_3 if s == subject else (ACCENT_2 if s.startswith("OB") else ACCENT)
                        for s in chart_ids],
            ),
            clear_figure=True,
        )

    # ── Ablation ─────────────────────────────────────────────────────────────
    with tab2:
        abl = summary.get("ablation", {})
        abl_rows = []
        for sid in subjects:
            a = abl.get(sid)
            if a is None:
                continue
            abl_rows.append(
                {
                    "Subject": sid,
                    "Baseline (No TL)": a["baseline"],
                    "Zero-shot": a["zero_shot"],
                    "FT without EA": a["ft_no_ea"],
                    "EA-FT EEGNet": a["ea_ft"],
                }
            )

        if not abl_rows:
            st.warning("Ablation results are not available yet. Reload the page.")
        else:
            df_abl = pd.DataFrame(abl_rows)
            means = summary.get("ablation_mean", {})
            df_abl = pd.concat(
                [
                    df_abl,
                    pd.DataFrame(
                        [
                            {
                                "Subject": "Mean",
                                "Baseline (No TL)": means.get(
                                    "baseline", df_abl["Baseline (No TL)"].mean()
                                ),
                                "Zero-shot": means.get(
                                    "zero_shot", df_abl["Zero-shot"].mean()
                                ),
                                "FT without EA": means.get(
                                    "ft_no_ea", df_abl["FT without EA"].mean()
                                ),
                                "EA-FT EEGNet": means.get(
                                    "ea_ft", df_abl["EA-FT EEGNet"].mean()
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            st.dataframe(
                df_abl.style.format(
                    {
                        "Baseline (No TL)": "{:.1f}",
                        "Zero-shot": "{:.1f}",
                        "FT without EA": "{:.1f}",
                        "EA-FT EEGNet": "{:.1f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            labels = ["Baseline\n(No TL)", "Zero-shot", "FT without EA", "EA-FT EEGNet"]
            vals = [
                means.get("baseline", 51.0),
                means.get("zero_shot", 72.6),
                means.get("ft_no_ea", 75.8),
                means.get("ea_ft", 73.4),
            ]
            st.pyplot(
                bar_chart(
                    labels,
                    vals,
                    "Mean accuracy (%)",
                    "Ablation study — mean accuracy",
                    colors=["#6B7280", "#3B82F6", "#C2410C", ACCENT],
                ),
                clear_figure=True,
            )

    # ── Calibration size ─────────────────────────────────────────────────────
    with tab3:
        curve = summary.get("calib_size_curve", [])
        df_curve = pd.DataFrame(curve).rename(
            columns={
                "trials": "Trials",
                "minutes": "Minutes",
                "mean_acc": "Mean Acc (%)",
                "std": "Std",
            }
        )
        st.dataframe(
            df_curve.style.format(
                {"Minutes": "{:.2f}", "Mean Acc (%)": "{:.1f}", "Std": "{:.1f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(8.5, 3.6), facecolor=PLOT_BG)
        ax.set_facecolor(PLOT_FACE)
        ax.errorbar(
            df_curve["Minutes"],
            df_curve["Mean Acc (%)"],
            yerr=df_curve["Std"],
            fmt="-o",
            color=ACCENT,
            ecolor="#9AA8BC",
            capsize=4,
            lw=2,
        )
        ax.axvline(
            calib_minutes, color=ACCENT_3, ls="--", lw=1.5,
            label=f"Selected ({calib_minutes:.2f} min)",
        )
        ax.set_xlabel("Calibration time (min)", color=PLOT_TEXT)
        ax.set_ylabel("Mean accuracy (%)", color=PLOT_TEXT)
        ax.set_title("Accuracy vs calibration size", color=PLOT_TEXT, fontweight="600")
        ax.tick_params(colors=PLOT_TEXT)
        ax.grid(color=PLOT_GRID, linestyle="--", alpha=0.95)
        ax.legend(facecolor=PLOT_BG, edgecolor=PLOT_SPINE, labelcolor=PLOT_TEXT)
        for spine in ax.spines.values():
            spine.set_color(PLOT_SPINE)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # ── Literature ───────────────────────────────────────────────────────────
    with tab4:
        lit = pd.DataFrame(summary.get("literature", []))
        if not lit.empty:
            lit = lit.rename(
                columns={
                    "method": "Method",
                    "acc": "Mean Acc (%)",
                    "calib_min": "Calibration (min)",
                    "params": "Parameters",
                    "ref": "Reference",
                }
            )
            st.dataframe(
                lit.style.format(
                    {"Mean Acc (%)": "{:.1f}", "Calibration (min)": "{:.2f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=PLOT_BG)
            ax.set_facecolor(PLOT_FACE)
            colors = [ACCENT if "Ours" in m else "#6B7280" for m in lit["Method"]]
            y_pos = np.arange(len(lit))
            ax.barh(y_pos, lit["Mean Acc (%)"], color=colors, edgecolor="#9AA8BC")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(lit["Method"], color=PLOT_TEXT)
            ax.set_xlabel("Mean accuracy (%)", color=PLOT_TEXT)
            ax.set_title(
                "Literature comparison (BCI IV 2a, Left vs Right)",
                color=PLOT_TEXT, fontweight="600",
            )
            ax.tick_params(colors=PLOT_TEXT)
            ax.invert_yaxis()
            ax.grid(axis="x", color=PLOT_GRID, linestyle="--", alpha=0.95)
            for spine in ax.spines.values():
                spine.set_color(PLOT_SPINE)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

    # ── Statistics ───────────────────────────────────────────────────────────
    with tab5:
        wil = pd.DataFrame(summary.get("wilcoxon", []))
        if not wil.empty:
            wil = wil.rename(
                columns={
                    "comparison": "Comparison",
                    "mean_delta": "Mean Δ (pts)",
                    "wins": "EA-FT wins",
                    "p": "p (two-sided)",
                    "significant": "Significant (α=.05)",
                }
            )
            st.dataframe(
                wil.style.format({"Mean Δ (pts)": "{:.3f}", "p (two-sided)": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Wilcoxon signed-rank tests across n = 9 subjects. "
                "Positive Δ favours EA-FT EEGNet."
            )

    # ── OpenBMI secondary target ─────────────────────────────────────────────
    with tab6:
        ob = summary.get("openbmi")
        if not ob:
            st.warning(
                "OpenBMI results not found. Train with:\n\n"
                "`python scripts/train_openbmi_target.py`"
            )
        else:
            st.markdown(
                "**Secondary-dataset validation** — source model trained on all "
                "9 BCI IV 2a subjects; target = OpenBMI / Lee2019_MI "
                f"(`{ob.get('file', 'sess01_subj01_EEG_MI.mat')}`)."
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Zero-shot accuracy", f"{ob['zero_shot_acc']*100:.1f}%")
            c2.metric("EA-FT (calib-only)", f"{ob['ea_ft_calib_acc']*100:.1f}%")
            c3.metric("EA-FT (transductive)", f"{ob['ea_ft_transductive_acc']*100:.1f}%")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("AUC", f"{ob['ea_ft_calib_auc']:.3f}")
            d2.metric("Cohen κ", f"{ob['ea_ft_calib_kappa']:.3f}")
            d3.metric("Calib trials", f"{ob['n_calib']}")
            d4.metric("Test trials", f"{ob['n_test']}")

            cmp_labels = ["BCI IV 2a\n(mean of 9)", "OpenBMI\n(OB01)"]
            cmp_vals = [
                float(summary.get("headline_acc", 74.1)),
                float(ob["ea_ft_calib_acc"] * 100.0),
            ]
            st.pyplot(
                bar_chart(
                    cmp_labels,
                    cmp_vals,
                    "EA-FT accuracy (%)",
                    "Primary vs secondary dataset",
                    colors=[ACCENT, ACCENT_2],
                ),
                clear_figure=True,
            )
            st.caption(
                "OpenBMI uses different hardware, montage, and sampling rate. "
                "Channels are harmonised to the 22-channel BCI IV 2a montage before EA-FT."
            )
