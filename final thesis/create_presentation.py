"""
Final Thesis Presentation — EA-FT EEGNet
Halwathurage H.O.M.B | 215526U
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import nsmap
from pptx.oxml import parse_xml
from copy import deepcopy
import os

# ── Palette (academic, matches friend's navy/red style) ──────────────
NAVY = RGBColor(0x0D, 0x1B, 0x2A)
NAVY2 = RGBColor(0x1B, 0x3A, 0x5F)
RED = RGBColor(0xC0, 0x39, 0x2B)
ACCENT = RGBColor(0x1A, 0x6B, 0x8A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF5, 0xF6, 0xF8)
GRAY = RGBColor(0x4A, 0x55, 0x68)
DARK = RGBColor(0x1A, 0x1A, 0x2E)
GREEN = RGBColor(0x1E, 0x7A, 0x46)
ORANGE = RGBColor(0xD3, 0x54, 0x00)

W, H = Inches(13.333), Inches(7.5)  # widescreen 16:9


def set_run(run, size=18, bold=False, color=DARK, font="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def add_bg(slide, color=LIGHT):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_text(slide, l, t, w, h, text, size=18, bold=False, color=DARK,
             align=PP_ALIGN.LEFT, font="Calibri", anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    try:
        tf._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr",
                                         MSO_ANCHOR.BOTTOM: "b"}[anchor])
    except Exception:
        pass
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_run(run, size, bold, color, font)
    return box


def add_para(tf, text, size=16, bold=False, color=DARK, align=PP_ALIGN.LEFT,
             space_before=6, space_after=4, font="Calibri", level=0):
    p = tf.add_paragraph()
    p.alignment = align
    p.level = level
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    set_run(run, size, bold, color, font)
    return p


def section_header(slide, title, subtitle=None):
    """Top navy bar + title for content slides."""
    add_rect(slide, 0, 0, W, Inches(0.9), NAVY)
    add_rect(slide, 0, Inches(0.9), W, Inches(0.06), RED)
    add_text(slide, Inches(0.5), Inches(0.18), Inches(12), Inches(0.55),
             title, size=28, bold=True, color=WHITE, font="Calibri")
    if subtitle:
        add_text(slide, Inches(0.5), Inches(1.1), Inches(12), Inches(0.4),
                 subtitle, size=14, bold=False, color=GRAY)


def footer(slide, page, total=26):
    add_text(slide, Inches(0.4), Inches(7.1), Inches(8), Inches(0.3),
             "EA-FT EEGNet  |  Halwathurage H.O.M.B  |  215526U",
             size=10, color=GRAY)
    add_text(slide, Inches(11.5), Inches(7.1), Inches(1.5), Inches(0.3),
             f"{page} / {total}", size=10, color=GRAY, align=PP_ALIGN.RIGHT)


def bullet_block(slide, l, t, w, h, items, size=15, color=DARK):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(4)
        p.space_after = Pt(4)
        p.level = 0
        run = p.add_run()
        run.text = "▸  " + item
        set_run(run, size, False, color)
    return box


def card(slide, l, t, w, h, title, body_lines, title_color=NAVY, accent=RED):
    shape = add_rect(slide, l, t, w, h, WHITE)
    # subtle left accent
    add_rect(slide, l, t, Inches(0.08), h, accent)
    add_text(slide, l + Inches(0.2), t + Inches(0.12), w - Inches(0.3), Inches(0.35),
             title, size=15, bold=True, color=title_color)
    box = slide.shapes.add_textbox(l + Inches(0.2), t + Inches(0.48),
                                   w - Inches(0.3), h - Inches(0.55))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(body_lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(2)
        p.space_after = Pt(2)
        run = p.add_run()
        run.text = line
        set_run(run, 12, False, GRAY)
    return shape


def add_table(slide, data, l, t, w, h, col_widths=None, header=True):
    rows, cols = len(data), len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, l, t, w, h)
    table = table_shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = cw
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for run in p.runs:
                    if header and r == 0:
                        set_run(run, 11, True, WHITE)
                    else:
                        set_run(run, 11, False, DARK)
            # fill
            fill = cell.fill
            fill.solid()
            if header and r == 0:
                fill.fore_color.rgb = NAVY2
            elif r % 2 == 0:
                fill.fore_color.rgb = RGBColor(0xEE, 0xF2, 0xF7)
            else:
                fill.fore_color.rgb = WHITE
    return table_shape


# ══════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width = W
prs.slide_height = H
blank = prs.slide_layouts[6]
TOTAL = 26
page = 0


def new_slide():
    global page
    page += 1
    s = prs.slides.add_slide(blank)
    add_bg(s, LIGHT)
    return s


# ── 1. TITLE ─────────────────────────────────────────────────────────
s = new_slide()
add_rect(s, 0, 0, W, H, NAVY)
add_rect(s, 0, Inches(5.9), W, Inches(1.6), NAVY2)
add_rect(s, 0, Inches(5.84), W, Inches(0.08), RED)

add_text(s, Inches(0.8), Inches(1.3), Inches(11.5), Inches(0.4),
         "CM4900 — Research Project in Artificial Intelligence",
         size=16, color=RGBColor(0xA0, 0xB4, 0xC8), align=PP_ALIGN.CENTER)
add_text(s, Inches(0.8), Inches(1.9), Inches(11.5), Inches(1.4),
         "Cross-Subject Transfer Learning Based Compact CNN\nfor EEG-Based Motor Imagery Classification",
         size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER, font="Calibri")
add_text(s, Inches(0.8), Inches(3.5), Inches(11.5), Inches(0.4),
         "EA-FT EEGNet Framework  ·  Final Thesis Presentation",
         size=18, color=RGBColor(0xE8, 0xC4, 0xC0), align=PP_ALIGN.CENTER)

add_text(s, Inches(0.8), Inches(6.1), Inches(11.5), Inches(1.2),
         "Halwathurage H.O.M.B  |  215526U\n"
         "Supervisor: Dr. Thilini Piyathilake\n"
         "Department of Computational Mathematics  ·  Faculty of Information Technology\n"
         "University of Moratuwa  ·  BSc Hons in Artificial Intelligence  ·  Batch 21  ·  August 2026",
         size=13, color=RGBColor(0xC8, 0xD4, 0xE0), align=PP_ALIGN.CENTER)


# ── 2. OUTLINE ───────────────────────────────────────────────────────
s = new_slide()
section_header(s, "Outline")
items = [
    ("1.", "Introduction", RED),
    ("2.", "Objectives", NAVY),
    ("3.", "Literature Review", RED),
    ("4.", "Research Gap", NAVY),
    ("5.", "Technology", RED),
    ("6.", "Novel Approach", NAVY),
    ("7.", "Design", RED),
    ("8.", "Implementation", NAVY),
    ("9.", "Evaluation", RED),
    ("10.", "Conclusion and Further Work", NAVY),
    ("11.", "References", RED),
]
# two columns
for i, (num, label, col) in enumerate(items):
    col_x = Inches(1.5) if i < 6 else Inches(7.2)
    row_y = Inches(1.4) + Inches(0.75) * (i % 6)
    add_text(s, col_x, row_y, Inches(0.6), Inches(0.5), num,
             size=22, bold=True, color=col)
    add_text(s, col_x + Inches(0.7), row_y, Inches(4.5), Inches(0.5), label,
             size=22, bold=True, color=col)
footer(s, page, TOTAL)


# ── 3. INTRODUCTION ──────────────────────────────────────────────────
s = new_slide()
section_header(s, "Introduction", "The Calibration Bottleneck in Motor Imagery BCIs")

card(s, Inches(0.4), Inches(1.5), Inches(4.0), Inches(2.4),
     "Brain–Computer Interfaces",
     ["Direct pathway: brain → device",
      "No neuromuscular routes needed",
      "Critical for ALS, SCI, stroke",
      "EEG: non-invasive, portable, real-time"],
     accent=ACCENT)

card(s, Inches(4.6), Inches(1.5), Inches(4.0), Inches(2.4),
     "Motor Imagery (MI)",
     ["Mental rehearsal of movement",
      "Modulates mu (8–13 Hz) & beta (13–30 Hz)",
      "Event-Related Desynchronization (ERD)",
      "Endogenous — no external stimulus"],
     accent=RED)

card(s, Inches(8.8), Inches(1.5), Inches(4.0), Inches(2.4),
     "The Core Problem",
     ["Extreme inter-subject variability",
      "Marginal distribution shift",
      "20–30 min calibration per user",
      "Blocks real-world BCI deployment"],
     accent=ORANGE)

add_rect(s, Inches(0.4), Inches(4.2), Inches(12.5), Inches(2.4), WHITE)
add_rect(s, Inches(0.4), Inches(4.2), Inches(0.1), Inches(2.4), NAVY)
add_text(s, Inches(0.7), Inches(4.35), Inches(12), Inches(0.4),
         "Proposed Solution — EA-FT EEGNet", size=16, bold=True, color=NAVY)
bullet_block(s, Inches(0.7), Inches(4.8), Inches(12), Inches(1.6), [
    "Euclidean Alignment (EA): unsupervised input-level whitening → standardize spatial covariance across subjects",
    "Conservative Fine-Tuning (FT): freeze universal layers; adapt only user-specific layers with ~5 min calibration",
    "Goal: cut calibration from 20–30 min to under 5 min while keeping competitive decoding accuracy",
], size=14)
footer(s, page, TOTAL)


# ── 4. OBJECTIVES ────────────────────────────────────────────────────
s = new_slide()
section_header(s, "Objectives")

objs = [
    ("O1", "Review", "Conduct a comprehensive review of EEG-based MI classification techniques and deep learning architectures."),
    ("O2", "Analyse", "Investigate theoretical frameworks and challenges of inter-subject variability and calibration in EEG-BCIs."),
    ("O3", "Develop", "Propose and implement EA-FT EEGNet — Euclidean Alignment + conservative fine-tuning for cross-subject transfer."),
    ("O4", "Evaluate", "Experimentally evaluate on BCI Competition IV 2a to show reduced calibration time with robust accuracy."),
]
for i, (tag, verb, text) in enumerate(objs):
    y = Inches(1.35) + Inches(1.3) * i
    add_rect(s, Inches(0.5), y, Inches(12.3), Inches(1.15), WHITE)
    add_rect(s, Inches(0.5), y, Inches(1.3), Inches(1.15), NAVY if i % 2 == 0 else RED)
    add_text(s, Inches(0.6), y + Inches(0.25), Inches(1.1), Inches(0.35),
             tag, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.6), y + Inches(0.65), Inches(1.1), Inches(0.3),
             verb, size=11, color=RGBColor(0xDD, 0xDD, 0xDD), align=PP_ALIGN.CENTER)
    add_text(s, Inches(2.0), y + Inches(0.3), Inches(10.5), Inches(0.7),
             text, size=15, color=DARK)
footer(s, page, TOTAL)


# ── 5. LITERATURE REVIEW ─────────────────────────────────────────────
s = new_slide()
section_header(s, "Literature Review", "Evolution of EEG-MI Decoding & Cross-Subject Transfer")

timeline = [
    ("1920s", "Berger", "First scalp EEG recordings"),
    ("1991", "Wolpaw", "SMR cursor control proof-of-concept"),
    ("2017", "DeepConvNet", "End-to-end CNN (~170k params)"),
    ("2018", "EEGNet", "Compact CNN (~2k params)"),
    ("2020", "He & Wu", "Euclidean Alignment for BCI TL"),
]
for i, (year, name, desc) in enumerate(timeline):
    x = Inches(0.4) + Inches(2.55) * i
    add_rect(s, x, Inches(1.4), Inches(2.4), Inches(1.8), WHITE)
    add_rect(s, x, Inches(1.4), Inches(2.4), Inches(0.08), RED if i % 2 else ACCENT)
    add_text(s, x + Inches(0.1), Inches(1.55), Inches(2.2), Inches(0.35),
             year, size=16, bold=True, color=RED if i % 2 else ACCENT, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.1), Inches(2.0), Inches(2.2), Inches(0.35),
             name, size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.1), Inches(2.45), Inches(2.2), Inches(0.55),
             desc, size=11, color=GRAY, align=PP_ALIGN.CENTER)

add_text(s, Inches(0.5), Inches(3.5), Inches(12), Inches(0.35),
         "Key Challenges Identified", size=16, bold=True, color=NAVY)

challenges = [
    ("Low SNR", "Skull attenuation; EOG/EMG artifacts"),
    ("Data scarcity", "Labeled EEG is expensive to collect"),
    ("Inter-subject shift", "Anatomy, impedance, baselines differ"),
    ("Calibration bottleneck", "20–30 min setup before use"),
    ("Catastrophic forgetting", "FT overwrites universal rhythms"),
]
for i, (t, d) in enumerate(challenges):
    x = Inches(0.4) + Inches(2.55) * i
    card(s, x, Inches(3.95), Inches(2.4), Inches(2.5), t, [d],
         accent=RED if i % 2 else NAVY)
footer(s, page, TOTAL)


# ── 6. LITERATURE — COMPARISON ───────────────────────────────────────
s = new_slide()
section_header(s, "Literature Review", "Comparison of Existing Approaches")

data = [
    ["Approach", "Strength", "Limitation"],
    ["CSP + LDA", "Light, interpretable", "Manual features; poor under shift"],
    ["DeepConvNet / ShallowConvNet", "Strong end-to-end learning", "170k / 44k params; long calibration"],
    ["EEGNet (Lawhern 2018)", "~2k params; multi-paradigm", "Still needs subject-specific training"],
    ["Riemannian / MDM", "Drift-resistant geometry", "Heavy; hard for real-time"],
    ["Domain Adaptation / TL", "Reuses source knowledge", "Fails without input alignment"],
]
add_table(s, data, Inches(0.5), Inches(1.4), Inches(12.3), Inches(4.2),
          col_widths=[Inches(3.5), Inches(4.2), Inches(4.6)])

add_text(s, Inches(0.5), Inches(5.9), Inches(12.3), Inches(0.8),
         "Insight: Complex architectures and downstream DA are well studied — but lightweight "
         "input-space alignment combined with conservative weight adaptation remains underexplored.",
         size=13, color=GRAY)
footer(s, page, TOTAL)


# ── 7. RESEARCH GAP ──────────────────────────────────────────────────
s = new_slide()
section_header(s, "Research Gap", "What is missing — and what this work fills")

# left existing
add_rect(s, Inches(0.4), Inches(1.4), Inches(5.9), Inches(5.0), WHITE)
add_rect(s, Inches(0.4), Inches(1.4), Inches(5.9), Inches(0.55), RED)
add_text(s, Inches(0.6), Inches(1.5), Inches(5.5), Inches(0.4),
         "EXISTING PRACTICE", size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
bullet_block(s, Inches(0.7), Inches(2.2), Inches(5.4), Inches(3.8), [
    "Focus on heavy CNNs / RNNs or complex DA",
    "Raw heterogeneous EEG fed directly to networks",
    "Fine-tuning forced to 'unlearn' source statistics",
    "Riemannian alignment: powerful but heavy",
    "No unified lightweight input + weight adaptation",
    "Calibration remains 15–30 minutes",
], size=14)

# right needed
add_rect(s, Inches(6.9), Inches(1.4), Inches(5.9), Inches(5.0), WHITE)
add_rect(s, Inches(6.9), Inches(1.4), Inches(5.9), Inches(0.55), GREEN)
add_text(s, Inches(7.1), Inches(1.5), Inches(5.5), Inches(0.4),
         "RESEARCH NEED", size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
bullet_block(s, Inches(7.2), Inches(2.2), Inches(5.4), Inches(3.8), [
    "Standardize EEG input space before CNN",
    "Preserve universal mu/beta filters (freeze)",
    "Adapt only user-specific layers (few-shot)",
    "Computationally light for real-time BCI",
    "Leakage-free, deployable evaluation protocol",
    "Calibration under 5 minutes",
], size=14)
footer(s, page, TOTAL)


# ── 8. TECHNOLOGY ────────────────────────────────────────────────────
s = new_slide()
section_header(s, "Technology", "Theoretical Foundations of the Extension")

techs = [
    ("Sensorimotor Rhythms", "MI generates ERD in mu (8–13 Hz) and beta (13–30 Hz) over contralateral motor cortex (C3/C4)."),
    ("EEGNet Architecture", "Temporal + depthwise + separable convolutions ≈ 2,450 parameters — efficient, low overfitting risk."),
    ("Euclidean Alignment", "Whitening: R̄ = (1/n) Σ XᵢXᵢᵀ → X̃ = R̄⁻¹/² X  so mean covariance → Identity."),
    ("Transfer Learning", "Pre-train on source subjects (LOSO); adapt target with few labeled trials."),
    ("Layer Freezing", "Block 1 frozen (912 params) = universal filters; Block 2 + classifier (1,538) = user-specific."),
    ("Leakage-Free Protocol", "Model selection on calibration validation split only; test scored once."),
]
for i, (title, body) in enumerate(techs):
    r, c = divmod(i, 3)
    x = Inches(0.4) + Inches(4.25) * c
    y = Inches(1.35) + Inches(2.7) * r
    card(s, x, y, Inches(4.1), Inches(2.45), title, [body],
         accent=RED if i % 2 else ACCENT)
footer(s, page, TOTAL)


# ── 9. NOVEL APPROACH ────────────────────────────────────────────────
s = new_slide()
section_header(s, "Novel Approach", "EA-FT EEGNet — Dual-Stage Adaptation")

add_text(s, Inches(0.5), Inches(1.25), Inches(12.3), Inches(0.6),
         "Hypothesis: Input-level alignment is a prerequisite for efficient few-shot transfer — "
         "standardize first, then adapt conservatively.",
         size=14, bold=True, color=NAVY)

# pipeline boxes
stages = [
    ("1. Acquire", "Raw multi-channel\nEEG (22 ch, 250 Hz)"),
    ("2. Clean", "Artifact reject\n+ 8–30 Hz filter"),
    ("3. Align (EA)", "Unsupervised\ncovariance whitening"),
    ("4. Pre-train", "EEGNet on source\nsubjects (LOSO)"),
    ("5. Fine-tune", "Freeze Block 1\nAdapt Block 2"),
    ("6. Predict", "Left vs Right\nmotor intent"),
]
for i, (title, body) in enumerate(stages):
    x = Inches(0.35) + Inches(2.15) * i
    add_rect(s, x, Inches(2.1), Inches(2.0), Inches(2.2), WHITE)
    add_rect(s, x, Inches(2.1), Inches(2.0), Inches(0.45), NAVY if i < 3 else RED)
    add_text(s, x + Inches(0.05), Inches(2.15), Inches(1.9), Inches(0.4),
             title, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.1), Inches(2.75), Inches(1.8), Inches(1.3),
             body, size=12, color=DARK, align=PP_ALIGN.CENTER)
    if i < 5:
        add_text(s, x + Inches(1.85), Inches(2.9), Inches(0.35), Inches(0.4),
                 "→", size=20, bold=True, color=RED, align=PP_ALIGN.CENTER)

# key properties
props = [
    ("Input-level", "EA removes\ncovariance shift"),
    ("Weight-level", "Conservative FT\nprevents forgetting"),
    ("Few-shot", "36 trials ≈\n4.8 minutes"),
    ("Compact", "2,450 params\n146 KB deploy"),
]
for i, (t, b) in enumerate(props):
    x = Inches(0.5) + Inches(3.2) * i
    add_rect(s, x, Inches(4.7), Inches(3.0), Inches(1.8), WHITE)
    add_rect(s, x, Inches(4.7), Inches(0.08), Inches(1.8), GREEN)
    add_text(s, x + Inches(0.2), Inches(4.85), Inches(2.7), Inches(0.35),
             t, size=14, bold=True, color=GREEN)
    add_text(s, x + Inches(0.2), Inches(5.3), Inches(2.7), Inches(1.0),
             b, size=13, color=GRAY)
footer(s, page, TOTAL)


# ── 10. DESIGN — ARCHITECTURE ────────────────────────────────────────
s = new_slide()
section_header(s, "Design", "Three-Zone System Architecture")

zones = [
    ("ZONE 1 — Preprocessing", NAVY, [
        "Variance-based artifact rejection (μ + 5σ)",
        "Zero-phase Butterworth bandpass 8–30 Hz",
        "Euclidean Alignment (calibration-only R̄)",
        "Output: whitened trial tensors",
    ]),
    ("ZONE 2 — ML Engine", ACCENT, [
        "EEGNet: F1=8, D=2, F2=16, kern=64",
        "Total parameters: 2,450",
        "LOSO pre-training (Adam, LR=1e-3)",
        "Source pool = 8 subjects → target held out",
    ]),
    ("ZONE 3 — Adaptation", RED, [
        "Freeze Block 1 (912 params)",
        "Train Block 2 + classifier (1,538)",
        "LR = 5e-5, early stop patience 10",
        "Val split of calibration (no test leak)",
    ]),
]
for i, (title, col, lines) in enumerate(zones):
    x = Inches(0.4) + Inches(4.25) * i
    add_rect(s, x, Inches(1.35), Inches(4.1), Inches(5.2), WHITE)
    add_rect(s, x, Inches(1.35), Inches(4.1), Inches(0.6), col)
    add_text(s, x + Inches(0.15), Inches(1.45), Inches(3.8), Inches(0.45),
             title, size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    bullet_block(s, x + Inches(0.25), Inches(2.2), Inches(3.7), Inches(4.0),
                 lines, size=14)
footer(s, page, TOTAL)


# ── 11. IMPLEMENTATION ───────────────────────────────────────────────
s = new_slide()
section_header(s, "Implementation", "Development Environment & Module Pipeline")

card(s, Inches(0.4), Inches(1.3), Inches(4.1), Inches(2.6),
     "Environment",
     ["Python 3.10 + PyTorch",
      "NumPy / SciPy / scikit-learn",
      "NVIDIA RTX 4050 GPU",
      "Seed = 42 (full reproducibility)",
      "BCI Competition IV 2a (.mat)"],
     accent=NAVY)

card(s, Inches(4.7), Inches(1.3), Inches(4.1), Inches(2.6),
     "Core Modules Built",
     ["Dataset parser (nested MATLAB structs)",
      "Artifact rejection + bandpass",
      "EA whitening (150 ms / subject)",
      "EEGNet + LOSO trainer",
      "Leakage-free fine-tuner"],
     accent=ACCENT)

card(s, Inches(9.0), Inches(1.3), Inches(3.9), Inches(2.6),
     "Rigor Measures",
     ["Calib-only EA (deployable)",
      "Val-split model selection",
      "True cadence: 8.03 s/trial",
      "3 ablation arms",
      "Wilcoxon signed-rank tests"],
     accent=RED)

add_text(s, Inches(0.5), Inches(4.2), Inches(12), Inches(0.35),
         "What Was Explicitly Built & Validated", size=15, bold=True, color=NAVY)

impl_items = [
    "Full end-to-end notebook pipeline (data → preprocess → pre-train → fine-tune → evaluate)",
    "Deployable EA: reference from calibration trials only — no test-set peeking",
    "Discovered & fixed test-set leakage in early protocol → all results regenerated",
    "Honest calibration time from measured inter-onset spacing (not analysis window)",
    "Ablation: Baseline / Zero-shot / FT-without-EA / EA-FT — same held-out trials",
    "Artifacts: trained models (146 KB), demo bundle, statistical tables",
]
bullet_block(s, Inches(0.5), Inches(4.6), Inches(12.3), Inches(2.2), impl_items, size=13)
footer(s, page, TOTAL)


# ── 12. DATASET ──────────────────────────────────────────────────────
s = new_slide()
section_header(s, "Evaluation — Dataset & Protocol", "BCI Competition IV Dataset 2a (Graz)")

meta = [
    ["Property", "Value"],
    ["Subjects", "9 (A01–A09)"],
    ["Channels", "22 EEG (+ 3 EOG discarded)"],
    ["Sampling rate", "250 Hz"],
    ["Trial window", "4.0 s (1,000 samples)"],
    ["Classes (raw)", "Left, Right, Feet, Tongue"],
    ["Target task", "Binary: Left vs Right hand"],
    ["Trials / subject", "288 → ~143 binary"],
    ["Artifact rejection", "0.5 % mean (12 of 2,592)"],
    ["Protocol", "Leave-One-Subject-Out"],
    ["Calibration", "36 trials ≈ 4.82 min"],
]
add_table(s, meta, Inches(0.4), Inches(1.3), Inches(6.2), Inches(5.4),
          col_widths=[Inches(2.8), Inches(3.4)])

add_rect(s, Inches(7.0), Inches(1.3), Inches(5.8), Inches(5.4), WHITE)
add_rect(s, Inches(7.0), Inches(1.3), Inches(5.8), Inches(0.5), NAVY)
add_text(s, Inches(7.2), Inches(1.38), Inches(5.4), Inches(0.4),
         "Leakage-Free Evaluation Rules", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
bullet_block(s, Inches(7.3), Inches(2.0), Inches(5.3), Inches(4.4), [
    "Target subject never in pre-training source pool",
    "EA reference from calibration only",
    "Fine-tune checkpoint selected on 25% calib validation",
    "Test trials scored exactly once",
    "Calibration time = n × 8.03 s (measured)",
    "Same test set across all ablation arms",
    "Metrics: Accuracy, AUC, Cohen's κ",
], size=13)
footer(s, page, TOTAL)


# ── 13. ALIGNMENT RESULTS ────────────────────────────────────────────
s = new_slide()
section_header(s, "Evaluation — Alignment Quality", "Does Euclidean Alignment actually whiten the data?")

data = [
    ["Subject", "‖R̄−I‖F Before", "After EA", "Reduction"],
    ["S01", "398.86", "4.686", "98.8%"],
    ["S02", "466.70", "4.686", "99.0%"],
    ["S03", "791.47", "4.686", "99.4%"],
    ["S04", "344.66", "4.686", "98.6%"],
    ["S05", "255.08", "4.686", "98.2%"],
    ["S06", "626.59", "4.686", "99.3%"],
    ["S07", "334.55", "4.686", "98.6%"],
    ["S08", "1310.53", "4.686", "99.6%"],
    ["S09", "1191.58", "4.686", "99.6%"],
    ["Mean", "635.56", "4.686", "99.0%"],
]
add_table(s, data, Inches(0.4), Inches(1.25), Inches(7.5), Inches(5.5),
          col_widths=[Inches(1.5), Inches(2.2), Inches(1.8), Inches(2.0)])

add_rect(s, Inches(8.2), Inches(1.25), Inches(4.7), Inches(5.5), WHITE)
add_rect(s, Inches(8.2), Inches(1.25), Inches(4.7), Inches(0.5), GREEN)
add_text(s, Inches(8.4), Inches(1.35), Inches(4.3), Inches(0.35),
         "Verified Completely", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
bullet_block(s, Inches(8.45), Inches(2.0), Inches(4.3), Inches(4.4), [
    "Mean covariance mismatch ↓ 99.0%",
    "Inter-subject spread: 5.14× → 1.00×",
    "All subjects collapse to same distance",
    "EA runtime: ~150 ms / subject",
    "Suitable for real-time deployment",
    "Geometric function: SUCCESS",
], size=13)
footer(s, page, TOTAL)


# ── 14. MAIN RESULTS ─────────────────────────────────────────────────
s = new_slide()
section_header(s, "Evaluation — Main Results", "EA-FT EEGNet after 36-trial (~4.8 min) calibration")

# big numbers
metrics = [
    ("74.1%", "Mean Accuracy", "Deployable (calib-only EA)"),
    ("0.809", "Mean AUC", "Threshold-independent"),
    ("0.471", "Mean Cohen's κ", "Chance-corrected"),
    ("4.82 min", "Calibration", "76–84% reduction"),
]
for i, (val, label, sub) in enumerate(metrics):
    x = Inches(0.4) + Inches(3.2) * i
    add_rect(s, x, Inches(1.3), Inches(3.05), Inches(2.0), WHITE)
    add_rect(s, x, Inches(1.3), Inches(3.05), Inches(0.08), RED if i == 0 else NAVY)
    add_text(s, x + Inches(0.1), Inches(1.45), Inches(2.85), Inches(0.7),
             val, size=32, bold=True, color=RED if i == 0 else NAVY, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.1), Inches(2.2), Inches(2.85), Inches(0.35),
             label, size=13, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.1), Inches(2.55), Inches(2.85), Inches(0.4),
             sub, size=11, color=GRAY, align=PP_ALIGN.CENTER)

data = [
    ["Subject", "Zero-Shot %", "EA-FT %", "Δ", "AUC", "κ"],
    ["S01", "61.7", "70.1", "+8.4", "0.814", "0.409"],
    ["S02", "66.7", "68.5", "+1.9", "0.720", "0.370"],
    ["S03", "94.4", "92.6", "−1.9", "0.980", "0.851"],
    ["S04", "69.8", "71.7", "+1.9", "0.803", "0.441"],
    ["S05", "61.1", "59.3", "−1.9", "0.606", "0.180"],
    ["S06", "63.6", "62.6", "−0.9", "0.745", "0.255"],
    ["S07", "62.0", "63.0", "+0.9", "0.689", "0.271"],
    ["S08", "82.4", "81.5", "−0.9", "0.954", "0.634"],
    ["S09", "91.7", "91.7", "0.0", "0.969", "0.833"],
    ["Mean", "72.6", "73.4*", "+0.8", "0.809", "0.471"],
]
add_table(s, data, Inches(0.8), Inches(3.5), Inches(11.7), Inches(3.3),
          col_widths=[Inches(1.5), Inches(2.0), Inches(2.0), Inches(1.5), Inches(2.2), Inches(2.5)])
# note about 74.1
add_text(s, Inches(0.5), Inches(6.85), Inches(12), Inches(0.25),
         "* Table shows transductive EA-FT (73.4%). Headline deployable result with calibration-only EA = 74.1%.",
         size=10, color=GRAY)
footer(s, page, TOTAL)


# ── 15. ABLATION ─────────────────────────────────────────────────────
s = new_slide()
section_header(s, "Evaluation — Ablation Study", "Which component actually delivers the performance?")

data = [
    ["Condition", "Mean Acc %", "Std", "Interpretation"],
    ["Baseline (no TL)", "51.0", "3.0", "Chance — fails without transfer"],
    ["Zero-Shot (no FT)", "72.6", "13.3", "Transfer alone ≈ full system"],
    ["FT without EA", "75.8", "15.7", "Highest mean, highest variance"],
    ["EA-FT EEGNet (Ours)", "73.4", "12.4", "Competitive + more robust"],
]
add_table(s, data, Inches(0.4), Inches(1.3), Inches(12.5), Inches(2.6),
          col_widths=[Inches(3.5), Inches(2.0), Inches(1.5), Inches(5.5)])

# finding cards
findings = [
    ("Transfer is Decisive",
     ["+22.4 pts vs from-scratch",
      "Wins on 9/9 subjects",
      "p = 0.0039, r = 1.000",
      "Strongest possible at n=9"]),
    ("EA = Robustness, not Accuracy",
     ["Geometry: 99% mismatch ↓",
      "Mean acc: not improved",
      "Std ↓ 15.7 → 12.4 pts",
      "Rescues weak transferrers"]),
    ("Fine-Tuning is Safe",
     ["Max drop: only 1.9 pts",
      "No catastrophic forgetting",
      "912 params frozen",
      "Conservative LR (5e-5)"]),
]
for i, (t, lines) in enumerate(findings):
    x = Inches(0.4) + Inches(4.25) * i
    card(s, x, Inches(4.2), Inches(4.1), Inches(2.5), t, lines,
         accent=GREEN if i == 0 else (ORANGE if i == 1 else ACCENT))
footer(s, page, TOTAL)


# ── 16. BENCHMARK ────────────────────────────────────────────────────
s = new_slide()
section_header(s, "Evaluation — Benchmark Comparison", "BCI Competition IV 2a — Left vs Right Hand MI")

data = [
    ["Method", "Acc %", "Calib. Time", "Params", "Reference"],
    ["DeepConvNet (no TL)", "62.4", "30 min", "170,000", "Schirrmeister 2017"],
    ["ShallowConvNet (no TL)", "64.1", "30 min", "44,000", "Schirrmeister 2017"],
    ["Standard EEGNet (no TL)", "65.8", "30 min", "~2,000", "Lawhern 2018"],
    ["EEGNet + Riemannian Align.", "74.2", "15 min", "~2,000", "He & Wu 2020"],
    ["EEGNet + DA (MDM)", "76.1", "20 min", "~2,000", "Barachant 2012"],
    ["EA-FT EEGNet (36 trials)", "74.1", "≈ 4.8 min", "2,450", "This work"],
    ["EA-FT EEGNet (18 trials)", "73.2", "≈ 2.4 min", "2,450", "This work"],
]
add_table(s, data, Inches(0.3), Inches(1.3), Inches(12.7), Inches(4.0),
          col_widths=[Inches(3.5), Inches(1.4), Inches(2.0), Inches(1.8), Inches(4.0)])

add_rect(s, Inches(0.4), Inches(5.5), Inches(12.5), Inches(1.2), WHITE)
add_rect(s, Inches(0.4), Inches(5.5), Inches(0.1), Inches(1.2), GREEN)
add_text(s, Inches(0.7), Inches(5.65), Inches(12), Inches(0.9),
         "Takeaway: Matches Riemannian Alignment accuracy (within 0.1 pts) at one-third the calibration time. "
         "Beats all non-transfer deep baselines by 8–12 pts at ~1/6 of their setup burden. "
         "At 18 trials (2.4 min): still 73.2% — 88–92% calibration reduction.",
         size=13, color=DARK)
footer(s, page, TOTAL)


# ── 17. CALIBRATION SENSITIVITY ──────────────────────────────────────
s = new_slide()
section_header(s, "Evaluation — Calibration Sensitivity", "Accuracy vs. number of calibration trials")

data = [
    ["Trials", "Time (min)", "Mean Acc %", "Std", "↓ vs 20 min", "↓ vs 30 min"],
    ["10", "1.34", "72.9", "12.0", "93.3%", "95.5%"],
    ["18", "2.41", "73.2", "11.9", "88.0%", "92.0%"],
    ["27", "3.61", "74.4", "11.4", "81.9%", "88.0%"],
    ["36 ★", "4.82", "73.7", "11.2", "75.9%", "83.9%"],
    ["54", "7.23", "74.4", "11.2", "63.9%", "75.9%"],
    ["72", "9.64", "74.4", "11.4", "51.8%", "67.9%"],
]
add_table(s, data, Inches(0.8), Inches(1.3), Inches(11.7), Inches(3.6),
          col_widths=[Inches(1.5), Inches(2.0), Inches(2.0), Inches(1.5), Inches(2.3), Inches(2.4)])

add_rect(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(1.5), WHITE)
bullet_block(s, Inches(0.7), Inches(5.35), Inches(12), Inches(1.3), [
    "Performance plateaus early — diminishing returns beyond ~27 trials",
    "Even 10 trials (~1.3 min) already achieve 72.9% mean accuracy",
    "Primary protocol: 36 trials (4.82 min) balances stability and user burden",
], size=14)
footer(s, page, TOTAL)


# ── 18. STATISTICS ───────────────────────────────────────────────────
s = new_slide()
section_header(s, "Evaluation — Statistical Analysis", "Wilcoxon signed-rank tests (n = 9 subjects, two-sided)")

data = [
    ["Comparison", "Median Δ", "Mean Δ", "Wins", "p-value", "r", "Sig?"],
    ["EA-FT vs From-scratch", "+16.98", "+22.39", "9/9", "0.0039", "+1.00", "Yes"],
    ["EA-FT vs Zero-shot", "0.00", "+0.83", "4/9", "0.5745", "+0.22", "No"],
    ["EA-FT vs FT without EA", "−3.70", "−2.41", "4/9", "0.4258", "−0.31", "No"],
]
add_table(s, data, Inches(0.3), Inches(1.3), Inches(12.7), Inches(2.4),
          col_widths=[Inches(3.2), Inches(1.5), Inches(1.5), Inches(1.3),
                      Inches(1.6), Inches(1.3), Inches(1.3)])

card(s, Inches(0.4), Inches(4.0), Inches(6.1), Inches(2.6),
     "Honest Interpretation",
     ["With n=9, smallest attainable p ≈ 0.004",
      "Non-significant ≠ 'no effect exists'",
      "Study powered only for large (~10 pt) effects",
      "Transfer effect is unambiguous and universal",
      "EA mean-accuracy effect unresolved at this n"],
     accent=ORANGE)

card(s, Inches(6.8), Inches(4.0), Inches(6.1), Inches(2.6),
     "Scientific Integrity",
     ["Ablation contradicted original hypothesis",
      "Result reported honestly — not retuned",
      "Hypothesis revised: transfer is prerequisite",
      "EA reframed as robustness mechanism",
      "Correct attribution > asserted victory"],
     accent=GREEN)
footer(s, page, TOTAL)


# ── 19. CONTRIBUTIONS ────────────────────────────────────────────────
s = new_slide()
section_header(s, "Research Contributions", "Five evidenced contributions (strongest first)")

contribs = [
    ("1", "Few-shot MI framework under 5-min budget",
     "74.1% acc / 0.809 AUC / 0.471 κ at 4.82 min — matches RA accuracy at ⅓ the time"),
    ("2", "Transfer — not alignment — enables few-shot decoding",
     "Pre-trained representation +22.4 pts vs from-scratch on all 9 subjects (p=0.0039, r=1.0)"),
    ("3", "EA reframed as robustness mechanism",
     "99% covariance mismatch ↓; rescues weak transferrers; reduces across-subject variance"),
    ("4", "Conservative FT eliminates catastrophic forgetting",
     "Max degradation 1.9 pts; 912 params frozen; safe personalization in few-shot regime"),
    ("5", "Fully reproducible, leakage-free evaluation protocol",
     "Calib-only EA · val-split selection · measured cadence · open ablation design"),
]
for i, (num, title, body) in enumerate(contribs):
    y = Inches(1.2) + Inches(1.05) * i
    add_rect(s, Inches(0.4), y, Inches(12.5), Inches(0.95), WHITE)
    add_rect(s, Inches(0.4), y, Inches(0.7), Inches(0.95), NAVY if i % 2 == 0 else RED)
    add_text(s, Inches(0.45), y + Inches(0.25), Inches(0.6), Inches(0.45),
             num, size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.3), y + Inches(0.12), Inches(11.3), Inches(0.35),
             title, size=14, bold=True, color=NAVY)
    add_text(s, Inches(1.3), y + Inches(0.48), Inches(11.3), Inches(0.4),
             body, size=12, color=GRAY)
footer(s, page, TOTAL)


# ── 20. OBJECTIVES ACHIEVED ──────────────────────────────────────────
s = new_slide()
section_header(s, "Achievement of Objectives")

data = [
    ["#", "Objective", "Status", "Evidence"],
    ["O1", "Comprehensive literature review", "✓ Achieved", "Ch. 2; 19+ sources; Table 1"],
    ["O2", "Analyse variability & calibration", "✓ Achieved", "Ch. 3; gap §2.6.1; empirical 5.14× spread"],
    ["O3", "Propose & implement EA-FT EEGNet", "✓ Achieved", "Ch. 4–6; full 3-zone pipeline"],
    ["O4", "Evaluate: less calib., robust acc.", "✓ Achieved*", "Ch. 7; 74.1% @ 4.82 min"],
]
add_table(s, data, Inches(0.3), Inches(1.3), Inches(12.7), Inches(3.0),
          col_widths=[Inches(0.8), Inches(4.0), Inches(2.2), Inches(5.7)])

add_rect(s, Inches(0.4), Inches(4.6), Inches(12.5), Inches(2.0), WHITE)
add_rect(s, Inches(0.4), Inches(4.6), Inches(0.1), Inches(2.0), ACCENT)
add_text(s, Inches(0.7), Inches(4.75), Inches(12), Inches(0.35),
         "* Objective 4 — Revised Attribution", size=14, bold=True, color=NAVY)
add_text(s, Inches(0.7), Inches(5.2), Inches(12), Inches(1.2),
         "The practical outcome is achieved (calibration ↓ 76–84%, accuracy competitive). "
         "Ablation shows the mechanism is cross-subject transfer, not EA as originally hypothesized. "
         "Reporting the correct mechanism strengthens — rather than weakens — the contribution.",
         size=13, color=DARK)
footer(s, page, TOTAL)


# ── 21. LIMITATIONS ──────────────────────────────────────────────────
s = new_slide()
section_header(s, "Limitations & Challenges", "Candid bounds on what this study can claim")

lims = [
    ("n = 9 subjects", "Unit of analysis is the subject; powered only for large effects"),
    ("Binary task only", "Left vs Right — easiest lateralized pair; 4-class untested"),
    ("Single dataset", "One lab, montage, amplifier, protocol"),
    ("Healthy participants", "Intended users have cortical reorganization"),
    ("Offline evaluation", "Cue-locked trials; online dynamics absent"),
    ("Artifact approx.", "Variance reject (0.5%) substituted for full ICA-ADJUST"),
]
for i, (t, b) in enumerate(lims):
    r, c = divmod(i, 3)
    x = Inches(0.4) + Inches(4.25) * c
    y = Inches(1.3) + Inches(1.8) * r
    card(s, x, y, Inches(4.1), Inches(1.6), t, [b], accent=ORANGE)

add_text(s, Inches(0.5), Inches(5.2), Inches(12.3), Inches(1.4),
         "Key challenges overcome: (1) discovered & eliminated test-set leakage; "
         "(2) established honest calibration timing from measured cadence; "
         "(3) confronted ablation result that contradicted hypothesis — reported without retuning; "
         "(4) parsed complex nested MATLAB dataset structures correctly.",
         size=13, color=GRAY)
footer(s, page, TOTAL)


# ── 22. CONCLUSION ───────────────────────────────────────────────────
s = new_slide()
section_header(s, "Conclusion", "What this research establishes")

bullet_block(s, Inches(0.5), Inches(1.3), Inches(12.3), Inches(3.2), [
    "EA-FT EEGNet brings working MI decoding within a five-minute setup budget (74.1% @ 4.82 min).",
    "Cross-subject transfer of a pre-trained representation is the true prerequisite for few-shot MI decoding.",
    "Conservative layer-selective fine-tuning makes personalization safe — catastrophic forgetting is absent.",
    "Euclidean Alignment earns its place by protecting users the system would otherwise fail (robustness).",
    "Inter-subject variability dominates every algorithmic effect — pointing toward subject-adaptive protocols.",
    "A 146 KB deployed model on consumer hardware is a concrete step toward usable assistive BCIs.",
], size=15)

add_rect(s, Inches(0.5), Inches(4.8), Inches(12.3), Inches(1.8), NAVY)
add_text(s, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.3),
         "“A research contribution that has been correctly attributed is more valuable\n"
         "than one that has merely been asserted.”",
         size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
footer(s, page, TOTAL)


# ── 23. FURTHER WORK ─────────────────────────────────────────────────
s = new_slide()
section_header(s, "Further Work", "Seven directions ranked by research value")

fw = [
    ("1", "Scale source domain", "Pre-train on MOABB / PhysioNet / OpenBMI — re-test EA hypothesis"),
    ("2", "Increase subject n", "40+ subjects for adequate power on 2–3 pt effects"),
    ("3", "Subject-adaptive protocols", "Route weak vs strong transferrers differently"),
    ("4", "Four-class + rejection", "Feet/tongue + abstain on low-confidence trials"),
    ("5", "Cross-hardware validation", "Dry electrodes, different montages/amplifiers"),
    ("6", "Closed-loop online BCI", "LSL + OpenBCI — measure true latency & drift"),
    ("7", "Clinical population", "Stroke / ALS — where EA may matter most"),
]
for i, (num, title, body) in enumerate(fw):
    y = Inches(1.2) + Inches(0.75) * i
    add_rect(s, Inches(0.4), y, Inches(0.55), Inches(0.6), RED if i < 3 else NAVY)
    add_text(s, Inches(0.4), y + Inches(0.1), Inches(0.55), Inches(0.4),
             num, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.15), y + Inches(0.05), Inches(4.5), Inches(0.5),
             title, size=14, bold=True, color=NAVY)
    add_text(s, Inches(5.8), y + Inches(0.05), Inches(7.0), Inches(0.5),
             body, size=13, color=GRAY)
footer(s, page, TOTAL)


# ── 24. REFERENCES ───────────────────────────────────────────────────
s = new_slide()
section_header(s, "References (Selected)")

refs = [
    "[1] Abiri et al., J. Neural Eng., 2019 — Comprehensive review of EEG-BCI paradigms",
    "[5] Wu, Xu & Lu, IEEE TCDS, 2022 — Transfer learning for EEG-BCIs (since 2016)",
    "[8] Lawhern et al., J. Neural Eng., 2018 — EEGNet: compact CNN for EEG-BCIs",
    "[12] Wolpaw et al., Electroencephalogr. Clin. Neurophysiol., 1991 — SMR cursor control",
    "[19] Nicolas-Alonso & Gomez-Gil, Sensors, 2012 — BCI review",
    "[20] Schirrmeister et al., Hum. Brain Mapp., 2017 — Deep ConvNets for EEG decoding",
    "[21] He & Wu, IEEE TBME, 2020 — Euclidean Space Data Alignment for BCI TL",
    "[22] Barachant et al., IEEE TBME, 2012 — Multiclass BCI via Riemannian geometry",
    "Dataset: BCI Competition IV Dataset 2a (Graz University of Technology)",
    "Full reference list: Thesis Chapter — References (22 cited sources)",
]
bullet_block(s, Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.5), refs, size=13)
footer(s, page, TOTAL)


# ── 25. DEMO / DELIVERABLES ──────────────────────────────────────────
s = new_slide()
section_header(s, "What Was Delivered", "Complete research package for evaluators")

deliverables = [
    ("Thesis Document", ["Full 8-chapter thesis", "Tables 1–16, Figures 1–28",
                         "Honest limitations & challenges", "Reproducible protocol"]),
    ("Implementation", ["End-to-end PyTorch notebook", "EA + LOSO + FT pipeline",
                        "3 ablation arms", "Statistical tests"]),
    ("Results Artifacts", ["9 adapted models (146 KB)", "Demo bundle + summary JSON",
                           "Per-subject metrics tables", "Calibration curves"]),
    ("Scientific Rigor", ["Leakage-free evaluation", "Deployable (calib-only) EA",
                          "Hypothesis revised by evidence", "Measured calibration time"]),
]
for i, (t, lines) in enumerate(deliverables):
    x = Inches(0.35) + Inches(3.25) * i
    card(s, x, Inches(1.4), Inches(3.1), Inches(5.0), t, lines,
         accent=RED if i % 2 else NAVY)
footer(s, page, TOTAL)


# ── 26. THANK YOU ────────────────────────────────────────────────────
s = new_slide()
add_rect(s, 0, 0, W, H, NAVY)
add_rect(s, 0, Inches(0), W, Inches(0.1), RED)
add_text(s, Inches(0.5), Inches(2.2), Inches(12.3), Inches(0.8),
         "Thank You", size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.5), Inches(3.2), Inches(12.3), Inches(0.5),
         "Questions & Discussion", size=22, color=RGBColor(0xE8, 0xC4, 0xC0), align=PP_ALIGN.CENTER)
add_text(s, Inches(0.5), Inches(4.3), Inches(12.3), Inches(1.5),
         "Halwathurage H.O.M.B  |  215526U\n"
         "Supervisor: Dr. Thilini Piyathilake\n"
         "EA-FT EEGNet  ·  University of Moratuwa  ·  August 2026",
         size=14, color=RGBColor(0xA0, 0xB4, 0xC8), align=PP_ALIGN.CENTER)

# ── Save ─────────────────────────────────────────────────────────────
out_dir = r"D:\AI Degree\Level 04 Sem 02\Research project in AI\Research project files\final thesis"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "Final_Thesis_Presentation_215526U_EA-FT_EEGNet.pptx")
prs.save(out_path)
print(f"Saved: {out_path}")
print(f"Slides: {len(prs.slides)}")
