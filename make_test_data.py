"""
Generate synthetic radiology report validation test data.
No PHI — entirely fictional reports.
"""
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Ground truth pool ──────────────────────────────────────────────────────
ORGANS = ["liver", "kidney", "lung", "adrenal gland", "pancreas", "thyroid", "lymph node"]
REC_CODES = ["REC1", "REC2a", "REC2b", "REC3a", "REC3b", "REC4"]
FINDING_TYPES = ["nodule", "cyst", "mass", "lesion", "calcification", "adenoma", "hemangioma"]
MODALITIES = ["CT", "MRI", "PET/CT", "Ultrasound", "None"]
TIMEFRAMES = ["1 month", "3 months", "6 months", "12 months", "immediate", "N/A"]

# REC code → reasonable followup modality + timeframe
REC_TO_FOLLOWUP = {
    "REC1": ("None",      "N/A"),
    "REC2a": ("CT",       "3 months"),
    "REC2b": ("CT",       "12 months"),
    "REC3a": ("MRI",      "6 months"),
    "REC3b": ("PET/CT",   "1 month"),
    "REC4":  ("PET/CT",   "immediate"),
}

REPORT_TEMPLATES = [
    "CT abdomen/pelvis with contrast. FINDINGS: A {size} {finding_type} is identified in the {organ}. "
    "The lesion demonstrates {enhancement}. No other significant findings. "
    "IMPRESSION: {size} {finding_type} of the {organ}. {recommendation}",

    "MRI {organ} without and with gadolinium contrast. FINDINGS: There is a {size} {finding_type} "
    "noted in the {organ}, which is {characteristics}. IMPRESSION: {finding_type} of the {organ} "
    "measuring {size}. {recommendation}",

    "CT chest with contrast. FINDINGS: A {size} {finding_type} is seen in the {organ}. "
    "Margins are {margins}. No lymphadenopathy. IMPRESSION: {size} {finding_type}, {organ}. "
    "{recommendation}",
]

ENHANCEMENTS = ["heterogeneous enhancement", "homogeneous enhancement", "no significant enhancement",
                "arterial phase enhancement with washout", "peripheral nodular enhancement"]
CHARACTERISTICS = ["well-circumscribed", "T2 hyperintense", "mildly heterogeneous",
                   "hypervascular with washout", "indeterminate in signal"]
MARGINS = ["well-defined", "lobulated", "irregular", "spiculated", "smooth"]
RECOMMENDATIONS = {
    "REC1": "No follow-up imaging recommended.",
    "REC2a": "Short-interval follow-up CT recommended in 3 months.",
    "REC2b": "Follow-up CT in 12 months recommended.",
    "REC3a": "MRI with contrast recommended for further characterization.",
    "REC3b": "Tissue sampling/biopsy recommended.",
    "REC4": "Urgent further evaluation recommended — findings suspicious for malignancy.",
}

def make_size():
    cm = round(random.uniform(0.4, 4.5), 1)
    return f"{cm} cm"

def corrupt(value, pool, error_rate):
    """Return a wrong value from pool with probability error_rate."""
    if random.random() < error_rate:
        others = [x for x in pool if x != value]
        return random.choice(others)
    return value

# ── Build rows ─────────────────────────────────────────────────────────────
rows = []
for i in range(30):
    organ       = random.choice(ORGANS)
    rec_code    = random.choice(REC_CODES)
    finding     = random.choice(FINDING_TYPES)
    size        = make_size()
    mod, time   = REC_TO_FOLLOWUP[rec_code]

    template = random.choice(REPORT_TEMPLATES)
    report_text = template.format(
        organ=organ,
        finding_type=finding,
        size=size,
        enhancement=random.choice(ENHANCEMENTS),
        characteristics=random.choice(CHARACTERISTICS),
        margins=random.choice(MARGINS),
        recommendation=RECOMMENDATIONS[rec_code],
    )

    # Model outputs — different error rates per column to make validation interesting
    rows.append({
        # Context
        "report_id":              f"RPT-{1000+i}",
        "report_text":            report_text,
        # Gold standard (3 columns)
        "gt_organ":               organ,
        "gt_rec_code":            rec_code,
        "gt_finding_type":        finding,
        # Model outputs (6 columns) — intentionally imperfect
        "model_organ":            corrupt(organ,    ORGANS,        error_rate=0.10),
        "model_rec_code":         corrupt(rec_code, REC_CODES,     error_rate=0.25),
        "model_finding_type":     corrupt(finding,  FINDING_TYPES, error_rate=0.15),
        "model_followup_modality":corrupt(mod,      MODALITIES,    error_rate=0.20),
        "model_followup_timeframe":corrupt(time,    TIMEFRAMES,    error_rate=0.20),
        "model_finding_size":     size if random.random() > 0.12 else make_size(),
    })

df = pd.DataFrame(rows)
out = "test_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows → {out}")
print(df[["gt_organ","gt_rec_code","gt_finding_type","model_organ","model_rec_code"]].head(8).to_string())
