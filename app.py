"""
LLM Output Validator — Streamlit App
Manually validate LLM/NLP extracted fields against ground truth, then export for confusion matrices.
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
import io
import random

try:
    import plotly.express as px
    import plotly.figure_factory as ff
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Validator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.gt-box {
    padding: 10px 14px;
    background: #e8f4fd;
    border-left: 4px solid #2196f3;
    border-radius: 6px;
    font-size: 1.05em;
    margin-bottom: 8px;
}
.model-box {
    padding: 14px 18px;
    background: #fff8e1;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    font-size: 1.2em;
    font-weight: 500;
    margin-bottom: 8px;
}
.verdict-correct   { color: #22c55e; font-weight: 700; }
.verdict-incorrect { color: #ef4444; font-weight: 700; }
.verdict-skipped   { color: #f59e0b; font-weight: 700; }
.shortcut-hint     { color: #888; font-size: 0.82em; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "page":         "setup",
    "df":           None,
    "project_name": "My Project",
    "reviewer":     "reviewer1",
    "gt_cols":      [],       # ground truth columns shown on left
    "val_cols":     [],       # validation (model output) columns
    "col_gt_map":   {},       # {val_col: gt_col} optional linkage
    "mode":         "by_column",
    "queue":        [],       # [(row_idx, col_name), ...]
    "pos":          0,
    "results":      {},       # {(row_idx, col_name): result_dict}
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

S = st.session_state

# ─────────────────────────────────────────────────────────────────────────────
# Keyboard shortcuts
# Arrow keys: → correct, ← incorrect, ↑ next/forward, ↓ back
# ─────────────────────────────────────────────────────────────────────────────
_KEYBOARD_JS = """
<script>
(function() {
  if (window._vlKeys) return;
  window._vlKeys = true;
  document.addEventListener('keydown', function(e) {
    var tag = document.activeElement ? document.activeElement.tagName : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    function clickBtn(label) {
      var btns = window.parent.document.querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) {
        if (btns[i].innerText.trim().indexOf(label) === 0) {
          btns[i].click(); return;
        }
      }
    }
    if (e.key === 'ArrowRight') { e.preventDefault(); clickBtn('✓'); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); clickBtn('✗'); }
    if (e.key === 'ArrowUp')    { e.preventDefault(); clickBtn('↑ Next'); }
    if (e.key === 'ArrowDown')  { e.preventDefault(); clickBtn('↓ Back'); }
  });
})();
</script>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_file(f) -> pd.DataFrame | None:
    name = f.name
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith((".json", ".jsonl")):
        try:
            return pd.read_json(f)
        except Exception:
            lines = f.read().decode().strip().splitlines()
            return pd.DataFrame([json.loads(ln) for ln in lines if ln.strip()])
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    st.error(f"Unsupported file type: {name}")
    return None


def build_queue(df: pd.DataFrame, val_cols: list[str], mode: str) -> list[tuple]:
    rows = list(df.index)
    if mode == "by_column":
        pairs = [(r, c) for c in val_cols for r in rows]
    elif mode == "by_row":
        pairs = [(r, c) for r in rows for c in val_cols]
    else:
        pairs = [(r, c) for c in val_cols for r in rows]
        random.shuffle(pairs)
    return pairs


def record_verdict(row_idx, col: str, verdict: str, notes: str = ""):
    gt_col = S.col_gt_map.get(col)
    gt_val = None
    if gt_col and gt_col in S.df.columns:
        gt_val = S.df.at[row_idx, gt_col]
    S.results[(row_idx, col)] = {
        "row_id":              row_idx,
        "validation_col":      col,
        "ground_truth_col":    gt_col,
        "ground_truth_value":  gt_val,
        "model_value":         S.df.at[row_idx, col],
        "verdict":             verdict,
        "notes":               notes,
        "reviewer":            S.reviewer,
        "timestamp":           datetime.now().isoformat(),
    }


def results_to_df() -> pd.DataFrame:
    if not S.results:
        return pd.DataFrame()
    return pd.DataFrame(list(S.results.values()))


def _render_value(label: str, val, box_class: str, key: str):
    val_str = str(val) if val is not None else "—"
    st.markdown(f"**{label}**")
    if len(val_str) > 250:
        st.text_area(label, value=val_str, height=140, disabled=True,
                     key=key, label_visibility="collapsed")
    else:
        st.markdown(f'<div class="{box_class}">{val_str}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Setup
# ─────────────────────────────────────────────────────────────────────────────

def page_setup():
    st.title("LLM Output Validator — Project Setup")

    c1, c2 = st.columns(2)
    with c1:
        S.project_name = st.text_input("Project name", value=S.project_name)
    with c2:
        S.reviewer = st.text_input("Reviewer ID", value=S.reviewer)

    uploaded = st.file_uploader(
        "Upload dataframe (CSV, JSON, JSONL, XLSX)",
        type=["csv", "json", "jsonl", "xlsx", "xls"],
    )
    if uploaded:
        df = load_file(uploaded)
        if df is not None:
            df = df.reset_index(drop=True)
            S.df = df
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    if S.df is None:
        st.info("Upload a file to configure the project.")
        return

    df = S.df
    cols = list(df.columns)

    st.subheader("Preview (first 5 rows)")
    st.dataframe(df.head(5), use_container_width=True)

    st.subheader("Column Roles")
    st.caption(
        "**Ground truth** = correct/gold values shown on the left. "
        "**Validation** = model outputs to verify, shown on the right one at a time."
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        gt = st.multiselect("Ground truth columns (left panel)", cols, default=S.gt_cols)
    with cc2:
        val = st.multiselect("Validation columns (model outputs)", cols, default=S.val_cols)

    if val and gt:
        st.subheader("Link each validation column to its ground truth (optional)")
        st.caption("Required for confusion matrices.")
        mapping = {}
        n = min(len(val), 4)
        row_cols = st.columns(n)
        for i, vc in enumerate(val):
            with row_cols[i % n]:
                default_gt = S.col_gt_map.get(vc, "(none)")
                opts = ["(none)"] + gt
                idx = opts.index(default_gt) if default_gt in opts else 0
                chosen = st.selectbox(f"`{vc}` ←→", opts, index=idx, key=f"gtmap_{vc}")
                if chosen != "(none)":
                    mapping[vc] = chosen
        S.col_gt_map = mapping

    st.subheader("Validation Options")
    oc1, oc2 = st.columns(2)
    with oc1:
        mode = st.radio(
            "Presentation order",
            ["by_column", "by_row", "random"],
            format_func={"by_column": "By column", "by_row": "By row", "random": "Random"}.__getitem__,
            horizontal=True,
            index=["by_column", "by_row", "random"].index(S.mode),
        )
    with oc2:
        sample_n = st.number_input(
            "Rows to validate (0 = all)", min_value=0, max_value=len(df), value=0, step=1
        )

    st.markdown("---")
    if st.button("▶ Start Validation", type="primary"):
        if not val:
            st.error("Select at least one validation column.")
            return
        S.gt_cols  = gt
        S.val_cols = val
        S.mode     = mode
        working_df = df.copy()
        if sample_n > 0:
            working_df = df.sample(n=int(sample_n), random_state=42).reset_index(drop=True)
            S.df = working_df
        S.queue = build_queue(S.df, val, mode)
        S.pos   = 0
        S.page  = "validate"
        st.rerun()

    # Quick confusion matrix without any manual validation
    if S.col_gt_map:
        st.markdown("---")
        st.subheader("Quick Confusion Matrix (full dataset, no manual validation)")
        qcm_col = st.selectbox("Column", list(S.col_gt_map.keys()), key="qcm_col")
        if st.button("Generate"):
            _render_confusion_matrix(S.df, qcm_col, S.col_gt_map[qcm_col])


# ─────────────────────────────────────────────────────────────────────────────
# Page: Validate
# ─────────────────────────────────────────────────────────────────────────────

def page_validate():
    queue = S.queue
    total = len(queue)

    if not queue:
        st.error("Queue is empty. Go back to setup.")
        if st.button("← Setup"):
            S.page = "setup"
            st.rerun()
        return

    # Sidebar
    with st.sidebar:
        st.markdown(f"### {S.project_name}")
        st.markdown(f"**Reviewer:** {S.reviewer}  \n**Mode:** {S.mode}")
        st.markdown("---")
        jump = st.number_input("Jump to item #", min_value=1, max_value=total,
                               value=S.pos + 1, step=1)
        if st.button("Go", key="jump_btn"):
            S.pos = int(jump) - 1
            st.rerun()
        st.markdown("---")
        if st.button("📊 View Results", use_container_width=True):
            S.page = "results"
            st.rerun()
        if st.button("⚙ Setup", use_container_width=True):
            S.page = "setup"
            st.rerun()
        if S.results:
            st.markdown("---\n**Progress by column:**")
            for vc in S.val_cols:
                col_r   = [r for r in S.results.values() if r["validation_col"] == vc]
                n_col   = sum(1 for (_, ci) in queue if ci == vc)
                done    = len(col_r)
                correct = sum(1 for r in col_r if r["verdict"] == "correct")
                acc     = f"{correct/done:.0%}" if done else "—"
                st.markdown(f"**{vc}**: {done}/{n_col} · {acc}")

    pos = S.pos

    # Header
    h1, h2 = st.columns([3, 1])
    with h1:
        st.title(f"Validating: {S.project_name}")
    with h2:
        done_count    = sum(1 for r in S.results.values() if r["verdict"] in ("correct", "incorrect"))
        correct_count = sum(1 for r in S.results.values() if r["verdict"] == "correct")
        st.markdown(f"**Item {min(pos+1, total)} of {total}**")
        if done_count:
            st.markdown(f"**Accuracy:** {correct_count/done_count:.1%} ({correct_count}/{done_count})")

    st.progress(min(pos / total, 1.0))

    if pos >= total:
        st.success("🎉 All items reviewed!")
        if st.button("📊 View Results", type="primary"):
            S.page = "results"
            st.rerun()
        return

    row_idx, col_name = queue[pos]
    row      = S.df.loc[row_idx]
    existing = S.results.get((row_idx, col_name))

    left, right = st.columns([1, 1], gap="large")

    # ── Left: ground truth ───────────────────────────────────────────────────
    with left:
        st.markdown("### Ground Truth")

        linked_gt = S.col_gt_map.get(col_name)

        # Show all GT columns
        for c in S.gt_cols:
            # Highlight the linked GT column more prominently
            box = "model-box" if c == linked_gt else "gt-box"
            _render_value(
                f"{c} {'← linked' if c == linked_gt else ''}",
                row.get(c, "—"), box, f"gt_{c}_{pos}"
            )

        # If linked GT isn't in the selected gt_cols, show it anyway
        if linked_gt and linked_gt not in S.gt_cols:
            st.markdown("---")
            _render_value(f"{linked_gt} (linked GT)", row.get(linked_gt, "—"),
                          "model-box", f"lgt_{pos}")

        st.caption(f"Row {row_idx}")

    # ── Right: model output + verdict ────────────────────────────────────────
    with right:
        st.markdown(f"### Model Output — `{col_name}`")

        model_val = row.get(col_name, "—")
        _render_value("Model output", model_val, "model-box", f"mv_{pos}")

        st.markdown("---")

        st.markdown(
            '<div class="shortcut-hint">'
            '→ correct &nbsp;|&nbsp; ← incorrect &nbsp;|&nbsp; ↑ next &nbsp;|&nbsp; ↓ back'
            '</div>',
            unsafe_allow_html=True,
        )

        notes = st.text_input(
            "Notes (optional)",
            value=existing.get("notes", "") if existing else "",
            placeholder="Corrections, observations…",
            key=f"notes_{pos}",
        )

        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            if st.button("✓ Correct", type="primary", key=f"yes_{pos}", use_container_width=True):
                record_verdict(row_idx, col_name, "correct", notes)
                S.pos = min(pos + 1, total)
                st.rerun()
        with bc2:
            if st.button("✗ Incorrect", key=f"no_{pos}", use_container_width=True):
                record_verdict(row_idx, col_name, "incorrect", notes)
                S.pos = min(pos + 1, total)
                st.rerun()
        with bc3:
            if st.button("↑ Next", key=f"next_{pos}", use_container_width=True):
                S.pos = min(pos + 1, total)
                st.rerun()
        with bc4:
            if st.button("↓ Back", key=f"back_{pos}", use_container_width=True):
                S.pos = max(pos - 1, 0)
                st.rerun()

        if existing:
            v   = existing["verdict"]
            cls = {"correct": "verdict-correct", "incorrect": "verdict-incorrect",
                   "skipped": "verdict-skipped"}.get(v, "")
            st.markdown(f'<div class="{cls}">Previously: {v.upper()}</div>',
                        unsafe_allow_html=True)

    st.components.v1.html(_KEYBOARD_JS, height=0)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix renderer (shared by setup quick-plot and results page)
# ─────────────────────────────────────────────────────────────────────────────

def _render_confusion_matrix(df: pd.DataFrame, val_col: str, gt_col: str, filter_mask=None):
    subset = df.copy()
    if filter_mask is not None:
        subset = subset[filter_mask]
    subset = subset[[gt_col, val_col]].dropna()
    if subset.empty:
        st.info("No data available.")
        return

    y_true = subset[gt_col].astype(str)
    y_pred = subset[val_col].astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    if HAS_SKLEARN and HAS_PLOTLY:
        normalize = st.checkbox("Normalize rows", value=False, key=f"norm_{val_col}_{gt_col}")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_disp = np.nan_to_num(cm_disp)
            ann = [[f"{v:.2f}" for v in r] for r in cm_disp]
        else:
            cm_disp = cm.astype(float)
            ann     = [[str(v) for v in r] for r in cm]

        fig = ff.create_annotated_heatmap(
            z=cm_disp, x=labels, y=labels,
            annotation_text=ann, colorscale="Blues", showscale=True,
        )
        fig.update_layout(
            title=f"{val_col} vs {gt_col}  (n={len(subset)})",
            xaxis_title=f"Predicted — {val_col}",
            yaxis_title=f"Actual — {gt_col}",
            yaxis_autorange="reversed",
            height=max(350, 80 * len(labels)),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Classification report"):
            st.code(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    else:
        st.dataframe(
            pd.crosstab(y_true, y_pred, rownames=[gt_col], colnames=[val_col]),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Results
# ─────────────────────────────────────────────────────────────────────────────

def page_results():
    st.title(f"Results — {S.project_name}")

    results_df = results_to_df()

    tab_summary, tab_cm, tab_raw = st.tabs(["Summary & Export", "Confusion Matrices", "Raw Data"])

    with tab_summary:
        if results_df.empty:
            st.info("No results yet.")
        else:
            total    = len(results_df)
            correct  = (results_df.verdict == "correct").sum()
            incorrect= (results_df.verdict == "incorrect").sum()
            skipped  = (results_df.verdict == "skipped").sum()
            reviewed = correct + incorrect

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Reviewed", total)
            m2.metric("Correct",  correct)
            m3.metric("Incorrect",incorrect)
            m4.metric("Skipped",  skipped)
            m5.metric("Accuracy", f"{correct/reviewed:.1%}" if reviewed else "—")

            st.subheader("By Column")
            by_col = (
                results_df.groupby("validation_col")
                .apply(lambda g: pd.Series({
                    "reviewed":  (g.verdict != "skipped").sum(),
                    "correct":   (g.verdict == "correct").sum(),
                    "incorrect": (g.verdict == "incorrect").sum(),
                    "skipped":   (g.verdict == "skipped").sum(),
                    "accuracy":  f"{(g.verdict=='correct').sum() / max((g.verdict!='skipped').sum(),1):.1%}",
                }), include_groups=False)
                .reset_index()
            )
            st.dataframe(by_col, use_container_width=True)

            if HAS_PLOTLY:
                counts = (
                    results_df[results_df.verdict != "skipped"]
                    .groupby(["validation_col", "verdict"]).size()
                    .reset_index(name="count")
                )
                fig = px.bar(
                    counts, x="validation_col", y="count", color="verdict", barmode="group",
                    color_discrete_map={"correct": "#22c55e", "incorrect": "#ef4444"},
                    title="Correct vs Incorrect by Column",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Export")
            dl1, dl2 = st.columns(2)
            with dl1:
                buf = io.StringIO()
                results_df.to_csv(buf, index=False)
                st.download_button(
                    "⬇ Download CSV",
                    data=buf.getvalue(),
                    file_name=f"{S.project_name.replace(' ','_')}_validation.csv",
                    mime="text/csv",
                )
            with dl2:
                st.download_button(
                    "⬇ Download JSON",
                    data=results_df.to_json(orient="records", indent=2),
                    file_name=f"{S.project_name.replace(' ','_')}_validation.json",
                    mime="application/json",
                )

            st.subheader("Loading in a notebook")
            st.code(
                'import pandas as pd\n'
                'df = pd.read_csv("..._validation.csv")\n\n'
                '# Confusion matrix for one column\n'
                'col = df[df.validation_col == "model_rec_code"]\n'
                'from sklearn.metrics import confusion_matrix\n'
                'cm = confusion_matrix(col.ground_truth_value, col.model_value)',
                language="python",
            )

    with tab_cm:
        st.subheader("Confusion Matrix Generator")

        cols_with_gt = [vc for vc in S.val_cols if S.col_gt_map.get(vc)]
        if not cols_with_gt:
            st.info("No ground truth links configured. Set them up in the Setup page.")
        else:
            st.markdown("#### From validated rows only")
            sel = st.selectbox("Column", cols_with_gt, key="cm_reviewed")
            if not results_df.empty:
                col_r = results_df[
                    (results_df.validation_col == sel) & (results_df.verdict != "skipped")
                ].copy()
                if col_r.empty:
                    st.info("No reviewed rows for this column yet.")
                else:
                    tmp = col_r.rename(columns={
                        "ground_truth_value": "__gt__", "model_value": "__pred__"
                    })
                    _render_confusion_matrix(tmp, "__pred__", "__gt__")
            else:
                st.info("No validation results yet.")

            st.markdown("---")
            st.markdown("#### From full dataset (all rows)")
            sel2 = st.selectbox("Column", cols_with_gt, key="cm_full")
            if sel2 and S.df is not None:
                _render_confusion_matrix(S.df, sel2, S.col_gt_map[sel2])

    with tab_raw:
        if results_df.empty:
            st.info("No results yet.")
        else:
            st.dataframe(results_df, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Continue Validation", type="primary"):
            S.page = "validate"
            st.rerun()
    with c2:
        if st.button("New Project"):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
if S.page == "setup":
    page_setup()
elif S.page == "validate":
    page_validate()
elif S.page == "results":
    page_results()
