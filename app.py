"""
LLM Output Validator — Streamlit App
Manually validate LLM/NLP extracted fields against ground truth, then export for confusion matrices.
Supports project save/resume so progress survives app restarts.
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import random
import re
import html as _html
from difflib import SequenceMatcher

try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        accuracy_score, f1_score, cohen_kappa_score,
    )
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
.linked-gt-box {
    padding: 10px 14px;
    background: #d4edda;
    border-left: 4px solid #28a745;
    border-radius: 6px;
    font-size: 1.1em;
    font-weight: 600;
    margin-bottom: 8px;
}
.explore-box {
    padding: 8px 12px;
    background: #f8f9fa;
    border-left: 4px solid #6c757d;
    border-radius: 6px;
    font-size: 0.98em;
    margin-bottom: 6px;
    white-space: pre-wrap;
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
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border: 1px solid #dee2e6;
}
.metric-label { font-size: 0.85em; color: #666; margin-bottom: 2px; }
.metric-value { font-size: 1.6em; font-weight: 700; color: #1a1a2e; }
.verdict-correct   { color: #22c55e; font-weight: 700; }
.verdict-incorrect { color: #ef4444; font-weight: 700; }
.verdict-skipped   { color: #f59e0b; font-weight: 700; }
.shortcut-hint { color: #555; font-size: 0.9em; margin-top: 4px;
                 background: #f0f0f0; padding: 6px 10px; border-radius: 4px;
                 display: inline-block; }
.shortcut-off  { color: #aaa; font-size: 0.9em; margin-top: 4px; font-style: italic; }
.save-info     { font-size: 0.8em; color: #888; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "page":               "setup",
    "df":                 None,
    "project_name":       "My Project",
    "reviewer":           "reviewer1",
    "save_dir":           "",          # full path to project save directory
    "gt_cols":            [],
    "col_gt_map":         {},          # {val_col: [gt_col, ...]}
    "val_cols":           [],
    "explore_cols":       [],          # extra columns togglable on left panel
    "explore_visible":    False,       # whether explore panel is currently open
    "col_metric_mode":    {},          # {val_col: "categorical" | "binary"}
    "col_positive_class": {},          # {val_col: "positive_label"}
    "mode":               "by_column",
    "queue":              [],          # [(row_idx, col_name), ...]
    "pos":                0,
    "results":            {},          # {(row_idx, col_name): result_dict}
    "row_notes":          {},          # {str(row_id): notes_str}  — one per row
    "kb_active":          False,
    "multi_col_mode":     False,       # validate all val_cols per row simultaneously
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

S = st.session_state

# ─────────────────────────────────────────────────────────────────────────────
# Keyboard shortcuts — attach to window.parent.document, not the iframe doc
# ─────────────────────────────────────────────────────────────────────────────

def _keyboard_component(active: bool):
    js = f"""
<script>
(function() {{
  var par = window.parent;
  if (par._vlKeyHandler) {{
    par.document.removeEventListener('keydown', par._vlKeyHandler);
    par._vlKeyHandler = null;
  }}
  if (!{'true' if active else 'false'}) return;
  function handler(e) {{
    var el  = par.document.activeElement;
    var tag = el ? el.tagName : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    function clickBtn(startsWith) {{
      var btns = par.document.querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) {{
        if (btns[i].innerText.trim().indexOf(startsWith) === 0) {{
          btns[i].click(); return;
        }}
      }}
    }}
    if (e.key === 'ArrowRight') {{ e.preventDefault(); clickBtn('✓'); }}
    if (e.key === 'ArrowLeft')  {{ e.preventDefault(); clickBtn('✗'); }}
    if (e.key === 'ArrowUp')    {{ e.preventDefault(); clickBtn('↑ Next'); }}
    if (e.key === 'ArrowDown')  {{ e.preventDefault(); clickBtn('↓ Back'); }}
  }}
  par._vlKeyHandler = handler;
  par.document.addEventListener('keydown', handler);
}})();
</script>
"""
    st.components.v1.html(js, height=0)


def _keyboard_component_multi(active: bool):
    """Keyboard shortcuts for multi-col mode.
    → = Mark All Correct   ← = Mark All Incorrect
    ↑ = Next row           ↓ = Back
    Targets the "✓ All Correct" / "✗ All Incorrect" buttons by prefix."""
    js = f"""
<script>
(function() {{
  var par = window.parent;
  if (par._vlKeyHandler) {{
    par.document.removeEventListener('keydown', par._vlKeyHandler);
    par._vlKeyHandler = null;
  }}
  if (!{'true' if active else 'false'}) return;
  function handler(e) {{
    var el  = par.document.activeElement;
    var tag = el ? el.tagName : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    function clickBtn(startsWith) {{
      var btns = par.document.querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) {{
        if (btns[i].innerText.trim().indexOf(startsWith) === 0) {{
          btns[i].click(); return;
        }}
      }}
    }}
    if (e.key === 'ArrowRight') {{ e.preventDefault(); clickBtn('✓ All Correct'); }}
    if (e.key === 'ArrowLeft')  {{ e.preventDefault(); clickBtn('✗ All Incorrect'); }}
    if (e.key === 'ArrowUp')    {{ e.preventDefault(); clickBtn('↑ Next'); }}
    if (e.key === 'ArrowDown')  {{ e.preventDefault(); clickBtn('↓ Back'); }}
  }}
  par._vlKeyHandler = handler;
  par.document.addEventListener('keydown', handler);
}})();
</script>
"""
    st.components.v1.html(js, height=0)


# ─────────────────────────────────────────────────────────────────────────────
# Resizable left/right panel split
# ─────────────────────────────────────────────────────────────────────────────

def _resize_component():
    """Inject a drag handle between the two content panels.

    Strategy: a zero-height marker div is placed as the FIRST child of the
    left column (inside `with left:`).  Walking up from that marker to the
    nearest [data-testid="column"] ancestor is unambiguous — unlike searching
    for sentinel elements placed *outside* the columns.

    A MutationObserver watches each column's style attribute so that the
    moment Streamlit's React reconciler resets the flex values after a
    navigation rerender, the saved width is immediately re-applied.
    Width is persisted in browser localStorage — no Streamlit round-trip.
    """
    js = """
<script>
(function() {
  var STORAGE_KEY = 'vl_left_panel_pct';
  var par = window.parent;
  var doc = par.document;

  /* ── Tear down any previous instance ── */
  var old = doc.getElementById('vl-drag-handle');
  if (old) old.remove();
  if (par._vlSplitObs) { par._vlSplitObs.disconnect(); par._vlSplitObs = null; }

  function getSavedPct() {
    var v = parseFloat(localStorage.getItem(STORAGE_KEY));
    return (isNaN(v) || v < 15 || v > 85) ? 50 : v;
  }
  function savePct(pct) { localStorage.setItem(STORAGE_KEY, pct.toFixed(1)); }

  /* ── Find columns by walking UP from a marker placed inside the left col ── */
  function findSplitBlock() {
    var marker = doc.getElementById('vl-left-col-marker');
    if (!marker) return null;
    var el = marker;
    while (el && el !== doc.body) {
      if (el.getAttribute && el.getAttribute('data-testid') === 'column') {
        var block = el.parentElement;
        if (block && block.getAttribute('data-testid') === 'stHorizontalBlock') {
          var cols = block.querySelectorAll(':scope > [data-testid="column"]');
          if (cols.length === 2) return { block: block, cols: cols };
        }
        break;
      }
      el = el.parentElement;
    }
    return null;
  }

  function applyPct(cols, pct) {
    cols[0].style.setProperty('flex', pct + ' 1 0%', 'important');
    cols[1].style.setProperty('flex', (100 - pct) + ' 1 0%', 'important');
  }

  function positionHandle(handle, block, cols) {
    var br = block.getBoundingClientRect();
    var lr = cols[0].getBoundingClientRect();
    var rr = cols[1].getBoundingClientRect();
    if (!br.width) return;
    handle.style.left   = Math.round((lr.right + rr.left) / 2 - br.left - 5) + 'px';
    handle.style.height = block.offsetHeight + 'px';
  }

  function setup(attempt) {
    var result = findSplitBlock();
    if (!result) {
      if (attempt < 20) setTimeout(function() { setup(attempt + 1); }, 150);
      return;
    }

    var pct = getSavedPct();
    var block = result.block, cols = result.cols;
    applyPct(cols, pct);
    block.style.position = 'relative';

    /* ── Build the drag handle ── */
    var handle = doc.createElement('div');
    handle.id    = 'vl-drag-handle';
    handle.title = 'Drag to resize panels';
    handle.style.cssText =
      'position:absolute;top:0;width:10px;cursor:col-resize;z-index:9999;' +
      'border-radius:5px;background:transparent;transition:background .15s;' +
      'user-select:none;-webkit-user-select:none;';
    positionHandle(handle, block, cols);
    block.appendChild(handle);

    handle.addEventListener('mouseenter', function() {
      handle.style.background = 'rgba(33,150,243,0.5)';
    });
    handle.addEventListener('mouseleave', function() {
      if (!par._vlDragging) handle.style.background = 'transparent';
    });
    handle.addEventListener('mousedown', function(e) {
      e.preventDefault();
      par._vlDragging = true;
      handle.style.background = 'rgba(33,150,243,0.85)';
      var blockRect = block.getBoundingClientRect();
      function onMove(ev) {
        var newPct = ((ev.clientX - blockRect.left) / blockRect.width) * 100;
        newPct = Math.min(Math.max(newPct, 15), 85);
        applyPct(cols, newPct);
        savePct(newPct);
        positionHandle(handle, block, cols);
      }
      function onUp() {
        par._vlDragging = false;
        handle.style.background = 'transparent';
        doc.removeEventListener('mousemove', onMove);
        doc.removeEventListener('mouseup', onUp);
      }
      doc.addEventListener('mousemove', onMove);
      doc.addEventListener('mouseup', onUp);
    });

    /* ── Watch for Streamlit resetting column flex styles after rerenders ──
         Targeting only the two column style attributes keeps this cheap.    */
    var obs = new MutationObserver(function() {
      applyPct(cols, getSavedPct());
      positionHandle(handle, block, cols);
    });
    obs.observe(cols[0], { attributes: true, attributeFilter: ['style'] });
    obs.observe(cols[1], { attributes: true, attributeFilter: ['style'] });
    par._vlSplitObs = obs;
  }

  setup(0);
})();
</script>
"""
    st.components.v1.html(js, height=0)


def _textbox_resize_component():
    """Persist the user-resized height of each [data-vl-box] div across rows.

    Every text box rendered by _render_value carries a data-vl-box attribute
    whose value is a stable label-derived key.  On each render we:
      1. Restore saved heights from localStorage so the box starts at the same
         size the user left it on the previous row.
      2. Attach a ResizeObserver so any drag-resize is immediately saved.
    The observer is replaced on every render — the old one's targets are stale
    DOM nodes from the previous row and are garbage-collected automatically.
    """
    js = """
<script>
(function() {
  var PREFIX = 'vl_box_h_';
  var par = window.parent;
  var doc = par.document;

  if (par._vlBoxObs) { par._vlBoxObs.disconnect(); par._vlBoxObs = null; }

  function applyAndWatch(attempt) {
    var boxes = doc.querySelectorAll('[data-vl-box]');
    if (!boxes.length) {
      if (attempt < 20) setTimeout(function() { applyAndWatch(attempt + 1); }, 150);
      return;
    }

    /* Restore saved heights */
    boxes.forEach(function(box) {
      var k   = box.getAttribute('data-vl-box');
      var saved = localStorage.getItem(PREFIX + k);
      if (saved) {
        box.style.height = saved;
      }
      /* Ensure resize handle is active (Streamlit may strip inline styles) */
      box.style.resize   = 'vertical';
      box.style.overflow = 'auto';
    });

    /* Save height whenever the user drags a box */
    var ro = new ResizeObserver(function(entries) {
      entries.forEach(function(entry) {
        var box = entry.target;
        var k   = box.getAttribute('data-vl-box');
        if (k && box.offsetHeight > 0) {
          localStorage.setItem(PREFIX + k, box.offsetHeight + 'px');
        }
      });
    });
    boxes.forEach(function(box) { ro.observe(box); });
    par._vlBoxObs = ro;
  }

  applyAndWatch(0);
})();
</script>
"""
    st.components.v1.html(js, height=0)


# ─────────────────────────────────────────────────────────────────────────────
# Native folder picker  (tkinter — local-only, never sends data anywhere)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_directory(target_key: str):
    """Open the OS native folder-picker and write the chosen path into
    st.session_state[target_key].

    Tkinter must run on the main thread, which Streamlit's event loop blocks.
    The fix: spawn a tiny child process that owns its own main thread, runs the
    dialog there, and returns the path via stdout.  No data leaves the machine.
    """
    import subprocess, sys, platform
    try:
        system = platform.system()
        if system == "Darwin":
            # macOS — AppleScript needs no extra Python packages
            script = 'POSIX path of (choose folder with prompt "Select project folder")'
            res = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            chosen = res.stdout.strip() if res.returncode == 0 else ""
        elif system == "Windows":
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
                "$d.Description='Select project folder';"
                "$d.ShowNewFolderButton=$true;"
                "if($d.ShowDialog() -eq 'OK'){Write-Output $d.SelectedPath}"
            )
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True, text=True, timeout=120,
            )
            chosen = res.stdout.strip() if res.returncode == 0 else ""
        else:
            # Linux — run tkinter in its own subprocess to avoid thread conflict
            tiny = (
                "import tkinter as tk, sys;"
                "from tkinter import filedialog;"
                "r=tk.Tk(); r.withdraw(); r.wm_attributes('-topmost',True);"
                "p=filedialog.askdirectory(title='Select project folder');"
                "r.destroy(); print(p) if p else None"
            )
            res = subprocess.run(
                [sys.executable, "-c", tiny],
                capture_output=True, text=True, timeout=120,
            )
            chosen = res.stdout.strip() if res.returncode == 0 else ""

        if chosen:
            st.session_state[target_key] = chosen
    except Exception as exc:
        st.error(f"Folder picker unavailable: {exc}. Please type the path manually.")


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Serialise numpy/pandas types that json.dumps chokes on."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def save_project():
    """Persist full project state to S.save_dir.  Silently skips if no dir set."""
    if not S.save_dir:
        return
    save_path = Path(S.save_dir)
    try:
        save_path.mkdir(parents=True, exist_ok=True)

        # ── Dataframe ────────────────────────────────────────────────────────
        if S.df is not None:
            S.df.to_csv(save_path / "data.csv", index=True)

        # ── Results: store as list of [row_id, col, dict] triples ────────────
        results_serial = [
            [int(row_id) if isinstance(row_id, (int, np.integer)) else row_id,
             col,
             {k: _json_default(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
              for k, v in res.items()}]
            for (row_id, col), res in S.results.items()
        ]

        state = {
            "version":            2,
            "project_name":       S.project_name,
            "reviewer":           S.reviewer,
            "save_dir":           str(S.save_dir),
            "gt_cols":            S.gt_cols,
            "val_cols":           S.val_cols,
            "explore_cols":       S.explore_cols,
            "col_gt_map":         S.col_gt_map,
            "col_metric_mode":    S.col_metric_mode,
            "col_positive_class": S.col_positive_class,
            "mode":               S.mode,
            "multi_col_mode":     S.multi_col_mode,
            "queue":              [[int(r) if isinstance(r, (int, np.integer)) else r, c]
                                   for r, c in S.queue],
            "pos":                int(S.pos),
            "results":            results_serial,
            "row_notes":          S.row_notes,
            "saved_at":           datetime.now().isoformat(),
        }

        with open(save_path / "project.json", "w") as f:
            json.dump(state, f, indent=2, default=_json_default)

    except Exception as e:
        st.warning(f"Auto-save failed: {e}")


def load_project(save_dir: str) -> tuple[bool, str]:
    """Load project from save_dir into session state.
    Returns (success, message)."""
    save_path = Path(save_dir)
    project_file = save_path / "project.json"
    data_file    = save_path / "data.csv"

    if not project_file.exists():
        return False, "No project.json found at that path."

    try:
        with open(project_file) as f:
            state = json.load(f)

        # Load dataframe
        if data_file.exists():
            df = pd.read_csv(data_file, index_col=0)
            df.index = df.index.astype(int)
            S.df = df
        else:
            return False, "data.csv not found — cannot restore dataframe."

        S.project_name       = state.get("project_name", "")
        S.reviewer           = state.get("reviewer", "reviewer1")
        S.save_dir           = str(save_dir)
        S.gt_cols            = state.get("gt_cols", [])
        S.val_cols           = state.get("val_cols", [])
        S.explore_cols       = state.get("explore_cols", [])
        S.col_gt_map         = state.get("col_gt_map", {})
        S.col_metric_mode    = state.get("col_metric_mode", {})
        S.col_positive_class = state.get("col_positive_class", {})
        S.mode               = state.get("mode", "by_column")
        S.multi_col_mode     = state.get("multi_col_mode", False)
        S.queue              = [tuple(q) for q in state.get("queue", [])]
        S.pos                = state.get("pos", 0)
        S.row_notes          = state.get("row_notes", {})

        # Restore results
        S.results = {}
        for item in state.get("results", []):
            row_id, col, res_dict = item
            S.results[(int(row_id), col)] = res_dict

        # Clear any stale row-notes widget state so text areas re-init from S.row_notes
        stale = [k for k in list(st.session_state.keys()) if k.startswith("rownotes_")]
        for k in stale:
            del st.session_state[k]

        saved_at = state.get("saved_at", "unknown")
        total    = len(S.queue)
        done     = sum(1 for r in S.results.values() if r.get("verdict") in ("correct", "incorrect"))
        return True, (f"Loaded **{S.project_name}** · {done}/{total} reviewed · "
                      f"saved {saved_at[:19].replace('T', ' ')}")

    except Exception as e:
        return False, f"Error loading project: {e}"


def project_summary(save_dir: str) -> dict | None:
    """Return brief metadata from a saved project without fully loading it."""
    try:
        with open(Path(save_dir) / "project.json") as f:
            s = json.load(f)
        total = len(s.get("queue", []))
        done  = sum(1 for r in s.get("results", []) if r[2].get("verdict") in ("correct", "incorrect"))
        return {
            "name":     s.get("project_name", "—"),
            "reviewer": s.get("reviewer", "—"),
            "progress": f"{done} / {total}",
            "saved":    s.get("saved_at", "—")[:19].replace("T", " "),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
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


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def record_verdict(row_idx, col: str, verdict: str):
    """Record a verdict. Notes live in S.row_notes, not per-verdict."""
    linked_gts     = _as_list(S.col_gt_map.get(col))
    primary_gt     = linked_gts[0] if linked_gts else None
    primary_gt_val = (
        S.df.at[row_idx, primary_gt]
        if primary_gt and primary_gt in S.df.columns else None
    )
    all_gt = {gc: str(S.df.at[row_idx, gc]) for gc in linked_gts if gc in S.df.columns}

    S.results[(row_idx, col)] = {
        "row_id":             row_idx,
        "validation_col":     col,
        "ground_truth_col":   primary_gt,
        "ground_truth_value": primary_gt_val,
        "all_ground_truth":   json.dumps(all_gt) if all_gt else None,
        "model_value":        S.df.at[row_idx, col],
        "verdict":            verdict,
        "reviewer":           S.reviewer,
        "timestamp":          datetime.now().isoformat(),
    }


def _flush_notes_and_save(row_idx):
    """Read the text area for this row from widget state and persist."""
    notes_key = f"rownotes_{row_idx}"
    current   = st.session_state.get(notes_key, "")
    if current is not None:
        S.row_notes[str(row_idx)] = current
    save_project()


def results_to_df() -> pd.DataFrame:
    """Full internal results dataframe — includes GT/model values for in-app metrics."""
    if not S.results:
        return pd.DataFrame()
    df = pd.DataFrame(list(S.results.values()))
    # Merge per-row notes as a single column
    df["row_notes"] = df["row_id"].astype(str).map(
        lambda rid: S.row_notes.get(rid, "")
    )
    return df


def results_to_export_df(extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Minimal export dataframe — only what's needed to compute metrics externally.
    Does NOT duplicate original data columns by default.
    Pass extra_cols to join specific columns from S.df by row_id."""
    if not S.results:
        return pd.DataFrame()
    rows = []
    for (row_id, col), res in S.results.items():
        rows.append({
            "row_id":         row_id,
            "validation_col": col,
            "verdict":        res.get("verdict"),
            "reviewer":       res.get("reviewer"),
            "timestamp":      res.get("timestamp"),
            "row_notes":      S.row_notes.get(str(row_id), ""),
        })
    df = pd.DataFrame(rows)
    if extra_cols and S.df is not None:
        valid = [c for c in extra_cols if c in S.df.columns]
        if valid:
            extra = S.df[valid].copy()
            extra.index.name = "row_id"
            extra = extra.reset_index()
            df = df.merge(extra, on="row_id", how="left")
    return df


def _find_and_highlight(gt_text: str, model_val: str) -> tuple[str, bool]:
    """Search for model_val (or a close match) inside gt_text.
    Returns (html_string, was_found).  The html_string is always HTML-escaped;
    matches are wrapped in a red highlight <span>.
    Three-pass strategy: exact → word-level → fuzzy substring."""

    SPAN = ('<span style="background:#e53e3e;color:#fff;padding:1px 4px;'
            'border-radius:3px;font-weight:600">{}</span>')

    def build_html(text: str, spans: list[tuple[int, int]]) -> str:
        """Build escaped HTML, injecting highlight spans at the given char ranges."""
        spans = sorted(set(spans))
        # Merge overlapping ranges
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out, last = "", 0
        for s, e in merged:
            out += _html.escape(text[last:s])
            out += SPAN.format(_html.escape(text[s:e]))
            last = e
        out += _html.escape(text[last:])
        return out

    model_str = model_val.strip() if model_val else ""
    if not model_str or model_str in ("—", "None", "nan"):
        return _html.escape(gt_text), False

    # ── Pass 1: exact case-insensitive ──────────────────────────────────────
    pattern = re.compile(re.escape(model_str), re.IGNORECASE)
    hits = [(m.start(), m.end()) for m in pattern.finditer(gt_text)]
    if hits:
        return build_html(gt_text, hits), True

    # ── Pass 2: word-level (words ≥ 4 chars from model_val) ─────────────────
    words = [w for w in re.findall(r"[A-Za-z0-9]+", model_str) if len(w) >= 4]
    if words:
        hits = []
        for word in words:
            wp = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            hits.extend((m.start(), m.end()) for m in wp.finditer(gt_text))
        if hits:
            return build_html(gt_text, hits), True

    # ── Pass 3: fuzzy substring (SequenceMatcher, skip for very long model vals)
    n = len(model_str)
    if n < 3 or n > 120:
        return _html.escape(gt_text), False

    gt_lower    = gt_text.lower()
    model_lower = model_str.lower()
    best_ratio, best_span = 0.0, (0, 0)

    for window in sorted({n, max(3, int(n * 0.8)), int(n * 1.2)}):
        for i in range(len(gt_lower) - window + 1):
            ratio = SequenceMatcher(
                None, model_lower, gt_lower[i : i + window], autojunk=False
            ).ratio()
            if ratio > best_ratio:
                best_ratio, best_span = ratio, (i, i + window)

    if best_ratio >= 0.82:
        return build_html(gt_text, [best_span]), True

    return _html.escape(gt_text), False


def _render_value(label: str, val, box_class: str, key: str,
                  highlight_val: str | None = None):
    """Render a value in a coloured, vertically-resizable box.
    Height is persisted across rows via localStorage keyed by a stable
    sanitised label so resizing one row carries over to the next.
    If highlight_val is given, regex-search for it in val and wrap hits in red."""
    val_str = str(val) if val is not None else "—"

    if highlight_val is not None and val_str not in ("—",):
        display_html, _ = _find_and_highlight(val_str, str(highlight_val))
    else:
        display_html = _html.escape(val_str)

    # Stable key for localStorage — derived from label, not row position
    storage_key = re.sub(r'\W+', '_', label).strip('_')

    # All boxes are vertically resizable; long content gets a taller default
    default_h = "height:220px;" if len(val_str) > 250 else "height:80px;"
    box_style  = f"overflow-y:auto;resize:vertical;white-space:pre-wrap;min-height:40px;{default_h}"

    st.markdown(f"**{label}**")
    st.markdown(
        f'<div class="{box_class}" style="{box_style}" data-vl-box="{storage_key}">'
        f'{display_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers  (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_binary_metrics(y_true_raw, y_pred_raw, pos_class: str) -> dict:
    if not HAS_SKLEARN:
        return {}
    try:
        pos = str(pos_class)
        yt  = (pd.Series(y_true_raw).astype(str) == pos).astype(int)
        yp  = (pd.Series(y_pred_raw).astype(str) == pos).astype(int)
        cm  = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        n    = len(yt)
        sens = tp / (tp + fn)           if (tp + fn)           else float("nan")
        spec = tn / (tn + fp)           if (tn + fp)           else float("nan")
        prec = tp / (tp + fp)           if (tp + fp)           else float("nan")
        f1   = (2 * prec * sens / (prec + sens)
                if not (np.isnan(prec) or np.isnan(sens)) and (prec + sens) > 0
                else float("nan"))
        acc  = (tp + tn) / n            if n                   else float("nan")
        return dict(n=n, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                    sensitivity=sens, specificity=spec, precision=prec, f1=f1, accuracy=acc)
    except Exception:
        return {}


def _compute_categorical_metrics(y_true_raw, y_pred_raw) -> dict:
    if not HAS_SKLEARN:
        return {}
    try:
        yt    = pd.Series(y_true_raw).astype(str)
        yp    = pd.Series(y_pred_raw).astype(str)
        acc   = accuracy_score(yt, yp)
        kappa = cohen_kappa_score(yt, yp)
        return dict(n=len(yt), accuracy=acc, kappa=kappa)
    except Exception:
        return {}


def _render_binary_summary(m: dict, pos_class: str):
    if not m:
        st.info("Not enough data to compute binary metrics.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    def card(col, label, val):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div></div>',
            unsafe_allow_html=True,
        )
    fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "—"
    card(c1, "Sensitivity / Recall", fmt(m["sensitivity"]))
    card(c2, "Specificity",          fmt(m["specificity"]))
    card(c3, "Precision",            fmt(m["precision"]))
    card(c4, "F1 Score",             fmt(m["f1"]))
    card(c5, "Accuracy",             fmt(m["accuracy"]))
    if HAS_PLOTLY:
        neg   = f"not {pos_class}"
        z     = [[m["tp"], m["fn"]], [m["fp"], m["tn"]]]
        ann   = [[f"TP\n{m['tp']}", f"FN\n{m['fn']}"], [f"FP\n{m['fp']}", f"TN\n{m['tn']}"]]
        fig   = ff.create_annotated_heatmap(
            z=z, x=[pos_class, neg], y=[pos_class, neg],
            annotation_text=ann,
            colorscale=[[0, "#fff3cd"], [1, "#28a745"]], showscale=False,
        )
        fig.update_layout(
            title=f"Confusion matrix  (positive = '{pos_class}',  n={m['n']})",
            xaxis_title="Predicted", yaxis_title="Actual",
            yaxis_autorange="reversed", height=320,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_categorical_summary(m: dict, y_true_raw, y_pred_raw):
    if not m:
        st.info("Not enough data to compute categorical metrics.")
        return
    c1, c2, c3 = st.columns(3)
    def card(col, label, val):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div></div>',
            unsafe_allow_html=True,
        )
    card(c1, "Accuracy",      f"{m['accuracy']:.3f}")
    card(c2, "Cohen's Kappa", f"{m['kappa']:.3f}")
    card(c3, "n reviewed",    str(m["n"]))
    _render_confusion_matrix_raw(y_true_raw, y_pred_raw, f"n={m['n']}")


def _render_confusion_matrix_raw(y_true_raw, y_pred_raw, title_suffix=""):
    if not (HAS_SKLEARN and HAS_PLOTLY):
        st.dataframe(pd.crosstab(
            pd.Series(y_true_raw).astype(str),
            pd.Series(y_pred_raw).astype(str),
        ), use_container_width=True)
        return
    yt     = pd.Series(y_true_raw).astype(str)
    yp     = pd.Series(y_pred_raw).astype(str)
    labels = sorted(set(yt) | set(yp))
    normalize = st.checkbox("Normalize rows", value=False,
                            key=f"norm_{'_'.join(labels[:3])}_{len(labels)}")
    cm = confusion_matrix(yt, yp, labels=labels)
    if normalize:
        cm_d = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_d = np.nan_to_num(cm_d)
        ann  = [[f"{v:.2f}" for v in r] for r in cm_d]
    else:
        cm_d = cm.astype(float)
        ann  = [[str(v) for v in r] for r in cm]
    fig = ff.create_annotated_heatmap(
        z=cm_d, x=labels, y=labels, annotation_text=ann,
        colorscale="Blues", showscale=True,
    )
    fig.update_layout(
        title=f"Confusion matrix  {title_suffix}",
        xaxis_title="Predicted", yaxis_title="Actual",
        yaxis_autorange="reversed", height=max(350, 80 * len(labels)),
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Classification report"):
        st.code(classification_report(yt, yp, labels=labels, zero_division=0))


def _render_confusion_matrix(df: pd.DataFrame, val_col: str, gt_col: str):
    subset = df[[gt_col, val_col]].dropna()
    if subset.empty:
        st.info("No data available.")
        return
    mode      = S.col_metric_mode.get(val_col, "categorical")
    pos_class = S.col_positive_class.get(val_col, "True")
    if mode == "binary":
        m = _compute_binary_metrics(subset[gt_col], subset[val_col], pos_class)
        _render_binary_summary(m, pos_class)
    else:
        m = _compute_categorical_metrics(subset[gt_col], subset[val_col])
        _render_categorical_summary(m, subset[gt_col], subset[val_col])


# ─────────────────────────────────────────────────────────────────────────────
# Page: Setup
# ─────────────────────────────────────────────────────────────────────────────

def page_setup():
    st.title("LLM Output Validator")

    # ── Flush any browse-picker results into widget keys BEFORE widgets render.
    #    We must do this first: Streamlit raises an error if you write to a
    #    widget's key after the widget has already rendered in this execution.
    for _desired, _widget in [
        ("resume_path_desired", "resume_path_input"),
        ("save_dir_desired",    "save_dir_widget"),
    ]:
        if st.session_state.get(_desired):
            st.session_state[_widget] = st.session_state.pop(_desired)

    # ── Resume existing project ──────────────────────────────────────────────
    st.subheader("Resume a saved project")
    rp_col, rp_btn = st.columns([5, 1])
    with rp_col:
        resume_input = st.text_input(
            "Path to existing project directory",
            placeholder="/Users/you/projects/my-validation",
            key="resume_path_input",
        )
    with rp_btn:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if st.button("📁 Browse", key="browse_resume", use_container_width=True):
            _pick_directory("resume_path_desired")   # ← staging key, not widget key
            st.rerun()
    if resume_input:
        summary = project_summary(resume_input)
        if summary:
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Project",  summary["name"])
            rc2.metric("Reviewer", summary["reviewer"])
            rc3.metric("Progress", summary["progress"])
            rc4.metric("Last saved", summary["saved"])
            if st.button("▶ Resume this project", type="primary"):
                ok, msg = load_project(resume_input)
                if ok:
                    st.success(msg)
                    S.page = "validate"
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.warning("No valid project found at that path.")

    st.divider()

    # ── New project ──────────────────────────────────────────────────────────
    st.subheader("Configure a new project")

    c1, c2 = st.columns(2)
    with c1:
        S.project_name = st.text_input("Project name", value=S.project_name)
    with c2:
        S.reviewer = st.text_input("Reviewer ID", value=S.reviewer)

    # Initialise widget state from session so Browse result survives reruns
    if "save_dir_widget" not in st.session_state:
        st.session_state["save_dir_widget"] = S.save_dir
    sd_col, sd_btn = st.columns([5, 1])
    with sd_col:
        save_dir_input = st.text_input(
            "Save directory (full path — created if it doesn't exist)",
            placeholder="/Users/you/projects/my-validation",
            key="save_dir_widget",
            help="Project state is auto-saved here after every verdict so you can resume later.",
        )
    with sd_btn:
        st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        if st.button("📁 Browse", key="browse_save_dir", use_container_width=True):
            _pick_directory("save_dir_desired")      # ← staging key, not widget key
            st.rerun()
    S.save_dir = save_dir_input

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
        st.info("Upload a file to continue.")
        return

    df   = S.df
    cols = list(df.columns)

    st.subheader("Preview (first 5 rows)")
    st.dataframe(df.head(5), use_container_width=True)

    # ── Column roles ────────────────────────────────────────────────────────
    st.subheader("Column Roles")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        gt = st.multiselect("Ground truth columns (left panel)", cols, default=[])
    with cc2:
        val = st.multiselect("Validation columns (model outputs)", cols, default=[])
    with cc3:
        explore = st.multiselect(
            "Exploration columns (togglable on left)",
            cols, default=[],
            help="Extra columns you can show/hide during review for context — "
                 "not validated, not highlighted.",
        )
    S.explore_cols = explore

    # ── GT linkage ───────────────────────────────────────────────────────────
    if val and gt:
        st.subheader("Link validation columns → ground truth")
        st.caption("Linked columns are highlighted during review. First linked = primary for metrics.")
        mapping: dict[str, list[str]] = {}
        link_cols = st.columns(min(len(val), 3))
        for i, vc in enumerate(val):
            with link_cols[i % len(link_cols)]:
                chosen = st.multiselect(
                    f"`{vc}`", options=gt, default=[],
                    key=f"gtmap_{vc}",
                )
                if chosen:
                    mapping[vc] = chosen
        S.col_gt_map = mapping

    # ── Metric type ──────────────────────────────────────────────────────────
    if val:
        st.subheader("Metric type per validation column")
        st.caption(
            "**Categorical** → accuracy + Cohen's κ + confusion matrix.  "
            "**Binary** → sensitivity, specificity, precision, recall, F1."
        )
        metric_modes:     dict[str, str] = {}
        positive_classes: dict[str, str] = {}
        for vc in val:
            mc1, mc2, mc3 = st.columns([2, 2, 2])
            with mc1:
                st.markdown(f"**`{vc}`**")
            with mc2:
                cur = S.col_metric_mode.get(vc, "categorical")
                choice = st.radio(
                    "type", ["categorical", "binary"],
                    format_func=lambda x: "Categorical" if x == "categorical" else "Binary",
                    horizontal=True,
                    index=0 if cur == "categorical" else 1,
                    key=f"mmode_{vc}",
                    label_visibility="collapsed",
                )
                metric_modes[vc] = choice
            with mc3:
                if choice == "binary":
                    pos = st.text_input(
                        "Positive class label",
                        value=S.col_positive_class.get(vc, "True"),
                        key=f"posclass_{vc}",
                    )
                    positive_classes[vc] = pos
                else:
                    st.caption("N/A")
        S.col_metric_mode    = metric_modes
        S.col_positive_class = positive_classes

    # ── Validation options ───────────────────────────────────────────────────
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

    multi_col_mode = st.checkbox(
        "Validate all columns simultaneously (show every model output per row on the right)",
        value=S.multi_col_mode,
        help="When checked, all validation columns are shown side-by-side on the right panel "
             "so you can verdict each one without advancing to the next row. "
             "The left panel and all existing functionality are unchanged.",
    )

    st.markdown("---")
    if st.button("▶ Start Validation", type="primary"):
        if not val:
            st.error("Select at least one validation column.")
            return
        if not S.save_dir:
            st.error("Please enter a save directory — required for auto-save.")
            return
        S.gt_cols        = gt
        S.val_cols       = val
        S.mode           = mode
        S.multi_col_mode = multi_col_mode
        working_df = df.copy()
        if sample_n > 0:
            working_df = df.sample(n=int(sample_n), random_state=42).reset_index(drop=True)
            S.df = working_df
        if S.multi_col_mode:
            rows = list(S.df.index)
            if mode == "random":
                random.shuffle(rows)
            S.queue = [(r, "__MULTI__") for r in rows]
        else:
            S.queue = build_queue(S.df, val, mode)
        S.pos       = 0
        S.row_notes = {}
        S.results   = {}
        save_project()   # initial save
        S.page = "validate"
        st.rerun()

    # ── Quick confusion matrix ────────────────────────────────────────────────
    cols_with_gt = [vc for vc, gts in S.col_gt_map.items() if gts]
    if cols_with_gt:
        st.markdown("---")
        st.subheader("Quick Confusion Matrix (full dataset)")
        qcm_val = st.selectbox("Validation column", cols_with_gt, key="qcm_val")
        linked  = _as_list(S.col_gt_map.get(qcm_val))
        qcm_gt  = st.selectbox("Ground truth column", linked, key="qcm_gt")
        if st.button("Generate", key="qcm_btn"):
            _render_confusion_matrix(S.df, qcm_val, qcm_gt)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Validate
# ─────────────────────────────────────────────────────────────────────────────

def page_validate():
    queue = S.queue
    total = len(queue)

    if not queue:
        st.error("Queue is empty — go back to Setup.")
        if st.button("← Setup"):
            S.page = "setup"
            st.rerun()
        return

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### {S.project_name}")
        st.markdown(f"**Reviewer:** {S.reviewer}  \n**Mode:** {S.mode}")
        if S.save_dir:
            st.markdown(f'<div class="save-info">💾 {S.save_dir}</div>', unsafe_allow_html=True)
        st.markdown("---")

        safe_pos = min(S.pos + 1, total)
        jump = st.number_input("Jump to item #", min_value=1, max_value=total,
                               value=safe_pos, step=1)
        if st.button("Go", key="jump_btn"):
            S.pos = int(jump) - 1
            st.rerun()

        if st.button("💾 Save now", use_container_width=True):
            save_project()
            st.success("Saved.")

        st.markdown("---")
        if st.button("📊 View Results", use_container_width=True):
            S.page = "results"
            st.rerun()
        if st.button("⚙ Setup", use_container_width=True):
            S.page = "setup"
            st.rerun()

        if S.results:
            st.markdown("---\n**Progress by column:**")
            # In multi-col mode every queue entry covers all val_cols for one row
            n_rows = len(set(r for r, _ in queue))
            for vc in S.val_cols:
                col_r   = [r for r in S.results.values() if r["validation_col"] == vc]
                n_col   = n_rows if S.multi_col_mode else sum(1 for (_, ci) in queue if ci == vc)
                done    = len(col_r)
                correct = sum(1 for r in col_r if r["verdict"] == "correct")
                acc     = f"{correct/done:.0%}" if done else "—"
                badge   = "🔵" if S.col_metric_mode.get(vc, "categorical") == "categorical" else "🟢"
                st.markdown(f"{badge} **{vc}**: {done}/{n_col} · {acc}")

    pos = S.pos

    # ── Header ───────────────────────────────────────────────────────────────
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
        done_btn, results_btn, setup_btn = st.columns(3)
        with done_btn:
            if st.button("📊 View Results", type="primary", use_container_width=True):
                S.page = "results"
                st.rerun()
        with results_btn:
            if st.button("⚙ Setup / New Project", use_container_width=True):
                S.page = "setup"
                st.rerun()
        with setup_btn:
            if st.button("🔄 Review Again (reset queue)", use_container_width=True):
                S.queue   = build_queue(S.df, S.val_cols, S.mode)
                S.pos     = 0
                S.results = {}
                save_project()
                st.rerun()
        _keyboard_component(False)
        return

    row_idx, col_name = queue[pos]
    row        = S.df.loc[row_idx]
    is_multi   = (col_name == "__MULTI__")

    # ── Notes widget — one per row, shared across all columns ────────────────
    notes_key = f"rownotes_{row_idx}"
    if notes_key not in st.session_state:
        st.session_state[notes_key] = S.row_notes.get(str(row_idx), "")

    # ── Variables used by single-col mode only ───────────────────────────────
    if not is_multi:
        existing       = S.results.get((row_idx, col_name))
        linked_gts     = _as_list(S.col_gt_map.get(col_name))
        metric_mode    = S.col_metric_mode.get(col_name, "categorical")
        pos_class      = S.col_positive_class.get(col_name, "True")
        model_val      = str(row.get(col_name, "")) if col_name in row.index else ""
        gt_highlight_map = {}   # unused in single-col mode
    else:
        existing    = None
        linked_gts  = list(S.gt_cols)   # all GT cols shown as linked
        metric_mode = None
        pos_class   = None
        model_val   = ""
        # Reverse of col_gt_map: for each GT col, what model value should be
        # highlighted in it?  Built so each GT col is highlighted with the
        # model value from its linked validation column.
        gt_highlight_map: dict[str, str] = {}
        for vc in S.val_cols:
            mv = str(row.get(vc, "")) if vc in row.index else ""
            for gc in _as_list(S.col_gt_map.get(vc)):
                if gc not in gt_highlight_map:   # first linked val_col wins
                    gt_highlight_map[gc] = mv

        # Auto-default: pre-fill all columns as "correct" when first arriving
        # at a row. Reviewer only needs to flip the incorrect ones.
        if not any(S.results.get((row_idx, vc)) for vc in S.val_cols):
            for vc in S.val_cols:
                record_verdict(row_idx, vc, "correct")
            save_project()

    left, right = st.columns([1, 1], gap="large")

    # ── Left: ground truth (identical for both modes) ────────────────────────
    with left:
        # Zero-height marker: JS walks UP from here to find this column div.
        st.markdown('<div id="vl-left-col-marker" style="height:0;overflow:hidden;margin:0;padding:0"></div>',
                    unsafe_allow_html=True)
        lh1, lh2 = st.columns([3, 1])
        with lh1:
            st.markdown("### Ground Truth")
        with lh2:
            if S.explore_cols:
                toggle_label = (
                    f"👁 Hide extras ({len(S.explore_cols)})"
                    if S.explore_visible
                    else f"👁 Show extras ({len(S.explore_cols)})"
                )
                if st.button(toggle_label, key=f"explore_toggle_{pos}",
                             use_container_width=True):
                    S.explore_visible = not S.explore_visible
                    st.rerun()

        for c in S.gt_cols:
            is_linked = c in linked_gts
            if is_multi:
                hval = gt_highlight_map.get(c)   # per-column model value
            else:
                hval = model_val if is_linked else None
            _render_value(
                f"{c}  ✦ compare" if is_linked else c,
                row.get(c, "—"),
                "linked-gt-box" if is_linked else "gt-box",
                f"gt_{c}_{pos}",
                highlight_val=hval,
            )
        for gc in [g for g in linked_gts if g not in S.gt_cols]:
            hval = gt_highlight_map.get(gc) if is_multi else model_val
            _render_value(f"{gc}  ✦ compare", row.get(gc, "—"),
                          "linked-gt-box", f"lgt_{gc}_{pos}",
                          highlight_val=hval)

        if S.explore_visible and S.explore_cols:
            st.markdown("---")
            st.markdown("**Exploration**")
            for c in S.explore_cols:
                _render_value(c, row.get(c, "—"), "explore-box", f"exp_{c}_{pos}")

        st.caption(f"Row {row_idx}")

    # ── Right: single-col verdict ─────────────────────────────────────────────
    with right:
        if not is_multi:
            badge_html = (
                f'<span style="background:#d4edda;color:#155724;padding:2px 8px;'
                f'border-radius:4px;font-size:0.8em;font-weight:600;">BINARY · pos="{pos_class}"</span>'
                if metric_mode == "binary" else
                f'<span style="background:#cce5ff;color:#004085;padding:2px 8px;'
                f'border-radius:4px;font-size:0.8em;font-weight:600;">CATEGORICAL</span>'
            )
            st.markdown(f"### Model Output — `{col_name}` &nbsp; {badge_html}",
                        unsafe_allow_html=True)
            _render_value("Model output", model_val or "—", "model-box", f"mv_{pos}")
            st.markdown("---")

            kb_col, hint_col = st.columns([1, 3])
            with kb_col:
                if st.button("⌨ Keys ON" if S.kb_active else "⌨ Keys OFF",
                             key=f"kb_toggle_{pos}"):
                    S.kb_active = not S.kb_active
                    st.rerun()
            with hint_col:
                if S.kb_active:
                    st.markdown(
                        '<span class="shortcut-hint">'
                        '→ correct &nbsp;|&nbsp; ← incorrect &nbsp;|&nbsp; ↑ next &nbsp;|&nbsp; ↓ back'
                        '</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="shortcut-off">keyboard shortcuts off</span>',
                                unsafe_allow_html=True)

            bc1, bc2, bc3, bc4 = st.columns(4)
            with bc1:
                if st.button("✓ Correct", type="primary", key=f"yes_{pos}", use_container_width=True):
                    _flush_notes_and_save(row_idx)
                    record_verdict(row_idx, col_name, "correct")
                    save_project()
                    S.pos = min(pos + 1, total)
                    st.rerun()
            with bc2:
                if st.button("✗ Incorrect", key=f"no_{pos}", use_container_width=True):
                    _flush_notes_and_save(row_idx)
                    record_verdict(row_idx, col_name, "incorrect")
                    save_project()
                    S.pos = min(pos + 1, total)
                    st.rerun()
            with bc3:
                if st.button("↑ Next", key=f"next_{pos}", use_container_width=True):
                    _flush_notes_and_save(row_idx)
                    S.pos = min(pos + 1, total)
                    st.rerun()
            with bc4:
                if st.button("↓ Back", key=f"back_{pos}", use_container_width=True):
                    _flush_notes_and_save(row_idx)
                    S.pos = max(pos - 1, 0)
                    st.rerun()

            if existing:
                v   = existing["verdict"]
                cls = {"correct": "verdict-correct", "incorrect": "verdict-incorrect",
                       "skipped": "verdict-skipped"}.get(v, "")
                st.markdown(f'<div class="{cls}">Previously: {v.upper()}</div>',
                            unsafe_allow_html=True)

        # ── Right: multi-col verdict panel ────────────────────────────────────
        else:
            n_judged = sum(
                1 for vc in S.val_cols
                if S.results.get((row_idx, vc), {}).get("verdict") in ("correct", "incorrect")
            )
            all_judged = (n_judged == len(S.val_cols))

            # Header row: title + keyboard toggle
            mh1, mh2 = st.columns([3, 1])
            with mh1:
                st.markdown("### Model Outputs — All Columns")
                st.caption(
                    f"{n_judged} / {len(S.val_cols)} judged"
                    + ("  ✓ all done" if all_judged else "")
                )
            with mh2:
                if st.button("⌨ Keys ON" if S.kb_active else "⌨ Keys OFF",
                             key=f"mc_kb_toggle_{pos}"):
                    S.kb_active = not S.kb_active
                    st.rerun()
                if S.kb_active:
                    st.markdown(
                        '<div style="font-size:0.78em;color:#555;margin-top:2px">'
                        '→ all ✓ &nbsp;← all ✗<br>↑ next &nbsp;↓ back</div>',
                        unsafe_allow_html=True,
                    )

            # Mark-all buttons
            ma1, ma2 = st.columns(2)
            with ma1:
                if st.button("✓ All Correct", key=f"mc_all_yes_{pos}",
                             use_container_width=True, type="primary"):
                    _flush_notes_and_save(row_idx)
                    for vc in S.val_cols:
                        record_verdict(row_idx, vc, "correct")
                    save_project()
                    st.rerun()
            with ma2:
                if st.button("✗ All Incorrect", key=f"mc_all_no_{pos}",
                             use_container_width=True):
                    _flush_notes_and_save(row_idx)
                    for vc in S.val_cols:
                        record_verdict(row_idx, vc, "incorrect")
                    save_project()
                    st.rerun()

            st.markdown("---")

            # One compact row per validation column
            for vc in S.val_cols:
                mv = str(row.get(vc, "—")) if vc in row.index else "—"
                vc_res = S.results.get((row_idx, vc))
                cur_v  = vc_res["verdict"] if vc_res else None
                mm     = S.col_metric_mode.get(vc, "categorical")
                pc     = S.col_positive_class.get(vc, "True")

                badge_html = (
                    f'<span style="background:#d4edda;color:#155724;padding:1px 5px;'
                    f'border-radius:3px;font-size:0.75em;font-weight:600;">BIN·{pc}</span>'
                    if mm == "binary" else
                    f'<span style="background:#cce5ff;color:#004085;padding:1px 5px;'
                    f'border-radius:3px;font-size:0.75em;font-weight:600;">CAT</span>'
                )
                if cur_v == "correct":
                    status_html = '<span class="verdict-correct" style="font-size:1.3em">✓</span>'
                elif cur_v == "incorrect":
                    status_html = '<span class="verdict-incorrect" style="font-size:1.3em">✗</span>'
                else:
                    status_html = '<span style="color:#bbb;font-size:1.3em">·</span>'

                ci, cy, cn, cs = st.columns([5, 1, 1, 0.6])
                with ci:
                    st.markdown(
                        f'<div style="font-weight:600;font-size:0.9em">'
                        f'`{vc}` &nbsp;{badge_html}</div>',
                        unsafe_allow_html=True,
                    )
                    mv_esc = _html.escape(mv)
                    st.markdown(
                        f'<div style="background:#fff8e1;border-left:3px solid #ffc107;'
                        f'padding:4px 8px;border-radius:4px;font-size:0.9em;'
                        f'max-height:72px;overflow-y:auto;white-space:pre-wrap;">'
                        f'{mv_esc}</div>',
                        unsafe_allow_html=True,
                    )
                with cy:
                    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
                    if st.button("✓", key=f"mc_yes_{vc}_{pos}",
                                 use_container_width=True, help="Correct",
                                 type="primary" if cur_v == "correct" else "secondary"):
                        _flush_notes_and_save(row_idx)
                        record_verdict(row_idx, vc, "correct")
                        save_project()
                        st.rerun()
                with cn:
                    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
                    if st.button("✗", key=f"mc_no_{vc}_{pos}",
                                 use_container_width=True, help="Incorrect"):
                        _flush_notes_and_save(row_idx)
                        record_verdict(row_idx, vc, "incorrect")
                        save_project()
                        st.rerun()
                with cs:
                    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align:center">{status_html}</div>',
                                unsafe_allow_html=True)

                st.markdown(
                    '<hr style="margin:4px 0;border:none;border-top:1px solid #eee">',
                    unsafe_allow_html=True,
                )

            # Navigation
            st.markdown("")
            nb1, nb2 = st.columns(2)
            with nb1:
                if st.button("↓ Back", key=f"mc_back_{pos}", use_container_width=True):
                    _flush_notes_and_save(row_idx)
                    S.pos = max(pos - 1, 0)
                    st.rerun()
            with nb2:
                if st.button("↑ Next", key=f"mc_next_{pos}",
                             use_container_width=True, type="primary"):
                    _flush_notes_and_save(row_idx)
                    S.pos = min(pos + 1, total)
                    st.rerun()

    # ── Row notes — full-width below both panels (both modes) ────────────────
    st.markdown("---")
    st.text_area(
        f"📝 Notes for row {row_idx}  *(shared across all columns — appends as you add more)*",
        key=notes_key,
        height=90,
        placeholder="Any observations, corrections, or flags for this row…",
    )

    if is_multi:
        # Multi-col keyboard: → all correct | ← all incorrect | ↑ next | ↓ back
        _keyboard_component_multi(S.kb_active)
    else:
        _keyboard_component(S.kb_active)
    _resize_component()
    _textbox_resize_component()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Results  (unchanged structure, updated notebook snippet)
# ─────────────────────────────────────────────────────────────────────────────

def page_results():
    st.title(f"Results — {S.project_name}")
    results_df = results_to_df()

    tab_summary, tab_metrics, tab_cm, tab_raw = st.tabs(
        ["Summary & Export", "Metrics by Column", "Confusion Matrices", "Raw Data"]
    )

    with tab_summary:
        if results_df.empty:
            st.info("No results yet.")
        else:
            total     = len(results_df)
            correct   = (results_df.verdict == "correct").sum()
            incorrect = (results_df.verdict == "incorrect").sum()
            skipped   = (results_df.verdict == "skipped").sum()
            reviewed  = correct + incorrect

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Reviewed",  total)
            m2.metric("Correct",   correct)
            m3.metric("Incorrect", incorrect)
            m4.metric("Skipped",   skipped)
            m5.metric("Accuracy",  f"{correct/reviewed:.1%}" if reviewed else "—")

            st.subheader("By Column")
            rows_sum = []
            for vc in S.val_cols:
                g    = results_df[results_df.validation_col == vc]
                mode = S.col_metric_mode.get(vc, "categorical")
                row  = {
                    "column":      vc,
                    "metric_type": mode,
                    "reviewed":    (g.verdict != "skipped").sum(),
                    "correct":     (g.verdict == "correct").sum(),
                    "incorrect":   (g.verdict == "incorrect").sum(),
                    "skipped":     (g.verdict == "skipped").sum(),
                }
                n_rev = row["reviewed"]
                row["accuracy"] = f"{row['correct']/n_rev:.1%}" if n_rev else "—"
                if n_rev and S.col_gt_map.get(vc):
                    sub = results_df[
                        (results_df.validation_col == vc) &
                        (results_df.verdict != "skipped") &
                        results_df.ground_truth_value.notna() &
                        results_df.model_value.notna()
                    ]
                    if mode == "binary":
                        bm = _compute_binary_metrics(
                            sub.ground_truth_value, sub.model_value,
                            S.col_positive_class.get(vc, "True"))
                        row["sensitivity"] = f"{bm.get('sensitivity', float('nan')):.3f}"
                        row["specificity"] = f"{bm.get('specificity', float('nan')):.3f}"
                        row["F1"]          = f"{bm.get('f1', float('nan')):.3f}"
                    else:
                        cm = _compute_categorical_metrics(sub.ground_truth_value, sub.model_value)
                        row["kappa"] = f"{cm.get('kappa', float('nan')):.3f}"
                rows_sum.append(row)
            st.dataframe(pd.DataFrame(rows_sum), use_container_width=True)

            if HAS_PLOTLY:
                counts = (
                    results_df[results_df.verdict != "skipped"]
                    .groupby(["validation_col", "verdict"]).size()
                    .reset_index(name="count")
                )
                fig = px.bar(counts, x="validation_col", y="count", color="verdict",
                             barmode="group",
                             color_discrete_map={"correct": "#22c55e", "incorrect": "#ef4444"},
                             title="Correct vs Incorrect by Column")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Export")
            # Optional: let the user include specific original-data columns in the export
            all_src_cols  = list(S.df.columns) if S.df is not None else []
            base_col_set  = {"row_id", "validation_col", "verdict", "reviewer",
                             "timestamp", "row_notes"}
            available_extra = [c for c in all_src_cols if c not in base_col_set]
            extra_cols = st.multiselect(
                "Include additional columns from original data (optional)",
                options=available_extra,
                default=[],
                help="These columns are joined by row_id — use to add GT values, "
                     "report text, or any other field you want alongside the verdicts.",
            )

            export_df = results_to_export_df(extra_cols=extra_cols or None)
            base_count = len(base_col_set & set(export_df.columns))
            st.caption(
                f"{len(export_df)} rows · {len(export_df.columns)} columns "
                f"(6 audit cols" +
                (f" + {len(extra_cols)} original-data cols" if extra_cols else "") +
                "). Join back to your source file via `row_id` for any remaining columns."
            )

            dl1, dl2 = st.columns(2)
            fname = S.project_name.replace(" ", "_")
            with dl1:
                buf = io.StringIO()
                export_df.to_csv(buf, index=False)
                st.download_button("⬇ Download CSV", data=buf.getvalue(),
                                   file_name=f"{fname}_validation.csv", mime="text/csv")
            with dl2:
                st.download_button("⬇ Download JSON",
                                   data=export_df.to_json(orient="records", indent=2),
                                   file_name=f"{fname}_validation.json",
                                   mime="application/json")

            st.subheader("Loading in a notebook")
            st.code(
                'import pandas as pd\n'
                'from sklearn.metrics import cohen_kappa_score, classification_report\n\n'
                '# Load the validation results (verdicts only — no duplicated source data)\n'
                'results = pd.read_csv("..._validation.csv")\n'
                '# Load your original data file to get GT and model values\n'
                'orig = pd.read_csv("your_original_data.csv")\n\n'
                '# --- Categorical column ---\n'
                'col, gt_col = "model_rec_code", "gt_rec_code"\n'
                'sub = (\n'
                '    results[(results.validation_col == col) & (results.verdict != "skipped")]\n'
                '    .merge(orig[["row_id", col, gt_col]].rename(columns={"row_id": "row_id"}),\n'
                '           on="row_id", how="left")\n'
                ')\n'
                'print(cohen_kappa_score(sub[gt_col], sub[col]))\n\n'
                '# --- Binary column ---\n'
                'col, gt_col = "model_finding_present", "gt_finding_present"\n'
                'sub = (\n'
                '    results[(results.validation_col == col) & (results.verdict != "skipped")]\n'
                '    .merge(orig[["row_id", col, gt_col]], on="row_id", how="left")\n'
                ')\n'
                'y_true = (sub[gt_col] == "True").astype(int)\n'
                'y_pred = (sub[col]    == "True").astype(int)\n'
                'print(classification_report(y_true, y_pred))\n\n'
                '# --- Row-level notes ---\n'
                'notes = results[["row_id", "row_notes"]].drop_duplicates()',
                language="python",
            )

    with tab_metrics:
        st.subheader("Detailed Metrics by Column")
        cols_with_gt = [vc for vc, gts in S.col_gt_map.items() if gts]
        if not cols_with_gt:
            st.info("Link validation columns to ground truth in Setup.")
        elif results_df.empty:
            st.info("No results yet.")
        else:
            sel       = st.selectbox("Select column", cols_with_gt, key="metrics_sel")
            mode      = S.col_metric_mode.get(sel, "categorical")
            pos_class = S.col_positive_class.get(sel, "True")
            sub = results_df[
                (results_df.validation_col == sel) &
                (results_df.verdict != "skipped") &
                results_df.ground_truth_value.notna() &
                results_df.model_value.notna()
            ]
            if sub.empty:
                st.info("No reviewed rows with GT values for this column.")
            else:
                st.markdown(
                    f"**Mode:** {'Binary  (positive = `' + pos_class + '`)' if mode == 'binary' else 'Categorical'}"
                    f"  ·  **n = {len(sub)}**"
                )
                # Gate heavy Plotly/sklearn render behind a button so the page
                # doesn't freeze when first loaded mid-validation.
                metrics_req_key = ("metrics", sel, mode, pos_class)
                if st.button("Compute metrics", key="compute_metrics_btn", type="primary"):
                    st.session_state["metrics_last_req"] = metrics_req_key
                if st.session_state.get("metrics_last_req") == metrics_req_key:
                    if mode == "binary":
                        m = _compute_binary_metrics(sub.ground_truth_value, sub.model_value, pos_class)
                        _render_binary_summary(m, pos_class)
                    else:
                        m = _compute_categorical_metrics(sub.ground_truth_value, sub.model_value)
                        _render_categorical_summary(m, sub.ground_truth_value, sub.model_value)

    with tab_cm:
        st.subheader("Confusion Matrix Generator")
        cols_with_gt = [vc for vc, gts in S.col_gt_map.items() if gts]
        if not cols_with_gt:
            st.info("No ground truth links configured.")
        else:
            # ── From validated rows ──────────────────────────────────────────
            st.markdown("#### From validated rows only")
            sel_r    = st.selectbox("Validation column", cols_with_gt, key="cm_rev_val")
            linked_r = _as_list(S.col_gt_map.get(sel_r))
            gt_sel_r = st.selectbox("Ground truth column", linked_r, key="cm_rev_gt")

            cm_rev_req_key = ("cm_rev", sel_r, gt_sel_r)
            if st.button("Generate (validated rows)", key="cm_rev_btn", type="primary"):
                st.session_state["cm_rev_last_req"] = cm_rev_req_key

            if st.session_state.get("cm_rev_last_req") == cm_rev_req_key:
                if results_df.empty:
                    st.info("No results yet.")
                else:
                    sub_r = results_df[
                        (results_df.validation_col == sel_r) &
                        (results_df.verdict != "skipped")
                    ].copy()
                    primary_gt = (_as_list(S.col_gt_map.get(sel_r)) or [None])[0]
                    if gt_sel_r != primary_gt:
                        sub_r["ground_truth_value"] = S.df.loc[
                            sub_r["row_id"].tolist(), gt_sel_r].values
                    sub_r = sub_r.dropna(subset=["ground_truth_value", "model_value"])
                    if sub_r.empty:
                        st.info("No reviewed rows yet.")
                    else:
                        mode_r = S.col_metric_mode.get(sel_r, "categorical")
                        pos_r  = S.col_positive_class.get(sel_r, "True")
                        if mode_r == "binary":
                            m = _compute_binary_metrics(sub_r.ground_truth_value,
                                                        sub_r.model_value, pos_r)
                            _render_binary_summary(m, pos_r)
                        else:
                            _render_confusion_matrix_raw(sub_r.ground_truth_value,
                                                         sub_r.model_value, f"n={len(sub_r)}")

            # ── From full dataset ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### From full dataset (all rows)")
            sel2    = st.selectbox("Validation column", cols_with_gt, key="cm_full_val")
            linked2 = _as_list(S.col_gt_map.get(sel2))
            gt2     = st.selectbox("Ground truth column", linked2, key="cm_full_gt")

            cm_full_req_key = ("cm_full", sel2, gt2)
            if st.button("Generate (full dataset)", key="cm_full_btn", type="primary"):
                st.session_state["cm_full_last_req"] = cm_full_req_key

            if st.session_state.get("cm_full_last_req") == cm_full_req_key:
                if S.df is not None:
                    _render_confusion_matrix(S.df, sel2, gt2)

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
