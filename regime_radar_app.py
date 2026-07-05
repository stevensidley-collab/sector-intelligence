"""
Regime Radar — standalone Streamlit app.

Runs in the same repo as sector-intelligence, sharing the same .env and
dependencies. Start with:

    uv run streamlit run regime_radar_app.py --server.port 8502

(Use a different port to run alongside sector-intelligence/app.py.)
"""

import html
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from regime_radar import MODELS, run_regime_radar

st.set_page_config(
    page_title="Regime Radar",
    page_icon="🔭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Archive — shared outputs/ directory with sector-intelligence
# ---------------------------------------------------------------------------
_ARCHIVE_DIR  = Path(__file__).parent / "outputs"
_ARCHIVE_FILE = _ARCHIVE_DIR / "archive.jsonl"
_ARCHIVE_DIR.mkdir(exist_ok=True)


def _append_archive(result: str, model: str) -> None:
    record = {
        "tab":       "radar",
        "trend":     "Regime radar scan",
        "result":    result,
        "model":     model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with _ARCHIVE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_radar_history() -> list[dict]:
    if not _ARCHIVE_FILE.exists():
        return []
    records = []
    with _ARCHIVE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("tab") == "radar":
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return list(reversed(records))  # newest first


# ---------------------------------------------------------------------------
# Link renderer
# ---------------------------------------------------------------------------
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _render(text: str) -> str:
    stash: dict = {}

    def _stash(m):
        token = f"\x00L{len(stash)}\x00"
        label = html.escape(m.group(1))
        url   = html.escape(m.group(2), quote=True)
        stash[token] = (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
        )
        return token

    escaped = html.escape(_LINK_RE.sub(_stash, text))
    for token, anchor in stash.items():
        escaped = escaped.replace(token, anchor)
    return escaped


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔭 Regime Radar")
st.caption(
    "Monthly capital-rotation scan · Low-confidence candidates to watch · "
    "Not investment advice"
)

with st.sidebar:
    st.header("Model")
    model_label = st.radio(
        "Reasoning model",
        options=list(MODELS.keys()),
        index=0,
    )
    selected_model = MODELS[model_label]
    st.caption("Sonnet reasons more sharply but costs ~5× more per scan.")
    st.divider()
    st.markdown(
        "**Cadence: monthly.**\n\n"
        "Capital rotation is slow. Run once per month — more frequent scans "
        "add noise, not signal."
    )

# Initialise history
if "radar_history" not in st.session_state:
    st.session_state["radar_history"] = _load_radar_history()

st.markdown(
    "Scans four leading signals — **Policy/Government → Fund Formation → "
    "Current-trade exhaustion → Commentator drift** — to detect whether "
    "large-scale capital is beginning to rotate out of the current singularity "
    "themes (AI, nuclear, quantum, energy, launch) and toward another sector.\n\n"
    "Most scans will find no credible rotation signal. That is the expected and "
    "valued outcome."
)

col_run, col_note = st.columns([1, 3])
with col_run:
    run_scan = st.button("Run monthly scan", type="primary")
with col_note:
    st.caption(
        "⏱ Searches all four signal types, all tracked commentators, and all "
        "capital allocators — expect 3–5 minutes."
    )

if run_scan:
    result = ""
    try:
        with st.spinner(
            "Scanning policy commitments, fund formation, current-trade exhaustion, "
            "and commentator + allocator drift…"
        ):
            result = run_regime_radar(model=selected_model)
    except Exception as e:
        result = f"⚠️ Radar scan failed: {e}"

    _append_archive(result, selected_model)
    st.session_state["radar_history"].insert(0, {
        "result":    result,
        "model":     selected_model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    st.session_state["radar_ran_this_session"] = True

# Show latest result only if run this session
history = st.session_state["radar_history"]
if st.session_state.get("radar_ran_this_session") and history:
    st.divider()
    st.markdown(_render(history[0]["result"]), unsafe_allow_html=True)

# Prior scans — collapsed
prior = history if not st.session_state.get("radar_ran_this_session") else history[1:]
if prior:
    st.divider()
    st.subheader(f"Prior scans ({len(prior)})")
    for entry in prior:
        model_tag = entry.get("model", "").split("-")[1] if entry.get("model") else ""
        label = f"**{entry['timestamp']}** · {model_tag}"
        with st.expander(label, expanded=False):
            st.markdown(_render(entry["result"]), unsafe_allow_html=True)
