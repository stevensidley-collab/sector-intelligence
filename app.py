"""
Sector Intelligence — Streamlit UI.

Three second-derivative analysis tabs:
  🤖 AI      — AI infrastructure demand trends
  ⚛️ Nuclear  — nuclear energy demand and supply trends
  🔮 Quantum  — quantum computing commercialisation trends

Each tab uses the same bottleneck-tracing engine with domain-specific
suggested prompts. Output is a verified hypothesis, not a recommendation.
"""

import html
import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from derivative_analysis import MODELS, run_derivative_analysis, run_third_order_analysis

# ---------------------------------------------------------------------------
# Persistent archive — one JSON record per line in outputs/archive.jsonl
# ---------------------------------------------------------------------------
_ARCHIVE_DIR  = Path(__file__).parent / "outputs"
_ARCHIVE_FILE = _ARCHIVE_DIR / "archive.jsonl"
_ARCHIVE_DIR.mkdir(exist_ok=True)


def _append_to_archive(tab: str, trend: str, result: str, model: str) -> None:
    record = {
        "tab":       tab,
        "trend":     trend,
        "result":    result,
        "model":     model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with _ARCHIVE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_archive_for_tab(tab: str) -> list[dict]:
    """
    Return all runs for this tab as paired entries (second + third order),
    newest first.

    On disk each run is stored as two consecutive records:
      1. The second-order result (trend = bare trend string)
      2. The third-order result (trend = "[THIRD ORDER] ..." prefix)
    We re-pair them here so history entries have both result and t3_result.
    Legacy single-record entries (no third-order partner) are kept as-is.
    """
    if not _ARCHIVE_FILE.exists():
        return []

    raw = []
    with _ARCHIVE_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("tab") == tab:
                    raw.append(rec)
            except json.JSONDecodeError:
                pass

    # Pair consecutive second + third records
    paired: list[dict] = []
    i = 0
    while i < len(raw):
        rec = raw[i]
        if rec["trend"].startswith("[THIRD ORDER]"):
            i += 1
            continue  # orphan third-order record — skip
        # Look ahead for a matching third-order record
        if i + 1 < len(raw) and raw[i + 1]["trend"].startswith("[THIRD ORDER]"):
            paired.append({
                "trend":     rec["trend"],
                "result":    rec["result"],
                "t3_result": raw[i + 1]["result"],
                "model":     rec.get("model", ""),
                "timestamp": rec.get("timestamp", ""),
            })
            i += 2
        else:
            paired.append({
                "trend":     rec["trend"],
                "result":    rec["result"],
                "t3_result": None,
                "model":     rec.get("model", ""),
                "timestamp": rec.get("timestamp", ""),
            })
            i += 1

    return list(reversed(paired))  # newest first

st.set_page_config(
    page_title="Sector Intelligence",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 Sector Intelligence")
st.caption("Second-derivative supply-chain analysis · Verified bottleneck hypotheses · Not investment advice")

# Model selector — persists across all tabs for the session
with st.sidebar:
    st.header("Model")
    model_label = st.radio(
        "Reasoning model",
        options=list(MODELS.keys()),
        index=0,  # Haiku by default
    )
    selected_model = MODELS[model_label]
    st.caption("Sonnet reasons more sharply but costs ~5× more per run.")

# ---------------------------------------------------------------------------
# Link renderer — opens every markdown link in a new tab
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
# Shared analysis widget — renders identically in each tab
# ---------------------------------------------------------------------------
def analysis_tab(key: str, suggested: list[str], model: str = "claude-haiku-4-5-20251001") -> None:
    """
    Render a trend selector + run button for one domain tab.

    Each run executes second-order analysis immediately followed by an automatic
    third-order upstream pass. Both results are stored together in session state
    and on disk, so they survive page interactions and app restarts.

    key        — unique string to namespace Streamlit widget keys
    suggested  — list of pre-written trend strings for the domain
    """
    # Initialise per-tab history in session_state, seeded from disk archive
    hist_key = f"{key}_history"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = _load_archive_for_tab(key)

    options = suggested + ["Custom — type below"]
    choice  = st.selectbox("Suggested trend", options, key=f"{key}_select")

    if choice == "Custom — type below":
        trend = st.text_area(
            "Custom trend",
            placeholder="Describe the demand force you want to trace…",
            height=80,
            key=f"{key}_custom",
        )
    else:
        st.text_area(
            "Trend (editable)",
            value=choice,
            height=80,
            key=f"{key}_display",
        )
        # Read back the widget value so edits the user makes are captured.
        # st.text_area with value= seeds the widget but doesn't override
        # subsequent edits — those live in session_state under the key.
        trend = st.session_state.get(f"{key}_display", choice)

    run = st.button("Run analysis", type="primary", key=f"{key}_run",
                    disabled=not trend.strip())

    if run and trend.strip():
        # --- Second-order pass ---
        with st.spinner("Stage 1 of 2 — tracing the value chain (second-order)…"):
            result = run_derivative_analysis(trend.strip(), model=model)

        # --- Third-order pass — automatic, model self-applies all gates ---
        t3_result = ""
        try:
            with st.spinner("Stage 2 of 2 — walking upstream (third-order)…"):
                t3_result = run_third_order_analysis(
                    second_order_verdict=result,
                    trend=trend.strip(),
                    model=model,
                )
        except Exception as e:
            t3_result = f"⚠️ Third-order pass failed: {e}"

        entry = {
            "trend":     trend.strip(),
            "result":    result,
            "t3_result": t3_result,   # "" means ran but empty; None reserved for pre-feature archive entries
            "model":     model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        _append_to_archive(key, trend.strip(), result, model)
        _append_to_archive(key, f"[THIRD ORDER] {trend.strip()}", t3_result, model)
        st.session_state[hist_key].insert(0, entry)

    elif not trend.strip():
        st.info("Select or enter a trend above and click **Run analysis**.")

    # --- Show output only when this session has produced a run -------------
    # On startup (history loaded from archive) we show nothing until the user
    # explicitly clicks Run — so the app doesn't look like it just ran something.
    history = st.session_state[hist_key]
    ran_this_session = st.session_state.get(f"{key}_ran_this_session", False)

    if run and trend.strip():
        st.session_state[f"{key}_ran_this_session"] = True
        ran_this_session = True

    if ran_this_session and history:
        latest = history[0]
        st.divider()
        st.subheader("Second-order analysis")
        st.markdown(_render(latest["result"]), unsafe_allow_html=True)
        t3 = latest.get("t3_result")
        if t3 is not None:  # None = old run predating this feature; "" = ran but returned empty
            st.divider()
            st.subheader("Third-order upstream pass")
            st.markdown(
                _render(t3) if t3 else "_Third-order analysis returned no output — the model may have hit a token limit. Try running again._",
                unsafe_allow_html=True,
            )

    # --- History (prior runs, collapsed) -----------------------------------
    if len(history) > 1 or (not ran_this_session and history):
        st.divider()
        prior = history if not ran_this_session else history[1:]
        st.subheader(f"Prior runs ({len(prior)})")
        for i, entry in enumerate(prior, start=1):
            model_tag = entry.get("model", "").split("-")[1] if entry.get("model") else ""
            label = (
                f"**{entry['timestamp']}** · {model_tag} — "
                f"{entry['trend'][:80]}{'…' if len(entry['trend']) > 80 else ''}"
            )
            with st.expander(label, expanded=False):
                st.markdown("**Second-order**")
                st.markdown(_render(entry["result"]), unsafe_allow_html=True)
                if entry.get("t3_result"):
                    st.divider()
                    st.markdown("**Third-order upstream pass**")
                    st.markdown(_render(entry["t3_result"]), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab definitions
# ---------------------------------------------------------------------------
tab_ai, tab_nuclear, tab_quantum = st.tabs([
    "🤖 AI",
    "⚛️ Nuclear",
    "🔮 Quantum",
])

with tab_ai:
    st.subheader("AI Infrastructure — Second-Derivative Analysis")
    st.markdown(
        "Trace AI demand trends downstream to find supply bottlenecks: "
        "the concentrated, hard-to-substitute chokepoints where pricing power sits."
    )
    analysis_tab("ai", [
        "AI compute demand driving data-centre electricity consumption",
        "Hyperscaler GPU cluster buildout driving demand for high-bandwidth memory",
        "AI inference workloads shifting to edge devices, driving demand for low-power silicon",
        "AI training scale-up driving demand for advanced packaging and CoWoS capacity",
    ], model=selected_model)

with tab_nuclear:
    st.subheader("Nuclear Energy — Second-Derivative Analysis")
    st.markdown(
        "Trace nuclear demand trends to find supply bottlenecks: "
        "where capacity is genuinely constrained along the fuel cycle and reactor supply chain."
    )
    analysis_tab("nuclear", [
        "AI data-centre power demand reigniting nuclear power purchase agreements",
        "Western governments de-risking uranium supply chains away from Russian enrichment",
        "SMR licensing momentum creating demand for specialised reactor-grade components",
        "Utility fleet-life extensions driving demand for nuclear fuel fabrication capacity",
    ], model=selected_model)

with tab_quantum:
    st.subheader("Quantum Computing — Second-Derivative Analysis")
    st.markdown(
        "Trace quantum commercialisation trends to find supply bottlenecks: "
        "where the enabling materials, components, and services are genuinely scarce."
    )
    analysis_tab("quantum", [
        "Quantum computing commercialisation driving demand for dilution refrigerators",
        "Photonic quantum computing scaling driving demand for ultra-low-loss optical fibre",
        "Quantum hardware scale-up driving demand for specialised microwave components and cryogenic electronics",
        "Government quantum investment programmes driving demand for trapped-ion and superconducting qubit foundry capacity",
    ], model=selected_model)

