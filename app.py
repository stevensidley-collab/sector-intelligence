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
    """Return all archived runs for this tab, newest first."""
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
                if rec.get("tab") == tab:
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return list(reversed(records))  # newest first

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
        trend = choice
        st.text_area(
            "Trend (editable)",
            value=trend,
            height=80,
            key=f"{key}_display",
        )

    run = st.button("Run analysis", type="primary", key=f"{key}_run",
                    disabled=not trend.strip())

    if run and trend.strip():
        with st.spinner("Searching and reasoning through the value chain…"):
            result = run_derivative_analysis(trend.strip(), model=model)

        # Persist to disk, then prepend to in-session history
        _append_to_archive(key, trend.strip(), result, model)
        st.session_state[hist_key].insert(0, {
            "trend":     trend.strip(),
            "result":    result,
            "model":     model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        st.markdown(_render(result), unsafe_allow_html=True)

    elif not trend.strip():
        st.info("Select or enter a trend above and click **Run analysis**.")

    # --- Third-order panel (shown only when at least one run exists) -------
    history = st.session_state[hist_key]
    if history:
        with st.expander("🔍 Go third order on a second-order beneficiary", expanded=False):
            st.caption(
                "Walk one rung upstream from a named bottleneck-holder. "
                "**Only valid when the second-order link beneath it was rated STRONG.** "
                "Moderate or weak links compound uncertainty — the tool will refuse them."
            )

            # Let the user pick which prior run to draw context from
            run_labels = [
                f"{e['timestamp']} — {e['trend'][:60]}{'…' if len(e['trend']) > 60 else ''}"
                for e in history
            ]
            chosen_idx = st.selectbox(
                "Base second-order run",
                range(len(run_labels)),
                format_func=lambda i: run_labels[i],
                key=f"{key}_t3_run_select",
            )
            chosen_entry = history[chosen_idx]

            beneficiary_input = st.text_input(
                "Beneficiary to analyse upstream (e.g. 'Ajinomoto / ABF substrates')",
                key=f"{key}_t3_beneficiary",
                placeholder="Company name / product position",
            )

            verdict_confirmed = st.checkbox(
                "I confirm the second-order link for this beneficiary was rated **STRONG** "
                "in the analysis above. (Do not tick if the verdict was Moderate, Weak, or Speculative.)",
                key=f"{key}_t3_confirmed",
            )

            run_t3 = st.button(
                "Run third-order analysis",
                key=f"{key}_t3_run",
                disabled=not (beneficiary_input.strip() and verdict_confirmed),
            )

            if run_t3 and beneficiary_input.strip() and verdict_confirmed:
                with st.spinner("Walking upstream — searching for third-order constraints…"):
                    t3_result = run_third_order_analysis(
                        beneficiary=beneficiary_input.strip(),
                        second_order_verdict=chosen_entry["result"],
                        trend=chosen_entry["trend"],
                        model=selected_model,
                    )
                _append_to_archive(
                    tab=key,
                    trend=f"[THIRD ORDER: {beneficiary_input.strip()}] {chosen_entry['trend']}",
                    result=t3_result,
                    model=selected_model,
                )
                st.session_state[hist_key].insert(0, {
                    "trend":     f"[THIRD ORDER: {beneficiary_input.strip()}] {chosen_entry['trend']}",
                    "result":    t3_result,
                    "model":     selected_model,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                st.markdown(_render(t3_result), unsafe_allow_html=True)

    # --- History -----------------------------------------------------------
    history = st.session_state[hist_key]  # refresh after any third-order insert
    if history:
        st.divider()
        st.subheader(f"History ({len(history)} run{'s' if len(history) != 1 else ''})")
        for i, entry in enumerate(history):
            model_tag = f" · {entry.get('model', '').split('-')[1] if entry.get('model') else ''}"
            label = f"**{entry['timestamp']}**{model_tag} — {entry['trend'][:80]}{'…' if len(entry['trend']) > 80 else ''}"
            with st.expander(label, expanded=(i == 0 and not run)):
                st.markdown(_render(entry["result"]), unsafe_allow_html=True)


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
