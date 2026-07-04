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
import re

import streamlit as st

from derivative_analysis import run_derivative_analysis

st.set_page_config(
    page_title="Sector Intelligence",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 Sector Intelligence")
st.caption("Second-derivative supply-chain analysis · Verified bottleneck hypotheses · Not investment advice")

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
def analysis_tab(key: str, suggested: list[str]) -> None:
    """
    Render a trend selector + run button for one domain tab.
    key        — unique string to namespace Streamlit widget keys
    suggested  — list of pre-written trend strings for the domain
    """
    # Initialise per-tab history in session_state
    hist_key = f"{key}_history"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []  # list of {trend, result, timestamp}

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
            result = run_derivative_analysis(trend.strip())

        # Prepend to history so most recent is first
        from datetime import datetime
        st.session_state[hist_key].insert(0, {
            "trend": trend.strip(),
            "result": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        st.markdown(_render(result), unsafe_allow_html=True)

    elif not trend.strip():
        st.info("Select or enter a trend above and click **Run analysis**.")

    # --- History -----------------------------------------------------------
    history = st.session_state[hist_key]
    if history:
        st.divider()
        st.subheader(f"History ({len(history)} run{'s' if len(history) != 1 else ''})")
        for i, entry in enumerate(history):
            label = f"**{entry['timestamp']}** — {entry['trend'][:80]}{'…' if len(entry['trend']) > 80 else ''}"
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
    ])

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
    ])

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
    ])
