"""
Sector Intelligence — Streamlit UI.

Primary:   Second-derivative supply-chain analysis (demand trend → bottleneck hypothesis)
Secondary: Nuclear sector brief (SEC filings + news, grouped by segment)
"""

import html
import re

import streamlit as st

from derivative_analysis import run_derivative_analysis
from nuclear_brief import SEGMENTS_ORDER, load_watchlist, run_nuclear_brief

st.set_page_config(
    page_title="Sector Intelligence",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 Sector Intelligence")
st.caption("Second-derivative analysis · Nuclear sector brief · Not investment advice")

# ---------------------------------------------------------------------------
# Link renderer — opens every markdown link in a new browser tab
# ---------------------------------------------------------------------------
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _render(text: str) -> str:
    """Escape raw content; convert [label](url) links to target=_blank anchors."""
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
# Tabs
# ---------------------------------------------------------------------------
tab_analysis, tab_brief = st.tabs(["🔬 Second-Derivative Analysis", "📋 Nuclear Brief"])


# ── Tab 1: Second-Derivative Analysis ──────────────────────────────────────
with tab_analysis:
    st.subheader("Second-Derivative Analysis")
    st.markdown(
        "Enter a demand trend. The tool traces the value chain to find genuine "
        "supply **bottlenecks** — concentrated, hard-to-substitute chokepoints — "
        "and names who sits on them. Verification searches are run before any "
        "claim is asserted. Output is a hypothesis, not a recommendation."
    )

    trend = st.text_area(
        "Trend",
        placeholder=(
            "e.g. AI compute demand driving electricity consumption, "
            "or onshoring of semiconductor fabs driving demand for ultra-pure water"
        ),
        height=80,
    )

    run_btn = st.button("Run analysis", type="primary", disabled=not trend.strip())

    if run_btn and trend.strip():
        with st.spinner("Running second-derivative analysis — searching and reasoning…"):
            result = run_derivative_analysis(trend.strip())
        st.markdown(_render(result), unsafe_allow_html=True)
    elif not trend.strip():
        st.info("Enter a trend above and click **Run analysis**.")


# ── Tab 2: Nuclear Brief ───────────────────────────────────────────────────
with tab_brief:
    st.subheader("Nuclear Sector Brief")
    st.markdown(
        "On-demand brief of SEC EDGAR filings and recent news, "
        "grouped by segment and split into CONFIRMED / CHATTER / CONTEXT tiers."
    )

    col_sel, col_meta = st.columns([2, 3])

    with col_sel:
        segment = st.selectbox(
            "Segment",
            options=["all"] + SEGMENTS_ORDER,
            format_func=lambda s: "All segments" if s == "all" else s.title(),
        )
        generate = st.button("Generate brief", type="primary")
        st.caption(
            "A full brief queries Tavily once per company (~39 calls). "
            "Segment-scoped briefs are faster."
        )

    with col_meta:
        entries   = load_watchlist(segment)
        tradeable = [e for e in entries if e.get("tradeable")]
        context   = [e for e in entries if not e.get("tradeable")]
        st.metric("Tradeable names", len(tradeable))
        st.metric("Context names", len(context))
        st.metric(
            "SEC filers",
            sum(1 for e in tradeable if e.get("sec_filer")),
        )

    if generate:
        label = "All segments" if segment == "all" else segment.title()
        with st.spinner(f"Assembling {label} brief — fetching filings and news…"):
            brief = run_nuclear_brief(segment)
        st.markdown(_render(brief), unsafe_allow_html=True)
