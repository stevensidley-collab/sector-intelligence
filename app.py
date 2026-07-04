"""
Sector Intelligence — Streamlit UI.

Generates nuclear sector briefs on demand from SEC EDGAR filings and
Tavily news. No real-time feeds, no trading signals.
"""

import html
import re

import streamlit as st

from nuclear_brief import SEGMENTS_ORDER, load_watchlist, run_nuclear_brief

st.set_page_config(page_title="Nuclear Sector Intelligence", page_icon="☢️", layout="wide")
st.title("☢️ Nuclear Sector Intelligence")
st.caption("SEC filings + news · No real-time feeds · Not investment advice")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Brief options")

    segment = st.selectbox(
        "Segment",
        options=["all"] + SEGMENTS_ORDER,
        format_func=lambda s: "All segments" if s == "all" else s.title(),
    )

    st.markdown("---")
    st.subheader("Watchlist")
    entries = load_watchlist(segment)
    tradeable = [e for e in entries if e.get("tradeable") and e.get("active")]
    context   = [e for e in entries if not e.get("tradeable") and e.get("active")]
    st.metric("Tradeable names", len(tradeable))
    st.metric("Context names", len(context))
    st.caption(
        f"Covering {len(tradeable)} tradeable and {len(context)} context "
        f"names · SEC filings checked for {sum(1 for e in tradeable if e.get('sec_filer'))} "
        "eligible filers"
    )

    st.markdown("---")
    generate = st.button("Generate brief", type="primary", use_container_width=True)
    st.caption(
        "A full brief queries Tavily once per company. "
        "Segment-scoped briefs are faster and cheaper."
    )

# ---------------------------------------------------------------------------
# Link renderer (opens in new tab)
# ---------------------------------------------------------------------------
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _render(text: str) -> str:
    """Convert markdown links to target=_blank anchors; escape everything else."""
    stash: dict = {}

    def _stash(m):
        token = f"\x00L{len(stash)}\x00"
        label = html.escape(m.group(1))
        url   = html.escape(m.group(2), quote=True)
        stash[token] = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
        return token

    escaped = html.escape(_LINK_RE.sub(_stash, text))
    for token, anchor in stash.items():
        escaped = escaped.replace(token, anchor)
    return escaped


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if generate:
    label = "All segments" if segment == "all" else segment.title()
    with st.spinner(f"Assembling {label} brief — fetching filings and news…"):
        brief = run_nuclear_brief(segment)

    st.markdown(
        _render(brief),
        unsafe_allow_html=True,
    )
else:
    st.info(
        "Select a segment in the sidebar (or leave on **All segments**) "
        "and click **Generate brief**."
    )
