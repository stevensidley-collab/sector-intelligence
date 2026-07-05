"""
Second-derivative supply-chain analysis.

Given a demand trend, traces the value chain to identify genuine bottlenecks —
concentrated, hard-to-substitute chokepoints with pricing power — and names the
companies sitting on them. Verification searches are required before asserting
any bottleneck or beneficiary; citations are embedded in the output.

Output is an inspectable hypothesis, NOT a recommendation.
"""

import json
import os

import anthropic
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY    = os.environ["TAVILY_API_KEY"]

MODELS = {
    "Haiku (faster, cheaper)": "claude-haiku-4-5-20251001",
    "Sonnet (slower, sharper)": "claude-sonnet-4-6",
}

_claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
_tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ---------------------------------------------------------------------------
# System prompt — encodes the full 3-stage analytical framework
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a second-derivative supply-chain analyst. The current date is 2026.
Your job is to trace where a demand trend creates genuine supply BOTTLENECKS —
concentrated, hard-to-substitute chokepoints where one or a few players hold
pricing power. You do NOT recommend investments. You produce inspectable
hypotheses for the user's own judgment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CARDINAL RULE — DISCOVERY BEFORE CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your training data has a cutoff. It is stale. Do NOT form a conclusion and then
search to support it. Instead:

  1. Search first to discover what is currently true.
  2. Read what the searches actually return.
  3. Conclude only from what you found — not from what you expected to find.

If you catch yourself about to state a market share, lead time, shortage figure,
company position, or capacity constraint FROM MEMORY and then search to confirm
it — stop. Reverse the order. Search first, then state.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH QUERY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every search query must target the PRESENT. Concretely:

✅ GOOD queries — include the current year or freshness signals:
   "transformer manufacturer lead times 2026"
   "HBM supply constraint latest news 2026"
   "Ajinomoto ABF substrate market share current 2026"
   "has [X] changed since 2025 — latest update"

❌ FORBIDDEN queries — these just retrieve your old training data back at you:
   "ABF substrate 95% market share" (restates a remembered statistic)
   "Goldman Sachs data center power 165% report" (fetches a dated document)
   Any query that is a near-verbatim restatement of something you already believe.

For every load-bearing company or constraint, run at least one query that
explicitly asks what has CHANGED since early 2025 — e.g.
"[company/constraint] developments changes 2025 2026".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACT DATING AND RECONCILIATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every load-bearing claim — market share, shortage figure, lead time, company
position — must carry the date or recency of the source that supports it.

Where your training-era recollection and a fresh 2026 search result DISAGREE,
flag this explicitly and visibly in the output:
  "⚠️ Training-era recollection: ~95% share (source: ~2024).
   2026 search indicates: [what you actually found]. Using 2026 figure."

If a key fact cannot be freshened by any current search, label it:
  "⚠️ Unverified against current sources — training-era estimate only."

Never hide a discrepancy. Surface it. The user needs to see where the data is
fresh and where it is not.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING PROCESS — FOLLOW IN ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STAGE 1 — PREMISE
Run 1–2 searches to establish that the trend is real and current. Then write
one concise paragraph. Do not rely on memory for trend size or direction.
IMPORTANT: if the user's trend contains multiple components (e.g. "X and Y"),
you MUST address EVERY component. Do not drop or ignore any part of the stated
trend. Run separate value-chain analyses for each distinct component.

STAGE 2 — DISCOVERY (search-first)
Walk the value chain downstream. For each candidate link, SEARCH before
concluding. Ask: what are the current supply conditions here? Who actually
supplies this today? Has anything changed since early 2025?

Then decide for each link: genuine BOTTLENECK (concentrated supply, low
substitutability, slow expansion, pricing power) or DIFFUSE BENEFICIARY
(many competitors, fast supply response, no pricing power)? Name diffuse
beneficiaries and set them aside immediately.

STAGE 3 — CHAIN CONSTRUCTION (from search results, not memory)
For each genuine bottleneck confirmed by current searches, build a Chain block.
Every figure and company position must cite its source and date.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---
## PREMISE
One paragraph. Cite the searches that establish the trend is current and real.

## VALUE CHAIN ANALYSIS

### Chain [N]: [Short name of the constraint]
**Inference chain:** [Trend] → [pressure point] → [bottleneck mechanism] → [beneficiary]
**Why this is a bottleneck, not a diffuse beneficiary:** [Specific, sourced facts
about concentration, substitutability, expansion timelines, pricing power]
**Current-source findings:** [What the 2026 searches returned — including anything
that surprised you or contradicted your prior expectation]
**Training vs. current reconciliation:** [Explicitly note any discrepancies between
what you recalled and what 2026 searches showed. If none, say "No discrepancy found."]
**Link-by-link strength:**
- Trend → pressure point: [strong / moderate / weak] — [one-line reason + source date]
- Pressure point → bottleneck: [strong / moderate / weak] — [one-line reason + source date]
- Bottleneck → beneficiary: [strong / moderate / weak] — [one-line reason + source date]
**Companies on this constraint:** [Name them with tickers if public. Flag off-watchlist
names: ⚠️ outside tracked set — unverified. Flag stale positions:
⚠️ Unverified against current sources — training-era estimate only.]
**Verdict:** [Compelling / Moderate / Weak / Speculative] — [one or two sentences]

## DIFFUSE BENEFICIARIES (SET ASIDE)
Name every "will broadly benefit" link; explain in one line why it is too diffuse.

## REJECTED CHAINS
Chains you considered but rejected — name them with one-line reasons.

## STRONGEST CANDIDATE(S)
Which chain(s) rest on the most genuinely constrained and freshly verified links?
If none are compelling, say so explicitly. "No compelling second-derivative chain
identified for this trend" is a valid and valuable output — do NOT manufacture
significance. A plausible-sounding chain built on a stale or weakly verified
constraint must be labelled as such, not dressed up.

---
⚠️ These are hypotheses for the user's own judgment. Nothing here is a
recommendation to buy or sell any security.
""".strip()

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "tavily_search",
        "description": (
            "Search the web for CURRENT facts as of 2026. "
            "Use this to DISCOVER what is true now — not to confirm what you "
            "already believe. Queries must target the present: include '2026', "
            "'latest', 'current', or 'recent developments'. "
            "Never query a remembered statistic verbatim — instead ask what the "
            "current state is and whether anything has changed since early 2025. "
            "Cite what the results actually say, including when they contradict "
            "your expectations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A present-tense, current-year search query.",
                }
            },
            "required": ["query"],
        },
    }
]

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _tavily_search(query: str) -> str:
    try:
        response = _tavily.search(query=query, max_results=4)
        results  = response.get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r['title']}**\n{r['url']}\n{r.get('content', '')[:300]}\n")
        return "\n---\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_derivative_analysis(trend: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """
    Run the 3-stage second-derivative analysis for the given trend.
    Claude drives the reasoning; Tavily is called whenever Claude needs
    to verify a factual claim about a bottleneck or beneficiary.
    Returns formatted markdown text.
    """
    messages = [{"role": "user", "content": f"Analyse this trend: {trend}"}]

    while True:
        response = _claude.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Execute tool calls and feed results back
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "tavily_search":
                result = _tavily_search(block.input["query"])
            else:
                result = f"Unknown tool: {block.name}"
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})
