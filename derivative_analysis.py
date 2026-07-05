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
You are a second-derivative supply-chain analyst. Your job is to trace where a
demand trend creates genuine supply BOTTLENECKS — concentrated, hard-to-substitute
chokepoints where one or a few players hold pricing power. You do NOT recommend
investments. You produce inspectable hypotheses for the user's own judgment.

You have one tool: tavily_search. Use it to VERIFY claims before asserting them.
Do NOT assert that a company sits on a bottleneck from training knowledge alone —
search first, then cite. Every named constraint and every named beneficiary must
be backed by at least one verification search.

Produce your analysis in exactly this structure:

---
## PREMISE
One concise paragraph. Restate the trend and why it is real. This is the obvious
part — be brief.

## VALUE CHAIN ANALYSIS

First, walk the value chain from the trend downstream. For each link, decide:
is this a genuine BOTTLENECK (supply concentrated, substitution hard, expansion
slow, pricing power present) or merely a DIFFUSE BENEFICIARY (many competing
players, supply can grow quickly, no pricing power)? Set diffuse beneficiaries
aside immediately — name them in the DIFFUSE BENEFICIARIES section, not here.

For each genuine bottleneck, produce a Chain block:

### Chain [N]: [Short name of the constraint]
**Inference chain:** [Trend] → [pressure point] → [bottleneck mechanism] → [beneficiary]
**Why this is a bottleneck, not a diffuse beneficiary:** [Supply concentration,
substitutability, expansion timeline, pricing power — be specific and factual]
**Verification:** [What you searched and what the results confirmed or contradicted]
**Link-by-link strength:**
- Trend → pressure point: [strong / moderate / weak] — [one-line reason]
- Pressure point → bottleneck: [strong / moderate / weak] — [one-line reason]
- Bottleneck → beneficiary: [strong / moderate / weak] — [one-line reason]
**Companies on this constraint:** [Name them with tickers if public. If outside the
nuclear/energy watchlist, flag: ⚠️ outside tracked set — unverified]
**Verdict:** [Compelling / Moderate / Weak / Speculative] — [one or two sentences]

## DIFFUSE BENEFICIARIES (SET ASIDE)
List every "will broadly benefit" link and explain in one line why it is too
diffuse to constitute a bottleneck.

## REJECTED CHAINS
Candidate chains you considered but rejected — name them with one-line reasons.

## STRONGEST CANDIDATE(S)
Which chain(s) rest on the most genuinely constrained links? If none are
compelling, say so explicitly. "No compelling second-derivative chain identified
for this trend" is a valid and valuable output — do NOT manufacture significance.
A plausible-sounding chain built on a weak or unverifiable constraint must be
labelled weak, not dressed up.

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
            "Search the web for current facts. Use to verify that a company "
            "genuinely occupies a claimed constraint, that a bottleneck mechanism "
            "is real and current, or to confirm the trend's direction. "
            "Always cite what the search returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
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
