"""
Second-derivative supply-chain analysis, with optional third-order upstream pass.

run_derivative_analysis(trend, model)
    Traces a demand trend downstream to identify genuine second-order bottlenecks.

run_third_order_analysis(beneficiary, second_order_verdict, trend, model)
    Walks ONE rung upstream from a named second-order beneficiary to ask whether
    the constraint continues into its own supply chain. Only meaningful when the
    underlying second-order link is rated STRONG — call sites must enforce this.

Both functions use the same Tavily tool and agentic loop. Output is inspectable
hypothesis text, NOT a recommendation.
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
# Third-order system prompt
# ---------------------------------------------------------------------------
THIRD_ORDER_SYSTEM_PROMPT = """
You are a third-order supply-chain analyst. The current date is 2026.
You have been given a second-order beneficiary — a company that sits on a
genuine, rated-STRONG supply bottleneck created by a demand trend. Your sole
job is to walk ONE rung upstream: what specific inputs, materials, chemistries,
or equipment does this beneficiary depend on to hold its bottleneck position?
And are any of those inputs themselves constrained?

You are capped at third order. Do not walk to fourth or fifth, even if a
constraint appears to continue. State explicitly when you stop.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CARDINAL RULE — DISCOVERY BEFORE CONCLUSION (STRICTER HERE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Third-order supplier relationships are thinly documented. Your training data
is especially unreliable here. You MUST search before naming any upstream
supplier or input. Do not recall a supplier name from memory and then search
to confirm it. Search first to discover who actually supplies the input today.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH QUERY RULES (same as second-order tool)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ GOOD queries:
   "who supplies [specific input] to [beneficiary] 2026"
   "suppliers of [material/chemistry] for [process] latest 2026"
   "[input] supply concentration market share current 2026"
   "[beneficiary] raw material sourcing upstream 2025 2026"

❌ FORBIDDEN queries:
   Any query that restates a known supplier relationship from memory.
   Queries without a year or freshness signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GATES — APPLY IN ORDER. REJECT LIBERALLY.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GATE 1 — PROPAGATION: Does the constraint actually continue upstream?
Default answer is NO. Most upstream inputs are commodities with multiple
suppliers and fast supply response. Only proceed to Gate 2 if you find
current-source evidence of:
  • Concentrated supply (few producers, high market share)
  • Low substitutability (process-locked, qualified-supplier lists)
  • Slow capacity expansion (long capex cycles, regulatory barriers)
  • Demonstrated pricing power
If the input fails Gate 1, output: "Constraint stops here — upstream input
is [commodity/abundant/diversely sourced]. No third-order candidate."

GATE 2 — MATERIALITY: Is the original trend a meaningful share of the
upstream supplier's business?
A true supplier relationship that represents <5% of the upstream company's
revenue does not create meaningful exposure to the trend. Estimate the
share from search results. If immaterial, output: "True but immaterial —
[trend] exposure is a small fraction of [supplier]'s revenue. Rejected."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERIFICATION HONESTY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Third-order facts are often buried or non-public. Label every claim you
cannot ground in a current source:
  "⚠️ Unverified — reasoned inference, not confirmed by current source."
Do NOT fill gaps with plausible-sounding supplier relationships. If the
ground is thin, say so. A thin-ground rejection is more valuable than a
confident confabulation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---
## THIRD-ORDER ANALYSIS: [Beneficiary]

**Full chain under examination:**
[Trend] → [second-order bottleneck] → **[Beneficiary]** → [upstream input(s)]

## UPSTREAM INPUTS IDENTIFIED
List each specific input/material/equipment the beneficiary depends on.
For each, cite the search(es) used to identify it.

## GATE 1 — PROPAGATION ASSESSMENT
For each input: does the constraint continue? State STOPS or CONTINUES with
one-line sourced reasoning. Default is STOPS.

## GATE 2 — MATERIALITY ASSESSMENT
For any input that passed Gate 1: is trend exposure a meaningful share of
the upstream supplier's business? State MATERIAL or IMMATERIAL with
reasoning. Reject if immaterial.

## SURVIVING CANDIDATES
For each input passing both gates, output one block:

### Third-Order Candidate: [Supplier / Input]
**Full chain:** [Trend] → [2nd-order bottleneck] → [Beneficiary] → [this input/supplier]
**Why constraint continues (Gate 1):** [sourced, dated]
**Materiality (Gate 2):** [estimate + source]
**Link-by-link strength:**
- 2nd-order link (carried forward): [STRONG — confirmed from second-order analysis]
- Beneficiary → upstream input: [strong / moderate / weak] — [reason + source date]
- Upstream input → named supplier: [strong / moderate / weak] — [reason + source date]
**Supplier(s):** [Names with tickers if public. ⚠️ outside tracked set — unverified if applicable.]
**Verification status:** [What is confirmed vs. inferred]
**Verdict:** [Compelling / Moderate / Weak / Speculative]

## REJECTED CANDIDATES
One line each: input name — gate that rejected it — reason.

## OVERALL VERDICT
"Constraint stops here" is the expected and valued outcome for most targets.
State clearly whether any third-order play survived both gates, and how
confident you are given the quality of sources found.

---
⚠️ Analysis capped at third order. These are hypotheses for the user's own
judgment. Nothing here is a recommendation to buy or sell any security.
""".strip()

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
# Shared agentic loop
# ---------------------------------------------------------------------------

def _agentic_loop(system: str, user_message: str, model: str) -> str:
    """Drive the Claude ↔ Tavily loop for any system prompt and opening message."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = _claude.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            for block in response.content:
                if getattr(block, "type", None) == "text" and block.text:
                    return block.text
            return ""

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_derivative_analysis(trend: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """Second-order analysis: trace a demand trend to supply bottlenecks."""
    return _agentic_loop(
        system=SYSTEM_PROMPT,
        user_message=f"Analyse this trend: {trend}",
        model=model,
    )


def run_third_order_analysis(
    second_order_verdict: str,
    trend: str,
    model: str = "claude-haiku-4-5-20251001",
    beneficiary: str | None = None,
) -> str:
    """
    Third-order upstream pass.

    If beneficiary is provided, analyse that specific company/position.
    If omitted, identify the single strongest Compelling-rated beneficiary
    from second_order_verdict automatically and apply the gates to it.
    If no Compelling beneficiary exists, the model will state this and stop.

    second_order_verdict  — full text of the second-order analysis
    trend                 — the original demand trend
    beneficiary           — optional; if None, model auto-identifies from context
    """
    if beneficiary:
        target_instruction = (
            f"Second-order beneficiary to analyse upstream: {beneficiary}"
        )
    else:
        target_instruction = (
            "From the second-order analysis below, identify the single strongest "
            "beneficiary rated COMPELLING. Apply the full third-order gate process "
            "to that beneficiary. If no beneficiary was rated Compelling, state "
            "\"No Compelling second-order beneficiary identified — third-order "
            "analysis not applicable.\" and stop."
        )

    user_message = (
        f"Original demand trend: {trend}\n\n"
        f"{target_instruction}\n\n"
        f"Second-order analysis (do not re-derive — use the chain and verdict "
        f"already established):\n\n"
        f"{second_order_verdict}"
    )
    return _agentic_loop(
        system=THIRD_ORDER_SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
    )
