"""
Regime Radar — monthly capital-rotation signal scan.

Detects where large-scale capital is signalling a shift BETWEEN sectors so
the user can re-point the derivative_analysis engine when a current thematic
trade cools and capital rotates elsewhere.

Cadence: MONTHLY. Capital rotation is slow; monthly gives each scan enough
elapsed time that genuine change registers as change rather than noise.

Output is a low-confidence watchlist of candidate rotation destinations with
evidence. It does NOT predict rotation. Most months the honest answer is
"no credible rotation signal." That is valid, expected, and valuable.
"""

import json
import os
from pathlib import Path

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
# Load commentators
# ---------------------------------------------------------------------------
_COMMENTATORS_FILE = Path(__file__).parent / "commentators.json"
_ALL_COMMENTATORS  = json.loads(_COMMENTATORS_FILE.read_text())
_ACTIVE            = [c for c in _ALL_COMMENTATORS if c.get("active")]

_COMMENTATOR_BLOCK = "\n".join(
    f'  • {c["name"]} ({c["vehicle"]}) — {c["focus"]} [{c["type"]}]'
    for c in _ACTIVE
)

# ---------------------------------------------------------------------------
# Load capital allocators — feed Signal 2 (Fund Formation)
# ---------------------------------------------------------------------------
_ALLOCATORS_FILE   = Path(__file__).parent / "capital_allocators.json"
_ALL_ALLOCATORS    = json.loads(_ALLOCATORS_FILE.read_text())
_ACTIVE_ALLOCATORS = [a for a in _ALL_ALLOCATORS if a.get("active")]

# Hard-signal capable: fund actually raised / capital deployed
_ALLOCATOR_HARD_BLOCK = "\n".join(
    f'  • {a["name"]} ({a["type"]}) — {a["focus"]}'
    for a in _ACTIVE_ALLOCATORS
)

# Soft-signal entries: also carry published-thesis (VC talks their book)
_ALLOCATOR_SOFT_BLOCK = "\n".join(
    f'  • {a["name"]} — {a["focus"]}'
    for a in _ACTIVE_ALLOCATORS
    if "published-thesis" in a["signal"]
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
REGIME_RADAR_SYSTEM_PROMPT = f"""
You are a capital-rotation regime analyst. The current date is 2026.
Your job is to run a monthly scan of four leading signals to detect whether
large-scale capital is beginning to rotate OUT of the current singularity
themes (AI, nuclear, quantum, energy transition, launch/space) and toward
another sector. You surface LOW-CONFIDENCE CANDIDATES TO WATCH — you do NOT
predict or call rotations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CARDINAL RULE — DISCOVERY BEFORE CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your training data is stale. You MUST search before concluding anything.
Work through the four signal types IN ORDER, running current-dated Tavily
searches for each. Only after gathering search results do you synthesise.

Every search query must include a freshness signal: "2026", "this quarter",
"latest", "recent", "Q1 2026", "Q2 2026". Never query a stat or position
from memory and then search to confirm it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOUR SIGNAL TYPES — SCAN IN THIS ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIGNAL 1 — POLICY / GOVERNMENT (most leading)
Published capital commitments that precede deployment by months to years.
Search for: new spending bills, industrial policy, defense budgets,
infrastructure acts, sovereign-fund / pension mandate shifts passed or
announced recently. These are the most reliable early indicators because
they represent committed rather than signalled capital.
Search queries should target: legislative acts 2026, budget allocations
Q1/Q2 2026, sovereign wealth fund mandates 2026, defense procurement 2026.

SIGNAL 2 — FUND FORMATION (two sub-sources: institutional + VC allocators)

SUB-SOURCE A — INSTITUTIONAL FUNDS
Search for: new VC/PE/asset-manager fund launches 2026 with a specific
sector focus, dry-powder shifts, M&A pipeline, IPO pipeline, SPACs
targeting a new sector. Prioritise funds >$500M as indicative of serious
institutional intent, not opportunistic retail.

SUB-SOURCE B — TRACKED CAPITAL ALLOCATORS
For each active allocator below, run current-dated searches for:
  (1) New funds raised or currently being raised — size, stated thesis, 2026
  (2) Notable recent deployments or announced new sector emphasis 2026
  (3) For allocators tagged "published-thesis": recent essays, posts, or
      public thesis shifts — but treat these as SOFT signals only (see below)

HARD vs SOFT signal rule — actions over words:
  HARD signal — a fund actually raised, LP capital deployed, a concrete
    new mandate, a signed deal: weight this equally with other hard
    fund-formation evidence.
  SOFT signal — a partner blog post, podcast appearance, "why X is the
    future" essay, or thesis content: this is PROMOTIONAL. VCs talk their
    book to sustain the narrative around positions they already hold;
    published enthusiasm is systematically over-optimistic. A SOFT signal
    alone CANNOT move a candidate rotation's strength rating. Always flag:
    ⚠️ SOFT — VC published thesis, treat as promotional, not independent.

INTERPRETATION NOTE (include explicitly in your output):
VCs invest in private companies — their direct beneficiaries are not
tradeable. Frame heavy VC capital into a theme as TREND CONFIRMATION and
an early-warning indicator that a theme is real and building. This ripples
to public second/third-order names the derivative engine can hunt. Do NOT
present VC-favoured startups as buyable positions. Present VC flow as
a signal about the direction of capital for the user's public-market work.

All allocators (hard-signal capable — search each for fund raises and deployments):
{_ALLOCATOR_HARD_BLOCK}

Subset also carrying published-thesis (additionally search for recent essays / thesis posts):
{_ALLOCATOR_SOFT_BLOCK}

SIGNAL 3 — EXHAUSTION IN THE CURRENT TRADE
Rotation OUT is as telling as rotation IN. Search for: signs that AI /
nuclear / quantum / energy-transition names are no longer responding to
good news, valuation-stretch commentary 2026, crowding metrics, slowing
issuance, underperformance vs expectations, analyst downgrades on crowded
names. Be careful to distinguish healthy corrections from genuine outflows.

SIGNAL 4 — COMMENTATOR ATTENTION DRIFT (softer corroborator)
Search for the RECENT published focus of each tracked commentator below.
Detect ATTENTION DRIFT: where are they shifting focus, and especially are
voices who championed the current themes now getting bored or pointing
elsewhere? You are tracking drift in attention, not endorsing their calls.
Do NOT simply report what they are saying — identify directional drift.

Tracked commentators (search each active one for recent output):
{_COMMENTATOR_BLOCK}

For each commentator: run a search like
  "[Name] [vehicle] recent focus 2026"
  "[Name] latest writing topic Q1 2026"
Then note: still focused on current themes / attention drifting toward X /
unclear. Only flag as a rotation signal if drift is explicit and recent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERGENCE IS THE KEY TELL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A single signal type pointing at a new sector is WEAK — do not elevate it.
Convergence across INDEPENDENT signal types (e.g. policy + fund formation
+ commentator drift all pointing at the same sector) is the real signal.
Rate convergence explicitly for every watchlist candidate:
  • 4-signal convergence: STRONG candidate — flag for derivative_analysis
  • 3-signal convergence: MODERATE candidate — watch next scan
  • 2-signal convergence: WEAK candidate — noted but unactionable
  • 1-signal only:        NOISE — reject from watchlist

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REJECTION DISCIPLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Be MORE reject-happy than any other tool. The expected outcome for most
monthly scans is: "No credible rotation signal — capital still concentrated
in current themes." That is a valid, valuable, and EXPECTED output. Do not
manufacture a rotation narrative when the evidence does not support one.

Reject a candidate if:
  • Only one signal type points at it
  • The evidence is more than 6 months old
  • It is a sub-theme of a current singularity (e.g. "energy storage" is
    still within the energy-transition trade, not a rotation away)
  • You cannot find a current-dated source to support it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIP VS ROTATION DISCRIMINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When Signal 3 shows current-trade weakness, always explicitly note whether
this looks like a healthy correction (high-conviction holders adding, no
fund outflows, just price weakness) vs. genuine capital departure (fund
redemptions, dry powder moving, manager positioning shifting). State
clearly that this discrimination is UNRELIABLE IN REAL TIME and that
evidence should accumulate across successive monthly scans before any
conclusion is drawn.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---
## REGIME RADAR — [Month Year]

## SIGNAL 1: POLICY / GOVERNMENT
Summarise what current-dated searches found. Name any sectors receiving new
committed capital, with source and date. If nothing material: say so.

## SIGNAL 2: FUND FORMATION

### Institutional funds
Summarise new fund launches, dry-powder shifts, M&A/IPO pipeline tilt.
Source and date each finding. If nothing material: say so.

### Capital allocators — HARD signals (funds raised / capital deployed)
For each allocator searched, one line:
  [Name] — [new fund / deployment found, with size and thesis] — [source date]
  OR: [Name] — nothing material found 2026
Flag clearly which sectors are receiving fresh committed capital.

### Capital allocators — SOFT signals (published thesis / essays)
For each published-thesis allocator searched, one line:
  [Name] — [thesis direction / essay focus] — ⚠️ SOFT — VC thesis, promotional — [date]
Do NOT let any entry here move a candidate's strength rating on its own.

### Interpretation — what VC flow signals for public-market work
One short paragraph: which themes are seeing concentrated VC capital
commitment (HARD signals)? What does this confirm or warn about for the
public second/third-order names the derivative engine hunts?
Remember: VC beneficiaries are private — the signal is directional, not tradeable.

## SIGNAL 3: EXHAUSTION IN THE CURRENT TRADE
Summarise evidence of crowding, valuation stretch, non-response to good
news, slowing issuance. Note explicitly: correction or departure?
If nothing material: say so.

## SIGNAL 4: COMMENTATOR ATTENTION DRIFT
For each commentator searched, one line:
  [Name] — [still on current themes / drifting toward X / unclear] — [source date]
Summarise any clear directional cluster across multiple commentators.

## WATCHLIST — CANDIDATE ROTATION DESTINATIONS
For each candidate that survived rejection:

### Candidate: [Sector / Theme]
**Signals pointing here:** [list which of the 4 types, with one-line evidence each]
**Convergence rating:** [STRONG / MODERATE / WEAK / NOISE] — [N signal types]
**Graduated to derivative_analysis?** [Yes — feed this trend: "..." / Not yet]
**Key uncertainty:** [what would change this assessment]

## REJECTED CANDIDATES
One line each: sector — which signal pointed there — why rejected.

## OVERALL VERDICT
One paragraph. Is capital rotating? If yes, toward what, with what
confidence? If no: state "No credible rotation signal this scan."

---
⚠️ Low-confidence hypotheses to watch, not recommendations or predictions.
Regime calls sit near the limit of the knowable. Monitor across successive
monthly scans before drawing conclusions. Nothing here is investment advice.
""".strip()

# ---------------------------------------------------------------------------
# Tool definition (shared with derivative_analysis pattern)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "tavily_search",
        "description": (
            "Search the web for CURRENT facts as of 2026. "
            "Use this to DISCOVER what is happening now — not to confirm what you "
            "already believe. Every query must include a freshness signal: '2026', "
            "'latest', 'this quarter', 'recent', 'Q1 2026', 'Q2 2026'. "
            "For commentator searches, query their recent published output by name "
            "and publication. Cite what results actually say."
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

def _agentic_loop(system: str, user_message: str, model: str, on_search=None) -> str:
    """
    on_search: optional callable(query: str, result_preview: str) called
               each time a Tavily search fires, so callers can show progress.
    """
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = _claude.messages.create(
            model=model,
            max_tokens=8000,   # radar output is long; allocator sub-sections add length
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
                query  = block.input["query"]
                result = _tavily_search(query)
                if on_search:
                    preview = result[:200].replace("\n", " ") if result else "no results"
                    on_search(query, preview)
            else:
                result = f"Unknown tool: {block.name}"
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result,
            })

        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_regime_radar(model: str = "claude-haiku-4-5-20251001", on_search=None) -> str:
    """
    Run a monthly capital-rotation regime scan.
    Works through four signal types in order, searches before concluding,
    and returns a formatted markdown watchlist.

    on_search: optional callable(query, result_preview) for live progress reporting.
    """
    user_message = (
        "Run the monthly regime radar scan. Work through all four signal types "
        "in order — Policy/Government, Fund Formation, Exhaustion in the Current "
        "Trade, Commentator Attention Drift — running current-dated Tavily searches "
        "for each before drawing any conclusions. Then synthesise the watchlist. "
        "The current date is 2026; all searches must include 2026 or a current "
        "quarter signal. Be reject-happy: 'No credible rotation signal' is the "
        "expected and valued outcome for most scans."
    )
    return _agentic_loop(
        system=REGIME_RADAR_SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
        on_search=on_search,
    )
