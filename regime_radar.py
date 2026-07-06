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
import re
from html.parser import HTMLParser
from pathlib import Path

import anthropic
import openai
import requests
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY    = os.environ["TAVILY_API_KEY"]

LOCAL_BASE_URL = "http://localhost:1234/v1"

MODELS = {
    "Haiku (faster, cheaper)": "claude-haiku-4-5-20251001",
    "Sonnet (slower, sharper)": "claude-sonnet-4-6",
    "Gemma 4 (local)": "local",
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

HARD vs SOFT signal rule — actions over words. Three tiers:
  HARD signal — a fund actually raised, LP capital deployed, a concrete
    new mandate, a signed deal: weight this equally with other hard
    fund-formation evidence.
  SOFT signal — a formal partner blog post, published thesis essay, or
    official fund letter: PROMOTIONAL. VCs sustain narratives around
    positions they already hold; published enthusiasm is systematically
    over-optimistic. Cannot move a candidate's strength rating alone.
    Flag: ⚠️ SOFT — VC published thesis, promotional.
  SOFTER signal — informal VC commentary: podcast remarks, conference
    quotes, Twitter/X posts, off-the-cuff interviews. Even cheaper to
    produce than formal essays, even more self-interested. Its only value
    is earliness — it may surface attention drift before formal writing
    does, which helps spot convergence sooner. Never let it drive a
    conclusion. Flag: ⚠️ SOFTER — informal VC commentary, treat as
    directional noise only.

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
STAGE 5 — SELECTIVE DEEP READ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After completing signals 1–4, review ALL sources surfaced (URLs from Tavily
results and commentator searches). Identify the 5–10 that are most
signal-rich — i.e. most likely to contain primary data, concrete capital
commitments, or substantive analysis rather than summaries or opinion.

For those selected sources, call web_fetch on each URL to retrieve the full
text. Read it. Do NOT fetch the rest — leave them as snippets.

Every source used in your output MUST be tagged with one of:
  [fully read]             — web_fetch succeeded, full content available
  [snippet only]           — Tavily snippet only, not fetched
  [referenced, not readable] — paywalled, 403, non-HTML, or fetch failed

Never represent a source as having been read when it was not. If a high-value
source is paywalled, label it "referenced, not readable" and note you could
only see the snippet or headline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 6 — SYNTHESISE (causal picture)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reason across the whole gathered set — fully-read sources and snippets
together — to build the causal picture. Ask:

  • Is there convergent evidence of a capital shift? Across which signal
    types? In which direction?
  • If yes: what is DRIVING it? A policy catalyst? Exhaustion in the
    current trade? A genuinely new thesis gaining independent traction?
  • If no: what is the evidence FOR stability — i.e. why is capital NOT
    rotating? Name the specific sources that support this.

Every causal claim must:
  (a) cite the specific source(s) it rests on, and
  (b) note whether those sources were [fully read] or [snippet only].

A conclusion grounded in fully-read primary sources is STRONG.
A conclusion resting only on snippets or paywalled reports must be
labelled TENTATIVE — sourced from snippets, not verified in depth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 7 — EXPLAIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce the "Why" section: a causal account of why there has or has not
been a capital shift this month. Structure it as a chain of reasoning:
each link stated explicitly, each link confidence-rated (high / medium /
low), each link noting the read-depth of its supporting sources.

A well-evidenced NULL result — "no meaningful change this month, and here
is the evidence for that" — is a FIRST-CLASS output, delivered with the
same confidence as a positive finding. Most months, the honest "why" is
"it hasn't changed because [X, Y, Z]." Prefer this over manufacturing a
narrative.

Do NOT construct a rotation story unless the evidence genuinely supports
it. The cardinal sin is a confident-sounding causal narrative built on
thin, snippet-only, or paywalled sources.

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

## DEEP-READ SOURCES
List every source you web_fetched, with outcome:
| Source | URL | Read depth | Signal value |
|--------|-----|------------|--------------|
(one row per fetched URL; snippet-only sources need not be listed here
unless they are load-bearing for a conclusion)

## SYNTHESIS — CAUSAL PICTURE
Three to five paragraphs. Reason across all gathered material. For each
causal claim, cite the source and its read-depth in brackets.
Example: "Policy spending on X accelerated ([source, fully read]) while
fund formation into Y shows no new activity ([source, snippet only] —
tentative)."
Close with a one-line summary: convergent / divergent / no signal.

## WHY THERE HAS (OR HAS NOT) BEEN A CHANGE
Chain of reasoning, each link on its own line:
  [Link 1] [confidence: high/medium/low] [source read-depth]
  [Link 2] ...
  → Overall conclusion

If the conclusion is "no change": state it plainly and cite the evidence.
Label the overall conclusion: WELL-GROUNDED (fully-read sources) /
TENTATIVE (snippet-only) / THIN (paywalled / unread sources only).

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
# Tool definitions
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
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch the FULL text content of a specific URL identified as signal-rich "
            "during the search phase. Use this for Stage 5 (Selective Deep Read) only — "
            "call it on the 5–10 most valuable URLs from your search results. "
            "Do NOT use it to browse broadly; use tavily_search for discovery. "
            "Returns the page text (truncated to ~4000 chars) or an error string "
            "indicating the page is paywalled / unfetchable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact URL to fetch.",
                }
            },
            "required": ["url"],
        },
    },
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


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text stripper using stdlib only."""
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip  = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip = False
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "br"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of whitespace / blank lines
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", raw)).strip()


def _web_fetch(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a URL and return its text content.
    Returns an error string (never raises) so the model can label the source
    as 'referenced, not readable' and move on.
    """
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "RegimeRadar/1.0 (research tool)"},
            allow_redirects=True,
        )
        if resp.status_code in (401, 402, 403):
            return f"PAYWALL_OR_BLOCKED: HTTP {resp.status_code} — mark as [referenced, not readable]"
        if resp.status_code != 200:
            return f"FETCH_FAILED: HTTP {resp.status_code} — mark as [referenced, not readable]"
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return f"NON_TEXT: content-type '{content_type}' — mark as [referenced, not readable]"
        extractor = _TextExtractor()
        extractor.feed(resp.text)
        text = extractor.get_text()
        if len(text) < 100:
            return "FETCH_FAILED: page returned too little text — mark as [referenced, not readable]"
        return text[:max_chars] + (f"\n\n[... truncated at {max_chars} chars]" if len(text) > max_chars else "")
    except Exception as e:
        return f"FETCH_FAILED: {e} — mark as [referenced, not readable]"


# ---------------------------------------------------------------------------
# Local model helpers
# ---------------------------------------------------------------------------

def _tools_to_openai(tools: list) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _get_local_model_id() -> str:
    try:
        client = openai.OpenAI(base_url=LOCAL_BASE_URL, api_key="lm-studio")
        for m in client.models.list().data:
            if "embed" not in m.id.lower():
                return m.id
    except Exception:
        pass
    return "google/gemma-4-e4b"


def _dispatch_tool(name: str, inputs: dict, on_step=None) -> str:
    if name == "tavily_search":
        query  = inputs.get("query", "")
        if on_step:
            on_step("search", query)
        return _tavily_search(query)
    elif name == "web_fetch":
        url = inputs.get("url", "")
        if on_step:
            on_step("fetch", url)
        return _web_fetch(url)
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agentic loops — Anthropic and local (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _agentic_loop_anthropic(system: str, user_message: str, model: str, on_step=None) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = _claude.messages.create(
            model=model,
            max_tokens=8000,
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
            result = _dispatch_tool(block.name, block.input, on_step)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})


def _agentic_loop_local(system: str, user_message: str, on_step=None) -> str:
    client       = openai.OpenAI(base_url=LOCAL_BASE_URL, api_key="lm-studio")
    model_id     = _get_local_model_id()
    openai_tools = _tools_to_openai(TOOLS)
    messages     = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_message},
    ]

    while True:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=openai_tools,
            max_tokens=8000,
        )

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if finish_reason != "tool_calls" or not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            try:
                inputs = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                inputs = {}
            result = _dispatch_tool(tc.function.name, inputs, on_step)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def _agentic_loop(system: str, user_message: str, model: str, on_step=None) -> str:
    if model == "local":
        return _agentic_loop_local(system, user_message, on_step)
    return _agentic_loop_anthropic(system, user_message, model, on_step)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_regime_radar(model: str = "claude-haiku-4-5-20251001", on_step=None) -> str:
    """
    Run a monthly capital-rotation regime scan.

    Seven stages: signal gathering (1–4) + selective deep read (5) +
    synthesis (6) + explanatory "why" conclusion (7).

    on_step: optional callable(kind: str, detail: str) for live progress.
             kind='search' → Tavily query; kind='fetch' → URL being read.
    """
    user_message = (
        "Run the monthly regime radar scan. "
        "Stage 1–4: work through all signal types in order — "
        "Policy/Government, Fund Formation (institutional + allocators), "
        "Exhaustion in the Current Trade, Commentator Attention Drift — "
        "running current-dated Tavily searches for each. "
        "Stage 5: identify the 5–10 most signal-rich URLs from your searches "
        "and web_fetch each one for the full text; tag every source with its "
        "read-depth ([fully read] / [snippet only] / [referenced, not readable]). "
        "Stage 6: synthesise causally across all gathered material, citing "
        "source and read-depth for every claim. "
        "Stage 7: produce the 'Why there has (or has not) been a change' section "
        "— link-by-link reasoning, confidence-rated, read-depth honest. "
        "The current date is 2026. Be reject-happy: a well-evidenced null result "
        "is a first-class output."
    )
    return _agentic_loop(
        system=REGIME_RADAR_SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
        on_step=on_step,
    )
