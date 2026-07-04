"""
Nuclear sector intelligence — data gathering module.

Assembles a structured brief from:
  - SEC EDGAR filings (8-K, 10-K, 10-Q, S-1, 40-F, 20-F, 6-K) for the last 30 days
  - Recent news via Tavily web search

Output is grouped by segment and split into three tiers:
  CONFIRMED  — material SEC filings with direct EDGAR links
  CHATTER    — news/commentary (unverified)
  CONTEXT    — non-tradeable names that move the sector (no investment framing)
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import requests as req_lib
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")
SEC_BASE   = "https://data.sec.gov"
EDGAR_BASE = "https://www.sec.gov"

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

FILING_FORMS  = {"8-K", "S-1", "10-Q", "10-K", "40-F", "20-F", "6-K"}
LOOKBACK_DAYS = 30
SEGMENTS_ORDER = ["upstream", "midstream", "reactor-tech", "components", "downstream"]

# SEC allows ~10 req/sec; 150 ms spacing is safe and avoids IP blocks
_SEC_MIN_INTERVAL = 0.15
_last_sec_request = 0.0

_company_tickers_cache: Optional[dict] = None


# ---------------------------------------------------------------------------
# SEC helpers
# ---------------------------------------------------------------------------

def _sec_session() -> req_lib.Session:
    if not SEC_USER_AGENT:
        raise EnvironmentError(
            "SEC_USER_AGENT is not set. Add it to .env — e.g. "
            "'SectorIntelligence/1.0 your@email.com'. "
            "The SEC returns 403 on every request without it."
        )
    s = req_lib.Session()
    s.headers.update({"User-Agent": SEC_USER_AGENT, "Accept": "application/json"})
    return s


def _sec_get(session: req_lib.Session, url: str) -> Optional[dict]:
    global _last_sec_request
    elapsed = time.monotonic() - _last_sec_request
    if elapsed < _SEC_MIN_INTERVAL:
        time.sleep(_SEC_MIN_INTERVAL - elapsed)
    _last_sec_request = time.monotonic()
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _load_company_tickers(session: req_lib.Session) -> dict:
    """Fetch and cache SEC's full ticker→CIK mapping."""
    global _company_tickers_cache
    if _company_tickers_cache is not None:
        return _company_tickers_cache
    data = _sec_get(session, f"{SEC_BASE}/files/company_tickers.json")
    if not data:
        _company_tickers_cache = {}
        return {}
    _company_tickers_cache = {
        entry["ticker"].upper(): str(entry["cik_str"])
        for entry in data.values()
    }
    return _company_tickers_cache


def _resolve_cik(ticker: str, session: req_lib.Session) -> Optional[str]:
    """Resolve a ticker to a CIK, stripping exchange suffixes (.L, .AX, .T…)."""
    ticker_map = _load_company_tickers(session)
    base = ticker.split(".")[0].upper()
    return ticker_map.get(base)


def _fetch_sec_filings(cik: str, session: req_lib.Session) -> list:
    """Return qualifying filings from the last 30 days for this CIK."""
    data = _sec_get(session, f"{SEC_BASE}/submissions/CIK{cik.zfill(10)}.json")
    if not data:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs       = recent.get("primaryDocument", [])
    descs      = recent.get("primaryDocDescription", [])

    cutoff  = (datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)).date()
    results = []

    for form, date_str, accession, doc, desc in zip(forms, dates, accessions, docs, descs):
        if form not in FILING_FORMS:
            continue
        try:
            filing_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if filing_date < cutoff:
            break  # submissions are reverse-chronological
        nodash = accession.replace("-", "")
        link   = f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{nodash}/{doc}"
        results.append({
            "form": form,
            "date": date_str,
            "description": desc or form,
            "link": link,
        })

    return results


# ---------------------------------------------------------------------------
# News helper
# ---------------------------------------------------------------------------

def _fetch_news(name: str) -> list:
    """Return up to 3 recent Tavily results for this company."""
    try:
        client   = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=f"{name} nuclear energy news", max_results=3)
        return [
            {"title": r["title"], "url": r["url"], "snippet": r.get("content", "")[:200]}
            for r in response.get("results", [])
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Watchlist loader
# ---------------------------------------------------------------------------

def load_watchlist(segment: str = "all") -> list:
    with open(WATCHLIST_FILE) as f:
        entries = json.load(f)
    entries = [e for e in entries if e.get("active")]
    if segment != "all":
        entries = [e for e in entries if e.get("segment", "").lower() == segment.lower()]
    return entries


# ---------------------------------------------------------------------------
# Brief assembler
# ---------------------------------------------------------------------------

def run_nuclear_brief(segment: str = "all") -> str:
    """
    Assemble and return a formatted nuclear sector intelligence brief.
    segment: one of 'upstream', 'midstream', 'reactor-tech', 'components',
             'downstream', or 'all'.
    """
    entries = load_watchlist(segment)
    if not entries:
        return f"No active watchlist entries found for segment '{segment}'."

    session  = _sec_session()
    by_seg: dict = {}

    for entry in entries:
        seg       = entry["segment"]
        name      = entry["name"]
        ticker    = entry.get("ticker")
        sec_filer = entry.get("sec_filer", False)
        tradeable = entry.get("tradeable", True)

        news = _fetch_news(name)
        by_seg.setdefault(seg, {"tradeable": [], "context": []})

        if not tradeable:
            # Context names: news only, no investment framing
            by_seg[seg]["context"].append({"name": name, "news": news})
            continue

        # Tradeable entry — attempt EDGAR only where eligible
        filings      = []
        filings_note = None

        if sec_filer and ticker:
            cik = _resolve_cik(ticker, session)
            if cik:
                filings = _fetch_sec_filings(cik, session)
            else:
                filings_note = "CIK not found in SEC EDGAR"
        elif not sec_filer:
            filings_note = "news-only coverage (not an SEC filer)"
        else:
            filings_note = "No ticker — cannot resolve CIK"

        by_seg[seg]["tradeable"].append({
            "name": name,
            "ticker": ticker,
            "sec_filer": sec_filer,
            "filings": filings,
            "filings_note": filings_note,
            "news": news,
        })

    # ---------------------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------------------
    sections = []

    for seg in SEGMENTS_ORDER:
        if seg not in by_seg:
            continue
        data       = by_seg[seg]
        tradeables = data.get("tradeable", [])
        context    = data.get("context", [])
        if not tradeables and not context:
            continue

        lines = [f"# {seg.upper()}"]

        if tradeables:
            # CONFIRMED block
            lines.append("\n## ✅ CONFIRMED (SEC filings)")
            any_filings = False
            for e in tradeables:
                if not e["filings"]:
                    continue
                any_filings = True
                lines.append(f"\n**{e['name']}** ({e['ticker']})")
                for f in e["filings"]:
                    lines.append(
                        f"  - [{f['form']} — {f['description']} ({f['date']})]({f['link']})"
                    )
            if not any_filings:
                lines.append("_No qualifying filings in the last 30 days._")

            # CHATTER block
            lines.append("\n## 📰 CHATTER (news/commentary — unverified)")
            for e in tradeables:
                label = f"\n**{e['name']}** ({e['ticker']})"
                if not e["sec_filer"]:
                    label += " — news-only coverage (not an SEC filer)"
                lines.append(label)
                if e["filings_note"] and e["sec_filer"]:
                    lines.append(f"  _Filings: {e['filings_note']}_")
                if e["news"]:
                    for n in e["news"]:
                        lines.append(f"  - [{n['title']}]({n['url']})")
                else:
                    lines.append("  _No recent news found._")

        if context:
            lines.append("\n## ⚙️ CONTEXT — moves the tradeable names, not positions")
            for e in context:
                lines.append(f"\n**{e['name']}** — CONTEXT, not a tradeable position")
                if e["news"]:
                    for n in e["news"]:
                        lines.append(f"  - [{n['title']}]({n['url']})")
                else:
                    lines.append("  _No recent news found._")

        sections.append("\n".join(lines))

    if not sections:
        return "No data assembled for the requested segment(s)."

    header = (
        f"# ☢️ Nuclear Sector Intelligence Brief\n"
        f"_Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · "
        f"Segment: {segment}_\n\n"
        "**Disclaimer:** CONFIRMED items are sourced from SEC EDGAR public filings. "
        "CHATTER items are unverified news. Nothing here is investment advice.\n"
    )

    return header + "\n\n---\n\n".join(sections)
