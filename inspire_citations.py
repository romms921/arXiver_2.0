"""
inspire_citations.py
====================
Per-paper citation counts from the INSPIRE-HEP REST API (inspirehep.net).

Reads arXiv ids + titles from datasets/arxiv_papers.csv and, for each paper:

  1. GET https://inspirehep.net/api/arxiv/<arxiv_id> (direct external-id lookup)
  2. Fallback: GET https://inspirehep.net/api/literature?q=find eprint <id>
  3. Last resort: title search (q=t "<title>"), first hit only

Records citation_count, citation_count_without_self_citations, the INSPIRE
record id, and basic publication info (journal, year, DOI).

Output: datasets/inspire_citations.csv with columns
  arxiv_id, title, inspire_id, citation_count, citations_no_self,
  journal, year, doi, inspire_url, status

Resume-safe (Ctrl+C safe): finished ids are tracked in
datasets/inspire_citations_state.json and skipped on the next run.

Access / compliance notes (verified 2026-09-18):
  - No API key needed for public read endpoints; most metadata is CC0
    (see INSPIRE Terms of Use). Do not bulk-harvest email addresses.
  - robots.txt explicitly allows /api/literature (it only disallows
    /api/accounts, /api/submissions, /api/holdingpen, /api/workflows,
    /editor, /tools, /journals, /data, ...).
  - Rate limit: 15 requests per 5 s per IP (HTTP 429 if exceeded). This
    script sleeps 0.6 s between calls and backs off 10 s on 429.
  - Coverage caveat: INSPIRE-HEP is HEP-focused. Pure astro-ph papers are
    often absent (status=not_found) — use ads_citations.py as the primary
    citation source for this repo's astro-ph corpus and INSPIRE as a
    complement for hep-ph/hep-th/gr-qc overlap.
  - If you use the API in a scholarly work, please cite:
    Moskovic et al., "The INSPIRE REST API", doi:10.5281/zenodo.5788550.

Usage:
  python inspire_citations.py               # full run, resumable, no key needed
  python inspire_citations.py --limit 10    # quick test on 10 papers
  python inspire_citations.py --restart     # ignore saved progress
"""

import argparse
import json
import os
import re
import signal
import threading
import time
import urllib.parse

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Configuration / paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "datasets", "arxiv_papers.csv")
OUTPUT_CSV = os.path.join(HERE, "datasets", "inspire_citations.csv")
STATE_PATH = os.path.join(HERE, "datasets", "inspire_citations_state.json")

API_BASE = "https://inspirehep.net/api"
HEADERS = {"User-Agent": "arxiver-inspire-citations/1.0", "Accept": "application/json"}

POLITE_DELAY = 0.6  # keeps us under 15 req / 5 s
RETRY_WAIT = 10  # backoff on HTTP 429

OUTPUT_COLUMNS = [
    "arxiv_id", "title", "inspire_id", "citation_count", "citations_no_self",
    "journal", "year", "doi", "inspire_url", "status",
]

# --------------------------------------------------------------------------- #
# Graceful shutdown
# --------------------------------------------------------------------------- #
_stop = threading.Event()


def _handle_sigint(signum, frame):
    if _stop.is_set():
        print("\n[!] Second interrupt - exiting now.", flush=True)
        os._exit(1)
    print("\n[!] Stop requested. Finishing current paper, then saving... "
          "(Ctrl+C again to force quit)", flush=True)
    _stop.set()


signal.signal(signal.SIGINT, _handle_sigint)

# --------------------------------------------------------------------------- #
# State + output
# --------------------------------------------------------------------------- #
_state_lock = threading.Lock()
_write_lock = threading.Lock()


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f).get("done", []))
        except Exception:
            pass
    return set()


def save_state(done_ids):
    with _state_lock:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"done": sorted(done_ids)}, f)
        os.replace(tmp, STATE_PATH)


def append_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    with _write_lock:
        header = not os.path.exists(OUTPUT_CSV)
        df.to_csv(OUTPUT_CSV, mode="a", header=header, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# arXiv id helpers
# --------------------------------------------------------------------------- #
_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def extract_arxiv_id(pdf_link):
    if not isinstance(pdf_link, str):
        return None
    m = _ID_RE.search(pdf_link)
    if m:
        return m.group(1)
    m = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})", pdf_link)
    return m.group(1) if m else None


def strip_version(arxiv_id):
    return re.sub(r"v\d+$", "", arxiv_id or "")


# --------------------------------------------------------------------------- #
# INSPIRE plumbing
# --------------------------------------------------------------------------- #
def inspire_get(url, params=None, timeout=30):
    """GET with 429 backoff. Returns parsed JSON, 'not_found' on 404, else None."""
    while True:
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            if r.status_code == 429:
                print(f"\n[Rate Limit] 429. Sleeping {RETRY_WAIT}s...", flush=True)
                time.sleep(RETRY_WAIT)
                continue
            if r.status_code == 404:
                time.sleep(POLITE_DELAY)
                return "not_found"
            r.raise_for_status()
            time.sleep(POLITE_DELAY)
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"\n[HTTP Error] {e}", flush=True)
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n[Request Error] {e}", flush=True)
            return None


def record_to_row(arxiv_id, title, rec_id, meta):
    pubs = meta.get("publication_info", []) or []
    journal = pubs[0].get("journal_title", "") if pubs else ""
    year = pubs[0].get("year", "") if pubs else ""
    dois = meta.get("dois", []) or []
    doi = dois[0].get("value", "") if dois else ""
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "inspire_id": str(rec_id),
        "citation_count": meta.get("citation_count", 0) or 0,
        "citations_no_self": meta.get("citation_count_without_self_citations", "") or "",
        "journal": journal or "",
        "year": year or "",
        "doi": doi or "",
        "inspire_url": f"https://inspirehep.net/literature/{rec_id}",
        "status": "success",
    }


def lookup_by_arxiv(arxiv_id):
    """Direct external-id lookup. Returns (rec_id, metadata) or None."""
    data = inspire_get(f"{API_BASE}/arxiv/{strip_version(arxiv_id)}")
    if isinstance(data, dict) and "metadata" in data:
        meta = data["metadata"]
        rec_id = meta.get("control_number") or data.get("id", "")
        return rec_id, meta
    return None


def search_eprint(arxiv_id):
    """Fallback: `find eprint <id>` search. Returns (rec_id, metadata) or None."""
    data = inspire_get(
        f"{API_BASE}/literature",
        params={"q": f"find eprint {strip_version(arxiv_id)}",
                "fields": "titles,citation_count,citation_count_without_self_citations,"
                          "control_number,publication_info,dois,arxiv_eprints",
                "size": "2"},
    )
    if isinstance(data, dict):
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            meta = hits[0].get("metadata", {})
            rec_id = meta.get("control_number") or hits[0].get("id", "")
            return rec_id, meta
    return None


def search_title(title):
    """Last resort: title search, first hit. Returns (rec_id, metadata) or None."""
    clean = re.sub(r"\s+", " ", (title or "").strip())
    if not clean:
        return None
    data = inspire_get(
        f"{API_BASE}/literature",
        params={"q": f't "{clean}"',
                "fields": "titles,citation_count,citation_count_without_self_citations,"
                          "control_number,publication_info,dois,arxiv_eprints",
                "size": "1"},
    )
    if isinstance(data, dict):
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            meta = hits[0].get("metadata", {})
            rec_id = meta.get("control_number") or hits[0].get("id", "")
            return rec_id, meta
    return None


def process_paper(arxiv_id, title):
    found = None
    if arxiv_id:
        found = lookup_by_arxiv(arxiv_id) or search_eprint(arxiv_id)
    if found is None:
        found = search_title(title)
    if found is None:
        return {"arxiv_id": arxiv_id, "title": title, "inspire_id": "",
                "citation_count": 0, "citations_no_self": "", "journal": "",
                "year": "", "doi": "",
                "inspire_url": "",
                "status": "not_found"}
    rec_id, meta = found
    return record_to_row(arxiv_id, title, rec_id, meta or {})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_work_list(limit):
    df = pd.read_csv(INPUT_CSV, usecols=["pdf_link", "title"])
    work, seen = [], set()
    for _, row in df.iterrows():
        aid = extract_arxiv_id(row.get("pdf_link"))
        key = aid or f"title:{row.get('title', '')}"
        if key and key not in seen:
            seen.add(key)
            work.append((aid or "", str(row.get("title", ""))))
    return work[:limit] if limit else work


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N papers (0 = all).")
    ap.add_argument("--restart", action="store_true", help="Ignore saved progress and start over.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    work = build_work_list(args.limit)
    done = set() if args.restart else load_state()
    todo = [(aid, t) for aid, t in work if (aid or t) not in done]
    print(f"Total: {len(work)} | already done: {len(work) - len(todo)} | to process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    processed = 0
    try:
        for aid, title in todo:
            if _stop.is_set():
                break
            row = process_paper(aid, title)
            append_rows([row])
            done.add(aid or title)
            processed += 1
            if processed % 25 == 0:
                save_state(done)
            print(f"  [{processed}/{len(todo)}] {aid or title[:60]}: "
                  f"{row['status']} (cited {row['citation_count']})", flush=True)
    finally:
        save_state(done)
        print(f"\nSaved progress: {len(done)} ids done. Output -> {OUTPUT_CSV}")
        if _stop.is_set():
            print("Stopped early; rerun the same command to resume.")


if __name__ == "__main__":
    main()
