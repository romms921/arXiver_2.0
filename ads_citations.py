"""
ads_citations.py
================
Per-paper citation counts from the NASA ADS API (api.adsabs.harvard.edu).

Reads arXiv ids + titles from datasets/arxiv_papers.csv, resolves each paper
in ADS (arXiv identifier first, title search as fallback), then fetches:

  - ads_bibcode
  - total_citations        (ADS citation_count)
  - non_self_citations     (citing papers sharing no author with the target)

Output: datasets/ads_citations.csv with columns
  arxiv_id, title, ads_bibcode, total_citations, non_self_citations, status

Resume-safe (Ctrl+C safe): finished ids are tracked in
datasets/ads_citations_state.json and skipped on the next run.

Access / compliance notes:
  - The ADS API is the sanctioned programmatic endpoint (there is no
    robots.txt on api.adsabs.harvard.edu; api use is governed by the ADS API
    Terms of Use, not by scraping rules). It REQUIRES a personal token:
    https://ui.adsabs.harvard.edu/user/settings/token
    Set it as ADS_API_KEY in the environment or a .env file.
  - Rate limits: ~5000 requests/day. This script sleeps ~4 s between API
    calls and backs off 60 s on HTTP 429, so a default run stays well under
    the quota. Do not circumvent token limits; request an increase at
    adshelp@cfa.harvard.edu if needed.
  - Please acknowledge ADS in publications:
    "This research has made use of the Astrophysics Data System, funded by
    NASA under Cooperative Agreement 80NSSC25M7105."

Usage:
  export ADS_API_KEY="your_token_here"
  python ads_citations.py                 # full run, resumable
  python ads_citations.py --limit 10      # quick test on 10 papers
  python ads_citations.py --batch-size 5  # batch arXiv-id resolution (fewer calls)
  python ads_citations.py --restart       # ignore saved progress
"""

import argparse
import json
import os
import re
import signal
import sys
import threading
import time

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Configuration / paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "datasets", "arxiv_papers.csv")
OUTPUT_CSV = os.path.join(HERE, "datasets", "ads_citations.csv")
STATE_PATH = os.path.join(HERE, "datasets", "ads_citations_state.json")

ADS_API_KEY = os.getenv("ADS_API_KEY", "")
ADS_URL = "https://api.adsabs.harvard.edu/v1/search/query"

POLITE_DELAY = 4.0  # seconds between ADS calls (~900/hr max, under the quota)
RETRY_WAIT = 60  # seconds to wait on HTTP 429
MAX_CITATIONS_TO_CHECK = 1000  # ADS `rows` cap for the citing-papers query

OUTPUT_COLUMNS = [
    "arxiv_id",
    "title",
    "ads_bibcode",
    "total_citations",
    "non_self_citations",
    "status",
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
    """arxiv.org/pdf/2601.00044 -> '2601.00044' (handles old-style ids too)."""
    if not isinstance(pdf_link, str):
        return None
    m = _ID_RE.search(pdf_link)
    if m:
        return m.group(1)
    m = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})", pdf_link)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# ADS plumbing
# --------------------------------------------------------------------------- #
def _headers():
    return {"Authorization": f"Bearer {ADS_API_KEY}"}


def ads_request(params, timeout=20):
    """GET the ADS search endpoint, handling 429/401. Returns JSON or None."""
    while True:
        try:
            r = requests.get(ADS_URL, headers=_headers(), params=params, timeout=timeout)
            if r.status_code == 429:
                reset = r.headers.get("X-RateLimit-Reset", "?")
                print(f"\n[Rate Limit] 429 (reset info: {reset}). "
                      f"Sleeping {RETRY_WAIT}s...", flush=True)
                time.sleep(RETRY_WAIT)
                continue
            if r.status_code == 401:
                print("\n[Error] 401 Unauthorized. Is ADS_API_KEY correct? "
                      "Get one at https://ui.adsabs.harvard.edu/user/settings/token",
                      flush=True)
                return None
            r.raise_for_status()
            time.sleep(POLITE_DELAY)
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"\n[HTTP Error] {e}", flush=True)
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n[Request Error] {e}", flush=True)
            return None


def resolve_batch(batch_rows):
    """Resolve a batch of (arxiv_id, title) pairs via one arXiv-identifier query.

    Returns {arxiv_id: doc}. Papers found here skip the per-paper title search.
    """
    mapping = {}
    ids = [aid for aid, _ in batch_rows if aid]
    if not ids:
        return mapping
    q = " OR ".join(f'identifier:"arXiv:{aid}"' for aid in ids)
    data = ads_request({
        "q": q,
        "fl": "bibcode,title,author,aff,citation_count,identifier",
        "rows": len(ids) * 2,
    })
    if not data:
        return mapping
    for doc in data.get("response", {}).get("docs", []):
        for iden in doc.get("identifier", []):
            if iden.startswith("arXiv:"):
                mapping[iden.split(":", 1)[1]] = doc
                break
    return mapping


def get_non_self_citations(bibcode, target_authors):
    """Count citing papers sharing no author with the target. None on failure."""
    data = ads_request({
        "q": f'citations("{bibcode}")',
        "fl": "author",
        "rows": MAX_CITATIONS_TO_CHECK,
    })
    if data is None:
        return None
    docs = data.get("response", {}).get("docs", [])
    if not docs:
        return 0
    targets = {a.lower().strip() for a in target_authors if isinstance(a, str)}
    non_self = 0
    for doc in docs:
        citing = {a.lower().strip() for a in doc.get("author", []) if isinstance(a, str)}
        if not (targets & citing):
            non_self += 1
    return non_self


def process_paper(arxiv_id, title, batch_doc):
    """Returns a result dict (always marks done; failures use status != success)."""
    doc = batch_doc
    if doc is None:
        clean = re.sub(r"[^\w\s]", " ", title or "").strip()
        if clean:
            data = ads_request({
                "q": f'title:"{clean}"',
                "fl": "bibcode,author,aff,citation_count",
                "rows": 1,
            })
            if data and data.get("response", {}).get("docs"):
                doc = data["response"]["docs"][0]
    if not doc:
        return {"arxiv_id": arxiv_id, "title": title, "ads_bibcode": "",
                "total_citations": 0, "non_self_citations": 0, "status": "not_found"}

    bibcode = doc.get("bibcode", "")
    total = doc.get("citation_count", 0) or 0
    authors = doc.get("author", []) or []
    non_self = total
    if 0 < total <= MAX_CITATIONS_TO_CHECK:
        ns = get_non_self_citations(bibcode, authors)
        if ns is not None:
            non_self = ns
    return {"arxiv_id": arxiv_id, "title": title, "ads_bibcode": bibcode,
            "total_citations": total, "non_self_citations": non_self,
            "status": "success"}


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
    ap.add_argument("--batch-size", type=int, default=10, help="Papers per batched ADS identifier query.")
    ap.add_argument("--restart", action="store_true", help="Ignore saved progress and start over.")
    args = ap.parse_args()

    if not ADS_API_KEY:
        print("ADS_API_KEY is not set. Get a token at "
              "https://ui.adsabs.harvard.edu/user/settings/token and set:\n"
              '  export ADS_API_KEY="your_token"  (or add it to a .env file)',
              flush=True)
        sys.exit(2)

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
        for i in range(0, len(todo), args.batch_size):
            if _stop.is_set():
                break
            batch = todo[i:i + args.batch_size]
            mapping = resolve_batch(batch)
            rows = []
            for aid, title in batch:
                if _stop.is_set():
                    break
                rows.append(process_paper(aid, title, mapping.get(aid)))
            append_rows(rows)
            for r in rows:
                done.add(r["arxiv_id"] or r["title"])
            processed += len(rows)
            if processed % 20 == 0:
                save_state(done)
            print(f"  ... {processed}/{len(todo)} papers", flush=True)
    finally:
        save_state(done)
        print(f"\nSaved progress: {len(done)} ids done. Output -> {OUTPUT_CSV}")
        if _stop.is_set():
            print("Stopped early; rerun the same command to resume.")


if __name__ == "__main__":
    main()
