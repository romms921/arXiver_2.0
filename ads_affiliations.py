"""
ads_affiliations.py
===================
Backfill missing author affiliations via the NASA ADS API.

Reads the unique authors whose affiliation is blank in
datasets/author_affiliations.csv (run author_affil.py first), queries ADS for
each author's most recent paper (author:"Name", sort by date desc), and maps
the ADS author list back to the matching author to pick their affiliation.

Output: datasets/ads_affiliations.csv with columns
  author, affiliation, ads_bibcode, status

An optional --input file (one author name per line, the `none_affil_authors.txt`
convention from the original prototype) can be used instead of / in addition
to the CSV scan.

Resume-safe (Ctrl+C safe): finished authors are tracked in
datasets/ads_affiliations_state.json and skipped on the next run.

Access / compliance: same as ads_citations.py — ADS API use requires a
personal token (ADS_API_KEY env / .env), ~5000 requests/day. This script does
one request per author plus a 60 s backoff on HTTP 429, with a short polite
pause between calls.

Usage:
  export ADS_API_KEY="your_token_here"
  python ads_affiliations.py               # backfill all missing affiliations
  python ads_affiliations.py --limit 20    # quick test on 20 authors
  python ads_affiliations.py --input none_affil_authors.txt
  python ads_affiliations.py --restart     # ignore saved progress
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
AUTHOR_CSV = os.path.join(HERE, "datasets", "author_affiliations.csv")
OUTPUT_CSV = os.path.join(HERE, "datasets", "ads_affiliations.csv")
STATE_PATH = os.path.join(HERE, "datasets", "ads_affiliations_state.json")

ADS_API_KEY = os.getenv("ADS_API_KEY", "")
ADS_URL = "https://api.adsabs.harvard.edu/v1/search/query"

POLITE_DELAY = 1.0  # seconds between calls (well under the daily quota)
RETRY_WAIT = 60

OUTPUT_COLUMNS = ["author", "affiliation", "ads_bibcode", "status"]

# --------------------------------------------------------------------------- #
# Graceful shutdown
# --------------------------------------------------------------------------- #
_stop = threading.Event()


def _handle_sigint(signum, frame):
    if _stop.is_set():
        print("\n[!] Second interrupt - exiting now.", flush=True)
        os._exit(1)
    print("\n[!] Stop requested. Finishing current author, then saving... "
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


def save_state(done):
    with _state_lock:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"done": sorted(done)}, f)
        os.replace(tmp, STATE_PATH)


def append_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    with _write_lock:
        header = not os.path.exists(OUTPUT_CSV)
        df.to_csv(OUTPUT_CSV, mode="a", header=header, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# ADS lookup
# --------------------------------------------------------------------------- #
def lookup_author(author_name, timeout=20):
    """Query ADS for the author's most recent paper. Returns a result dict."""
    while True:
        try:
            r = requests.get(
                ADS_URL,
                headers={"Authorization": f"Bearer {ADS_API_KEY}"},
                params={
                    "q": f'author:"{author_name}"',
                    "fl": "author,aff,bibcode",
                    "sort": "date desc",
                    "rows": 1,
                },
                timeout=timeout,
            )
            if r.status_code == 429:
                print(f"\n[Rate Limit] 429. Sleeping {RETRY_WAIT}s...", flush=True)
                time.sleep(RETRY_WAIT)
                continue
            if r.status_code == 401:
                return {"author": author_name, "affiliation": "", "ads_bibcode": "",
                        "status": "auth_error"}
            r.raise_for_status()
            time.sleep(POLITE_DELAY)
            docs = r.json().get("response", {}).get("docs", [])
            if not docs:
                return {"author": author_name, "affiliation": "", "ads_bibcode": "",
                        "status": "not_found"}
            return match_affiliation(author_name, docs[0])
        except requests.exceptions.RequestException as e:
            print(f"\n[Request Error] {author_name}: {e}", flush=True)
            return {"author": author_name, "affiliation": "", "ads_bibcode": "",
                    "status": "error"}


def match_affiliation(author_name, doc):
    """Map the ADS author/aff parallel lists back onto the queried name."""
    authors = doc.get("author", []) or []
    affs = doc.get("aff", []) or []
    bibcode = doc.get("bibcode", "")

    parts = author_name.replace(".", " ").split()
    if not parts:
        return {"author": author_name, "affiliation": "", "ads_bibcode": bibcode,
                "status": "invalid_name"}
    target_last = parts[-1].lower().strip()

    for i, ads_name in enumerate(authors):
        ads_last = ads_name.split(",")[0].lower().strip()
        if ads_last == target_last and i < len(affs):
            aff = (affs[i] or "").replace("\n", " ").replace("\r", " ").strip()
            if aff and aff != "-":
                return {"author": author_name, "affiliation": aff,
                        "ads_bibcode": bibcode, "status": "success"}
            return {"author": author_name, "affiliation": "", "ads_bibcode": bibcode,
                    "status": "affil_missing"}
    if len(authors) == 1 and affs and affs[0] not in ("", "-"):
        aff = affs[0].replace("\n", " ").replace("\r", " ").strip()
        return {"author": author_name, "affiliation": aff,
                "ads_bibcode": bibcode, "status": "success"}
    return {"author": author_name, "affiliation": "", "ads_bibcode": bibcode,
            "status": "name_mismatch"}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def authors_from_csv():
    if not os.path.exists(AUTHOR_CSV):
        return []
    df = pd.read_csv(AUTHOR_CSV, usecols=["author", "affiliation"])
    mask = df["affiliation"].isna() | (df["affiliation"].astype(str).str.strip() == "")
    return sorted({str(a).strip() for a in df.loc[mask, "author"] if str(a).strip()})


def authors_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None, help="Optional .txt of author names (one per line).")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N authors (0 = all).")
    ap.add_argument("--restart", action="store_true", help="Ignore saved progress and start over.")
    args = ap.parse_args()

    if not ADS_API_KEY:
        print("ADS_API_KEY is not set. Get a token at "
              "https://ui.adsabs.harvard.edu/user/settings/token and set:\n"
              '  export ADS_API_KEY="your_token"  (or add it to a .env file)',
              flush=True)
        sys.exit(2)

    names = authors_from_csv()
    if args.input:
        names = sorted(set(names) | set(authors_from_txt(args.input)))
    if args.limit:
        names = names[:args.limit]

    done = set() if args.restart else load_state()
    todo = [n for n in names if n not in done]
    print(f"Authors missing affiliations: {len(names)} | already done: "
          f"{len(names) - len(todo)} | to process: {len(todo)}")
    if not todo:
        print("Nothing to do. (Run author_affil.py first if author_affiliations.csv "
              "does not exist yet.)")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    processed = 0
    try:
        for name in todo:
            if _stop.is_set():
                break
            res = lookup_author(name)
            if res["status"] == "auth_error":
                print("\n[!] API key rejected (401). Stopping.", flush=True)
                break
            append_rows([res])
            done.add(name)
            processed += 1
            if processed % 25 == 0:
                save_state(done)
            print(f"  [{processed}/{len(todo)}] {name}: {res['status']}", flush=True)
    finally:
        save_state(done)
        print(f"\nSaved progress: {len(done)} authors done. Output -> {OUTPUT_CSV}")
        if _stop.is_set():
            print("Stopped early; rerun the same command to resume.")


if __name__ == "__main__":
    main()
