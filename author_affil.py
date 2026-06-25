"""
author_affil.py
================
Extract author -> affiliation (and an inferred gender) for every arXiv paper
listed in datasets/arxiv_papers.csv.

Pipeline per paper:
  1. Pull the arXiv id out of the `pdf_link` column (e.g. arxiv.org/pdf/2601.00044 -> 2601.00044).
  2. Download the LaTeX source (arxiv.org/e-print/<id>), cached on disk.
  3. Send the relevant TeX (preamble + author block) to a local Ollama model.
  4. The model returns author/affiliation pairs and infers each author's gender.
     When it cannot infer gender from the name alone it calls the `web_search`
     tool (DuckDuckGo, no API key) and decides from the results.
  5. Rows are appended to datasets/author_affiliations.csv.

Designed to be stopped (Ctrl+C) and resumed: finished ids are tracked in a
state file and skipped on the next run. Downloads are cached so a resume never
re-fetches source it already has.

Batch / GPU:
  --batch-size 1   -> sequential, gentle on a small GPU (default)
  --batch-size 8   -> N papers in flight at once (set OLLAMA_NUM_PARALLEL on the
                      Ollama server to actually run them in parallel on a big GPU)

Examples:
  python author_affil.py                       # sequential, full run, resumable
  python author_affil.py --batch-size 8        # batched for a big GPU
  python author_affil.py --limit 20            # quick smoke test on 20 papers
  python author_affil.py --no-search           # disable the web_search tool
"""

import argparse
import gzip
import io
import json
import os
import re
import signal
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------- #
# Configuration / paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(HERE, "datasets", "arxiv_papers.csv")
OUTPUT_CSV = os.path.join(HERE, "datasets", "author_affiliations.csv")
STATE_PATH = os.path.join(HERE, "datasets", "author_affil_state.json")
CACHE_DIR = os.path.join(HERE, "cache", "latex")

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

ARXIV_HEADERS = {"User-Agent": "arxiver-author-affil/1.0"}
ARXIV_DELAY = 15.0          # polite per-thread delay between arXiv source fetches
OUTPUT_COLUMNS = [
    "arxiv_id", "author", "affiliation", "gender", "gender_method", "gender_confidence",
]

# --------------------------------------------------------------------------- #
# Graceful shutdown
# --------------------------------------------------------------------------- #
_stop = threading.Event()


def _handle_sigint(signum, frame):
    if _stop.is_set():
        print("\n[!] Second interrupt - exiting now.", flush=True)
        os._exit(1)
    print("\n[!] Stop requested. Finishing in-flight papers, then saving... "
          "(Ctrl+C again to force quit)", flush=True)
    _stop.set()


signal.signal(signal.SIGINT, _handle_sigint)


# --------------------------------------------------------------------------- #
# State (resume support)
# --------------------------------------------------------------------------- #
_state_lock = threading.Lock()


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("done", []))
        except Exception:
            pass
    return set()


def save_state(done_ids):
    with _state_lock:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"done": sorted(done_ids)}, f)
        os.replace(tmp, STATE_PATH)


# --------------------------------------------------------------------------- #
# Output writing (thread-safe, incremental)
# --------------------------------------------------------------------------- #
_write_lock = threading.Lock()


def append_rows(rows):
    """Append a list of dicts to the output CSV, writing the header once."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    with _write_lock:
        header = not os.path.exists(OUTPUT_CSV)
        df.to_csv(OUTPUT_CSV, mode="a", header=header, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# arXiv id + LaTeX source download
# --------------------------------------------------------------------------- #
_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def extract_arxiv_id(pdf_link):
    """arxiv.org/pdf/2601.00044 -> '2601.00044' (handles old-style ids too)."""
    if not isinstance(pdf_link, str):
        return None
    m = _ID_RE.search(pdf_link)
    if m:
        return m.group(1)
    # old-style ids, e.g. astro-ph/0601001
    m = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})", pdf_link)
    return m.group(1) if m else None


def _cache_path(arxiv_id):
    safe = arxiv_id.replace("/", "_")
    return os.path.join(CACHE_DIR, safe + ".tex")


def download_source(arxiv_id, session):
    """Return (status, tex) for an id, using an on-disk cache.

    status is one of:
      "ok"       -> tex is the LaTeX source
      "nosource" -> arXiv has no LaTeX source (PDF-only); cached, never refetched
      "error"    -> transient failure (network); not cached, retry next run

    Cache values: the TeX string, or the sentinel 'NO_TEX' for "nosource".
    """
    cpath = _cache_path(arxiv_id)
    if os.path.exists(cpath):
        with open(cpath, "r", encoding="utf-8", errors="replace") as f:
            cached = f.read()
        return ("nosource", "") if cached == "NO_TEX" else ("ok", cached)

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        resp = session.get(url, headers=ARXIV_HEADERS, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [{arxiv_id}] download failed: {e}", flush=True)
        return ("error", "")
    time.sleep(ARXIV_DELAY)  # be polite to arXiv

    tex = _extract_tex_from_bytes(resp.content)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cpath, "w", encoding="utf-8", errors="replace") as f:
        f.write(tex if tex else "NO_TEX")
    return ("ok", tex) if tex else ("nosource", "")


def _decode(b):
    """arXiv .tex files are usually UTF-8 but some are latin-1; try both before
    falling back to lossy replacement (avoids turning accents into U+FFFD)."""
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def _extract_tex_from_bytes(raw):
    """arXiv e-print payloads are: a gzipped tar, a single gzipped file, or raw."""
    # Try gzipped tar
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            parts = []
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".tex"):
                    fh = tar.extractfile(member)
                    if fh:
                        parts.append(_decode(fh.read()))
            if parts:
                return _order_tex(parts)
    except (tarfile.TarError, EOFError):
        pass
    # Try single gzipped tex file
    try:
        data = _decode(gzip.decompress(raw))
        if "\\documentclass" in data or "\\begin{document}" in data:
            return data
    except (OSError, EOFError):
        pass
    # Raw text fallback
    try:
        data = _decode(raw)
        if "\\documentclass" in data:
            return data
    except Exception:
        pass
    return ""


def _order_tex(parts):
    """Put the main file (the one with \\documentclass) first."""
    main = [p for p in parts if "\\documentclass" in p]
    rest = [p for p in parts if "\\documentclass" not in p]
    return "\n".join(main + rest)


# \author occurrences (the real author list); excludes \authorrunning via \b.
_AUTHOR_RE = re.compile(r"\\author\b")
# Affiliation markers/definitions, e.g. \affiliation{...}, \affil, \institute, and
# the inner \affiliation of \newcommand{\JHU}{\affiliation{...}} macro definitions.
_AFFIL_RE = re.compile(
    r"\\(affiliation|affil|altaffil(?:iation)?|institute|institution)\b", re.IGNORECASE)
# Lines in the preamble worth keeping: title + anything that *defines* or names
# an affiliation/author. Critically this captures \newcommand{\JHU}{\affiliation{...}}
# style macros so the model can resolve macro-based affiliations used in the body.
_PREAMBLE_KW = ("affil", "institut", r"\inst", "address", r"\thanks", r"\email",
                "orcid", r"\author", r"\correspond", "altaffil")


def _filtered_preamble(preamble, cap):
    """Keep only preamble lines that define affiliations/authors or the title."""
    keep = []
    for line in preamble.splitlines():
        low = line.lower()
        if r"\documentclass" in low or r"\title" in low or any(k in low for k in _PREAMBLE_KW):
            keep.append(line)
    out = "\n".join(keep)
    return out[:cap]


def relevant_tex(tex, preamble_cap=7000, block_cap=16000):
    """Carve out the author/affiliation region, wherever it lives.

    Astro papers put the author block at the top OR the bottom, and frequently
    define affiliations as preamble macros (\\newcommand{\\JHU}{\\affiliation{...}})
    that are merely *referenced* in the author list. So we send two things:
      1. Filtered preamble  -> the affiliation macro *definitions* + title.
      2. The author cluster -> the contiguous run of \\author commands (with their
         macro *calls* and \\email/\\affiliation lines), found anywhere in the doc.
    """
    if not tex:
        return ""

    bd = tex.find(r"\begin{document}")
    preamble = tex[:bd] if bd != -1 else tex[:preamble_cap]
    preamble_kept = _filtered_preamble(preamble, preamble_cap)

    # Locate the author cluster across the WHOLE document (top or bottom).
    marks = [m.start() for m in _AUTHOR_RE.finditer(tex)]
    if marks:
        a_lo, a_hi = min(marks), max(marks)
        # Pull in affiliation definitions sitting just above the author list
        # (common in AASTeX: \newcommand{\JHU}{\affiliation{...}} then \author... \JHU).
        affil_marks = [m.start() for m in _AFFIL_RE.finditer(tex)]
        preceding = [p for p in affil_marks if a_lo - 8000 <= p < a_lo]
        lo = max(0, (min(preceding) if preceding else a_lo) - 150)
        hi = min(len(tex), a_hi + 600)
        block = tex[lo:hi]
        if len(block) > block_cap:
            block = block[:block_cap]
    else:
        # No \author at all: fall back to the region up to \maketitle / first section.
        m = re.search(r"\\maketitle", tex) or re.search(r"\\section", tex)
        end = (m.end() + 1500) if m and r"\maketitle" in m.group(0) else (
            m.start() if m else len(tex))
        block = tex[:min(end, block_cap)]

    parts = []
    if preamble_kept.strip():
        parts.append("% --- preamble (affiliation definitions / title) ---\n" + preamble_kept)
    parts.append("% --- author block ---\n" + block)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# web_search tool (DuckDuckGo HTML, no API key)
# --------------------------------------------------------------------------- #
def web_search(query, session, max_results=5):
    try:
        resp = session.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"search error: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    out = []
    for res in soup.select(".result__body")[:max_results]:
        title = res.select_one(".result__a")
        snippet = res.select_one(".result__snippet")
        line = " - ".join(
            t.get_text(" ", strip=True) for t in (title, snippet) if t
        )
        if line:
            out.append(line)
    return "\n".join(out) if out else "no results"


# --------------------------------------------------------------------------- #
# Ollama extraction (tool-enabled chat)
# --------------------------------------------------------------------------- #
EXTRACT_PROMPT = """You extract author metadata from the LaTeX source of a scientific paper.

Return a JSON object: {"authors": [{...}, ...]} with ONE entry per author, in order.
Each entry has: author, affiliation, gender, gender_method, gender_confidence.

Rules:
- Find EVERY author. Names come from \\author, \\author[orcid]{Name}, \\authors, or an
  author list separated by \\and / commas. Do not skip anyone and do not invent anyone.
- Map each author to their affiliation. Affiliations come from \\affiliation, \\affil,
  \\altaffiliation, \\institute, \\thanks, or footnote markers. They may be defined as
  macros, e.g. \\newcommand{\\JHU}{\\affiliation{Johns Hopkins...}} and then referenced
  as \\JHU right after an author; resolve those macros to the real institution text.
  If several, give the primary (first) one. If truly unknown, use "".
- gender: infer "male"/"female" from the given (first) name when reasonably sure and set
  gender_method="name". If the name is ambiguous/unfamiliar, set gender="unknown",
  gender_method="unknown" (a later step will search the web). gender_confidence is
  high|medium|low.
- Output JSON only."""

GENDER_PROMPT = """You determine the gender of researchers. For each name you are unsure
about, call the web_search tool (e.g. query '"Full Name" <affiliation> he OR she') and
read the snippets for pronouns or other gender cues. When done, reply with ONLY a JSON
object mapping each full name to "male", "female", or "unknown":
{"Full Name": "male", "Other Name": "unknown"}"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web to help determine an author's gender.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
}


# Structured-output schema so Ollama is forced to emit parseable JSON.
AUTHORS_SCHEMA = {
    "type": "object",
    "properties": {
        "authors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "author": {"type": "string"},
                    "affiliation": {"type": "string"},
                    "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                    "gender_method": {"type": "string", "enum": ["name", "search", "unknown"]},
                    "gender_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["author", "affiliation", "gender", "gender_method", "gender_confidence"],
            },
        }
    },
    "required": ["authors"],
}


def _ollama_chat(model, messages, tools, fmt=None, num_ctx=8192, timeout=900):
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": "30m",  # keep the model warm between papers
               "options": {"temperature": 0, "num_ctx": num_ctx}}
    if tools:
        payload["tools"] = tools
    if fmt is not None:
        payload["format"] = fmt
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _parse_json_object(text):
    """Pull the first {...} JSON object out of a model reply."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def extract_authors(arxiv_id, tex_snippet, model, session, use_search):
    """Two passes: (1) structured author/affiliation extraction, (2) gender via search.

    Returns a list of author dicts, or None on a transient/hard failure.
    """
    # --- Pass 1: forced-JSON structured extraction (no tools) ---
    messages = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user",
         "content": f"arXiv id: {arxiv_id}\n\nLaTeX source:\n```\n{tex_snippet}\n```"},
    ]
    try:
        data = _ollama_chat(model, messages, tools=None, fmt=AUTHORS_SCHEMA)
    except Exception as e:
        print(f"  [{arxiv_id}] ollama error (extract): {e}", flush=True)
        return None
    parsed = _parse_json_object(data.get("message", {}).get("content", ""))
    if not isinstance(parsed, dict):
        return None
    authors = parsed.get("authors", [])
    if not isinstance(authors, list):
        return None

    # --- Pass 2: resolve unknown genders with the web_search tool ---
    if use_search:
        unknown = [a for a in authors
                   if isinstance(a, dict)
                   and (a.get("gender") or "unknown") == "unknown"
                   and (a.get("author") or "").strip()]
        if unknown:
            resolved = resolve_genders(arxiv_id, unknown, model, session)
            for a in unknown:
                g = resolved.get((a.get("author") or "").strip())
                if g in ("male", "female"):
                    a["gender"] = g
                    a["gender_method"] = "search"
                    a["gender_confidence"] = "low"
    return authors


def resolve_genders(arxiv_id, unknown_authors, model, session, max_tool_calls=12):
    """Tool-enabled chat: let the model web_search and return {name: gender}."""
    listing = "\n".join(
        f"- {a.get('author')} ({a.get('affiliation') or 'affiliation unknown'})"
        for a in unknown_authors
    )
    messages = [
        {"role": "system", "content": GENDER_PROMPT},
        {"role": "user", "content": f"Determine the gender of these researchers:\n{listing}"},
    ]
    for _ in range(max_tool_calls + 1):
        try:
            data = _ollama_chat(model, messages, tools=[WEB_SEARCH_TOOL])
        except Exception as e:
            print(f"  [{arxiv_id}] ollama error (gender): {e}", flush=True)
            return {}
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            messages.append(msg)
            for call in tool_calls:
                fn = call.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = _parse_json_object(args) or {}
                query = args.get("query", "")
                result = web_search(query, session) if query else "no query"
                messages.append({"role": "tool", "name": fn.get("name", "web_search"),
                                 "content": result})
            continue
        parsed = _parse_json_object(msg.get("content", ""))
        return parsed if isinstance(parsed, dict) else {}
    return {}


# --------------------------------------------------------------------------- #
# Per-paper worker
# --------------------------------------------------------------------------- #
def process_paper(arxiv_id, model, use_search):
    """Returns (arxiv_id, rows, ok).

    ok=False  -> transient failure (download/Ollama); NOT marked done, retried next run.
    ok=True   -> paper handled; rows may be [] (no LaTeX source / no authors found).
    """
    session = requests.Session()
    try:
        status, tex = download_source(arxiv_id, session)
        if status == "error":
            return arxiv_id, [], False
        if status == "nosource":
            return arxiv_id, [], True
        snippet = relevant_tex(tex)
        if not snippet.strip():
            return arxiv_id, [], True
        authors = extract_authors(arxiv_id, snippet, model, session, use_search)
        if authors is None:
            return arxiv_id, [], False  # ollama/parse failure -> retry later
        rows = []
        for a in authors:
            if not isinstance(a, dict):
                continue
            name = (a.get("author") or "").strip()
            if not name:
                continue
            rows.append({
                "arxiv_id": arxiv_id,
                "author": name,
                "affiliation": (a.get("affiliation") or "").strip(),
                "gender": (a.get("gender") or "unknown").strip().lower(),
                "gender_method": (a.get("gender_method") or "unknown").strip().lower(),
                "gender_confidence": (a.get("gender_confidence") or "").strip().lower(),
            })
        return arxiv_id, rows, True
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_id_list(limit):
    df = pd.read_csv(INPUT_CSV, usecols=["pdf_link"])
    ids = []
    seen = set()
    for link in df["pdf_link"]:
        aid = extract_arxiv_id(link)
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)
    if limit:
        ids = ids[:limit]
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Papers processed concurrently. 1 = sequential (small GPU). "
                         ">1 = batched (set OLLAMA_NUM_PARALLEL on the server for a big GPU).")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N papers (0 = all).")
    ap.add_argument("--no-search", action="store_true", help="Disable the web_search gender tool.")
    ap.add_argument("--restart", action="store_true", help="Ignore saved progress and start over.")
    args = ap.parse_args()

    use_search = not args.no_search
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    all_ids = build_id_list(args.limit)
    done = set() if args.restart else load_state()
    todo = [i for i in all_ids if i not in done]

    print(f"Model: {args.model} | batch-size: {args.batch_size} | "
          f"web_search: {'on' if use_search else 'off'}")
    print(f"Total: {len(all_ids)} | already done: {len(all_ids) - len(todo)} | "
          f"to process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    processed = 0
    failed = 0

    def _finish(aid, rows, ok):
        nonlocal processed, failed
        if not ok:
            failed += 1
            print(f"  [{aid}] FAILED (will retry next run)", flush=True)
            return
        append_rows(rows)
        done.add(aid)
        processed += 1
        if processed % 25 == 0:
            save_state(done)
        print(f"  [{aid}] {len(rows)} authors  ({processed}/{len(todo)})", flush=True)

    try:
        if args.batch_size <= 1:
            for aid in todo:
                if _stop.is_set():
                    break
                rid, rows, ok = process_paper(aid, args.model, use_search)
                _finish(rid, rows, ok)
        else:
            with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
                futures = {}
                it = iter(todo)
                # prime the pool
                for _ in range(args.batch_size):
                    try:
                        aid = next(it)
                    except StopIteration:
                        break
                    futures[pool.submit(process_paper, aid, args.model, use_search)] = aid
                while futures:
                    for fut in as_completed(list(futures)):
                        del futures[fut]
                        rid, rows, ok = fut.result()
                        _finish(rid, rows, ok)
                        if not _stop.is_set():
                            try:
                                aid = next(it)
                                futures[pool.submit(process_paper, aid, args.model, use_search)] = aid
                            except StopIteration:
                                pass
                        break  # re-evaluate as_completed over the refreshed set
    finally:
        save_state(done)
        print(f"\nSaved progress: {len(done)} ids done, {failed} failed (retryable). "
              f"Output -> {OUTPUT_CSV}")
        if _stop.is_set():
            print("Stopped early; rerun the same command to resume.")


if __name__ == "__main__":
    main()
