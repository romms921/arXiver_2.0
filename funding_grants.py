"""
funding_grants.py
=================
Extract funding sources and grant / award numbers from the *Acknowledgements*
(or *Funding*) section of arXiv papers, using the LaTeX source already cached by
author_affil.py in cache/latex/.

This script does NOT download anything - it only reads the local cache. Run
author_affil.py first (or at least let it cache the source) so the .tex files
exist. Papers whose source isn't cached yet are skipped (not marked done), so a
later run will pick them up once they're cached.

Pipeline per paper:
  1. Read the cached LaTeX for the arXiv id.
  2. Locate the acknowledgements / funding text (handles many LaTeX conventions,
     see locate_acknowledgements()).
  3. Extract (funder, grant_id) pairs:
       - default: a local Ollama model (handles the messy variety of grant
         formats), or
       - --regex-only: a fast, offline, LLM-free heuristic pass.
  4. Append rows to datasets/funding_grants.csv.

Resumable (Ctrl+C safe), with optional batching - same conventions as
author_affil.py.

Examples:
  python funding_grants.py                  # LLM extraction, resumable
  python funding_grants.py --limit 10       # quick test on 10 papers
  python funding_grants.py --regex-only     # no Ollama, fast heuristic pass
  python funding_grants.py --batch-size 8   # batched for a big GPU
"""

import argparse
import json
import os
import re
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Reuse the stable, shared helpers from author_affil so the cache layout, id
# parsing and Ollama plumbing stay consistent across the two scripts.
import author_affil as aa

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = aa.INPUT_CSV
CACHE_DIR = aa.CACHE_DIR
OUTPUT_CSV = os.path.join(HERE, "datasets", "funding_grants.csv")
STATE_PATH = os.path.join(HERE, "datasets", "funding_state.json")

DEFAULT_MODEL = aa.DEFAULT_MODEL
ACK_MAX_CHARS = 9000        # ack/funding text is short; cap what we feed the model
OUTPUT_COLUMNS = ["arxiv_id", "funder", "grant_id", "method", "context"]

# --------------------------------------------------------------------------- #
# Graceful shutdown (own handler; overrides the one author_affil registered)
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
# State + output (thread-safe, incremental)
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
# Locating the acknowledgements / funding text
# --------------------------------------------------------------------------- #
# Strip LaTeX line comments so commented-out sections (e.g. "%\begin{acknowledgments}"
# or "%%%% ACKNOWLEDGEMENTS %%%%") are never mistaken for the real thing.
# A "%" preceded by a backslash (\%) is a literal percent, not a comment.
def strip_comments(tex):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())


# Headings whose title mentions acknowledgements / funding / financial support.
_HEAD_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph|chapter)\*?\s*\{[^{}]*?"
    r"(?:acknowledg\w*|funding|financial support|grant support)[^{}]*\}",
    re.I,
)
# AASTeX-style command form: \acknowledgments / \acknowledgements (with or without braces).
_ACK_CMD_RE = re.compile(r"\\acknowledg(?:e?ments?|ement)\b\s*", re.I)
# A&A / AASTeX environment form.
_ACK_ENV_RE = re.compile(
    r"\\begin\{(acknowledg\w*)\}(.*?)\\end\{\1\}", re.I | re.S)
# MDPI-style command: \funding{...}
_FUNDING_CMD_RE = re.compile(r"\\funding\b\s*", re.I)
# Where an acknowledgements block ends: the next structural marker.
_ENDERS_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph|chapter)\b"
    r"|\\bibliography\b|\\begin\{thebibliography\}|\\printbibliography\b"
    r"|\\appendix\b|\\begin\{appendices\}|\\end\{document\}"
    r"|\\begin\{contribution\}|\\begin\{contributions\}",
)


def _read_balanced_braces(tex, open_pos):
    """Given index of a '{', return (content, index_after_closing_brace)."""
    depth = 0
    for i in range(open_pos, len(tex)):
        c = tex[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return tex[open_pos + 1:i], i + 1
    return tex[open_pos + 1:], len(tex)


def _block_after(tex, start):
    """Text from `start` up to the next structural marker (or end of doc)."""
    m = _ENDERS_RE.search(tex, start)
    return tex[start:(m.start() if m else len(tex))]


# Strong funding cues, for papers that put acknowledgements in plain prose with
# no heading or environment (it happens - the markup is sometimes commented out).
_FUND_CUE_RE = re.compile(
    r"acknowledge[sd]?\b|funded by|financial support|supported by|"
    r"grant\s+(?:no|number|agreement)|under\s+(?:grant|award)|fellowship",
    re.I,
)
_BIB_RE = re.compile(
    r"\\begin\{thebibliography\}|\\bibliography\b|\\printbibliography\b|\\end\{document\}")


def _fallback_ack(tex):
    """No marked section found: harvest funding-cue paragraphs from the tail.

    Conservative: only looks after the last \\section (or the document midpoint)
    and stops at the bibliography, so intro/methods prose can't leak in.
    """
    last_sec = [m.start() for m in re.finditer(r"\\section\b", tex)]
    region_start = max(last_sec[-1] if last_sec else 0, len(tex) // 3)
    tail = tex[region_start:]
    cut = _BIB_RE.search(tail)
    if cut:
        tail = tail[:cut.start()]
    keep = [p.strip() for p in re.split(r"\n\s*\n", tail) if _FUND_CUE_RE.search(p)]
    return "\n\n".join(keep)


def locate_acknowledgements(tex):
    """Return the acknowledgements/funding text found anywhere in the document.

    Collects every matching block (a paper may have both an Acknowledgements
    section and a separate Funding paragraph), merges overlaps, and caps length.
    """
    if not tex:
        return ""
    tex = strip_comments(tex)
    spans = []  # (start, end)

    # 1. \begin{acknowledg...} ... \end{acknowledg...}
    for m in _ACK_ENV_RE.finditer(tex):
        spans.append((m.start(2), m.end(2)))

    # 2. \section*{Acknowledgements} / \paragraph*{Funding:} style headings
    for m in _HEAD_RE.finditer(tex):
        block = _block_after(tex, m.end())
        spans.append((m.end(), m.end() + len(block)))

    # 3. \funding{...}
    for m in _FUNDING_CMD_RE.finditer(tex):
        brace = tex.find("{", m.end())
        if brace != -1 and brace - m.end() < 5:
            content, _ = _read_balanced_braces(tex, brace)
            spans.append((brace, brace + len(content)))

    # 4. \acknowledgments command form (skip the \begin/\end already handled).
    for m in _ACK_CMD_RE.finditer(tex):
        tail = tex[m.end():m.end() + 1]
        if tex[max(0, m.start() - 6):m.start()].endswith(("begin{", "\\end{")):
            continue  # part of an environment, handled above
        if tail == "{":  # \acknowledgments{...}
            content, _ = _read_balanced_braces(tex, m.end())
            spans.append((m.end(), m.end() + len(content)))
        else:            # \acknowledgments <free text...>
            block = _block_after(tex, m.end())
            spans.append((m.end(), m.end() + len(block)))

    if not spans:
        return _fallback_ack(tex)[:ACK_MAX_CHARS]

    # Merge overlapping/adjacent spans, then concatenate in document order.
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    text = "\n\n".join(tex[s:e].strip() for s, e in merged)
    return text[:ACK_MAX_CHARS]


# --------------------------------------------------------------------------- #
# Grant extraction - LLM
# --------------------------------------------------------------------------- #
GRANTS_PROMPT = """You read the acknowledgements / funding text of a research paper
and extract the funding agencies and their grant/award numbers.

Return JSON: {"grants": [{"funder": "...", "grant_id": "..."}, ...]}

Rules:
- One entry per grant/award number. If a funder gave several numbers, emit one
  entry per number with the same funder.
- "grant_id" is the identifier exactly as written (e.g. "AST-2108414",
  "ANR-24-CE92-0044", "KL 1358/22-1", "GO 17502", "855130"). Keep its original form.
- "funder" is the agency/foundation/program name (e.g. "NSF", "ERC",
  "National Natural Science Foundation of China"). If the text gives no clear
  funder for a number, use "".
- Only include real grant/award/contract/project numbers. Do NOT include people's
  names, telescope proposal text that isn't an ID, DOIs, URLs, or software.
- If there are no grant numbers, return {"grants": []}. Do not invent anything."""

GRANTS_SCHEMA = {
    "type": "object",
    "properties": {
        "grants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "funder": {"type": "string"},
                    "grant_id": {"type": "string"},
                },
                "required": ["funder", "grant_id"],
            },
        }
    },
    "required": ["grants"],
}


def llm_grants(arxiv_id, ack_text, model):
    """Return list of {funder, grant_id} dicts, or None on a transient failure."""
    messages = [
        {"role": "system", "content": GRANTS_PROMPT},
        {"role": "user", "content": f"Acknowledgements / funding text:\n```\n{ack_text}\n```"},
    ]
    try:
        data = aa._ollama_chat(model, messages, tools=None, fmt=GRANTS_SCHEMA, num_ctx=8192)
    except Exception as e:
        print(f"  [{arxiv_id}] ollama error: {e}", flush=True)
        return None
    parsed = aa._parse_json_object(data.get("message", {}).get("content", ""))
    if not isinstance(parsed, dict):
        return None
    grants = parsed.get("grants", [])
    return grants if isinstance(grants, list) else None


# --------------------------------------------------------------------------- #
# Grant extraction - regex fallback (offline, no Ollama)
# --------------------------------------------------------------------------- #
# Trigger words that usually precede a grant/award identifier.
_TRIGGER = (r"grants?(?:\s+agreement)?(?:\s+(?:nos?|numbers?|id))?\.?|"
            r"awards?(?:\s+(?:nos?|numbers?))?\.?|contracts?(?:\s+(?:nos?|numbers?))?\.?|"
            r"agreements?(?:\s+(?:nos?|numbers?))?\.?|funding\s+id|projects?(?:\s+id)?|"
            r"fellowships?|under")
# An identifier: contains at least one digit, may have letters/-/./ separators,
# length >= 4. An optional leading agency prefix is matched case-SENSITIVELY
# (?-i:...) so trigger words like "grant"/"grants" aren't swallowed into the id.
_ID = r"(?:(?-i:[A-Z]{1,6})[-\s]?)?[0-9][\w./\-]{2,}[0-9A-Za-z]"
_GRANT_RE = re.compile(rf"(?:{_TRIGGER})[\s:]*[#]?\s*({_ID})", re.I)
# Strip LaTeX accents/markup before regex matching so IDs aren't split by macros.
_TEX_CLEAN_RE = re.compile(r"\\[a-zA-Z]+\*?|[{}~$\\]")


def regex_grants(ack_text):
    """Heuristic, LLM-free extraction. Lower precision; for a quick offline pass."""
    text = _TEX_CLEAN_RE.sub(" ", ack_text)
    out, seen = [], set()
    for m in _GRANT_RE.finditer(text):
        gid = m.group(1).strip(" .,;-/")
        # discard things that are clearly years or too short to be IDs
        if len(re.sub(r"\D", "", gid)) < 3:
            continue
        if re.fullmatch(r"(19|20)\d{2}", gid):
            continue
        if gid in seen:
            continue
        seen.add(gid)
        ctx = text[max(0, m.start() - 40):m.end() + 10].strip()
        ctx = re.sub(r"\s+", " ", ctx)
        out.append({"funder": "", "grant_id": gid, "context": ctx})
    return out


# --------------------------------------------------------------------------- #
# Per-paper worker
# --------------------------------------------------------------------------- #
def read_cached_tex(arxiv_id):
    cpath = os.path.join(CACHE_DIR, arxiv_id.replace("/", "_") + ".tex")
    if not os.path.exists(cpath):
        return None
    with open(cpath, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    return None if data == "NO_TEX" else data


def process_paper(arxiv_id, model, use_llm):
    """Returns (arxiv_id, rows, mark_done).

    mark_done=False -> not cached yet, or transient LLM failure: leave for a
                       later run. mark_done=True -> attempted (rows may be []).
    """
    tex = read_cached_tex(arxiv_id)
    if tex is None:
        return arxiv_id, [], False  # source not cached / no LaTeX -> try later

    ack = locate_acknowledgements(tex)
    if not ack.strip():
        return arxiv_id, [], True   # no acknowledgements section found

    if use_llm:
        grants = llm_grants(arxiv_id, ack, model)
        if grants is None:
            return arxiv_id, [], False  # transient -> retry
        method = "llm"
    else:
        grants = regex_grants(ack)
        method = "regex"

    rows = []
    for g in grants:
        if not isinstance(g, dict):
            continue
        gid = (g.get("grant_id") or "").strip()
        if not gid:
            continue
        rows.append({
            "arxiv_id": arxiv_id,
            "funder": (g.get("funder") or "").strip(),
            "grant_id": gid,
            "method": method,
            "context": (g.get("context") or "").strip(),
        })
    return arxiv_id, rows, True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_id_list(limit):
    df = pd.read_csv(INPUT_CSV, usecols=["pdf_link"])
    ids, seen = [], set()
    for link in df["pdf_link"]:
        aid = aa.extract_arxiv_id(link)
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)
    return ids[:limit] if limit else ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Papers processed concurrently. 1 = sequential.")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N papers.")
    ap.add_argument("--regex-only", action="store_true",
                    help="Skip Ollama; use the offline regex heuristic instead.")
    ap.add_argument("--restart", action="store_true", help="Ignore saved progress.")
    args = ap.parse_args()

    use_llm = not args.regex_only
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_ids = build_id_list(args.limit)
    done = set() if args.restart else load_state()
    todo = [i for i in all_ids if i not in done]

    print(f"Mode: {'LLM (' + args.model + ')' if use_llm else 'regex-only'} | "
          f"batch-size: {args.batch_size}")
    print(f"Total: {len(all_ids)} | done: {len(all_ids) - len(todo)} | to process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    processed = skipped = failed = 0

    def _finish(aid, rows, mark_done):
        nonlocal processed, skipped, failed
        if not mark_done:
            # Distinguish "not cached" (silent skip) from a real failure isn't
            # tracked separately here; both are simply retried next run.
            skipped += 1
            return
        append_rows(rows)
        done.add(aid)
        processed += 1
        if processed % 50 == 0:
            save_state(done)
        if rows:
            print(f"  [{aid}] {len(rows)} grant(s)  ({processed}/{len(todo)})", flush=True)

    try:
        if args.batch_size <= 1:
            for aid in todo:
                if _stop.is_set():
                    break
                _finish(*process_paper(aid, args.model, use_llm))
        else:
            with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
                it = iter(todo)
                futures = {}
                for _ in range(args.batch_size):
                    try:
                        aid = next(it)
                    except StopIteration:
                        break
                    futures[pool.submit(process_paper, aid, args.model, use_llm)] = aid
                while futures:
                    for fut in as_completed(list(futures)):
                        del futures[fut]
                        _finish(*fut.result())
                        if not _stop.is_set():
                            try:
                                aid = next(it)
                                futures[pool.submit(process_paper, aid, args.model, use_llm)] = aid
                            except StopIteration:
                                pass
                        break
    finally:
        save_state(done)
        print(f"\nSaved: {processed} processed, {skipped} skipped (uncached/retry). "
              f"Output -> {OUTPUT_CSV}")
        if _stop.is_set():
            print("Stopped early; rerun the same command to resume.")


if __name__ == "__main__":
    main()
