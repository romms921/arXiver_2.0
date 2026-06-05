# arXiv LaTeX mining

Two scripts that mine arXiv paper LaTeX with a **local LLM (Ollama)**:

| Script | What it produces |
|--------|------------------|
| `author_affil.py` | one row per author: `author, affiliation, gender` |
| `funding_grants.py` | one row per grant: `funder, grant_id` (from the Acknowledgements/Funding section) |

Both are **stop/resume safe**, **cache-aware**, and run one paper at a time
(small GPU) or many in parallel (big GPU). `funding_grants.py` reuses the LaTeX
that `author_affil.py` caches, so **run `author_affil.py` first** (or at least
let it cache the source).

---

# 1. Author → Affiliation extractor

`author_affil.py` reads a CSV of arXiv papers, downloads each paper's LaTeX
source, and uses a local LLM to extract every author, their affiliation, and an
inferred gender into a new CSV.

---

## What it does (per paper)

1. Reads the `pdf_link` column from `datasets/arxiv_papers.csv`
   (e.g. `arxiv.org/pdf/2601.00044`) and extracts the arXiv id.
2. Downloads the LaTeX source from `arxiv.org/e-print/<id>` and caches it in
   `cache/latex/`.
3. Finds the author block (top *or* bottom of the file) and resolves
   macro-based affiliations like `\newcommand{\JHU}{\affiliation{...}}`.
4. **Pass 1** — sends the snippet to Ollama and gets back structured JSON of
   `author, affiliation, gender` (gender guessed from the first name).
5. **Pass 2** — for any author whose gender is still unknown, the model calls a
   `web_search` tool (DuckDuckGo, no API key) and infers gender from the results.
6. Appends rows to `datasets/author_affiliations.csv`.

Output columns: `arxiv_id, author, affiliation, gender, gender_method, gender_confidence`

---

## Prerequisites

1. **Python 3.9+**
2. **Python packages:**
   ```
   pip install -r requirements.txt
   ```
3. **Ollama** (the local LLM runtime) — install from <https://ollama.com>, then
   pull the model:
   ```
   ollama pull gemma4:e2b
   ```
   Make sure the Ollama server is running (`ollama serve`, or it starts
   automatically with the desktop app). The model needs ~7.2 GB of free RAM/VRAM.
4. An input CSV at `datasets/arxiv_papers.csv` with a `pdf_link` column.

---

## Usage

```bash
# Full run, one paper at a time (gentle on a small GPU). Resumable.
python author_affil.py

# Quick test on the first 5 papers
python author_affil.py --limit 5

# Batched for a big GPU: 8 papers in flight at once
python author_affil.py --batch-size 8

# Disable the gender web-search step (faster; gender only from names)
python author_affil.py --no-search

# Use a different model
python author_affil.py --model gemma4:e4b

# Ignore saved progress and start completely over
python author_affil.py --restart
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `gemma4:e2b` | Ollama model name |
| `--batch-size` | `1` | Papers processed concurrently (1 = sequential) |
| `--limit` | `0` (all) | Only process the first N papers |
| `--no-search` | off | Disable the `web_search` gender tool |
| `--restart` | off | Ignore saved progress and reprocess everything |

---

## Start / stop / resume

- Progress is tracked in `datasets/author_affil_state.json`; finished ids are
  skipped on the next run.
- Output is written incrementally to `datasets/author_affiliations.csv`.
- Press **Ctrl+C** once to stop gracefully — it finishes in-flight papers, saves
  progress, and exits. Re-run the same command to continue where it left off.
- Transient failures (network / Ollama errors) are **not** marked done, so they
  are retried automatically on the next run.

---

## Small GPU vs. big GPU

- **Small GPU / limited RAM:** keep `--batch-size 1` (default). Each paper takes
  a while; that's normal when the model is partly running on CPU.
- **Big GPU:** raise `--batch-size`, and let the Ollama server run them in
  parallel by setting `OLLAMA_NUM_PARALLEL` before starting it:
  ```bash
  # Linux/macOS
  OLLAMA_NUM_PARALLEL=8 ollama serve
  ```
  ```powershell
  # Windows (PowerShell)
  $env:OLLAMA_NUM_PARALLEL=8; ollama serve
  ```

---

## Configuration via environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:e2b` | Default model (overridden by `--model`) |

---

## Notes & limitations

- Many astrophysics papers don't put affiliations in the LaTeX at all, so some
  `affiliation` cells will be blank — that's a data-availability limit.
- Affiliation text may still contain LaTeX accent commands (e.g. `\"{u}`); strip
  them in a post-processing pass if you need clean text.
- Gender inference is heuristic (name + web snippets) and will not be perfect.
- Downloads are rate-limited to be polite to arXiv.

---

# 2. Funding / grant-number extractor

`funding_grants.py` finds the **Acknowledgements / Funding** text in each paper's
cached LaTeX and extracts the funding agencies and their grant/award numbers.

It **only reads the local cache** (`cache/latex/`) — it never downloads. Run
`author_affil.py` first so the source is cached; papers not yet cached are
skipped and picked up on a later run.

Output: `datasets/funding_grants.csv` with columns
`arxiv_id, funder, grant_id, method, context`.

## Two extraction modes

- **LLM (default):** sends the acknowledgements text to Ollama. Best at messy,
  varied grant formats and at attaching the right funder to each number
  (e.g. splitting "HST grants GO 17502 and AR 17572" into two rows).
- **`--regex-only`:** a fast, offline heuristic with no Ollama. Lower precision
  (misses the second of "X and Y" lists, leaves `funder` blank) but good for a
  quick first pass. The `context` column shows the surrounding text.

## Usage

```bash
python funding_grants.py                 # LLM extraction, resumable
python funding_grants.py --limit 10      # quick test on 10 papers
python funding_grants.py --regex-only    # no Ollama, fast heuristic pass
python funding_grants.py --batch-size 8  # batched for a big GPU
python funding_grants.py --restart       # ignore saved progress
```

State file: `datasets/funding_state.json`. Same Ctrl+C / resume behaviour as
`author_affil.py`.

## How it finds the section (edge cases handled)

- `\section*{Acknowledgements}` / `Acknowledgments` (UK & US spelling),
  also `\subsection`, `\subsubsection`, `\paragraph`, and `\chapter`.
- A separate `\section*{Funding}` or `\paragraph*{Funding:}` block (captured in
  addition to the acknowledgements).
- The `\begin{acknowledgements}...\end{...}` environment (A&A / AASTeX).
- The `\acknowledgments` command form (with or without braces) and MDPI
  `\funding{...}`.
- **Commented-out** markup (`%\begin{acknowledgments}`, `%%% ACKNOWLEDGEMENTS %%%`)
  is ignored — comments are stripped first.
- Papers with **no heading at all** (acknowledgements written as plain prose
  near the end): a conservative fallback harvests funding-cue paragraphs from the
  tail of the paper, stopping at the bibliography.

## Limitations

- Grant-number formats are extremely varied; the LLM is good but not perfect on a
  small model (it may occasionally mislabel a funder). `--regex-only` trades
  recall/precision for speed.
- If a paper genuinely has no acknowledgements, it produces no rows (still marked
  done so it isn't retried).
