"""
ollama_summarize_claude_markdown.py — Summarize Claude Code session logs via
local Ollama. Map-reduce summarization with section-aware chunking on the
Claude markdown format ("# Prompt N ..." / "# Response N ..." headers).

Approach:
  1. PARSE:   Extract Session ID and Title from the file preamble.
  2. MAP:     Split document into chunks, summarize each independently.
  3. REDUCE:  Hierarchically combine chunk summaries until they fit in one
              prompt, then generate the final per-session summary.
  4. WRITE:   One <basename>_summary.md per input file in --output-dir.

CRITICAL DESIGN CONSTRAINT:
  num_ctx is fixed for a single run (NEVER changes between Ollama calls
  within one invocation). Changing num_ctx between calls forces Ollama
  to restart its runner process, which can trigger delays and possibly
  undesirable system activity/behavior in the middle of processing files.

  The actual num_ctx value is selected ONCE in main() based on
  --parallel via PARALLEL_CONFIGS:
    -p 1: num_ctx=32768  (single slot, full context)
    -p 2: num_ctx=16384
    -p 3: num_ctx=8192
    -p 4: num_ctx=8192   (legacy default, fits 4x 8K slots in VRAM)
  num_ctx scales DOWN as parallel slots go up so total KV cache stays
  roughly constant in VRAM (parallel * num_ctx * ~150KB/token on
  gemma4-e4b GQA architecture).

  Two SIZE decisions are deliberately decoupled from num_ctx:
    - Map chunk size (CHUNK_TARGET_TOKENS) is fixed and small. It is
      bounded by how much OUTPUT a map call can emit before hitting the
      num_predict cap, not by available input context -- big chunks
      truncate the map summary and silently drop identifiers.
    - The single-pass threshold (SINGLE_PASS_MAX_TOKENS) IS a context-
      capacity decision: a doc small enough to fit one call skips
      map-reduce entirely (faster, and no reduce step to compress
      identifiers away).

Hardware reference: W6800 32 GB GPU (AMD), Windows or Linux host.
Default model:      gemma4:e4b-it-q8_0  (override with --model)

Usage:
    python ollama_summarize_claude_markdown.py path/to/folder
    python ollama_summarize_claude_markdown.py path/to/folder -m llama3.1:8b
    python ollama_summarize_claude_markdown.py path/to/folder -p 4
    python ollama_summarize_claude_markdown.py path/to/folder --resume
"""

# pylint: disable=missing-function-docstring,broad-exception-raised,broad-exception-caught

import argparse
import concurrent.futures
import datetime
import os
import re
import sys
import threading
import time
import traceback
import requests

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e4b-it-q8_0"

# Per-parallel configs. NUM_CTX, NUM_PREDICT_MAP, and NUM_PREDICT_REDUCE
# are selected from this table in main() based on --parallel. Once
# selected they are constant for the run -- num_ctx in particular never
# changes between Ollama calls (which would restart the runner). num_ctx
# scales down as slots go up so total KV cache stays roughly constant in
# VRAM.
#
# The generation caps scale WITH num_ctx, deliberately. A faithful
# summary that echoes every identifier hits the num_predict cap and
# truncates mid-output if the cap is small. High-context configs have
# abundant unused context (at -p 1 a map prompt is only ~5K of the 32K
# window), so we spend it on output headroom. Low-context configs
# (-p 3/4) can't afford that and will truncate identifier-dense output --
# an accepted small-VRAM tradeoff.
#
# Chunk size is intentionally NOT in this table -- see
# CHUNK_TARGET_TOKENS below for why map chunks stay small regardless of
# how large num_ctx is.
PARALLEL_CONFIGS = {
    1: {"num_ctx": 32768, "num_predict_map": 8192, "num_predict_reduce": 6144},
    2: {"num_ctx": 16384, "num_predict_map": 6144, "num_predict_reduce": 6144},
    3: {"num_ctx":  8192, "num_predict_map": 3072, "num_predict_reduce": 4096},
    4: {"num_ctx":  8192, "num_predict_map": 3072, "num_predict_reduce": 4096},
}

# Initial values; NUM_CTX / NUM_PREDICT_MAP / NUM_PREDICT_REDUCE are all
# rebound in main() once --parallel is known.
NUM_CTX = 8192
NUM_PREDICT_MAP = 3072

# Map chunk size, in tokens. Fixed and intentionally SMALL -- it does
# NOT scale with num_ctx. The binding constraint on a chunk is not how
# much input context is available, but how much OUTPUT the map call can
# produce before hitting NUM_PREDICT_MAP: a faithful summary that keeps
# every identifier in the chunk grows with the number of facts in that
# chunk. An oversized chunk makes the map response hit the num_predict
# cap and truncate silently, dropping every identifier past the cut.
# 4000 stays within MAX_INPUT_TOKENS_MAP for the smallest-context config
# (-p 3/4: 8192 - 400 - 3072 = 4720). Output completeness for a chunk
# this size is handled by NUM_PREDICT_MAP, which scales up on high-context
# configs (see PARALLEL_CONFIGS).
CHUNK_TARGET_TOKENS = 4000

NUM_BATCH = 2048
TEMPERATURE = 0.1

# Initial value; rebound in main() from PARALLEL_CONFIGS. High-context
# configs get a larger reduce cap so the final summary of an
# identifier-dense corpus isn't truncated.
NUM_PREDICT_REDUCE = 4096

# Prompt template overhead — the instructions themselves consume tokens.
# Conservative buffer that covers all four prompts:
#   MAP_PROMPT          ~280 tok (SPEAKER_BLOCK + capture bullets)
#   SINGLE_PASS_PROMPT  ~330 tok (SPEAKER_BLOCK + capture + 4 headings)
#   REDUCE_GROUP_PROMPT ~160 tok (SPEAKER_REMINDER + identifier rule)
#   REDUCE_FINAL_PROMPT ~220 tok (SPEAKER_REMINDER + 4 headings +
#                                 identifier rule + meta)
# SINGLE_PASS_PROMPT is still the binding case; 400 leaves ~70 tok of
# safety margin above it.
PROMPT_OVERHEAD_TOKENS = 400

# Phase timeouts (seconds). Sized for gemma at -p 4 on W6800; harmless if
# a faster model completes well before the timeout fires.
TIMEOUT_MAP_S = 120
TIMEOUT_REDUCE_S = 240
TIMEOUT_FINAL_S = 360

# Chars-per-token estimator for the gemma tokenizer on this kind of
# content. 3.2 is measured: an 80,639-char test doc reported ~24,400
# prompt tokens (= 3.3 chars/tok); 3.2 sits just under that so we slightly
# OVER-count tokens, keeping boundary math on the safe side. (The previous
# 3.8 under-counted by ~13% -- the unsafe direction -- and a stale comment
# wrongly called it conservative.)
#
# This is still a heuristic. estimate_tokens() drives every boundary
# decision (chunk fits, single-pass threshold, reduce-group packing) while
# the engine enforces EXACT counts, so the two can disagree -- most
# dangerously on the strict 8192-token -p3/4 configs, where an
# underestimate on dense markdown can overflow the window. A local
# tokenizer (or calibrating this constant from the prompt_eval_count
# Ollama returns on each call) would remove that gap; see project notes.
CHARS_PER_TOKEN = 3.2

# Max items per reduce group (even if more would fit by token budget).
# Caps the per-group context so the model can attend to each item.
REDUCE_GROUP_SIZE = 10

# Retry on transient HTTP / model failures.
MAX_RETRIES = 2
RETRY_DELAY_S = 5

# Hard cap on reduce rounds (safety net; usually 2-3 rounds is plenty).
MAX_REDUCE_ROUNDS = 10

# If a reduce round shrinks the corpus by less than this fraction, the
# model is no longer compressing — force the final pass instead of looping.
STALL_THRESHOLD = 0.05

# Token budget for input text, per phase. NUM_CTX minus prompt overhead
# minus the phase's generation cap. Initial values for module-load; all
# three are rebound in main() after --parallel selects the actual
# NUM_CTX / NUM_PREDICT_MAP.
MAX_INPUT_TOKENS_MAP = NUM_CTX - PROMPT_OVERHEAD_TOKENS - NUM_PREDICT_MAP
MAX_INPUT_TOKENS_REDUCE = NUM_CTX - PROMPT_OVERHEAD_TOKENS - NUM_PREDICT_REDUCE

# Single-pass threshold. A document whose whole text fits one call (with
# room for the reduce-side generation cap) skips map-reduce entirely and
# is summarized in a single pass -- faster, and lossless in the sense
# that there is no reduce step to compress identifiers away. This is a
# CONTEXT-capacity decision, independent of CHUNK_TARGET_TOKENS (an
# output-completeness decision). Equals MAX_INPUT_TOKENS_REDUCE; named
# separately so the single-pass call site reads clearly.
SINGLE_PASS_MAX_TOKENS = MAX_INPUT_TOKENS_REDUCE

# Set in main() from --model / --parallel.
MODEL = DEFAULT_MODEL
PARALLEL = 1

# Set in main() from --diag-dir. When non-None, every map chunk's output,
# every reduce group's output, and the final-reduce output is written to
# <DIAG_DIR>/<input-file-basename>/<phase-tagged-name>.txt. When None,
# all _diag_write() calls are no-ops -- zero behavior change.
DIAG_DIR: "str | None" = None

# Shared executors set in main().
#   _OLLAMA_POOL handles every call_ollama HTTP request. max_workers=PARALLEL
#   is the actual throttle that keeps concurrent requests under
#   OLLAMA_NUM_PARALLEL.
#   _FILE_POOL runs per-file worker threads, each of which executes the full
#   process_file pipeline top-to-bottom for one file. Those file workers
#   submit their ollama work to _OLLAMA_POOL and block on the futures —
#   they consume file-pool slots, not ollama-pool slots, so files can
#   overlap without violating the ollama concurrency cap.
_OLLAMA_POOL: "concurrent.futures.ThreadPoolExecutor | None" = None
_FILE_POOL: "concurrent.futures.ThreadPoolExecutor | None" = None


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
# Lock around tlog's print so concurrent file workers don't interleave
# mid-line. Output between consecutive tlog calls can still appear in
# arbitrary order across files, but each individual line stays intact.
_LOG_LOCK = threading.Lock()


def get_timestamp():
    """UTC timestamp with Z suffix for correlation with ollama logs."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tlog(msg, **kwargs):
    """Print with UTC timestamp prefix. Flushes per call so logs survive
    abrupt termination. Thread-safe via _LOG_LOCK.

    A leading '\\n' in msg is emitted BEFORE the timestamp so callers can
    add a visual blank line without producing an awkward 'timestamp on
    its own line, content below' rendering."""
    kwargs.setdefault("flush", True)
    blank = ""
    while msg.startswith("\n"):
        blank += "\n"
        msg = msg[1:]
    with _LOG_LOCK:
        print(f"{blank}{get_timestamp()} {msg}", **kwargs)


def format_duration(seconds: float) -> str:
    """Format seconds as 'Ns' / 'Mm Ss' / 'Hh Mm Ss'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def exc_summary(ex: BaseException) -> str:
    """Render an exception as type, message, and traceback for log output.
    Gives a human reading the log enough to identify the source line at a
    glance without re-running under a debugger."""
    return "".join(traceback.format_exception(ex))


def _diag_write(per_file_diag_dir: "str | None", name: str, content: str) -> None:
    """Write `content` to `{per_file_diag_dir}/{name}` for post-run
    inspection. No-op when diag is disabled (per_file_diag_dir is None),
    so callers can be unconditional. Creates the dir as needed. Called
    only from the main thread after futures resolve, so no locking is
    required."""
    if not per_file_diag_dir:
        return
    os.makedirs(per_file_diag_dir, exist_ok=True)
    path = os.path.join(per_file_diag_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


# ---------------------------------------------------------------------------
# SESSION METADATA
# ---------------------------------------------------------------------------
SESSION_ID_RE = re.compile(r"^Session ID:\s*(.+)", re.MULTILINE)
TITLE_RE = re.compile(r"^Title:\s*(.+)", re.MULTILINE)


def parse_session_metadata(text: str) -> tuple[str, str]:
    m_id = SESSION_ID_RE.search(text[:2000])
    m_title = TITLE_RE.search(text[:2000])
    session_id = m_id.group(1).strip() if m_id else "unknown"
    title = m_title.group(1).strip() if m_title else "unspecified"
    # Claude markdown files may carry the literal "<not-specified>" — that
    # renders as a blank HTML tag, so normalize before downstream writes.
    if not title or (title.startswith("<") and title.endswith(">")):
        title = "unspecified"
    return session_id, title


# ---------------------------------------------------------------------------
# CHUNKING — section-aware on Claude markdown headers
# ---------------------------------------------------------------------------
# Original Claude markdown headers look like:
#   "# Prompt 42 20260111-170800"   or   "# Response 7 20260111-170815"
SECTION_HEADER_RE = re.compile(
    r"^# (Prompt|Response) \d+ (\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})",
    re.MULTILINE,
)

# After normalization, headers look like "# Prompt (2026-01-11 17:08:00)".
SECTION_NORMALIZED_RE = re.compile(
    r"^# (?:Prompt|Response) \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\)",
    re.MULTILINE,
)


def normalize_headers(text: str) -> str:
    """Rewrite '# Prompt 42 20260111-170800' as '# Prompt (2026-01-11 17:08:00)'
    so the model sees a human-readable timestamp at zero instruction cost.
    Idempotent: re-applying changes nothing."""
    def _repl(m):
        speaker, y, mo, d, h, mi, s = m.groups()
        return f"# {speaker} ({y}-{mo}-{d} {h}:{mi}:{s})"
    return SECTION_HEADER_RE.sub(_repl, text)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def split_into_sections(text: str) -> list[str]:
    """Split a normalized document at Prompt/Response headers. Each returned
    string starts with its header. Content before the first header
    (preamble), if any, becomes section 0."""
    matches = list(SECTION_NORMALIZED_RE.finditer(text))
    if not matches:
        return [text]

    sections = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(preamble)

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())
    return sections


def _pack_by_separator(text: str, separator: str, max_chars: int) -> list[str]:
    """Greedily pack pieces (split by separator) into chunks <= max_chars.
    A piece individually larger than max_chars becomes its own chunk;
    caller refines further if needed."""
    pieces = text.split(separator)
    chunks = []
    current = []
    current_len = 0
    for piece in pieces:
        added = len(piece) + (len(separator) if current else 0)
        if current and current_len + added > max_chars:
            chunks.append(separator.join(current))
            current = [piece]
            current_len = len(piece)
        else:
            current.append(piece)
            current_len += added
    if current:
        chunks.append(separator.join(current))
    return chunks


def split_oversized_section(text: str, max_chars: int) -> list[str]:
    """Sub-split a section that exceeds max_chars. Cascades:
    paragraph (\\n\\n) -> sentence (". ") -> raw character cut. The raw
    cut is the guaranteed bound; earlier levels just attempt, and any
    still-over-budget output gets refined by the next level."""
    if len(text) <= max_chars:
        return [text]

    chunks = _pack_by_separator(text, "\n\n", max_chars)

    refined = []
    for c in chunks:
        if len(c) <= max_chars:
            refined.append(c)
        else:
            refined.extend(_pack_by_separator(c, ". ", max_chars))

    final = []
    for c in refined:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i : i + max_chars])
    return final


def chunk_sections(sections: list[str]) -> list[str]:
    """Pack adjacent sections into chunks <= CHUNK_TARGET_TOKENS. A single
    section exceeding the target is emitted on its own; if it would
    also exceed the model's input budget, sub-split first."""
    target_chars = int(CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN)
    max_input_chars = int(MAX_INPUT_TOKENS_MAP * CHARS_PER_TOKEN)
    chunks = []
    current_parts = []
    current_len = 0

    for section in sections:
        sec_len = len(section)
        if sec_len > target_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            if sec_len > max_input_chars:
                sub = split_oversized_section(section, max_input_chars)
                chunks.extend(sub)
                tlog(
                    f"    WARNING: oversized section "
                    f"(~{estimate_tokens(section)} tok > {MAX_INPUT_TOKENS_MAP} budget) "
                    f"sub-split into {len(sub)} chunks to avoid silent Ollama truncation."
                )
            else:
                chunks.append(section)
            continue
        if current_len + sec_len > target_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(section)
        current_len += sec_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


def chunk_text(text: str) -> list[str]:
    """Primary chunking entry point. If the document has Prompt/Response
    headers (Claude markdown format, already normalized), split
    section-aware; else fall back to paragraph-based splitting."""
    if SECTION_NORMALIZED_RE.search(text):
        sections = split_into_sections(text)
        chunks = chunk_sections(sections)
        tlog(
            f"  (Section-aware chunking: {len(sections)} sections "
            f"-> {len(chunks)} chunks)"
        )
        return chunks

    tlog("  (No section headers found -- using paragraph-based chunking)")
    target_chars = int(CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN)
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > target_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                if current_len + len(sent) > target_chars and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(sent)
                current_len += len(sent)
        elif current_len + para_len > target_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


# ---------------------------------------------------------------------------
# BUDGET-AWARE GROUPING (for the reduce phase)
# ---------------------------------------------------------------------------
def build_groups_by_token_budget(
    items: list[str],
    max_tokens: "int | None" = None,
    max_per_group: int = REDUCE_GROUP_SIZE,
) -> list[list[str]]:
    """Pack items into groups where each group's combined tokens stay
    under max_tokens, and group size stays under max_per_group. An item
    that individually exceeds max_tokens gets its own (over-budget)
    group; the model will silently truncate it — preferable to changing
    num_ctx mid-run.

    max_tokens defaults to the run's MAX_INPUT_TOKENS_REDUCE, resolved at
    CALL time. It must NOT be a default-argument expression: that would
    bind the module value as it stood at import (before main() rebinds it
    from PARALLEL_CONFIGS), so grouping would silently use the wrong
    budget on every config but the import-time default."""
    if max_tokens is None:
        max_tokens = MAX_INPUT_TOKENS_REDUCE
    groups = []
    current_group = []
    current_tokens = 0

    for item in items:
        item_tokens = estimate_tokens(item)

        if item_tokens > max_tokens:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
            groups.append([item])
            tlog(
                f"    WARNING: single item exceeds budget "
                f"(~{item_tokens} tok > {max_tokens} max). It will be truncated."
            )
            continue

        if (
            current_tokens + item_tokens > max_tokens
            or len(current_group) >= max_per_group
        ) and current_group:
            groups.append(current_group)
            current_group = []
            current_tokens = 0

        current_group.append(item)
        current_tokens += item_tokens

    if current_group:
        groups.append(current_group)
    return groups


# ---------------------------------------------------------------------------
# OLLAMA CALL WITH RETRY
# ---------------------------------------------------------------------------
# Some models (gemma in particular) inconsistently echo the prompt's trailing
# cue heading at the start of their response (e.g. "### FINAL SUMMARY:"
# followed by the actual summary). Strip that cue if it appears as the very
# first non-whitespace content; do NOT strip headings the model uses inside
# its body, which are typically wanted.
RESPONSE_CUE_PREFIXES = (
    "### FINAL SUMMARY:",
    "### CONSOLIDATED SUMMARY:",
    "### SUMMARY:",
)


def strip_leading_cue(response: str) -> str:
    s = response.lstrip()
    for cue in RESPONSE_CUE_PREFIXES:
        if s.startswith(cue):
            return s[len(cue):].lstrip()
    return response


def call_ollama(
    prompt: str,
    label: str = "",
    timeout: int = TIMEOUT_MAP_S,
    max_retries: int = MAX_RETRIES,
    num_predict: int = NUM_PREDICT_MAP,
    raw_diag_path: "str | None" = None,
) -> str:
    """POST to Ollama with retry. Uses the module-level NUM_CTX (set by
    main() from PARALLEL_CONFIGS; never changes mid-run). Uses the
    module-level MODEL set from --model.

    When `raw_diag_path` is provided, the model's PRE-strip response is
    written there before the leading-cue strip is applied. Useful for
    diagnosing degenerate generation (e.g. model produced 4096 tokens
    but the stripped result is empty -- the raw file will show whether
    the model emitted just whitespace, just the cue heading, or
    something else)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_batch": NUM_BATCH,
            "num_predict": num_predict,
            "temperature": TEMPERATURE,
        },
    }

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0

            tok_gen = data.get("eval_count", 0)
            tok_prompt = data.get("prompt_eval_count", 0)
            tlog(
                f"  [{label}] prompt={tok_prompt} tok, generated={tok_gen} tok, "
                f"time={elapsed:.1f}s  (timeout={timeout}s)"
            )
            # Cap-hit detection: if eval_count reached num_predict, the
            # model was cut off mid-generation -- silent output truncation.
            # Surface it loudly so the issue is visible.
            if tok_gen >= num_predict:
                tlog(
                    f"    WARNING: [{label}] hit num_predict cap "
                    f"({tok_gen} >= {num_predict} tok). Output likely truncated."
                )

            raw_response = data.get("response", "")
            if raw_diag_path:
                os.makedirs(os.path.dirname(raw_diag_path), exist_ok=True)
                with open(raw_diag_path, "w", encoding="utf-8") as f:
                    f.write(raw_response)

            # Brief raw-response shape log so degenerate output is
            # detectable from the log alone (e.g. "raw=4 chars" after
            # eval_count=4096 says the model emitted only the cue).
            tlog(
                f"    [{label}] raw response: {len(raw_response)} chars, "
                f"starts with: {raw_response[:60]!r}"
            )
            return strip_leading_cue(raw_response)

        except Exception as e:
            elapsed = time.time() - t0
            if attempt < max_retries:
                tlog(
                    f"  [{label}] attempt {attempt} failed after {elapsed:.0f}s "
                    f"(timeout={timeout}s): {e}"
                )
                tlog(f"  Retrying in {RETRY_DELAY_S}s...")
                time.sleep(RETRY_DELAY_S)
            else:
                tlog(
                    f"  [{label}] FAILED after {attempt} attempts "
                    f"(timeout={timeout}s):\n{exc_summary(e)}",
                    file=sys.stderr,
                )
                return ""


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
# Shared speaker-identification block. Used verbatim in MAP_PROMPT and
# SINGLE_PASS_PROMPT (which both see raw transcript and must attribute
# speakers from header patterns). The reduce prompts include a much
# shorter reminder rather than the full block, since by the reduce
# stage the upstream summaries already use "Developer" / "LLMDev"
# terminology.
SPEAKER_BLOCK = """\
Speaker identification: Sections begin with headers like
"# Prompt (2026-01-11 17:08:00)" or "# Response (2026-01-11 17:08:15)".
Prompt sections are spoken by the Developer; Response sections are spoken
by the LLMDev. Exception: if a Prompt section contains "user's action",
"user needs", or "user did a great job", it refers to the LLMDev.
Prompt sections containing the word "ultrathink" are spoken by Developer.
Prompt sections beginning with "This session is being continued from a
previous conversation" are spoken by LLMDev, except for the subsection
beginning with "All User Messages:" which is spoken by Developer."""


# Short reminder used in reduce prompts so attribution survives chains of
# consolidation even if upstream summaries get terse.
SPEAKER_REMINDER = (
    'In the summaries below, "Developer" refers to the human user and '
    '"LLMDev" refers to Claude Code. Preserve this attribution in your '
    'output.'
)


# Shared capture guidance: what each section/transcript summary must
# retain. Used verbatim by MAP_PROMPT and SINGLE_PASS_PROMPT so the two
# paths can't drift apart. The last bullet asks the model to keep
# concrete specifics (names, errors, version/ticket/CVE numbers) verbatim
# rather than generalizing them away -- the main thing that makes a
# session summary useful to read back later. Phrased for real, mostly
# conversational sessions: preserve specifics that ARE there; never
# invent or pad with placeholders.
CAPTURE_BLOCK = """\
- What the Developer asked for or what the LLMDev did
- Key decisions, files changed, commands run, or errors encountered
- Any blockers or open questions
- Timestamps from section headers (YYYY-MM-DD HH:MM:SS) where they
  anchor important events
- Concrete specifics, kept verbatim when they appear: file and function
  names, commands, error messages or codes, library/product/tool names,
  and any ticket, issue, or CVE numbers. Don't generalize them into vague
  terms ("a config file", "some error") or invent ones that aren't there."""


MAP_PROMPT = f"""\
You are summarizing one section of a Claude Code session transcript.

{SPEAKER_BLOCK}

### INSTRUCTIONS:
Summarize this section. Capture:
{CAPTURE_BLOCK}

Use concise bullets. Preserve specifics (file names, function names,
error messages). Omit small talk and routine acknowledgements.

### SECTION TEXT:
{{chunk}}

### SUMMARY:"""


REDUCE_GROUP_PROMPT = f"""\
Below are summaries from consecutive sections of a Claude Code session
transcript, in chronological order.

{SPEAKER_REMINDER}

### TASK:
Consolidate these into a single combined summary that is shorter than
the input. Compress by cutting prose -- merge restated observations and
drop filler and routine acknowledgements.

Preserve concrete specifics verbatim -- file and function names,
commands, error messages or codes, library/product names, and any
ticket, issue, or CVE numbers. Merge only points that genuinely restate
each other; keep distinct points distinct. Don't generalize a specific
into a vague term to save space.

Use concise bullets.

### SECTION SUMMARIES:
{{items_text}}

### CONSOLIDATED SUMMARY:"""


REDUCE_FINAL_PROMPT = f"""\
Below are consolidated summaries from a Claude Code session.
Source file: {{filename}}
Session ID: {{session_id}}
Session title: {{title}}

{SPEAKER_REMINDER}

### TASK:
Produce a final session summary.

Organise under these headings (omit any heading with no content):

1. **Goal** — what the Developer was trying to accomplish
2. **Key Actions** — what the LLMDev did (files changed, commands run, decisions made)
3. **Outcomes** — what was accomplished
4. **Open Items** — unresolved questions, deferred work, blockers

Use concise bullets. Preserve concrete specifics from the summaries
below verbatim -- file and function names, commands, error messages or
codes, library/product names, and any ticket, issue, or CVE numbers --
rather than generalizing them away. Keep the summary readable and
narrative; don't reduce it to a bare list of names.

### SUMMARIES:
{{items_text}}

### FINAL SUMMARY:"""


# Single-pass prompt: used when a document is small enough that the
# whole transcript fits in one Ollama call without chunking. The map/
# reduce path doesn't apply, so this prompt has to bundle everything --
# the speaker block (raw transcript input), the MAP capture guidance
# (what to extract from each section), AND the REDUCE_FINAL output
# structure (the four headings). Without this, single-pass would have
# to fall back to REDUCE_FINAL_PROMPT, which lacks the speaker block
# and the explicit capture bullets and would underperform on real
# transcripts containing ultrathink/continued-session patterns.
SINGLE_PASS_PROMPT = f"""\
Below is a complete Claude Code session transcript. The sections appear
in chronological order.
Source file: {{filename}}
Session ID: {{session_id}}
Session title: {{title}}

{SPEAKER_BLOCK}

### TASK:
Read the full transcript and produce a final session summary. Your
summary MUST be substantially shorter than the transcript.

When summarizing, capture:
{CAPTURE_BLOCK}

Preserve all distinct events and decisions. Merge duplicate observations
and drop redundancy. Use concise bullets. Preserve specifics (file names,
function names, error messages, timestamps). Omit small talk and routine
acknowledgements.

Organise under these headings (omit any heading with no content):

1. **Goal** — what the Developer was trying to accomplish
2. **Key Actions** — what the LLMDev did (files changed, commands run, decisions made)
3. **Outcomes** — what was accomplished
4. **Open Items** — unresolved questions, deferred work, blockers

### TRANSCRIPT:
{{text}}

### FINAL SUMMARY:"""


# ---------------------------------------------------------------------------
# MAP PHASE
# ---------------------------------------------------------------------------
def _map_one_chunk(i: int, chunk: str, total: int) -> tuple[int, str]:
    """Summarize one chunk. Runs in the main thread (sequential) or in a
    worker thread (parallel). The ENTER/EXIT logs print the OS thread id
    so you can correlate which thread handled which chunk when output
    from multiple workers interleaves."""
    try:
        tlog(
            f"_map_one_chunk ENTER: ident={threading.get_ident()} "
            f"native_tid={threading.get_native_id()}"
        )
        prompt = MAP_PROMPT.format(chunk=chunk)
        est = estimate_tokens(prompt)
        tlog(f"  Chunk {i+1}/{total}  (~{est} tokens in prompt)")
        result = call_ollama(
            prompt,
            label=f"map {i+1}/{total}",
            timeout=TIMEOUT_MAP_S,
            num_predict=NUM_PREDICT_MAP,
        )
        return i, result
    finally:
        tlog(
            f"_map_one_chunk EXIT: ident={threading.get_ident()} "
            f"native_tid={threading.get_native_id()}"
        )


def map_phase(chunks: list[str], per_file_diag_dir: "str | None" = None) -> list[str]:
    total = len(chunks)

    # Submit every chunk to the shared ollama pool. max_workers=PARALLEL on
    # that pool is the throttle; submit() itself never blocks. Multiple
    # file workers can submit concurrently — their chunks all queue here
    # and execute up to PARALLEL at a time across the whole run.
    futures = {
        _OLLAMA_POOL.submit(_map_one_chunk, i, chunk, total): i
        for i, chunk in enumerate(chunks)
    }
    indexed = {}
    for future in concurrent.futures.as_completed(futures):
        idx = futures[future]
        result = future.result()[1]
        indexed[idx] = result
        # Diag: write each chunk's raw map output (before "[Section N]"
        # wrapping) as soon as its future resolves -- not after the
        # whole map phase completes. Lets a human watching the diag
        # dir see partial results in real time and abort early if
        # they look wrong.
        _diag_write(
            per_file_diag_dir,
            f"map-result-chunk{idx+1:03d}.txt",
            result,
        )
    ordered_results = [(i, indexed[i]) for i in range(total)]

    summaries = []
    skipped = 0
    for i, result in ordered_results:
        if not result:
            tlog(f"  WARNING: Chunk {i+1} produced no output, skipping.")
            skipped += 1
            continue
        summaries.append(f"[Section {i+1}]\n{result}")

    if skipped:
        tlog(f"  ({skipped}/{total} chunks failed)")
    return summaries


# ---------------------------------------------------------------------------
# REDUCE PHASE (hierarchical)
# ---------------------------------------------------------------------------
def _reduce_one_group(
    round_num: int, gi: int, group: list[str], total_groups: int
) -> str:
    """Consolidate one group of summaries. ENTER/EXIT logs as in
    _map_one_chunk so parallel workers are easy to follow."""
    try:
        tlog(
            f"_reduce_one_group ENTER: ident={threading.get_ident()} "
            f"native_tid={threading.get_native_id()}"
        )
        group_text = "\n\n".join(group)
        prompt = REDUCE_GROUP_PROMPT.format(items_text=group_text)
        return call_ollama(
            prompt,
            label=f"reduce r{round_num} g{gi+1}/{total_groups}",
            timeout=TIMEOUT_REDUCE_S,
            num_predict=NUM_PREDICT_REDUCE,
        )
    finally:
        tlog(
            f"_reduce_one_group EXIT: ident={threading.get_ident()} "
            f"native_tid={threading.get_native_id()}"
        )


def _forced_final(items: list[str], final_kwargs: dict, reason: str) -> str:
    """Produce the final pass with whatever items fit in budget. Drops
    overflow with a warning — simpler than appending overflow verbatim,
    at the cost of some data loss in pathological inputs."""
    kept = []
    running = 0
    for item in items:
        t = estimate_tokens(item)
        if running + t > MAX_INPUT_TOKENS_REDUCE and kept:
            break
        kept.append(item)
        running += t
    if len(kept) < len(items):
        tlog(
            f"  WARNING: dropping {len(items) - len(kept)} item(s) that "
            f"don't fit in budget."
        )
    prompt = REDUCE_FINAL_PROMPT.format(items_text="\n\n".join(kept), **final_kwargs)
    return call_ollama(
        prompt,
        label=f"forced final ({reason})",
        timeout=TIMEOUT_FINAL_S,
        num_predict=NUM_PREDICT_REDUCE,
    )


def reduce_phase(
    summaries: list[str],
    filename: str,
    session_id: str,
    title: str,
    per_file_diag_dir: "str | None" = None,
) -> str:
    """Hierarchically reduce per-chunk summaries into one digest. Every
    call uses the run's fixed NUM_CTX. Groups are built by token budget.

    Termination:
      - Everything fits in one prompt -> final pass and return.
      - Round failed to compress meaningfully -> forced final.
      - MAX_REDUCE_ROUNDS exceeded -> forced final.

    Diag: when `per_file_diag_dir` is set, writes one file per group per
    round (including single-item pass-throughs, for a complete picture)
    and writes the terminal output as `reduce-combined.txt`."""
    items = list(summaries)
    round_num = 0
    prev_tokens = None
    final_kwargs = {"filename": filename, "session_id": session_id, "title": title}

    while True:
        combined_text = "\n\n".join(items)
        combined_tokens = estimate_tokens(combined_text)

        tlog(
            f"\n  reduce round {round_num}: {len(items)} items, "
            f"~{combined_tokens} tokens"
        )

        if combined_tokens <= MAX_INPUT_TOKENS_REDUCE:
            tlog("  Fits in one pass -- generating final summary.")
            prompt = REDUCE_FINAL_PROMPT.format(items_text=combined_text, **final_kwargs)
            result = call_ollama(
                prompt,
                label="final reduce",
                timeout=TIMEOUT_FINAL_S,
                num_predict=NUM_PREDICT_REDUCE,
            )
            _diag_write(per_file_diag_dir, "reduce-combined.txt", result)
            return result

        # Stall detection: if the model isn't compressing, force final
        # rather than burning GPU cycles on identical rounds.
        if prev_tokens is not None:
            reduction = (prev_tokens - combined_tokens) / prev_tokens
            if reduction < STALL_THRESHOLD:
                tlog(
                    f"  STALL: tokens shrank only {reduction:.1%} "
                    f"({prev_tokens} -> {combined_tokens}). Forcing final."
                )
                result = _forced_final(items, final_kwargs, "stall")
                _diag_write(per_file_diag_dir, "reduce-combined.txt", result)
                return result

        prev_tokens = combined_tokens

        groups = build_groups_by_token_budget(items)
        tlog(
            f"  Too large for one pass. Split into {len(groups)} groups "
            f"(budget: {MAX_INPUT_TOKENS_REDUCE} tok/group)."
        )

        # Single-item groups pass through unchanged; only multi-item
        # groups need a model call. Diag for pass-throughs is eagerly
        # written here -- their content is known the moment groups are
        # split, no need to wait for futures.
        group_meta = []      # (gi, group, is_single)
        model_groups = []    # (gi, group)
        for gi, group in enumerate(groups):
            group_tokens = estimate_tokens("\n\n".join(group))
            tlog(
                f"    Group {gi+1}/{len(groups)}  ({len(group)} items, "
                f"~{group_tokens} tokens)"
            )
            if len(group) == 1:
                tlog("    Skipping single-item group (pass-through).")
                group_meta.append((gi, group, True))
                _diag_write(
                    per_file_diag_dir,
                    f"reduce-round{round_num:02d}-grp{gi+1:03d}.txt",
                    group[0],
                )
            else:
                group_meta.append((gi, group, False))
                model_groups.append((gi, group))

        # Dispatch all multi-item groups to the shared ollama pool.
        # Each group's diag file is written as soon as its future
        # resolves -- not after the whole round completes -- so a
        # human watching the diag dir sees progress in real time.
        model_results = {}
        futures = {
            _OLLAMA_POOL.submit(
                _reduce_one_group, round_num, gi, group, len(groups)
            ): gi
            for gi, group in model_groups
        }
        for future in concurrent.futures.as_completed(futures):
            gi = futures[future]
            result = future.result()
            model_results[gi] = result
            _diag_write(
                per_file_diag_dir,
                f"reduce-round{round_num:02d}-grp{gi+1:03d}.txt",
                result,
            )

        # Reassemble results in original order.
        new_items = []
        for gi, group, is_single in group_meta:
            if is_single:
                new_items.append(group[0])
                continue
            result = model_results.get(gi, "")
            if result:
                new_items.append(f"[Group {gi+1}]\n{result}")
            else:
                tlog(f"    WARNING: Group {gi+1} failed; keeping items verbatim.")
                new_items.extend(group)

        items = new_items
        round_num += 1

        if round_num > MAX_REDUCE_ROUNDS:
            tlog(
                f"  WARNING: Exceeded {MAX_REDUCE_ROUNDS} reduce rounds. "
                f"Forcing final pass."
            )
            result = _forced_final(items, final_kwargs, "max-rounds")
            _diag_write(per_file_diag_dir, "reduce-combined.txt", result)
            return result


# ---------------------------------------------------------------------------
# PROCESS ONE FILE
# ---------------------------------------------------------------------------
def process_file(filepath: str) -> tuple[str, str, str, str, str]:
    """Returns (digest, session_id, title, filename, method) where method
    is "single-pass" or "map-reduce" -- the path actually taken, for an
    accurate per-summary footer."""
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath) / 1024
    tlog(f"\n{'='*60}")
    tlog(f"Processing: {filename}  ({filesize:.0f} KB)")
    tlog(f"{'='*60}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    session_id, title = parse_session_metadata(text)
    tlog(f"Session ID: {session_id}")
    tlog(f"Title: {title}")

    # Per-file diag subdir: <DIAG_DIR>/<filename-without-extension>/.
    # Created lazily by _diag_write on first write; we also create it
    # eagerly here so the log line below points at a real path.
    per_file_diag_dir = None
    if DIAG_DIR:
        basename = os.path.splitext(filename)[0]
        per_file_diag_dir = os.path.join(DIAG_DIR, basename)
        os.makedirs(per_file_diag_dir, exist_ok=True)
        tlog(f"Diag dir: {per_file_diag_dir}")

    # Normalize Claude markdown headers once, up front. No-op if the
    # document doesn't use the Claude format.
    text = normalize_headers(text)

    total_tokens_est = estimate_tokens(text)
    tlog(f"Estimated tokens: ~{total_tokens_est}")

    # Documents that fit one call: skip map-reduce, go straight to a
    # single pass. Threshold is SINGLE_PASS_MAX_TOKENS (context capacity),
    # NOT the map chunk size -- a doc can be far larger than one chunk yet
    # still fit a single call, and single-pass beats map-reduce on
    # identifier preservation because there is no lossy reduce step.
    # SINGLE_PASS_PROMPT bundles the speaker block + capture bullets +
    # 4-heading output structure so this path gives the same prompt
    # content quality as the orchestrator's naive prompt -- no missing
    # LLMDev definition, no missing capture guidance.
    if total_tokens_est <= SINGLE_PASS_MAX_TOKENS:
        tlog("Document fits one call -- single-pass summarization.")
        prompt = SINGLE_PASS_PROMPT.format(
            filename=filename, session_id=session_id, title=title, text=text
        )
        raw_diag_path = (
            os.path.join(per_file_diag_dir, "map-result-singlepass-raw.txt")
            if per_file_diag_dir else None
        )
        digest = call_ollama(
            prompt,
            label="single-pass",
            timeout=TIMEOUT_FINAL_S,
            num_predict=NUM_PREDICT_REDUCE,
            raw_diag_path=raw_diag_path,
        )
        # Single-pass acts as both "map" and "final reduce" for small docs;
        # record the output under the map-side filename to keep the diag
        # naming intuitive when comparing chunked vs single-pass runs.
        _diag_write(per_file_diag_dir, "map-result-singlepass.txt", digest)
        if digest and digest.strip():
            return digest, session_id, title, filename, "single-pass"
        # Single-pass produced empty/degenerate output. Rather than give
        # up, fall through to the chunked map-reduce path: smaller
        # per-call prompts using MAP_PROMPT have been more reliable than
        # SINGLE_PASS_PROMPT at the upper end of the context window. The
        # raw response is preserved in map-result-singlepass-raw.txt
        # (when -Diag was passed) for post-hoc investigation.
        tlog(
            "WARNING: single-pass returned empty/degenerate output. "
            "Falling back to chunked map-reduce."
        )

    chunks = chunk_text(text)
    tlog(f"Split into {len(chunks)} chunks\n")
    summaries = map_phase(chunks, per_file_diag_dir=per_file_diag_dir)

    if not summaries:
        tlog(f"\n  No usable summaries for {filename}.")
        return "", session_id, title, filename, "map-reduce"

    tlog(f"\nMap phase complete: {len(summaries)} chunk summaries.")
    digest = reduce_phase(
        summaries, filename, session_id, title,
        per_file_diag_dir=per_file_diag_dir,
    )
    return digest, session_id, title, filename, "map-reduce"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def summary_path_for(output_dir: str, source_filename: str) -> str:
    base = os.path.splitext(source_filename)[0]
    return os.path.join(output_dir, f"{base}_summary.md")


CONSOLIDATED_FILENAME = "consolidated_summary.md"


def write_consolidated_summary(output_dir: str) -> str | None:
    """Mechanically concatenate every per-session *_summary.md in
    output_dir into one consolidated_summary.md. No model call.
    Includes a TOC built from each file's '**Title:**' metadata line,
    with '---' separators between session blocks.
    Returns the path written, or None if no per-session summaries exist."""
    summary_files = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith("_summary.md") and f != CONSOLIDATED_FILENAME
    )
    if not summary_files:
        return None

    # Build TOC entries with explicit anchors. Using <a id="..."> rather
    # than relying on auto-generated heading anchors keeps links working
    # across renderers (VSCode preview, GitHub, Pandoc) since each
    # implements heading-slug rules slightly differently.
    toc_entries = []
    for i, path in enumerate(summary_files, 1):
        fname = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(500)
        m_title = re.search(r"^\*\*Title:\*\*\s*(.+)", head, re.MULTILINE)
        title = m_title.group(1).strip() if m_title else "unspecified"
        # Defensive: catch any angle-bracket placeholders (e.g. older
        # per-session files written before parse_session_metadata
        # normalized "<not-specified>") so the TOC doesn't render blank.
        if not title or (title.startswith("<") and title.endswith(">")):
            title = "unspecified"
        toc_entries.append((f"session-{i}", fname, title))

    consolidated_path = os.path.join(output_dir, CONSOLIDATED_FILENAME)
    today = datetime.date.today().isoformat()
    with open(consolidated_path, "w", encoding="utf-8") as out:
        out.write("# Consolidated Session Summaries\n\n")
        out.write(f"_Sessions: {len(summary_files)} • Assembled: {today}_\n\n")
        out.write("## Table of Contents\n\n")
        for i, (anchor, fname, title) in enumerate(toc_entries, 1):
            out.write(f"{i}. [{title}](#{anchor}) — `{fname}`\n")
        out.write("\n---\n\n")
        for i, (path, (anchor, _fname, _title)) in enumerate(
            zip(summary_files, toc_entries)
        ):
            if i > 0:
                out.write("\n\n---\n\n")
            out.write(f'<a id="{anchor}"></a>\n\n')
            with open(path, "r", encoding="utf-8") as f:
                out.write(f.read())

    return consolidated_path


def get_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Claude Code session logs via local Ollama "
        "(map-reduce, section-aware chunking)."
    )
    parser.add_argument(
        "path",
        help="Path to a Claude markdown file or a folder of .md/.txt files.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./summaries",
        help="Directory to write <basename>_summary.md files into "
        "(default: ./summaries).",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int, default=1,
        help="Number of concurrent Ollama requests (default: 1). "
        "Requires Ollama started with OLLAMA_NUM_PARALLEL >= N. "
        "Each slot uses ~1 GB VRAM for KV cache at num_ctx=8192.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any source file whose <basename>_summary.md already "
        "exists in --output-dir.",
    )
    parser.add_argument(
        "--diag-dir",
        default=None,
        help="If set, write per-chunk map outputs, per-group reduce "
        "outputs, and the final reduce output for each processed file "
        "to <diag-dir>/<input-basename>/. Useful for inspecting where "
        "specifics (file names, identifier tags, etc.) are dropped in "
        "the pipeline. No effect when omitted.",
    )
    return parser


def process_one_file(filepath: str, output_dir: str) -> str:
    """Run the full process_file pipeline for one file, write the
    per-session summary, log per-file timing. Returns 'processed' on
    success or 'failed' if the model produced no usable digest.

    Designed to run inside a _FILE_POOL worker thread. Its ollama work is
    submitted to _OLLAMA_POOL via map_phase / reduce_phase / call_ollama;
    this thread blocks waiting on those futures, which is fine because
    blocking a file worker doesn't consume an ollama slot."""
    filename = os.path.basename(filepath)
    out_path = summary_path_for(output_dir, filename)

    t0 = time.time()
    digest, session_id, title, fname, method = process_file(filepath)
    elapsed = time.time() - t0

    if not digest or not digest.strip():
        tlog(f"\n  No summary produced for {fname}.")
        tlog(f"  Time: {elapsed:.0f}s")
        return "failed"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Summary: {fname}\n")
        f.write(f"**Session ID:** {session_id}\n")
        f.write(f"**Title:** {title}\n\n")
        f.write(digest)
        f.write(
            f"\n\n---\n_Generated in {elapsed:.0f}s using model={MODEL}, "
            f"{method} summarization._\n"
        )

    tlog(f"\n  Summary written to: {out_path}")
    tlog(f"  Time: {elapsed:.0f}s")
    return "processed"


def main():
    # pylint: disable=global-statement
    global MODEL, PARALLEL, DIAG_DIR, _OLLAMA_POOL, _FILE_POOL
    global NUM_CTX, NUM_PREDICT_MAP, NUM_PREDICT_REDUCE
    global MAX_INPUT_TOKENS_MAP, MAX_INPUT_TOKENS_REDUCE, SINGLE_PASS_MAX_TOKENS

    # Force UTF-8 on stdout/stderr. Default Windows console encoding is
    # cp1252 (charmap), which can't represent characters like √ (U+221A)
    # found in some session titles -- writing those via print() would raise
    # UnicodeEncodeError. errors='replace' is belt-and-suspenders for any
    # console that genuinely lacks UTF-8 support (we'd see '?' placeholders
    # rather than a crash).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass  # Non-TextIOWrapper streams (rare; e.g. test capture)

    run_t0 = time.time()
    args = get_argparse().parse_args()
    MODEL = args.model
    PARALLEL = args.parallel
    if args.diag_dir:
        DIAG_DIR = os.path.abspath(args.diag_dir)
        os.makedirs(DIAG_DIR, exist_ok=True)
        tlog(f"Diag dir (top-level): {DIAG_DIR}")

    # Bind num_ctx / num_predict_map from PARALLEL_CONFIGS. Falls back to
    # the -p=4 config if an unrecognized parallel value is passed (e.g.
    # -p=8 on a much bigger GPU); user gets a warning.
    cfg = PARALLEL_CONFIGS.get(PARALLEL)
    if cfg is None:
        cfg = PARALLEL_CONFIGS[4]
        tlog(
            f"WARNING: --parallel={PARALLEL} not in PARALLEL_CONFIGS; "
            f"using -p=4 config (num_ctx={cfg['num_ctx']}). Edit "
            f"PARALLEL_CONFIGS to add a tuned config for this slot count."
        )
    NUM_CTX = cfg["num_ctx"]
    NUM_PREDICT_MAP = cfg["num_predict_map"]
    NUM_PREDICT_REDUCE = cfg["num_predict_reduce"]
    MAX_INPUT_TOKENS_MAP = NUM_CTX - PROMPT_OVERHEAD_TOKENS - NUM_PREDICT_MAP
    MAX_INPUT_TOKENS_REDUCE = NUM_CTX - PROMPT_OVERHEAD_TOKENS - NUM_PREDICT_REDUCE
    SINGLE_PASS_MAX_TOKENS = MAX_INPUT_TOKENS_REDUCE

    # CHUNK_TARGET_TOKENS is fixed and small, but guard against a config
    # edit that pushes it past the per-call map input budget (it would
    # only get sub-split by chunk_sections, but a loud note beats silent
    # surprise).
    if CHUNK_TARGET_TOKENS > MAX_INPUT_TOKENS_MAP:
        tlog(
            f"WARNING: CHUNK_TARGET_TOKENS ({CHUNK_TARGET_TOKENS}) exceeds the "
            f"map input budget ({MAX_INPUT_TOKENS_MAP}); chunks will be sub-split."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    tlog(f"Model: {MODEL}")
    tlog(
        f"Context (from PARALLEL_CONFIGS[{PARALLEL}]): num_ctx={NUM_CTX}, "
        f"num_predict_map={NUM_PREDICT_MAP}, num_predict_reduce={NUM_PREDICT_REDUCE}"
    )
    tlog(
        f"Timeouts: map={TIMEOUT_MAP_S}s, reduce={TIMEOUT_REDUCE_S}s, "
        f"final={TIMEOUT_FINAL_S}s"
    )
    tlog(f"Chunk target: {CHUNK_TARGET_TOKENS} tokens (fixed, output-bound)")
    tlog(f"Single-pass threshold: {SINGLE_PASS_MAX_TOKENS} tokens")
    tlog(
        f"Input budgets: map={MAX_INPUT_TOKENS_MAP} tok/chunk, "
        f"reduce={MAX_INPUT_TOKENS_REDUCE} tok/group"
    )
    tlog(f"Parallel: {PARALLEL} concurrent request(s)")
    if PARALLEL > 1:
        tlog(
            f"  NOTE: Ollama must be started with "
            f"OLLAMA_NUM_PARALLEL={PARALLEL} or higher."
        )
    tlog("")

    # Collect input files.
    if os.path.isfile(args.path):
        all_files = [args.path]
    elif os.path.isdir(args.path):
        exts = {".md", ".txt", ".markdown"}
        all_files = sorted(
            os.path.join(args.path, f)
            for f in os.listdir(args.path)
            if os.path.splitext(f)[1].lower() in exts
        )
    else:
        tlog(f"Error: {args.path} not found.", file=sys.stderr)
        sys.exit(1)

    tlog(f"Found {len(all_files)} file(s).")

    # Sequential resume check (cheap stat() — no point parallelizing).
    files_to_process = []
    resumed = 0
    for filepath in all_files:
        filename = os.path.basename(filepath)
        out_path = summary_path_for(args.output_dir, filename)
        if args.resume and os.path.isfile(out_path):
            tlog(f"  RESUME: {filename} already has a summary, skipping.")
            resumed += 1
        else:
            files_to_process.append(filepath)

    processed = 0
    failed = 0

    # Both pools sized at PARALLEL.
    #   _OLLAMA_POOL is the real throttle — matches OLLAMA_NUM_PARALLEL.
    #   _FILE_POOL just supplies enough file-worker threads to keep the
    #   ollama pool fed when individual files only have a few chunks
    #   each (otherwise ollama slots would sit idle while a single file
    #   processes serially).
    _OLLAMA_POOL = concurrent.futures.ThreadPoolExecutor(
        max_workers=PARALLEL, thread_name_prefix="ollama"
    )
    _FILE_POOL = concurrent.futures.ThreadPoolExecutor(
        max_workers=PARALLEL, thread_name_prefix="file"
    )
    try:
        future_to_path = {
            _FILE_POOL.submit(process_one_file, fp, args.output_dir): fp
            for fp in files_to_process
        }
        for future in concurrent.futures.as_completed(future_to_path):
            try:
                if future.result() == "processed":
                    processed += 1
                else:
                    failed += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                fp = future_to_path[future]
                tlog(
                    f"  ERROR processing {fp}:\n{exc_summary(e)}",
                    file=sys.stderr,
                )
                failed += 1
    finally:
        # File workers must drain first — they still hold references to
        # ollama futures. Only then tear down the ollama pool.
        _FILE_POOL.shutdown(wait=True)
        _OLLAMA_POOL.shutdown(wait=True)

    tlog("\n--- Run summary ---")
    tlog(f"  Processed: {processed}")
    if args.resume:
        tlog(f"  Resumed:   {resumed}")
    if failed:
        tlog(f"  Failed:    {failed}")

    consolidated_path = write_consolidated_summary(args.output_dir)
    if consolidated_path:
        tlog(f"  Consolidated summary: {consolidated_path}")
    else:
        tlog("  No per-session summaries found -- skipping consolidated file.")

    run_elapsed = time.time() - run_t0
    tlog(f"  Total time: {format_duration(run_elapsed)}")
    tlog("Done.")


if __name__ == "__main__":
    main()
