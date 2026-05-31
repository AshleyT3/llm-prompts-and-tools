# Prompt: Generate a Map-Reduce Summarizer for Claude Code Session Logs

Use the following as a complete request to a capable code-generation model
(Claude, GPT-4, etc.). It produces a single-file Python 3 script that
summarizes Claude Code session transcripts via a local Ollama install.

The goal of the generated script is **simplicity for tutorial reading**
combined with **enough real-world robustness that it works on a folder of
real sessions today**. Don't add features beyond what's described.

---

## What I want

A single Python 3 file, `ollama_summarize_claude_markdown.py`, that reads
Claude Code session Markdown files from a folder and writes one
`<basename>_summary.md` per input file plus one `consolidated_summary.md`
into an output folder. Uses a local Ollama server to do the actual
summarization via map-reduce (or a single-pass shortcut for small docs).

## Runtime context

- **Default model:** `gemma4:e4b-it-q8_0`. Override-able with `--model
  <ollama-tag>`. Other models (e.g. `llama3.1:8b`) should "just work"
  without further configuration.
- **Ollama URL:** `http://localhost:11434/api/generate`.
- **Target hardware:** workstation-class GPU with ~32 GB VRAM. AMD
  (W6800) is the original calibration target; NVIDIA also fine.

## CRITICAL CONSTRAINT: num_ctx is constant for a single run

`num_ctx` must NEVER change between Ollama HTTP calls within one
invocation of the script. Changing `num_ctx` mid-run causes Ollama to
restart its runner process, which can trigger delays and possibly
undesirable system activity/behavior in the middle of processing files.
Generally speaking, runner restarts are wasteful.

The actual `num_ctx` value is selected ONCE in `main()` from a
`PARALLEL_CONFIGS` table keyed by `--parallel`:

```python
PARALLEL_CONFIGS = {
    1: {"num_ctx": 32768, "num_predict_map": 8192, "num_predict_reduce": 6144},
    2: {"num_ctx": 16384, "num_predict_map": 6144, "num_predict_reduce": 6144},
    3: {"num_ctx":  8192, "num_predict_map": 3072, "num_predict_reduce": 4096},
    4: {"num_ctx":  8192, "num_predict_map": 3072, "num_predict_reduce": 4096},
}
```

After lookup, `NUM_CTX`, `NUM_PREDICT_MAP`, `NUM_PREDICT_REDUCE`,
`MAX_INPUT_TOKENS_MAP`, `MAX_INPUT_TOKENS_REDUCE`, and
`SINGLE_PASS_MAX_TOKENS` are rebound to the per-config values. They stay
constant for the rest of the run. All chunk sizes, reduce-group sizes,
generation caps, and prompt overhead budgets must fit inside that run's
`NUM_CTX`.

Two values are deliberately NOT in this table:

- **`CHUNK_TARGET_TOKENS`** (map chunk size) is a fixed module constant,
  intentionally small and **independent of `num_ctx`**. The binding
  constraint on a chunk is not how much input context is available but
  how much OUTPUT a map call can emit before hitting `num_predict`: a
  faithful summary that preserves specifics is token-dense, so an
  oversized chunk truncates the map summary mid-output and silently drops
  content. See Required tuning constants.
- The **generation caps** (`num_predict_map`, `num_predict_reduce`) DO
  scale with `num_ctx`, deliberately. High-context configs (`-p 1`: a
  map prompt is only ~5K of the 32K window) have spare context to spend
  on output headroom so dense summaries don't truncate. The strict 8192
  configs (`-p 3/4`) can't afford it and will truncate identifier-dense
  output — an accepted small-VRAM tradeoff.

Rationale for scaling: per-slot KV cache scales with `parallel * num_ctx`.
On the target W6800 (32 GB VRAM), keeping total KV cache around 5 GiB
across all configs leaves comfortable room for the ~11 GiB model
weights plus compute overhead. At `-p 1`, a single slot gets the full
context window, so realistic Claude sessions can fit single-pass; at
`-p 4`, chunks must be small enough to multiplex across four 8K slots.

If a `--parallel` value not in the table is passed (e.g. `-p 8` on a
larger GPU), fall back to the `-p 4` config with a warning telling the
user to edit `PARALLEL_CONFIGS` to add a tuned entry.

## CRITICAL CONSTRAINT: single model for the entire run

The Ollama tag passed via `--model` is used for **every** `call_ollama`
invocation in the run — map, reduce-group, final-reduce, single-pass,
forced-final. Do not introduce a separate model for any phase.

Reasons:

- **Runner stability.** Switching models mid-run — like changing `num_ctx` —
  forces Ollama to unload the current model and load the next one, which
  can trigger delays and possibly undesirable system activity/behavior in
  the middle of processing files. Generally speaking, runner restarts are
  wasteful.
- **Predictable VRAM footprint.** With one model loaded for the whole
  run, the user observes their actual memory cost once at startup and
  knows it won't change. `--parallel N` then has a clear, additive
  meaning: model weights + N × `num_ctx` KV cache. Multi-model
  configurations make that arithmetic — and the failure modes around
  hitting VRAM limits — much harder to reason about.

## Input format: Claude markdown files

These files are produced by a separate conversion tool (out of scope).
A typical file looks like:

```
Session ID: <uuid>
Title: <some title or "<not-specified>">
... preamble ...

# Prompt 1 20260111-170800
... user message ...

# Response 1 20260111-170815
... assistant response (possibly with tool calls, edits, etc.) ...

# Prompt 2 20260111-170900
...
```

Headers alternate between `# Prompt N <YYYYMMDD-HHMMSS>` and
`# Response N <YYYYMMDD-HHMMSS>`.

The script must:

- Parse `Session ID:` and `Title:` from the first ~2000 chars.
- Normalize `Title: <not-specified>` (or any title matching `<...>`,
  meaning empty angle-bracket placeholder) to the literal `unspecified`.
- Normalize headers to `# Prompt (YYYY-MM-DD HH:MM:SS)` / `# Response
  (YYYY-MM-DD HH:MM:SS)` once, up front, so the model sees a
  human-readable timestamp at zero instruction cost. The normalization
  function must be idempotent.

## Approach: map-reduce, with single-pass shortcut

1. **PARSE** — Extract Session ID and Title from preamble.
2. **MAP** — Split document into section-aware chunks (see Chunking
   below); summarize each chunk independently.
3. **REDUCE** — Hierarchically combine chunk summaries until they fit
   in a single prompt, then generate the final per-session summary.
4. **WRITE** — One `<basename>_summary.md` per input file in
   `--output-dir`.
5. **CONSOLIDATE** — Mechanically concatenate per-session summaries
   into `<output-dir>/consolidated_summary.md` with a TOC. No model
   call for this step.

### Single-pass shortcut with chunked fallback

If a file's estimated tokens ≤ `SINGLE_PASS_MAX_TOKENS`, skip the map
phase entirely and run `SINGLE_PASS_PROMPT` directly on the whole
document (see prompts section). One Ollama call total for the file.

The threshold is `SINGLE_PASS_MAX_TOKENS` (a **context-capacity**
decision: does the whole doc plus its generation cap fit one call?), NOT
`CHUNK_TARGET_TOKENS` (an output-completeness decision about map chunk
size). The two are independent: a document can be many chunks long yet
still fit a single call, and single-pass is preferred when it fits
because it has no lossy reduce step to compress specifics away.
`SINGLE_PASS_MAX_TOKENS` equals `MAX_INPUT_TOKENS_REDUCE` (i.e.
`NUM_CTX - PROMPT_OVERHEAD_TOKENS - NUM_PREDICT_REDUCE`).

**Fallback**: if the single-pass call returns an empty or whitespace-only
digest (model produced degenerate output despite generating tokens —
this happens occasionally near the upper end of the context window),
log a warning and **fall through to the chunked map-reduce path**.
Smaller per-call prompts using `MAP_PROMPT` have proven more reliable
than `SINGLE_PASS_PROMPT` at high token counts, so the chunked path
acts as a safety net.

In code, the small-doc branch should look approximately like:

```python
if total_tokens_est <= SINGLE_PASS_MAX_TOKENS:
    digest = call_ollama(SINGLE_PASS_PROMPT.format(...), ...)
    if digest and digest.strip():
        return digest, ...                 # success: one call, done
    tlog("WARNING: single-pass returned empty/degenerate output. "
         "Falling back to chunked map-reduce.")
    # fall through to chunked path below
chunks = chunk_text(text)
...
```

## CLI surface

Exactly these flags. No others.

- `path` (positional, required): file or directory of `.md` / `.txt` /
  `.markdown` files.
- `--output-dir`, `-o` (default `./summaries`)
- `--model`, `-m` (default `gemma4:e4b-it-q8_0`) — passed as a real
  Ollama tag string. **No "profile" abstraction.**
- `--parallel`, `-p` (default `1`): caps concurrent Ollama calls; also
  selects `num_ctx`/`num_predict_map`/`num_predict_reduce` from
  `PARALLEL_CONFIGS`. Document that the user must start Ollama with
  `OLLAMA_NUM_PARALLEL >= N`.
- `--resume`: skip any source file whose `<basename>_summary.md`
  already exists in `--output-dir`.
- `--diag-dir <path>` (optional, default off): when set, write per-phase
  intermediate outputs (each chunk's map output, each reduce group's
  output, the final reduce output, the raw pre-strip response for the
  single-pass call) to `<diag-dir>/<input-basename>/`. Lets a human
  inspect where specifics get dropped between map and reduce. No
  performance impact when omitted (all diag writes are no-ops when
  the path is `None`).

## Architecture: two thread pools

Use two `concurrent.futures.ThreadPoolExecutor` instances created in
`main()`, both `max_workers=PARALLEL`:

- `_OLLAMA_POOL` — every `call_ollama` HTTP request is submitted here.
  This pool's `max_workers` IS the actual GPU-side concurrency throttle.
- `_FILE_POOL` — runs `process_one_file(filepath, output_dir)` for each
  non-resumed file. Each file-worker thread executes the full pipeline
  (parse → chunk → map → reduce → write) **synchronously**, blocking on
  the futures it submitted to the ollama pool. File workers don't
  consume ollama-pool slots while blocked.

Main thread submits one file-worker per non-resumed file and collects
results via `concurrent.futures.as_completed`. On shutdown, drain
`_FILE_POOL` first, then `_OLLAMA_POOL`.

**Why two pools:** when one file has fewer chunks than `PARALLEL`, the
remaining ollama slots get used by chunks from other files. Single-pool
variants either serialize files (wasting GPU) or risk a nested-pool
deadlock pattern.

**Why no semaphore or async/await:** `ThreadPoolExecutor.max_workers`
already enforces the concurrency cap. Idle Python file-worker threads
are cheap (they hold the GIL release while blocked on futures).

## Chunking

### Section-aware (primary)

If the document contains normalized headers `# Prompt (...)` or
`# Response (...)`:

1. Split at headers into sections (treat any preamble before the first
   header as section 0).
2. Greedily pack consecutive sections into chunks ≤
   `CHUNK_TARGET_TOKENS`.
3. If a single section exceeds `CHUNK_TARGET_TOKENS` but fits within
   `MAX_INPUT_TOKENS_MAP`, emit it as its own chunk.
4. If a single section exceeds `MAX_INPUT_TOKENS_MAP`, sub-split using
   a **3-level cascade**: paragraph (`\n\n`) → sentence (`. `) → raw
   character cut. Each level only refines the chunks that the previous
   level left over-budget. The raw cut is the guaranteed bound.

### Paragraph fallback

If no normalized Prompt/Response headers are found, fall back to
paragraph-based splitting on `\n\n`.

### Token estimation

`estimate_tokens(text)` returns `int(len(text) / CHARS_PER_TOKEN)` with
`CHARS_PER_TOKEN = 3.2` (measured for the gemma tokenizer on this kind
of content). No tokenizer dependency.

This is a heuristic and it drives every boundary decision (chunk fit,
single-pass threshold, reduce-group packing) while the engine enforces
EXACT counts — so the two can disagree, most dangerously on the strict
8192-token `-p 3/4` configs where an underestimate on dense markdown can
overflow the window. `3.2` is set to slightly OVER-count tokens (real is
~3.3 chars/tok), keeping boundary math on the safe side. A local
tokenizer, or calibrating `CHARS_PER_TOKEN` from the `prompt_eval_count`
Ollama returns on each call, would remove the gap — left as a documented
future improvement, not built (keeps the script dependency-free).

## Reduce loop

`reduce_phase(summaries, filename, session_id, title, per_file_diag_dir=None)`:

Per round:

- If combined tokens ≤ `MAX_INPUT_TOKENS_REDUCE` → run the **final**
  reduce prompt and return.
- Else build groups via budget-aware grouping (see below). Submit
  multi-item groups to `_OLLAMA_POOL`. Pass-through single-item groups
  verbatim (they don't need a model call). Reassemble results in
  original order.
- **Stall detection:** if the corpus shrank by less than
  `STALL_THRESHOLD` (5%) since the previous round, the model isn't
  compressing — force the final prompt with whatever items fit in
  budget, discard the overflow with a warning, and return.
- Hard cap: `MAX_REDUCE_ROUNDS = 10`. On exceed, force final.

### Budget-aware grouping

Pack items into groups where:

- Combined tokens per group ≤ `MAX_INPUT_TOKENS_REDUCE`
- Items per group ≤ `REDUCE_GROUP_SIZE` (10)
- An item that individually exceeds the budget gets its own group with
  a warning (it'll be truncated by the model, which is preferable to
  changing `num_ctx`).

**Gotcha:** the `max_tokens` parameter must default to `None` and be
resolved to `MAX_INPUT_TOKENS_REDUCE` *inside* the function body at call
time — NOT via a default-argument expression (`max_tokens=MAX_INPUT_TOKENS_REDUCE`).
A default expression is evaluated once at import, binding the module
value as it stood *before* `main()` rebinds it from `PARALLEL_CONFIGS`,
so grouping would silently use the wrong budget on every config.

## Required tuning constants

Use these values exactly. They are calibrated for
`gemma4:e4b-it-q8_0` on a W6800. They are oversized for terse models
like `llama3.1:8b`, but harmless — those models stop generating earlier
than the cap.

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e4b-it-q8_0"

# Per-parallel config table. NUM_CTX, NUM_PREDICT_MAP, and
# NUM_PREDICT_REDUCE are bound in main() from this table based on
# --parallel; the bound values stay constant for the rest of the run.
# Generation caps scale WITH num_ctx so dense summaries don't truncate
# on high-context configs. Chunk size is NOT in this table (see below).
PARALLEL_CONFIGS = {
    1: {"num_ctx": 32768, "num_predict_map": 8192, "num_predict_reduce": 6144},
    2: {"num_ctx": 16384, "num_predict_map": 6144, "num_predict_reduce": 6144},
    3: {"num_ctx":  8192, "num_predict_map": 3072, "num_predict_reduce": 4096},
    4: {"num_ctx":  8192, "num_predict_map": 3072, "num_predict_reduce": 4096},
}

# Initial values for module load; NUM_CTX / NUM_PREDICT_MAP /
# NUM_PREDICT_REDUCE are all rebound in main() once --parallel is known.
NUM_CTX = 8192
NUM_PREDICT_MAP = 3072

# Map chunk size, in tokens. FIXED and intentionally small -- it does
# NOT scale with num_ctx and is NOT in PARALLEL_CONFIGS. It is bounded by
# how much OUTPUT a map call can emit before hitting NUM_PREDICT_MAP, not
# by input context. 4000 stays within MAX_INPUT_TOKENS_MAP even for the
# smallest config (-p 3/4: 8192 - 400 - 3072 = 4720); output completeness
# for a chunk this size is handled by NUM_PREDICT_MAP scaling up on
# high-context configs.
CHUNK_TARGET_TOKENS = 4000

NUM_BATCH = 2048
TEMPERATURE = 0.1

# Final-reduce generation cap. Initial value; rebound in main() from
# PARALLEL_CONFIGS (high-context configs get a larger cap so the final
# summary of a specifics-dense corpus isn't truncated).
NUM_PREDICT_REDUCE = 4096

# Prompt template overhead (tokens). Covers all prompts including the
# embedded SPEAKER_BLOCK and CAPTURE_BLOCK in MAP_PROMPT and
# SINGLE_PASS_PROMPT.
PROMPT_OVERHEAD_TOKENS = 400

TIMEOUT_MAP_S = 120
TIMEOUT_REDUCE_S = 240
TIMEOUT_FINAL_S = 360

# Chars-per-token estimator (heuristic; see Token estimation). 3.2 is
# measured for gemma on this content and set to slightly over-count so
# boundary math errs safe.
CHARS_PER_TOKEN = 3.2

REDUCE_GROUP_SIZE = 10
MAX_RETRIES = 2
RETRY_DELAY_S = 5
MAX_REDUCE_ROUNDS = 10
STALL_THRESHOLD = 0.05

# Token budgets and the single-pass threshold. Bound in main() after the
# per-parallel rebind. SINGLE_PASS_MAX_TOKENS == MAX_INPUT_TOKENS_REDUCE,
# named separately so the single-pass call site reads clearly.
MAX_INPUT_TOKENS_MAP = NUM_CTX - PROMPT_OVERHEAD_TOKENS - NUM_PREDICT_MAP
MAX_INPUT_TOKENS_REDUCE = NUM_CTX - PROMPT_OVERHEAD_TOKENS - NUM_PREDICT_REDUCE
SINGLE_PASS_MAX_TOKENS = MAX_INPUT_TOKENS_REDUCE

# Module-level globals (set in main() from parsed args).
MODEL = DEFAULT_MODEL
PARALLEL = 1
DIAG_DIR: "str | None" = None
```

## Use these prompts verbatim

The speaker-identification rules encode hard-won conventions about how
the Claude markdown format labels Developer vs LLMDev turns — do not
paraphrase or simplify them. `SPEAKER_BLOCK`, `SPEAKER_REMINDER`, and
`CAPTURE_BLOCK` are **shared constants** referenced by other prompts; do
not inline the block contents into each prompt.

**Prompt philosophy — preserve specifics, do not overfit.** The target
is real, mostly conversational dev↔Claude sessions (which may mention
file names, libraries, error strings, CVE numbers, or informal bug
nicknames, but have no structured ID scheme). Prompts must ask the model
to keep concrete specifics *that actually appear* — without inventing
them, without prioritizing identifier-listing over a readable narrative,
and without any test-specific tag vocabulary. Earlier revisions overfit
to a synthetic ID-grid test (instructions like "list every identifier",
"completeness of identifiers takes priority over prose", `Fix-RXX`
examples); those were removed because they degrade summaries of real
sessions. Keep the general phrasing below.

### SPEAKER_BLOCK (shared, used in MAP_PROMPT and SINGLE_PASS_PROMPT)

````
Speaker identification: Sections begin with headers like
"# Prompt (2026-01-11 17:08:00)" or "# Response (2026-01-11 17:08:15)".
Prompt sections are spoken by the Developer; Response sections are spoken
by the LLMDev. Exception: if a Prompt section contains "user's action",
"user needs", or "user did a great job", it refers to the LLMDev.
Prompt sections containing the word "ultrathink" are spoken by Developer.
Prompt sections beginning with "This session is being continued from a
previous conversation" are spoken by LLMDev, except for the subsection
beginning with "All User Messages:" which is spoken by Developer.
````

### SPEAKER_REMINDER (shared, used in REDUCE_GROUP_PROMPT and REDUCE_FINAL_PROMPT)

A one-line reminder so attribution survives multi-round reduces even
if an upstream summary starts losing track of the speaker labels.

````
In the summaries below, "Developer" refers to the human user and "LLMDev" refers to Claude Code. Preserve this attribution in your output.
````

### CAPTURE_BLOCK (shared, used in MAP_PROMPT and SINGLE_PASS_PROMPT)

The list of what each section/transcript summary must retain. Shared by
both prompts so the two paths can't drift apart (they did once: only one
explicitly asked to keep specifics, so recall differed wildly on the
same document). The last bullet is the de-test-ified specifics rule —
keep it general; do not reintroduce ID-grid vocabulary.

````
- What the Developer asked for or what the LLMDev did
- Key decisions, files changed, commands run, or errors encountered
- Any blockers or open questions
- Timestamps from section headers (YYYY-MM-DD HH:MM:SS) where they
  anchor important events
- Concrete specifics, kept verbatim when they appear: file and function
  names, commands, error messages or codes, library/product/tool names,
  and any ticket, issue, or CVE numbers. Don't generalize them into vague
  terms ("a config file", "some error") or invent ones that aren't there.
````

### MAP_PROMPT

Used for each chunk during the map phase. Note: `{SPEAKER_BLOCK}` and
`{CAPTURE_BLOCK}` are substituted at module load via f-string; `{chunk}`
is filled in at call time via `.format(chunk=...)`.

````
You are summarizing one section of a Claude Code session transcript.

{SPEAKER_BLOCK}

### INSTRUCTIONS:
Summarize this section. Capture:
{CAPTURE_BLOCK}

Use concise bullets. Preserve specifics (file names, function names,
error messages). Omit small talk and routine acknowledgements.

### SECTION TEXT:
{chunk}

### SUMMARY:
````

### REDUCE_GROUP_PROMPT

````
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
{items_text}

### CONSOLIDATED SUMMARY:
````

### REDUCE_FINAL_PROMPT

````
Below are consolidated summaries from a Claude Code session.
Source file: {filename}
Session ID: {session_id}
Session title: {title}

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
{items_text}

### FINAL SUMMARY:
````

### SINGLE_PASS_PROMPT

Used when a document is small enough that the whole transcript fits in
one Ollama call without chunking. The map/reduce path doesn't apply,
so this prompt bundles everything the model needs in one place: the
speaker block (raw transcript input), the shared `CAPTURE_BLOCK`
guidance, AND the REDUCE_FINAL output structure (the four headings).
Without this,
single-pass would have to fall back to REDUCE_FINAL_PROMPT, which
lacks the speaker block and would underperform on real transcripts
containing `ultrathink`/`continued session` patterns.

````
Below is a complete Claude Code session transcript. The sections appear
in chronological order.
Source file: {filename}
Session ID: {session_id}
Session title: {title}

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
{text}

### FINAL SUMMARY:
````

## Diag mode

When `--diag-dir <path>` is passed, the script preserves enough
intermediate state from each processed file that an investigator can
see exactly where details got dropped between phases — without
re-running.

### Layout

For each processed file (basename without extension):

```
<diag-dir>/<basename>/
    map-result-singlepass.txt       (only when single-pass ran;
                                     stripped digest)
    map-result-singlepass-raw.txt   (only when single-pass ran;
                                     pre-strip raw response, useful
                                     for diagnosing degenerate output)
    map-result-chunk001.txt         (one per chunk; raw map output
                                     BEFORE the "[Section N]" wrapping)
    map-result-chunk002.txt
    ...
    reduce-round00-grp001.txt       (one per group per round; includes
                                     pass-through single-item groups
                                     so slot numbering is contiguous)
    reduce-round00-grp002.txt
    reduce-round01-grp001.txt       (only if a second round happened)
    ...
    reduce-combined.txt             (terminal output: fits-in-one-pass,
                                     stall, or max-rounds final)
```

The per-file subdir is created lazily by a `_diag_write(dir, name,
content)` helper that no-ops when `dir` is `None`. Callers can be
unconditional.

### Timing: write as soon as data is available, not at the end

Each chunk's `map-result-chunkNNN.txt` is written **inside the
`as_completed` loop**, the moment that chunk's future resolves — not
batched at the end of the map phase. Same for reduce groups: each
group's diag file appears as its future completes. Pass-through groups
(single-item, no model call) are written **eagerly** at the start of
the round, before any futures are launched, since their content is
known immediately.

This lets a human watching the diag dir during a run see partial
results in real time and abort early if the first chunk's output looks
wrong, instead of waiting for the whole map phase.

### Raw response capture in `call_ollama`

`call_ollama` accepts an optional `raw_diag_path: str | None = None`
parameter. When provided, it writes the **pre-strip** response from
Ollama to that path before applying `strip_leading_cue`. This is the
mechanism behind `map-result-singlepass-raw.txt`. Only the single-pass
call site passes a `raw_diag_path` (chunk and reduce calls don't need
it — their stripped outputs already go to diag).

Additionally, `call_ollama` always logs a one-line shape hint after
each call, regardless of whether diag is enabled:

```
[label] raw response: 4029 chars, starts with: '### FINAL SUMMARY:\n\n**1. Goal**...'
```

This makes degenerate output visible from the log alone (e.g.
`raw=4 chars` after `eval_count=4096` says the model emitted only the
cue heading and 4090 tokens of whitespace).

## Implementation details and gotchas (real issues — don't omit)

- **Cap-hit warning** — after every Ollama call, compare `eval_count`
  to `num_predict`. If equal, the model was cut off mid-generation;
  log a WARNING (including the call's phase label — see Phase labels
  below) so silent truncation is visible and attributable to the
  specific phase that produced it.

- **Cue echoing** — some models (gemma in particular) echo the prompt's
  trailing cue heading back as the first line of their response (e.g.
  `### FINAL SUMMARY:` followed by the actual content). Strip a single
  leading cue matching `### FINAL SUMMARY:`, `### CONSOLIDATED SUMMARY:`,
  or `### SUMMARY:` (with optional leading whitespace) from every model
  response. Apply this inside `call_ollama` so all callers benefit.
  Don't strip cues that appear inside the body, only at the very start.
  The raw response (pre-strip) is preserved for diag via
  `raw_diag_path` so investigators can see what was actually emitted.

- **Retry with backoff** — `MAX_RETRIES` attempts with `RETRY_DELAY_S`
  delay on HTTP/timeout exceptions. Final attempt returns empty string
  rather than raising.

- **Failure isolation** — in the main `as_completed` loop, catch
  exceptions from individual file futures and count them as failures —
  don't let one bad file kill the whole run.

- **Exception reporting with traceback** — every place that catches
  and logs an exception (the final-attempt failure inside `call_ollama`,
  the `as_completed` catch in `main`) MUST emit the full traceback —
  not just `str(exception)`. A reader scanning the log should be able
  to see the source line at a glance, without re-running under a
  debugger. Provide a small helper using the modern single-arg form:

  ```python
  def exc_summary(ex: BaseException) -> str:
      return "".join(traceback.format_exception(ex))
  ```

  Use it in both log sites. Mid-retry failures inside `call_ollama`
  can stay terse (message only) — we don't want N tracebacks per file
  when the call eventually succeeds. Only the final, give-up log line
  needs the full stack.

- **Per-session output format**:

  ```
  # Summary: <source_filename>
  **Session ID:** <id>
  **Title:** <title>

  <model digest>

  ---
  _Generated in Xs using model=<tag>, <method> summarization._
  ```

  `<method>` is the path actually taken — `single-pass` or `map-reduce` —
  not a fixed string. `process_file` returns the method so the footer is
  accurate for demo, triage, and archival reference.

- **Consolidated output format**:

  ```
  # Consolidated Session Summaries

  _Sessions: N • Assembled: YYYY-MM-DD_

  ## Table of Contents

  1. [<Title>](#session-1) — `<filename>`
  2. [<Title>](#session-2) — `<filename>`
  ...

  ---

  <a id="session-1"></a>

  <contents of session 1's _summary.md, verbatim>

  ---

  <a id="session-2"></a>

  ...
  ```

  Use **explicit `<a id="...">` anchors** (not heading-derived
  auto-anchors). Different markdown renderers (VSCode preview, GitHub,
  Pandoc) compute heading anchors with slightly different slug rules;
  explicit anchors work in all of them.

  Defensively re-normalize titles when building the TOC: if a title is
  empty or matches `<...>`, replace with `unspecified` (so any stale
  per-session files written by older runs don't render blank in the
  TOC).

  Skip the consolidator if no `*_summary.md` files exist (after
  excluding `consolidated_summary.md` itself).

- **Phase labels** — every `call_ollama` invocation must accept and use
  a `label` argument that identifies its phase. Required labels:
  `map i/N` (chunk index and total), `reduce r<round> g<i>/<total>`
  (round and group index), `single-pass` (the small-doc shortcut),
  `final reduce` (the final pass after hierarchical reduce), and
  `forced final (<reason>)` (the stall or max-rounds fallback). The
  label appears in the call's per-call log line, the cap-hit warning,
  and the raw-response shape hint, so phase context is preserved
  across the entire log.

- **Required diagnostic data in stdout** — log formatting is at the
  implementer's discretion (banner-style, file-tag-prefixed, one
  line per event, etc.), but the following data fields MUST be
  visible somewhere in the log output. The intent is that an operator
  reading the log can answer: what was processed, how big, how long,
  did anything truncate, and where did it land — without re-running
  the script.

  Per source file:
  - filename and file size on disk (KB or bytes)
  - Session ID and Title
  - estimated token count of the document
  - chunk count (or `single-pass` if the document took the shortcut)
  - per-file diag dir path (if `--diag-dir` was passed)
  - absolute path of the written `_summary.md`
  - per-file wall-clock elapsed time, emitted as a log line — not
    only in the markdown file's footer

  Per Ollama call (every phase, no exceptions):
  - the call's phase label
  - prompt tokens (`prompt_eval_count` from the response)
  - generated tokens (`eval_count` from the response)
  - call elapsed wall-clock time
  - the configured timeout for that call
  - raw response shape hint (length + leading content snippet)

  Per reduce round:
  - round number
  - item count and combined token count at the start of the round
  - per-group dispatch (group index, item count, token count)
  - stall and forced-final transitions, when they occur

  At startup (after parsing args):
  - selected model
  - selected `num_ctx`, `num_predict_map`, `num_predict_reduce`
    (with a `(from PARALLEL_CONFIGS[N])` annotation so the source
    of the values is obvious)
  - `chunk_target`, the single-pass threshold, and derived input budgets
  - parallel slot count
  - diag dir top-level path (if set)

- **UTF-8 throughout for all text I/O** — Windows defaults text streams
  and files to cp1252 (charmap), which cannot represent characters like
  √, ², ʸ, em-dashes, or emoji — all of which appear in real session
  data (titles, code, prose). Without explicit UTF-8 the script will
  raise `UnicodeEncodeError` from `print()` or `UnicodeDecodeError`
  from file reads mid-run and abort one file's processing.

  - At the top of `main()`, BEFORE the first `tlog` call, reconfigure
    `sys.stdout` and `sys.stderr` to UTF-8:

    ```python
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass  # Non-TextIOWrapper streams (rare; e.g. test capture)
    ```

    `errors="replace"` is belt-and-suspenders for legacy consoles that
    genuinely lack UTF-8 display: you'd see `?` placeholders instead
    of a crash.

  - Every `open()` call for text files MUST pass `encoding="utf-8"`.
    For read paths, also pass `errors="replace"` so a single malformed
    byte sequence doesn't abort processing — the model just sees a
    `?` at that position.

- **Logging** — small `tlog(msg)` helper:
  - UTC ISO-format timestamp prefix with `Z` suffix.
  - `flush=True` per call so logs survive abrupt termination.
  - **Thread safety:** wrap the `print` in a module-level
    `threading.Lock` so concurrent file workers' output doesn't
    interleave mid-line.
  - **Leading newlines:** if `tlog` receives a message starting with
    one or more `\n`, emit those newlines BEFORE the timestamp so
    visual blank-line separators actually render blank (not "timestamp
    + space on its own line, content below").

- **Total run time** — capture `time.time()` at the very start of
  `main()`; print at the end. Use a `format_duration(seconds)` helper
  that returns `Ns` / `Mm Ss` / `Hh Mm Ss` depending on magnitude.

- **Thread-id ENTER/EXIT logs** — inside `_map_one_chunk` and
  `_reduce_one_group`, log `threading.get_ident()` and
  `threading.get_native_id()` at entry and exit (use `try/finally`).
  These let the reader correlate interleaved logs back to specific
  worker threads when `-p > 1` is in play. They're also a small
  introduction to Python threading for tutorial readers.

## Style guidelines

- Single file. No package layout.
- Module-level constants at the top. One source of truth per value.
- Module-level globals for `MODEL`, `PARALLEL`, `DIAG_DIR`,
  `_OLLAMA_POOL`, `_FILE_POOL`, and the parallel-dependent values
  (`NUM_CTX`, `NUM_PREDICT_MAP`, `NUM_PREDICT_REDUCE`,
  `MAX_INPUT_TOKENS_MAP`, `MAX_INPUT_TOKENS_REDUCE`,
  `SINGLE_PASS_MAX_TOKENS`). All bound once in `main()` from parsed args
  + `PARALLEL_CONFIGS` lookup. `CHUNK_TARGET_TOKENS` is a fixed constant,
  NOT parallel-dependent.
- Functions over classes. No state-machine objects. Each file worker
  thread's call stack IS its state.
- Comments explain WHY (constraints, gotchas), not WHAT (the code
  already shows that). Default to no comment unless the reason is
  non-obvious.
- Type hints on public function signatures.
- Keep `if __name__ == "__main__": main()` at the bottom.

## Explicitly NOT included (resist temptation to add)

These are choices code-generation models commonly volunteer as "more
helpful" or "more modern." Don't.

- **No streaming Ollama responses.** Use `stream: false` and read the
  complete JSON response.
- **No `/api/chat` endpoint.** Use `/api/generate`.
- **No `logging` library / loguru / structlog.** Use the simple `tlog`
  helper as described.
- **No tokenizer library** (tiktoken, transformers, etc.). Use the
  `CHARS_PER_TOKEN` heuristic for budgeting.
- **No async/await.** Use threads as described in the architecture
  section.
- **No extra CLI flags** beyond the six listed in the CLI surface
  section.
- **No per-call diag for chunk/reduce raw responses.** Only the
  single-pass call captures a raw-response sidecar (because that's
  the path most susceptible to degenerate output). Chunk and reduce
  calls write only their stripped outputs to diag — adding raw
  capture there would multiply file count without proportional
  diagnostic value.
