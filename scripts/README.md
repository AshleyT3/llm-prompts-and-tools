# scripts/

| File | Description |
|------|-------------|
| [`claude_to_markdown.py`](claude_to_markdown.py) | Convert Claude Code JSONL conversation history to browsable Markdown files | 
| [`claude_to_markdown_creation_prompt.md`](claude_to_markdown_creation_prompt.md) | A prompt that specifies the functionality of `claude_to_markdown.py` — feed it to any capable LLM to generate your own version, tweak it first to tailor the behavior, or simply read it as a precise functional spec |
| [`ollama_summarize_claude_markdown.py`](ollama_summarize_claude_markdown.py) | Summarize the Markdown sessions produced by `claude_to_markdown.py` using a local LLM via Ollama — one summary per session plus a consolidated digest, all offline |
| [`ollama_summarize_claude_markdown_creation_prompt.md`](ollama_summarize_claude_markdown_creation_prompt.md) | A prompt that specifies the functionality of `ollama_summarize_claude_markdown.py` — feed it to any capable LLM to generate your own version, tweak it first to tailor the behavior, or simply read it as a precise functional spec |

---

## claude_to_markdown.py

**Demo video:** [Search Your Claude Code Sessions Offline | JSONL to Markdown](https://www.youtube.com/watch?v=RP_PrUr5TmI)

https://www.youtube.com/watch?v=RP_PrUr5TmI

### The problem

Claude Code stores every conversation as a JSONL file on disk (typically under `~/.claude/projects/`). These files are not human-readable as-is. Searching across sessions, reviewing past work, or sharing a conversation requires opening raw JSON — which is tedious.

### The solution

`claude_to_markdown.py` converts those JSONL files into clean Markdown, one `.md` file per session. The output files are named with a timestamp and project prefix so they sort chronologically and are immediately browsable in VS Code (or any editor with Markdown support).

**What you get:**
- One `.md` file per Claude Code session
- Files named `YYYYMMDD-HHMM-<project>-<session-id>.jsonl.md` for easy sorting
- Each prompt/response pair rendered under numbered `# Prompt N` / `# Response N` headings with local timestamps
- Session ID and custom title at the top of each file
- Works on a single file, a project directory, or your entire Claude projects tree

### Requirements

- Python 3.7+
- No third-party dependencies (stdlib only)

### Usage

```bash
# Convert all sessions under your Claude projects directory
python scripts/claude_to_markdown.py ~/.claude/projects/ -o ./claude-history/

# Convert a single session file
python scripts/claude_to_markdown.py path/to/session.jsonl -o ./output/

# Only convert sessions from the last 7 days
python scripts/claude_to_markdown.py ~/.claude/projects/ --days 7 -o ./output/

# Disable automatic prefixes (timestamp and project name)
python scripts/claude_to_markdown.py ~/.claude/projects/ --no-prefix-timestamp --no-prefix-project
```

### CLI reference

```
usage: claude_to_markdown [-h] [--output OUTPUT] [--no-prefix-timestamp]
                           [--no-prefix-project] [--days N] [--include-untimed]
                           paths [paths ...]

positional arguments:
  paths                 JSONL files or directories to convert (directories are
                        searched recursively)

optional arguments:
  --output, -o          Output directory (default: place .md files next to source)
  --no-prefix-timestamp Disable timestamp prefix on output filenames
  --no-prefix-project   Disable project-name prefix on output filenames
  --days N              Only process files modified within the last N days
  --include-untimed     Include sessions with no timestamp (skipped by default)
```

### The creation prompt

[`claude_to_markdown_creation_prompt.md`](claude_to_markdown_creation_prompt.md) is a prompt that specifies the functionality of `claude_to_markdown.py`. You can use it to:

- Generate your own version of the script with any capable AI code generator
- Tweak the prompt first to customize the behavior (output format, naming conventions, extra features)
- Read it as documentation or functional specification for the claude_to_markdown script

---

## ollama_summarize_claude_markdown.py

**Demo video:** TBD

### The problem

Once you've converted your sessions to Markdown with `claude_to_markdown.py`, you may have dozens or hundreds of transcripts — far more than you can re-read. You want a short, faithful summary of each session, and a single overview across all of them, without sending any of that conversation history to a cloud service.

### The solution

`ollama_summarize_claude_markdown.py` summarizes those Markdown sessions entirely on your own machine using a local LLM via [Ollama](https://ollama.com/). It writes one `<basename>_summary.md` per input file and one `consolidated_summary.md` (with a table of contents) across the whole batch.

It uses a map-reduce strategy: each session is split into section-aware chunks (on the `# Prompt N` / `# Response N` headings), each chunk is summarized independently, and the chunk summaries are hierarchically combined into a final per-session summary. Small sessions that fit in a single model call skip straight to a single-pass summary. Each summary is organized under **Goal**, **Key Actions**, **Outcomes**, and **Open Items** headings, and is tuned to preserve concrete specifics (file names, commands, error messages, ticket/CVE numbers) rather than generalize them away.

**What you get:**
- One `<basename>_summary.md` per session, with a Goal / Key Actions / Outcomes / Open Items structure
- One `consolidated_summary.md` digest with a table of contents linking each session
- Everything runs locally — no conversation data leaves your machine
- Works on a single Markdown file or a whole folder of them

### Requirements

- Python 3.9+
- The [`requests`](https://pypi.org/project/requests/) library (`pip install requests`)
- A running [Ollama](https://ollama.com/) server with a model pulled (default: `gemma4:e4b-it-q8_0`)
- A workstation-class GPU with roughly 32 GB of VRAM is the calibration target for the default model and parallel settings; smaller setups can use a smaller model or lower `--parallel`

### Usage

```bash
# Summarize every Markdown session in a folder (writes to ./summaries)
python scripts/ollama_summarize_claude_markdown.py ./claude-history/

# Use a different Ollama model
python scripts/ollama_summarize_claude_markdown.py ./claude-history/ -m gemma4:e4b-it-q8_0

# Run several sessions concurrently (start Ollama with OLLAMA_NUM_PARALLEL >= 4)
python scripts/ollama_summarize_claude_markdown.py ./claude-history/ -p 4

# Skip sessions that already have a summary in the output directory
python scripts/ollama_summarize_claude_markdown.py ./claude-history/ --resume
```

### CLI reference

```
usage: ollama_summarize_claude_markdown.py [-h] [--output-dir OUTPUT_DIR]
                                           [--model MODEL] [--parallel PARALLEL]
                                           [--resume] [--diag-dir DIAG_DIR]
                                           path

positional arguments:
  path                  Path to a Claude markdown file or a folder of
                        .md/.txt/.markdown files

optional arguments:
  --output-dir, -o      Directory to write <basename>_summary.md files into
                        (default: ./summaries)
  --model, -m           Ollama model tag to use (default: gemma4:e4b-it-q8_0)
  --parallel, -p        Number of concurrent Ollama requests (default: 1).
                        Requires Ollama started with OLLAMA_NUM_PARALLEL >= N
  --resume              Skip any source file whose <basename>_summary.md
                        already exists in --output-dir
  --diag-dir            If set, write per-phase intermediate outputs for each
                        file into <diag-dir>/<input-basename>/, for inspecting
                        where details are dropped between phases
```

### The creation prompt

[`ollama_summarize_claude_markdown_creation_prompt.md`](ollama_summarize_claude_markdown_creation_prompt.md) is a prompt that specifies the functionality of `ollama_summarize_claude_markdown.py`. You can use it to:

- Generate your own version of the script with any capable AI code generator
- Tweak the prompt first to customize the behavior (model, chunking, prompts, output structure)
- Read it as documentation or functional specification for the ollama_summarize_claude_markdown script

## run_ollama_summarize_claude_markdown.ps1

A PowerShell wrapper (Windows / PowerShell 7+) around `ollama_summarize_claude_markdown.py` that saves you from retyping the same boilerplate each run. It:

- Creates a fresh timestamped output directory (`<OutputRootDirectory>\yyyyMMdd-HHmmss\`) per run
- Runs the summarizer (located side-by-side with the script) with output unbuffered
- Tees all output to `log.txt` inside that directory
- Surfaces a clear troubleshooting hint if Python isn't on `PATH` or `requests` isn't installed

### Usage

```powershell
# Summarize a folder; results land in C:\out\<timestamp>\
.\run_ollama_summarize_claude_markdown.ps1 C:\sessions-md C:\out

# Pick a model (short name or full Ollama tag), run concurrently, resume
.\run_ollama_summarize_claude_markdown.ps1 C:\sessions-md C:\out -Model phi4 -Parallel 4 -Resume

# Capture per-phase diagnostics
.\run_ollama_summarize_claude_markdown.ps1 C:\sessions-md C:\out -DiagDir C:\diag
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `SourceProjectMarkdownDirectory` | (positional, required) Markdown file or folder to summarize |
| `OutputRootDirectory` | (positional, required) Parent directory; a timestamped run subdir is created under it |
| `-Model` | Short name or full Ollama tag. Known short names (`gemma4`, `phi4`, `llama3.1`, `gpt-oss`) map to their full tag; anything else is passed through as-is. Default: `gemma4` |
| `-Parallel` | Concurrent Ollama requests. Default: `1` |
| `-Resume` | Skip sources that already have a summary in the output directory. Off by default |
| `-DiagDir` | Capture per-phase diagnostic outputs under this directory. Unset by default |

The `-Model`, `-Parallel`, `-Resume`, and `-DiagDir` options map directly onto the Python script's `--model`, `--parallel`, `--resume`, and `--diag-dir`; optional flags are only passed through when set, so the Python defaults apply otherwise.
