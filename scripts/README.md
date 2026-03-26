# scripts/

| File | Description |
|------|-------------|
| [`claude_to_markdown.py`](claude_to_markdown.py) | Convert Claude Code JSONL conversation history to browsable Markdown files |
| [`claude_to_markdown_creation_prompt.md`](claude_to_markdown_creation_prompt.md) | A prompt that specifies the functionality of `claude_to_markdown.py` — feed it to any capable LLM to generate your own version, tweak it first to tailor the behavior, or simply read it as a precise functional spec |

---

## claude_to_markdown.py

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
