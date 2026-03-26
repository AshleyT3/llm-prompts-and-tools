# Prompt: Create claude_to_markdown.py

Create a single-file Python script called `claude_to_markdown.py` that converts Claude Code conversation JSONL files into readable Markdown. The script must use only Python stdlib (no third-party dependencies). Target Python 3.7+.

## Claude Code JSONL Format

Claude Code stores conversations as JSONL files (one JSON object per line). Each line has a top-level `type` field and a `timestamp` field (ISO8601 UTC, e.g. `"2026-02-11T07:32:54.143Z"`).

Relevant record types:

- `"type": "user"` — a user message. Contains `message.role = "user"` and `message.content` which is either a plain string OR an array of content blocks. Array items may be dicts with `"type": "tool_result"` (skip these) or dicts with a `"text"` field (extract these), or plain strings (extract these).
- `"type": "assistant"` — an assistant message. Contains `message.role = "assistant"` and `message.content` which is either a plain string OR an array of content blocks. From the array, extract ONLY items where `type == "text"`. Skip `type == "thinking"` and `type == "tool_use"`.
- `"type": "custom-title"` — contains a `customTitle` string field with the user-assigned session title.
- All other types (`"system"`, `"file-history-snapshot"`, signature properties, etc.) should be ignored entirely.

When content is an array, join extracted text parts with `\n`.

## Session File Layout

Claude Code organizes sessions on disk as:

```
<project-dir>/
  <session-uuid>.jsonl          # top-level session file
  <session-uuid>/               # optional session subdirectory (same stem as the .jsonl)
    <tool-run-uuid>.jsonl       # sub-session files from tool runs, etc.
    <subdir>/<more>.jsonl       # may be nested arbitrarily
```

When converting a session, ALWAYS merge messages from the top-level `<uuid>.jsonl` AND all `**/*.jsonl` files found recursively within the sibling `<uuid>/` directory (if it exists). Sort all merged messages by timestamp.

## Recursive Project Discovery

The script accepts one or more positional `paths` arguments, each of which may be a `.jsonl` file or a directory.

**Explicit .jsonl file path:** Process only that session file (plus its session subdirectory as described above). Do NOT process sibling `.jsonl` files in the same directory.

**Directory path:** Search recursively for "project directories." A project directory is the first directory in each branch of the tree that contains `*.jsonl` files. Once found:
- Collect ALL `*.jsonl` files at that level (each is a top-level session file).
- Process each session file (including its session subdirectory).
- Do NOT descend further in that branch — projects do not nest.
- Continue searching sibling branches that haven't yielded a project directory yet.

Example: given path `~/a/`:
```
~/a/
  b/
    s1.jsonl        ← found → "b" is a project dir
    s2.jsonl        ← also processed (sibling in same project dir)
    s1/sub.jsonl    ← merged into s1's output (session subdir)
    c/
      x.jsonl       ← NOT found (b already claimed, no further descent)
  d/
    e/
      s3.jsonl      ← found → "e" is a project dir
```

Deduplicate discovered files (preserving order) in case the user provides overlapping paths.

## Content De-escaping

Before writing to markdown, de-escape JSON-escaped content in this order:
1. `\\` → `\`
2. `\"` → `"`
3. `` \` `` → `` ` ``

## Markdown Output Format

For each session, generate a markdown file structured as:

```
Session ID: <uuid-stem-of-jsonl-file>
Title: <customTitle from JSONL, or "<not-specified>" if absent>

# Prompt 1 <local-timestamp>

<user message content>

# Response 1 <local-timestamp>

<assistant message content>

# Prompt 2 <local-timestamp>

...
```

- A shared counter increments on each user message. The paired assistant response uses the same counter value.
- Each heading includes a local-time timestamp formatted as `YYYYMMDD-HHMMSS`, converted from the message's UTC ISO8601 timestamp to the system's local timezone. Use `datetime.fromisoformat()` after replacing trailing `Z` with `+00:00` (for Python 3.7 compat), then `.astimezone()` for local time.

## Output Filename Construction

The base filename is the original `.jsonl` filename. Prefixes are applied in this order (both enabled by default):

1. **Project prefix** (if enabled): prepend `<project-dir-name>-` where project-dir-name is the `.parent.name` of the session `.jsonl` file.
2. **Timestamp prefix** (if enabled): prepend `<YYYYMMDD-HHMM>-` derived from the earliest timestamp found in the JSONL file. The earliest timestamp is found by scanning all lines for the minimum `timestamp` value, then formatting it as `YYYYMMDD-HHMM`.

The final filename is `<timestamp-prefix><project-prefix><original.jsonl>.md`.

Example: `20260210-2205-myproject-abc123.jsonl.md`

Output location: if `--output` is specified, write to that directory (create it if needed). Otherwise, write the `.md` file next to the source `.jsonl` file.

## CLI Interface

No subcommands. All arguments are top-level. Use `argparse` with `prog='claude_to_markdown'`.

**Positional arguments:**
- `paths` (one or more): JSONL files or directories to convert. Directories are searched recursively as described above.

**Optional arguments:**
- `--output`, `-o`: Output directory. Default: place `.md` files next to source.
- `--no-prefix-timestamp`: Disable timestamp prefix (it is ON by default).
- `--no-prefix-project`: Disable project-name prefix (it is ON by default).
- `--days N`: Only process `.jsonl` files whose filesystem mtime is within the last N days. Filter is applied after discovery. If no files pass the filter, print an error to stderr showing total found vs. filtered count, and exit 1.
- `--include-untimed`: When timestamp-prefixing is active, files with no timestamp are **skipped by default** (they tend to be empty/useless). Pass this flag to include them anyway. Has no effect when `--no-prefix-timestamp` is used.

**Output during execution:**
- Print a header banner: `"Claude Code JSONL to Markdown Converter"` followed by a line of `=` (60 chars).
- Print active settings: days filter (if set), file count, prefix-timestamp yes/no, prefix-project yes/no, output directory or "side-by-side with source files". When prefix-timestamp is active, also print `Include untimed: yes` or `Include untimed: no (skipped by default)`.
- For each file: `[i/total] Converting: <filename>...` then `  -> <output-path>` on success, `  (skipped — no timestamp found)` when skipped, or `  ERROR: <message>` to stderr on failure.
- Print a summary line: `Done: X succeeded, Y failed`. If any files were skipped, insert `, Y skipped` between succeeded and failed counts (omit when zero): e.g. `Done: 5 succeeded, 3 skipped, 0 failed`.
- Exit 1 if any files failed. Skipped files do not count as failures.
- Handle KeyboardInterrupt gracefully (print message to stderr, exit 1).

## Script Structure

Write as a single self-contained `.py` file with a `#!/usr/bin/env python3` shebang. Organize into logical sections with separator comments:
- JSONL discovery (`discover_in_directory`, `resolve_paths`)
- Content extraction (`extract_user_content`, `extract_assistant_content`)
- Helpers (`de_escape_content`, `get_earliest_timestamp`, `format_timestamp_local`, `extract_custom_title`, `find_session_subdir_files`)
- Parsing and markdown generation (`parse_jsonl_to_messages`, `generate_markdown`)
- Conversion (`convert_jsonl_to_markdown`, `convert_batch`)
- CLI (`main`)

Include `if __name__ == '__main__': main()` at the bottom.
