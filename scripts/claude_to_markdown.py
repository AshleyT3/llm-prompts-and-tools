#!/usr/bin/env python3
"""Convert Claude Code JSONL conversation files to readable Markdown.
Copyright (c) 2026 - Ashley R. Thomas
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# JSONL discovery
# ---------------------------------------------------------------------------

def discover_in_directory(root: Path) -> List[Path]:
    """
    Recursively find project directories and return their top-level session
    JSONL files.

    A "project directory" is the first directory in each branch of the tree
    that contains *.jsonl files.  Once found, all *.jsonl at that level are
    collected and no further descent occurs in that branch (projects don't
    nest).  Sibling branches continue to be searched.
    """
    jsonl_here = sorted(root.glob('*.jsonl'))

    if jsonl_here:
        # This is a project directory — return all session jsonl files here
        return jsonl_here

    # No jsonl at this level — recurse into subdirectories
    result: List[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            result.extend(discover_in_directory(child))
    return result


def resolve_paths(paths: List[Path]) -> List[Path]:
    """
    Resolve CLI paths to a list of top-level session JSONL files.

    - Explicit .jsonl file → just that file
    - Directory → recursive project-directory discovery
    """
    result: List[Path] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            if path.suffix == '.jsonl':
                result.append(path)
        elif path.is_dir():
            result.extend(discover_in_directory(path))

    # Deduplicate while preserving order
    seen = set()
    deduped: List[Path] = []
    for p in result:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_user_content(message_data: Dict) -> Optional[str]:
    """Extract text content from user messages."""
    content = message_data.get('content')
    if not content:
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'tool_result':
                    continue
                if 'text' in item:
                    parts.append(item['text'])
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return '\n'.join(parts)

    return None


def extract_assistant_content(message_data: Dict) -> Optional[str]:
    """Extract text content from assistant messages (skip thinking/tool_use)."""
    content = message_data.get('content')
    if not content:
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text = item.get('text', '')
                if text:
                    parts.append(text)
        if parts:
            return '\n'.join(parts)

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def de_escape_content(content: str) -> str:
    """De-escape JSON-escaped markdown."""
    content = content.replace('\\\\', '\\')
    content = content.replace('\\"', '"')
    content = content.replace('\\`', '`')
    return content


def get_earliest_timestamp(jsonl_path: Path) -> Optional[str]:
    """Return earliest timestamp in a JSONL file formatted as YYYYMMDD-hhmm."""
    earliest = None

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = data.get('timestamp')
            if ts and (earliest is None or ts < earliest):
                earliest = ts

    if earliest is None:
        return None

    ts_str = earliest.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime('%Y%m%d-%H%M')
    except (ValueError, AttributeError):
        return None


def format_timestamp_local(iso_timestamp: str) -> str:
    """Convert ISO8601 UTC timestamp to local time as YYYYMMDD-HHMMSS."""
    if not iso_timestamp:
        return ''
    ts_str = iso_timestamp.replace('Z', '+00:00')
    try:
        dt_utc = datetime.fromisoformat(ts_str)
        dt_local = dt_utc.astimezone()
        return dt_local.strftime('%Y%m%d-%H%M%S')
    except (ValueError, AttributeError):
        return ''


def extract_custom_title(jsonl_path: Path) -> Optional[str]:
    """Scan a JSONL file for a custom-title entry and return its value."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get('type') == 'custom-title':
                return data.get('customTitle')
    return None


def find_session_subdir_files(session_jsonl_path: Path) -> List[Path]:
    """Find all JSONL files in the session-specific subdirectory."""
    session_dir = session_jsonl_path.parent / session_jsonl_path.stem
    if session_dir.is_dir():
        return sorted(session_dir.glob('**/*.jsonl'))
    return []


# ---------------------------------------------------------------------------
# Parsing and markdown generation
# ---------------------------------------------------------------------------

def parse_jsonl_to_messages(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file and extract user/assistant messages sorted by timestamp."""
    messages = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num} in {jsonl_path}: {e}",
                      file=sys.stderr)
                continue

            msg_type = data.get('type')
            if msg_type not in ('user', 'assistant'):
                continue

            message_data = data.get('message', {})
            role = message_data.get('role')
            timestamp = data.get('timestamp', '')

            if role == 'user':
                content = extract_user_content(message_data)
            elif role == 'assistant':
                content = extract_assistant_content(message_data)
            else:
                content = None

            if not content:
                continue

            messages.append({
                'role': role,
                'content': content,
                'timestamp': timestamp,
            })

    messages.sort(key=lambda x: x['timestamp'])
    return messages


def generate_markdown(messages: List[Dict[str, Any]],
                      session_id: str = '',
                      custom_title: Optional[str] = None) -> str:
    """Generate markdown from messages list."""
    sections = []

    if session_id:
        sections.append(f"Session ID: {session_id}\n")
        title = custom_title if custom_title else '<not-specified>'
        sections.append(f"Title: {title}\n\n")

    counter = 0
    for message in messages:
        role = message['role']
        content = de_escape_content(message['content'])
        ts_formatted = format_timestamp_local(message.get('timestamp', ''))
        ts_suffix = f" {ts_formatted}" if ts_formatted else ''

        if role == 'user':
            counter += 1
            sections.append(f"# Prompt {counter}{ts_suffix}\n\n{content}\n\n")
        elif role == 'assistant':
            sections.append(f"# Response {counter}{ts_suffix}\n\n{content}\n\n")

    return ''.join(sections)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_jsonl_to_markdown(session_jsonl_path: Path,
                              output_dir: Optional[Path] = None,
                              prefix_timestamp: bool = True,
                              prefix_project: bool = True,
                              include_untimed: bool = False) -> Optional[Path]:
    """
    Convert a top-level session JSONL file (and its session subdirectory)
    to Markdown.
    """
    if not session_jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {session_jsonl_path}")

    # Parse messages from main session file
    messages = parse_jsonl_to_messages(session_jsonl_path)

    # Find and merge messages from session-specific subdirectory
    subdir_files = find_session_subdir_files(session_jsonl_path)
    for sub_path in subdir_files:
        sub_messages = parse_jsonl_to_messages(sub_path)
        messages.extend(sub_messages)

    # Re-sort after merging
    messages.sort(key=lambda x: x['timestamp'])

    # Extract metadata
    session_id = session_jsonl_path.stem
    custom_title = extract_custom_title(session_jsonl_path)

    # Generate markdown
    markdown = generate_markdown(messages, session_id=session_id,
                                 custom_title=custom_title)

    # Build output filename with optional prefixes
    filename = session_jsonl_path.name
    if prefix_project:
        project_dir_name = session_jsonl_path.parent.name
        filename = f"{project_dir_name}-{filename}"
    write_file = True
    if prefix_timestamp:
        ts_prefix = get_earliest_timestamp(session_jsonl_path)
        if ts_prefix:
            filename = f"{ts_prefix}-{filename}"
        elif not include_untimed:
            write_file = False

    if not write_file:
        return None

    # Determine output path
    if output_dir:
        output_path = output_dir / f"{filename}.md"
    else:
        output_path = session_jsonl_path.parent / f"{filename}.md"

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    return output_path


def convert_batch(jsonl_paths: List[Path],
                  output_dir: Optional[Path] = None,
                  prefix_timestamp: bool = True,
                  prefix_project: bool = True,
                  include_untimed: bool = False,
                  verbose: bool = True) -> Tuple[int, int, int]:
    """Convert multiple JSONL files to Markdown.

    Returns (success, skipped, failed).
    """
    success = 0
    skipped = 0
    failed = 0
    total = len(jsonl_paths)

    for i, jsonl_path in enumerate(jsonl_paths, 1):
        if verbose:
            print(f"[{i}/{total}] Converting: {jsonl_path.name}...")

        try:
            output_path = convert_jsonl_to_markdown(
                jsonl_path, output_dir,
                prefix_timestamp=prefix_timestamp,
                prefix_project=prefix_project,
                include_untimed=include_untimed,
            )
            if output_path is None:
                skipped += 1
                if verbose:
                    print(f"  (skipped — no timestamp found)")
            else:
                success += 1
                if verbose:
                    print(f"  -> {output_path}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}", file=sys.stderr)

    return success, skipped, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='claude_to_markdown',
        description='Convert Claude Code JSONL conversation files to Markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all projects found under a directory tree
  %(prog)s ~/claude-projects/ -o ./output/

  # Convert a single session file
  %(prog)s session.jsonl -o ./output/

  # Disable timestamp and project prefixes
  %(prog)s ~/projects/ --no-prefix-timestamp --no-prefix-project

  # Only files modified in the last 7 days
  %(prog)s ~/projects/ --days 7 -o ./output/
""",
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='JSONL files or directories to convert (directories are searched recursively)',
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory (default: place .md files next to source)',
    )
    parser.add_argument(
        '--no-prefix-timestamp',
        action='store_true',
        default=False,
        help='Disable timestamp prefix on output filenames (enabled by default)',
    )
    parser.add_argument(
        '--no-prefix-project',
        action='store_true',
        default=False,
        help='Disable project-name prefix on output filenames (enabled by default)',
    )
    parser.add_argument(
        '--days',
        type=int,
        metavar='N',
        help='Only process .jsonl files modified within the last N days',
    )
    parser.add_argument(
        '--include-untimed',
        action='store_true',
        default=False,
        help='Include files with no timestamp even when --prefix-timestamp is active '
             '(by default such files are skipped)',
    )

    args = parser.parse_args()

    prefix_timestamp = not args.no_prefix_timestamp
    prefix_project = not args.no_prefix_project
    include_untimed = args.include_untimed

    # Resolve input paths
    input_paths = [Path(p).expanduser().resolve() for p in args.paths]

    try:
        jsonl_files = resolve_paths(input_paths)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not jsonl_files:
        print("Error: No .jsonl files found in specified paths", file=sys.stderr)
        sys.exit(1)

    # Filter by modification time if --days specified
    if args.days is not None:
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=args.days)
        original_count = len(jsonl_files)

        jsonl_files = [
            f for f in jsonl_files
            if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) >= cutoff_time
        ]

        if not jsonl_files:
            print(f"Error: No .jsonl files modified within the last {args.days} day(s)",
                  file=sys.stderr)
            print(f"  (Found {original_count} file(s) total, but none match the time filter)",
                  file=sys.stderr)
            sys.exit(1)

    # Resolve output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("Claude Code JSONL to Markdown Converter")
    print("=" * 60)
    print()
    if args.days is not None:
        print(f"Filter: files modified within the last {args.days} day(s)")
    print(f"Files to convert: {len(jsonl_files)}")
    print(f"Prefix timestamp: {'yes' if prefix_timestamp else 'no'}")
    print(f"Prefix project:   {'yes' if prefix_project else 'no'}")
    if prefix_timestamp:
        print(f"Include untimed:  {'yes' if include_untimed else 'no (skipped by default)'}")
    if output_dir:
        print(f"Output directory:  {output_dir}")
    else:
        print("Output: side-by-side with source files")
    print()

    # Convert
    try:
        success, skipped, failed = convert_batch(
            jsonl_files, output_dir,
            prefix_timestamp=prefix_timestamp,
            prefix_project=prefix_project,
            include_untimed=include_untimed,
        )
    except KeyboardInterrupt:
        print("\n\nConversion interrupted by user", file=sys.stderr)
        sys.exit(1)

    # Summary
    print()
    print("=" * 60)
    skipped_str = f", {skipped} skipped" if skipped else ""
    print(f"Done: {success} succeeded{skipped_str}, {failed} failed")

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
