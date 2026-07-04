#!/usr/bin/env python3
"""
Extract JSON from fenced code blocks in text.

Parses text (typically LLM output) and extracts JSON content from
fenced code blocks (```json ... ```) or raw JSON arrays/objects.

Usage:
    python extract_json.py --input /path/to/text_file
    echo "```json\n[{\"id\": \"test\"}]\n```" | python extract_json.py

Output:
    Parsed JSON to stdout.

Logic ported from: Fuser/subgraph_extractor.py (_extract_json_block)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_JSON_BLOCK_RE = re.compile(
    r"^```[ \t]*(json)?[ \t]*\n([\s\S]*?)^```[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


def extract_json_block(text: str) -> str:
    """Extract the last fenced JSON block or fallback to best-effort slice."""
    matches = list(_JSON_BLOCK_RE.finditer(text))
    chosen: re.Match[str] | None = None
    for m in reversed(matches):
        lang = (m.group(1) or "").strip().lower()
        if lang == "json":
            chosen = m
            break
    if chosen is None and matches:
        chosen = matches[-1]
    if chosen is not None:
        return chosen.group(2)
    # Fallback: attempt to slice between first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    # Try object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def extract_python_block(text: str) -> str:
    """Extract the last fenced Python code block."""
    pattern = re.compile(
        r"^```[ \t]*python[ \t]*\n([\s\S]*?)^```[ \t]*$",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group(1)
    # Fallback: generic code block
    pattern2 = re.compile(
        r"^```[ \t]*\n([\s\S]*?)^```[ \t]*$",
        re.MULTILINE,
    )
    matches2 = list(pattern2.finditer(text))
    if matches2:
        return matches2[-1].group(1)
    return ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract JSON from fenced code blocks in text"
    )
    p.add_argument(
        "--input",
        default=None,
        help="Path to text file (reads from stdin if not provided)",
    )
    p.add_argument(
        "--type",
        choices=["json", "python"],
        default="json",
        help="Type of content to extract (default: json)",
    )
    args = p.parse_args(argv)

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    if args.type == "python":
        result = extract_python_block(text)
        if result:
            print(result)
            return 0
        print("No Python code block found", file=sys.stderr)
        return 1

    raw = extract_json_block(text)
    if not raw:
        print("No JSON block found in input", file=sys.stderr)
        return 1

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}", file=sys.stderr)
        print(f"Raw extracted text:\n{raw[:500]}", file=sys.stderr)
        return 1

    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
