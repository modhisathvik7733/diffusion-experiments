"""Post-processing for DreamOn raw output.

DreamOn produces full-sequence output that may contain:
  - the original prefix verbatim (we want only the middle)
  - the original suffix verbatim
  - <|endoftext|> / <|im_end|> markers when generation completes
  - trailing runs of `!` (decoded from unfilled mask positions)
  - leftover <|mask|> / <|expand|> / <|beginoftext|> literal strings

This module strips all of the above and returns just the completion text.
"""
from __future__ import annotations

import re

_EOS_MARKERS = ("<|endoftext|>", "<|eos|>", "<|im_end|>")
_SPECIAL_LITERALS = ("<|beginoftext|>", "<|mask|>", "<|expand|>", "<|delete|>")
_TRAILING_BANG_RE = re.compile(r"!{4,}\s*$")


def clean_completion(full: str, prefix: str, suffix: str, language: str | None = None) -> str:
    """Extract the middle completion from a DreamOn full-sequence output."""
    # 1. Strip from first EOS marker onward
    text = full
    for marker in _EOS_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]

    # 2. Remove leftover special-token literals
    for special in _SPECIAL_LITERALS:
        text = text.replace(special, "")

    # 3. Locate the prefix and suffix to slice out the middle
    if prefix and prefix in text:
        text = text[text.index(prefix) + len(prefix) :]
    suffix_stripped = suffix.lstrip("\n") if suffix else ""
    marker = suffix_stripped[:30].strip() if suffix_stripped.strip() else ""
    if marker and marker in text:
        text = text.split(marker, 1)[0]

    # 4. Strip trailing `!!!!` runs (unfilled mask leftovers)
    text = _TRAILING_BANG_RE.sub("", text)

    # 5. Single-line mode: truncate at first newline
    if language == "line-mode" and "\n" in text:
        text = text.split("\n", 1)[0]

    return text.rstrip()
