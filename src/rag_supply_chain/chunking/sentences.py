"""Lightweight sentence splitter — regex-based, no heavy NLP dependency.

Good enough for technical prose: splits on '.', '!', '?' followed by
whitespace and an uppercase letter, digit, or quote. Swapping in a proper
sentence tokenizer later is a drop-in replacement for this function.
"""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]
