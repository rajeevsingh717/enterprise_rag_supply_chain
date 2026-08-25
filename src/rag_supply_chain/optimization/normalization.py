"""Safe normalization immediately before embedding.

Only formatting and unambiguous page/separator boilerplate are removed. A
content-token invariant prevents accidental loss of meaningful text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

_PAGE_MARKER = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of|/)\s*\d+\s*$", re.IGNORECASE)
_SEPARATOR = re.compile(r"^\s*[-_=*]{3,}\s*$")
_CONTENT_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_APPROX_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class NormalizationGuardError(ValueError):
    """Normalization would remove or alter meaningful content."""


@dataclass(frozen=True)
class NormalizationMetrics:
    before_tokens: int
    after_tokens: int
    removed_tokens: int
    reduction_fraction: float
    target_fraction: float
    target_met: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _remove_known_boilerplate(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _PAGE_MARKER.fullmatch(raw_line) or _SEPARATOR.fullmatch(raw_line):
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    without_boilerplate = _remove_known_boilerplate(text)
    expected_tokens = _CONTENT_TOKEN.findall(without_boilerplate.lower())
    normalized = " ".join(without_boilerplate.split())
    if _CONTENT_TOKEN.findall(normalized.lower()) != expected_tokens:
        raise NormalizationGuardError("normalization changed meaningful content tokens")
    return normalized


def approximate_token_count(text: str) -> int:
    return len(_APPROX_TOKEN.findall(text))


def measure_normalization(
    texts: list[str],
    token_count: Callable[[str], int] = approximate_token_count,
    target_fraction: float = 0.15,
) -> NormalizationMetrics:
    before = sum(token_count(text) for text in texts)
    after = sum(token_count(normalize_text(text)) for text in texts)
    removed = before - after
    reduction = removed / before if before else 0.0
    return NormalizationMetrics(before, after, removed, reduction, target_fraction, reduction >= target_fraction)
