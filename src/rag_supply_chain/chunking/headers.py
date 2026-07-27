"""Section-header extraction (§3.A hierarchical_context).

Walks Markdown or HTML source and splits it into `Segment`s of body text,
each tagged with the breadcrumb of section headers active at that point
(e.g. `("System Design", "Networking", "Subnets")`). A header-level jump
(e.g. H1 straight to H3) pads the skipped levels with `""` so breadcrumb
tuples stay positionally meaningful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

_MD_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BODY_TAGS = {"p", "li", "td", "blockquote"}


@dataclass
class Segment:
    text: str
    breadcrumb: tuple[str, ...]


def _push_header(stack: list[str], level: int, title: str) -> None:
    del stack[level - 1 :]
    stack.extend([""] * (level - 1 - len(stack)))
    stack.append(title)


def segment_markdown(text: str) -> list[Segment]:
    stack: list[str] = []
    segments: list[Segment] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            segments.append(Segment(body, tuple(stack)))
        buffer.clear()

    for line in text.splitlines():
        m = _MD_HEADER.match(line)
        if m:
            flush()
            _push_header(stack, len(m.group(1)), m.group(2).strip())
        else:
            buffer.append(line)
    flush()
    return segments


def segment_html(text: str) -> list[Segment]:
    soup = BeautifulSoup(text, "html.parser")
    stack: list[str] = []
    segments: list[Segment] = []

    for el in soup.find_all(True):
        if el.name in _HEADING_TAGS:
            _push_header(stack, _HEADING_TAGS[el.name], el.get_text(strip=True))
        elif el.name in _BODY_TAGS:
            body = el.get_text(" ", strip=True)
            if body:
                segments.append(Segment(body, tuple(stack)))
    return segments
