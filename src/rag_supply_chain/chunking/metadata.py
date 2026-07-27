"""Mandatory metadata extraction for ingested documents (§2.B / §3.A).

`doc_id`, `source_uri`, and `timestamp` come from the ingestion event and are
never overridden. `version` and `title` are the document's own claims about
itself: tried first via structured extraction (Markdown YAML-style
frontmatter, HTML `<meta>`/`<title>` tags, PDF document info), then via a
regex scan of the raw text ("Version: 2.3" style lines) if that comes up
empty, then a hardcoded default — never a hard failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_REGEX_VERSION = re.compile(r"(?im)^\s*(?:doc[_ ]?)?version\s*[:=]\s*(.+)$")
_REGEX_TITLE = re.compile(r"(?im)^\s*title\s*[:=]\s*(.+)$")
_DEFAULT_VERSION = "1"


@dataclass
class DocumentMetadata:
    doc_id: str
    source_uri: str
    timestamp: float
    version: str
    title: str | None = None


def extract_metadata(
    *,
    doc_id: str,
    source_uri: str,
    timestamp: float,
    mime_type: str,
    path: Path,
    text: str,
) -> DocumentMetadata:
    version: str | None = None
    title: str | None = None

    if mime_type == "text/markdown":
        version, title = _from_markdown_frontmatter(text)
    elif mime_type == "text/html":
        version, title = _from_html_meta(text)
    elif mime_type == "application/pdf":
        version, title = _from_pdf_info(path)

    if version is None:
        version = _regex_fallback(_REGEX_VERSION, text)
    if title is None:
        title = _regex_fallback(_REGEX_TITLE, text)

    return DocumentMetadata(
        doc_id=doc_id,
        source_uri=source_uri,
        timestamp=timestamp,
        version=version or _DEFAULT_VERSION,
        title=title,
    )


def _kv(block: str, key: str) -> str | None:
    m = re.search(rf"(?im)^\s*{key}\s*:\s*(.+)$", block)
    return m.group(1).strip().strip("\"'") if m else None


def _from_markdown_frontmatter(text: str) -> tuple[str | None, str | None]:
    m = FRONTMATTER.match(text)
    if not m:
        return None, None
    block = m.group(1)
    return _kv(block, "version"), _kv(block, "title")


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML-style frontmatter block, if present.

    The chunker should never see frontmatter as document content — it's
    metadata, already consumed by `extract_metadata`, not something a
    reader (or the LLM) should see injected into a retrieved chunk.
    """
    return FRONTMATTER.sub("", text, count=1)


def _from_html_meta(text: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(text, "html.parser")
    version = None
    tag = soup.find("meta", attrs={"name": "version"})
    if tag and tag.get("content"):
        version = tag["content"].strip()
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    return version, title


def _from_pdf_info(path: Path) -> tuple[str | None, str | None]:
    reader = PdfReader(str(path))
    info = reader.metadata
    title = info.title.strip() if info and info.title else None
    return None, title  # pypdf's info dict has no standard "version" field


def _regex_fallback(pattern: re.Pattern[str], text: str) -> str | None:
    # only scan the first ~40 lines: a "Version:" line found deep in prose
    # (e.g. inside a code sample) is a false positive, not real metadata.
    head = "\n".join(text.splitlines()[:40])
    m = pattern.search(head)
    return m.group(1).strip() if m else None
