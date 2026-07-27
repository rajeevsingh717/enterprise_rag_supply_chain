"""Document validation: MIME type + corruption checks for the ingestion engine.

Only files that pass validation are eligible for the `documents.raw` topic;
everything else routes to `documents.dlq` with a reason (§2.B ingestion rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}


@dataclass
class ValidationResult:
    ok: bool
    mime_type: str | None
    reason: str | None = None


def validate_file(path: Path) -> ValidationResult:
    mime_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if mime_type is None:
        return ValidationResult(False, None, f"unsupported extension {path.suffix!r}")

    try:
        if mime_type == "application/pdf":
            _validate_pdf(path)
        elif mime_type == "text/markdown":
            _validate_text(path)
        elif mime_type == "text/html":
            _validate_html(path)
    except Exception as e:  # noqa: BLE001 - any parse failure means corrupt/invalid
        return ValidationResult(False, mime_type, str(e).splitlines()[0])

    return ValidationResult(True, mime_type)


def _validate_pdf(path: Path) -> None:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("encrypted PDF")
    if len(reader.pages) == 0:
        raise ValueError("PDF has zero pages")
    reader.pages[0].extract_text()  # forces a read of the page's content stream


def _validate_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("empty markdown file")


def _validate_html(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(text, "html.parser")
    if not soup.find():
        raise ValueError("no HTML tags found")
