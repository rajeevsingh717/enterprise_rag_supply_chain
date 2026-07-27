"""Tests for the ingestion validation + routing logic (Phase 1 "done when" check).

No live Kafka/Redpanda needed: DocumentProducer is swapped for an in-memory
stub so these run under plain `pytest`.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from rag_supply_chain.ingestion.validation import validate_file
from rag_supply_chain.ingestion.watcher import process_file


class StubProducer:
    def __init__(self) -> None:
        self.raw: list[tuple[str, Path, str | None]] = []
        self.dlq: list[tuple[str, Path, str | None, str]] = []

    def produce_raw(self, doc_id, path, mime_type):
        self.raw.append((doc_id, path, mime_type))

    def produce_dlq(self, doc_id, path, mime_type, reason):
        self.dlq.append((doc_id, path, mime_type, reason))


def _write_valid_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)


def test_valid_pdf_passes_validation(tmp_path: Path) -> None:
    pdf = tmp_path / "good.pdf"
    _write_valid_pdf(pdf)

    result = validate_file(pdf)

    assert result.ok
    assert result.mime_type == "application/pdf"


def test_corrupt_pdf_fails_validation(tmp_path: Path) -> None:
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not actually a pdf")

    result = validate_file(pdf)

    assert not result.ok
    assert result.reason


def test_unsupported_extension_fails_validation(tmp_path: Path) -> None:
    f = tmp_path / "doc.exe"
    f.write_bytes(b"binary")

    result = validate_file(f)

    assert not result.ok


def test_valid_markdown_and_html_pass_validation(tmp_path: Path) -> None:
    md = tmp_path / "notes.md"
    md.write_text("# Title\n\nSome content.")
    html = tmp_path / "page.html"
    html.write_text("<html><body><p>hi</p></body></html>")

    assert validate_file(md).ok
    assert validate_file(html).ok


def test_empty_markdown_fails_validation(tmp_path: Path) -> None:
    md = tmp_path / "empty.md"
    md.write_text("   \n  ")

    result = validate_file(md)

    assert not result.ok


def test_good_and_corrupt_pdf_route_to_different_topics(tmp_path: Path) -> None:
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.pdf"
    _write_valid_pdf(good)
    bad.write_bytes(b"garbage")

    producer = StubProducer()
    process_file(good, producer)
    process_file(bad, producer)

    assert len(producer.raw) == 1
    assert producer.raw[0][1] == good
    assert len(producer.dlq) == 1
    assert producer.dlq[0][1] == bad
    assert producer.dlq[0][3]  # reason is non-empty
