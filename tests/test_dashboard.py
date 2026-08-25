from __future__ import annotations

import json
from pathlib import Path

from rag_supply_chain.dashboard import build_dashboard_data


def test_dashboard_uses_committed_baseline_and_labels_sample_query(tmp_path: Path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "sample_docs").mkdir()
    (tmp_path / "eval/results-local.json").write_text(
        json.dumps({"case_count": 1, "metrics": {"retrieval_hit": 1.0}})
    )
    (tmp_path / "sample_docs/system_design.md").write_text("Page 1 of 2\nUseful content")
    checks = {"Qdrant": lambda: (True, "reachable, 1 collection(s)")}

    data = build_dashboard_data(tmp_path, checks)

    assert data["evaluation"]["case_count"] == 1
    assert data["sample_query"]["label"].startswith("Illustrative sample")
    assert data["services"] == [
        {"name": "Qdrant", "ok": True, "detail": "reachable, 1 collection(s)"}
    ]
    assert data["normalization"]["removed_tokens"] > 0
