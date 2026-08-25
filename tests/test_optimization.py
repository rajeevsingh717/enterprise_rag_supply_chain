from __future__ import annotations

from rag_supply_chain.optimization.normalization import (
    measure_normalization,
    normalize_text,
)


def test_normalize_text_removes_only_known_boilerplate_and_whitespace() -> None:
    text = "Header\n\nPage 2 of 9\n---\n  Important   content.\n"
    assert normalize_text(text) == "Header Important content."


def test_normalization_measurement_reports_observed_result_not_target() -> None:
    result = measure_normalization(["word    word"], token_count=len)
    assert result.before_tokens == 12
    assert result.after_tokens == 9
    assert result.removed_tokens == 3
    assert result.reduction_fraction == 0.25
    assert result.target_met is True


def test_normalization_preserves_meaningful_tokens_in_order() -> None:
    normalized = normalize_text("Alpha\nPage 1 / 2\n beta,  GAMMA.")
    assert normalized.lower().replace(",", "").replace(".", "").split() == [
        "alpha",
        "beta",
        "gamma",
    ]
