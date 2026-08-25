from __future__ import annotations

from dataclasses import dataclass

from rag_supply_chain.telemetry.usage import TokenUsage, usage_from_response


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def test_usage_uses_provider_counts_and_explicit_rates() -> None:
    response = type("Response", (), {"usage": Usage(100, 20)})()
    result = usage_from_response(response, 3.0, 15.0)
    assert result == TokenUsage(100, 20, 0, 0, 0.0006)


def test_cost_is_unknown_without_rates_or_with_unpriced_cache_tokens() -> None:
    no_rates = usage_from_response(type("R", (), {"usage": Usage(10, 2)})(), None, None)
    cached = usage_from_response(type("R", (), {"usage": Usage(10, 2, 5)})(), 3.0, 15.0)
    assert no_rates.cost_usd is None
    assert cached.cost_usd is None
