"""Provider-reported token usage and explicitly configured cost estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


def usage_from_response(
    response: object,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> TokenUsage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return TokenUsage()
    input_tokens = int(getattr(raw, "input_tokens", 0) or 0)
    output_tokens = int(getattr(raw, "output_tokens", 0) or 0)
    cache_creation = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
    cost = None
    if (
        input_cost_per_million is not None
        and output_cost_per_million is not None
        and cache_creation == 0
        and cache_read == 0
    ):
        cost = (
            input_tokens * input_cost_per_million + output_tokens * output_cost_per_million
        ) / 1_000_000
    return TokenUsage(input_tokens, output_tokens, cache_creation, cache_read, cost)
