"""Cost estimation engine with configurable safety buffer.

The engine estimates the cost of running a task on a given model before the
request is executed. It applies a safety buffer (default 15%) to account for
tokenization variance and output length uncertainty.

The buffer percentage is read from the routing policies at construction time
so it can be changed via policies.yaml without code changes.

Example:
    engine = CostEngine(buffer_pct=0.15)
    estimate = await engine.estimate(model_spec, task_descriptor)
    print(f"Estimated cost: ${estimate.estimated_cost_usd:.4f}")
"""

from __future__ import annotations

from tidus.cost.tokenizers import count_tokens
from tidus.models.cost import CostEstimate
from tidus.models.model_registry import ModelSpec
from tidus.models.task import TaskDescriptor

# Default buffer if not supplied from policies.yaml
DEFAULT_BUFFER_PCT = 0.15


class CostEngine:
    """Estimates request cost before execution, applying a safety buffer."""

    def __init__(self, buffer_pct: float = DEFAULT_BUFFER_PCT) -> None:
        if not (0.0 <= buffer_pct <= 1.0):
            raise ValueError(f"buffer_pct must be in [0, 1], got {buffer_pct}")
        self._buffer_pct = buffer_pct

    # ── Public API ────────────────────────────────────────────────────────────

    async def estimate(self, model: ModelSpec, task: TaskDescriptor) -> CostEstimate:
        """Estimate the cost of running task on model.

        Uses the provider-native tokenizer for accurate input token counts,
        then applies the safety buffer to both input and output counts before
        computing the dollar cost.

        Args:
            model: The candidate ModelSpec to price.
            task:  The TaskDescriptor containing messages and output estimate.

        Returns:
            CostEstimate with raw and buffered counts and the dollar figure.
        """
        raw_input = await count_tokens(model, task.messages)
        raw_output = task.estimated_output_tokens

        buffered_input = int(raw_input * (1.0 + self._buffer_pct))
        buffered_output = int(raw_output * (1.0 + self._buffer_pct))

        cost_usd = (
            buffered_input / 1000.0 * model.input_price
            + buffered_output / 1000.0 * model.output_price
        )

        return CostEstimate(
            model_id=model.model_id,
            raw_input_tokens=raw_input,
            raw_output_tokens=raw_output,
            buffered_input_tokens=buffered_input,
            buffered_output_tokens=buffered_output,
            estimated_cost_usd=cost_usd,
            buffer_pct=self._buffer_pct,
        )

    def estimate_from_counts(
        self,
        model: ModelSpec,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate:
        """Synchronous cost estimate from pre-counted token counts.

        Used after execution when actual counts are known (no buffer needed
        but kept here for consistency — pass buffer_pct=0 for exact costs).

        Example:
            actual = engine.estimate_from_counts(spec, actual_in, actual_out)
        """
        buffered_input = int(input_tokens * (1.0 + self._buffer_pct))
        buffered_output = int(output_tokens * (1.0 + self._buffer_pct))

        cost_usd = (
            buffered_input / 1000.0 * model.input_price
            + buffered_output / 1000.0 * model.output_price
        )

        return CostEstimate(
            model_id=model.model_id,
            raw_input_tokens=input_tokens,
            raw_output_tokens=output_tokens,
            buffered_input_tokens=buffered_input,
            buffered_output_tokens=buffered_output,
            estimated_cost_usd=cost_usd,
            buffer_pct=self._buffer_pct,
        )
