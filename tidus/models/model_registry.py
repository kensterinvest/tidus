from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class ModelTier(int, Enum):
    """Cost tiers — lower number = more expensive, more capable."""
    premium = 1
    mid = 2
    economy = 3
    local = 4


class TokenizerType(str, Enum):
    tiktoken_cl100k = "tiktoken_cl100k"   # GPT-3.5, DeepSeek, Grok, Kimi
    tiktoken_o200k = "tiktoken_o200k"     # GPT-4o, GPT-4.1, GPT-5
    anthropic = "anthropic"               # Claude (count_tokens API endpoint)
    sentencepiece = "sentencepiece"       # Mistral family
    google = "google"                     # Gemini (generativeai SDK)
    ollama = "ollama"                     # Local models via Ollama tokenize endpoint


class Capability(str, Enum):
    chat = "chat"
    code = "code"
    reasoning = "reasoning"
    extraction = "extraction"
    classification = "classification"
    summarization = "summarization"
    creative = "creative"
    multimodal = "multimodal"
    long_context = "long_context"
    agents = "agents"


class ModelSpec(BaseModel):
    """Full specification for a single AI model in the registry.

    Example:
        spec = ModelSpec(
            model_id="claude-haiku-4-5",
            vendor="anthropic",
            tier=ModelTier.economy,
            max_context=200000,
            input_price=0.0008,
            output_price=0.004,
            tokenizer=TokenizerType.anthropic,
            capabilities=[Capability.chat, Capability.extraction],
            min_complexity="simple",
            max_complexity="moderate",
        )
    """

    model_id: str = Field(..., description="Unique identifier matching adapter lookup")
    display_name: str = Field("", description="Human-readable name shown in dashboard")
    vendor: str = Field(..., description="Matches AbstractModelAdapter.vendor constant")
    tier: ModelTier

    # Pricing (USD per 1K tokens; 0.0 for local)
    max_context: int = Field(..., gt=0, description="Maximum context window in tokens")
    input_price: float = Field(..., ge=0.0, description="USD per 1K input tokens")
    output_price: float = Field(..., ge=0.0, description="USD per 1K output tokens")

    tokenizer: TokenizerType
    latency_p50_ms: int = Field(1000, gt=0, description="Baseline median latency; updated by health probe")

    capabilities: list[Capability] = Field(default_factory=list)
    min_complexity: str = Field("simple", description="Minimum complexity this model should handle")
    max_complexity: str = Field("critical", description="Maximum complexity this model should handle")

    is_local: bool = Field(False, description="True = self-hosted, no API cost")
    enabled: bool = Field(True, description="False = excluded from routing (auto-set on probe failure)")
    deprecated: bool = Field(False, description="True = vendor EOL; route via fallbacks only")

    fallbacks: list[str] = Field(
        default_factory=list,
        description="Ordered fallback model_ids: cheaper → mid → premium → local",
    )

    last_price_check: date = Field(default_factory=date.today, description="Date of last price sync")
    last_health_check: datetime | None = None

    @property
    def is_free(self) -> bool:
        return self.input_price == 0.0 and self.output_price == 0.0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Quick cost estimate without safety buffer."""
        return (input_tokens / 1000 * self.input_price) + (output_tokens / 1000 * self.output_price)
