# Adapters

Tidus uses a vendor-agnostic adapter layer so every AI provider is called through the same interface. Adding a new vendor requires one file and a YAML entry — zero changes to the router, API, or cost engine.

## Built-in Adapters (v0.1.0)

| Adapter | Vendor | Auth Env Var | Token Counting | Models |
|---|---|---|---|---|
| `openai_adapter.py` | OpenAI | `OPENAI_API_KEY` | `tiktoken` (local) | gpt-4.1, gpt-4o-mini, o3, o4-mini, gpt-oss-120b, gpt-5-codex, codex-mini-latest |
| `anthropic_adapter.py` | Anthropic | `ANTHROPIC_API_KEY` | Anthropic count_tokens API | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 |
| `google_adapter.py` | Google | `GOOGLE_API_KEY` | google-generativeai SDK | gemini-3.1-pro, gemini-3.1-flash, gemini-nano |
| `mistral_adapter.py` | Mistral | `MISTRAL_API_KEY` | sentencepiece (local) | mistral-large-3, mistral-medium, mistral-small, codestral, devstral |
| `deepseek_adapter.py` | DeepSeek | `DEEPSEEK_API_KEY` | tiktoken cl100k (local) | deepseek-r1, deepseek-v3 |
| `xai_adapter.py` | xAI | `XAI_API_KEY` | tiktoken cl100k (local) | grok-3 |
| `moonshot_adapter.py` | Moonshot AI | `MOONSHOT_API_KEY` | tiktoken cl100k (local) | kimi-k2.5 |
| `ollama_adapter.py` | Ollama (local) | none | Ollama tokenize endpoint | llama4-maverick, llama4-scout, mistral-small-ollama, phi-4-ollama, gemma-3-ollama |

## Adapter Interface

All adapters implement `AbstractModelAdapter`:

```python
class AbstractModelAdapter(ABC):
    vendor: str                        # e.g. "openai", "anthropic"
    supported_model_ids: list[str]

    async def complete(self, model_id: str, task: TaskDescriptor) -> AdapterResponse: ...
    async def stream_complete(self, model_id: str, task: TaskDescriptor) -> AsyncIterator[str]: ...
    async def health_check(self, model_id: str) -> bool: ...
    async def count_tokens(self, model_id: str, messages: list[dict]) -> int: ...
```

`AdapterResponse` fields: `model_id`, `content`, `input_tokens`, `output_tokens`, `latency_ms`, `finish_reason`.

## Adding a New Adapter

### 1. Create the adapter file

```python
# tidus/adapters/myvendor_adapter.py
from tidus.adapters.base import AbstractModelAdapter, AdapterResponse, register_adapter

@register_adapter
class MyVendorAdapter(AbstractModelAdapter):
    vendor = "myvendor"
    supported_model_ids = ["mymodel-large", "mymodel-small"]

    async def complete(self, model_id: str, task) -> AdapterResponse:
        # Call your vendor's API here
        ...

    async def health_check(self, model_id: str) -> bool:
        # Quick liveness check (e.g. list models endpoint)
        ...

    async def count_tokens(self, model_id: str, messages: list[dict]) -> int:
        # Use tiktoken or your vendor's tokenizer
        ...
```

The `@register_adapter` decorator automatically registers the adapter by `vendor` string when the file is imported.

### 2. Import in adapter_factory.py

```python
# tidus/adapters/adapter_factory.py
from tidus.adapters import myvendor_adapter  # triggers @register_adapter
```

### 3. Add models to config/models.yaml

```yaml
- model_id: "mymodel-large"
  vendor: "myvendor"
  tier: 2
  max_context: 128000
  input_price: 0.001
  output_price: 0.003
  tokenizer: "tiktoken_cl100k"
  capabilities: [chat, code, reasoning]
  min_complexity: moderate
  max_complexity: complex
  enabled: true
```

That's it. The router, cost engine, and dashboard all pick up the new vendor automatically.

## OpenAI-Compatible Adapters

DeepSeek, xAI, and Moonshot all use the OpenAI Python SDK with a custom `base_url`. This pattern works for any vendor with an OpenAI-compatible API:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
)
```

## Graceful Degradation

If a vendor SDK is not installed (e.g. `mistralai` missing), the adapter raises `RuntimeError` at call time — not at import time. Tidus logs the error and attempts fallbacks. You can run Tidus with only the SDKs you need.

## Health Probes

Every 5 minutes, `HealthProbe` calls `adapter.health_check(model_id)` for all enabled models. After 3 consecutive failures, the model is auto-disabled (`enabled=False`) and a `model_auto_disabled` log event is emitted.

The health probe results are visible on the dashboard Registry Health panel.
