"""OpenRouter LLM access with per-role model routing and retry/backoff."""
import os
import time

from openai import OpenAI

_DEFAULT_CLIENT = None


def _client() -> OpenAI:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    return _DEFAULT_CLIENT


def _model_for(role: str) -> str:
    """MODEL_<ROLE> overrides MODEL; MODEL defaults to a strong slug."""
    default = os.environ.get("MODEL", "anthropic/claude-opus-4")
    return os.environ.get(f"MODEL_{role.upper()}", default)


def call_llm(role, messages, system=None, *, client=None, sleep_fn=time.sleep,
             max_tokens=1500, temperature=0.0, max_retries=4) -> str:
    client = client or _client()
    model = _model_for(role)
    fallback = os.environ.get("MODEL_FALLBACK")
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # transient: 429/5xx/network
            last_err = e
            if fallback and attempt == max_retries - 2:
                model = fallback   # last-ditch: swap to the fallback model
            sleep_fn(min(2 ** attempt, 8))
    raise last_err
