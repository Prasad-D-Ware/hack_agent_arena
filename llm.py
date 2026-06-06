"""OpenAI-compatible LLM access with per-role model routing and retry/backoff."""
import os
import time

from openai import OpenAI

_DEFAULT_CLIENT = None
_DEFAULT_CLIENT_CONFIG = None


def _provider_config() -> tuple[str, str | None, str]:
    provider = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()
    if provider == "groq":
        return (
            os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            os.environ.get("GROQ_API_KEY"),
            "llama-3.3-70b-versatile",
        )
    return (
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        os.environ.get("OPENROUTER_API_KEY"),
        "meta-llama/llama-3.3-70b-instruct:free",
    )


def _float_env(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _verbose() -> bool:
    return os.environ.get("LLM_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _client() -> OpenAI:
    global _DEFAULT_CLIENT, _DEFAULT_CLIENT_CONFIG
    base_url, api_key, _ = _provider_config()
    timeout = _float_env("LLM_TIMEOUT", "60")
    config = (base_url, api_key, timeout)
    if _DEFAULT_CLIENT is None or _DEFAULT_CLIENT_CONFIG != config:
        _DEFAULT_CLIENT = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        _DEFAULT_CLIENT_CONFIG = config
    return _DEFAULT_CLIENT


def _model_for(role: str) -> str:
    """MODEL_<ROLE> overrides MODEL; MODEL defaults depend on LLM_PROVIDER."""
    _, _, provider_default = _provider_config()
    default = os.environ.get("MODEL", provider_default)
    return os.environ.get(f"MODEL_{role.upper()}", default)


def call_llm(role, messages, system=None, *, client=None, sleep_fn=time.sleep,
             max_tokens=4000, temperature=0.0, max_retries=4) -> str:
    client = client or _client()
    model = _model_for(role)
    fallback = os.environ.get("MODEL_FALLBACK")
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    last_err = None
    for attempt in range(max_retries):
        started = time.monotonic()
        try:
            if _verbose():
                print(f"  [llm] {role} -> {model} (attempt {attempt + 1}/{max_retries})", flush=True)
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                max_tokens=max_tokens, temperature=temperature,
            )
            if _verbose():
                print(f"  [llm] {role} <- ok ({time.monotonic() - started:.1f}s)", flush=True)
            return resp.choices[0].message.content or ""
        except Exception as e:  # transient: 429/5xx/network
            last_err = e
            if _verbose():
                print(f"  [llm] {role} <- error ({time.monotonic() - started:.1f}s): {e}", flush=True)
            if fallback and model != fallback:
                model = fallback
                if _verbose():
                    print(f"  [llm] {role} switching to fallback model {model}", flush=True)
            elif fallback and attempt == max_retries - 2:
                model = fallback   # last-ditch: swap to the fallback model
            sleep_fn(min(2 ** attempt, 8))
    raise last_err
