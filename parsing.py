"""Pull the single python code block out of an LLM reply."""
import re

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _FENCE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()
