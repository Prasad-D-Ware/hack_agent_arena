"""Pull the single python code block out of an LLM reply."""
import re

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.IGNORECASE)
# Detects responses cut off mid-token-budget: open string literal or open bracket
def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models (e.g. qwen3)."""
    return _THINK.sub("", text or "").strip()


def is_truncated(code: str) -> bool:
    """Return True when code looks cut off mid-token-budget.

    Checks two signals on the last non-empty line:
    - Unclosed string literal (odd number of unescaped quote chars of same type)
    - Bare keyword or operator at end with no value (e.g. "for song_id", "x =")
    """
    if not code:
        return False
    last_line = code.rstrip().splitlines()[-1].rstrip()
    if not last_line:
        return False
    # Unclosed string: count unescaped quotes; odd count means open literal
    for q in ('"', "'"):
        # Remove escaped quotes then count remaining
        cleaned = re.sub(r'\\' + q, '', last_line)
        if cleaned.count(q) % 2 != 0:
            return True
    # Line ends with a keyword or operator with no RHS (e.g. "for x", "x =", "return")
    if re.search(r'(?:=|,|and\b|or\b|not\b|return\b)\s*$', last_line):
        return True
    # "for <var>" with no "in" — body is cut off
    if re.match(r'^\s*for\s+\w[\w,\s]*$', last_line) and ' in ' not in last_line:
        return True
    return False


def extract_code(text: str) -> str:
    text = strip_thinking(text)
    matches = [m.strip() for m in _FENCE.findall(text) if m.strip()]
    code = matches[-1] if matches else text.strip()
    while code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*\n?", "", code, count=1).strip()
    while code.endswith("```"):
        code = re.sub(r"\n?```$", "", code, count=1).strip()
    return code
