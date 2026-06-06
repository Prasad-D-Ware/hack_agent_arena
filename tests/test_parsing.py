from parsing import extract_code


def test_extracts_fenced_python_block():
    text = "Here you go:\n```python\nprint(1)\n```\nthanks"
    assert extract_code(text) == "print(1)"


def test_extracts_unlabeled_fence():
    text = "```\nx = 2\n```"
    assert extract_code(text) == "x = 2"


def test_no_fence_returns_stripped_text():
    assert extract_code("  print(3)  ") == "print(3)"
