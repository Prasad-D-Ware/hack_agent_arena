from parsing import extract_code, is_truncated


def test_extracts_fenced_python_block():
    text = "Here you go:\n```python\nprint(1)\n```\nthanks"
    assert extract_code(text) == "print(1)"


def test_extracts_unlabeled_fence():
    text = "```\nx = 2\n```"
    assert extract_code(text) == "x = 2"


def test_no_fence_returns_stripped_text():
    assert extract_code("  print(3)  ") == "print(3)"


def test_strips_accidental_nested_fence():
    text = "```python\n```python\nprint(4)\n```\n```"
    assert extract_code(text) == "print(4)"


def test_strips_think_block_before_code():
    text = "<think>lots of reasoning here</think>\n```python\nprint(5)\n```"
    assert extract_code(text) == "print(5)"


def test_think_only_reply_returns_empty():
    text = "<think>I need to figure out what to do next...</think>"
    assert extract_code(text) == ""


def test_is_truncated_unclosed_string():
    code = 'result = apis.spotify.show_song_library(access_token="eyJhbGci'
    assert is_truncated(code) is True


def test_is_truncated_incomplete_for_loop():
    code = "for song_id"
    assert is_truncated(code) is True


def test_is_truncated_complete_code():
    assert is_truncated('print("hello")') is False
    assert is_truncated("x = 1\nprint(x)") is False


def test_is_truncated_empty():
    assert is_truncated("") is False
