from state import Subgoal, StepRecord, error_signature


def test_subgoal_defaults():
    sg = Subgoal(id=1, description="log into spotify")
    assert sg.status == "pending" and sg.result is None and sg.attempts == 0


def test_steprecord_holds_fields():
    s = StepRecord(subgoal_id=2, code="print(1)", output="1", error_signature=None)
    assert s.subgoal_id == 2 and s.code == "print(1)"


def test_error_signature_detects_and_normalizes():
    out = "Traceback (most recent call last):\n  ...\nValueError: amount 1234 invalid"
    assert error_signature(out) == "ValueError: amount # invalid"


def test_error_signature_picks_last_error_line():
    out = "KeyError: 'a'\nsome text\nTypeError: bad"
    assert error_signature(out) == "TypeError: bad"


def test_error_signature_none_for_plain_output():
    assert error_signature("hello world") is None
    assert error_signature("") is None
