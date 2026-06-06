from state import Subgoal, StepRecord, error_signature
from state import Blackboard


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


def test_add_step_records_and_counts():
    bb = Blackboard(task_instruction="t")
    sig = bb.add_step(1, "print(1)", "1")
    assert sig is None
    assert bb.interactions_used == 1 and len(bb.steps) == 1


def test_add_step_returns_error_signature():
    bb = Blackboard(task_instruction="t")
    sig = bb.add_step(1, "x", "ValueError: nope 5")
    assert sig == "ValueError: nope #"
    assert bb.steps[-1].error_signature == "ValueError: nope #"


def test_has_repeated_error_true_on_same_signature_twice():
    bb = Blackboard(task_instruction="t")
    bb.add_step(1, "x", "KeyError: 'a'")
    bb.add_step(1, "x", "KeyError: 'a'")
    assert bb.has_repeated_error(1) is True


def test_has_repeated_error_false_for_different_subgoals():
    bb = Blackboard(task_instruction="t")
    bb.add_step(1, "x", "KeyError: 'a'")
    bb.add_step(2, "x", "KeyError: 'a'")
    assert bb.has_repeated_error(1) is False


def test_recent_error_signatures_filters_by_subgoal():
    bb = Blackboard(task_instruction="t")
    bb.add_step(1, "x", "KeyError: 'a'")
    bb.add_step(1, "y", "ok")
    assert bb.recent_error_signatures(1, 2) == ["KeyError: 'a'"]
