from verifier import verify_subgoal, verify_final, _parse_verdict
from state import Blackboard, Subgoal
from tests.conftest import FakeLLM


def test_parse_verdict_pass():
    assert _parse_verdict("PASS") == (True, "")


def test_parse_verdict_fail_with_reason():
    ok, reason = _parse_verdict("FAIL: token missing")
    assert ok is False and reason == "token missing"


def test_parse_verdict_unknown_is_conservative_fail():
    ok, reason = _parse_verdict("hmm not sure")
    assert ok is False and "hmm" in reason


def test_verify_subgoal_passes():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "log in", result="got token")
    llm = FakeLLM({"verifier": ["PASS"]})
    assert verify_subgoal(sg, bb, llm=llm) == (True, "")


def test_verify_final_fail_feedback_reaches_caller():
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({"verifier": ["FAIL: answer should be a number"]})
    ok, reason = verify_final(bb, "apis.supervisor.complete_task(answer='x')", llm=llm)
    assert ok is False and "number" in reason
    # the completion code is shown to the verifier
    assert "complete_task" in llm.calls[0][1][0]["content"]
