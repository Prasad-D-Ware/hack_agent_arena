from planner import make_plan, replan, _parse_plan
from state import Blackboard


def test_parse_plan_handles_numbered_and_bulleted():
    text = "1. log in\n2) list songs\n- like first\n\nnotes here"
    plan = _parse_plan(text)
    assert [sg.description for sg in plan] == ["log in", "list songs", "like first"]
    assert [sg.id for sg in plan] == [1, 2, 3]


def test_make_plan_uses_llm_output(make_llm=None):
    from tests.conftest import FakeLLM
    bb = Blackboard(task_instruction="play my discover weekly")
    llm = FakeLLM({"planner": ["1. log into spotify\n2. play discover weekly"]})
    plan = make_plan(bb, hints="", llm=llm)
    assert [sg.description for sg in plan] == ["log into spotify", "play discover weekly"]
    # the prompt should carry the task text
    assert "discover weekly" in llm.calls[0][1][0]["content"]


def test_make_plan_falls_back_to_single_subgoal_when_empty():
    from tests.conftest import FakeLLM
    bb = Blackboard(task_instruction="do the thing")
    llm = FakeLLM({"planner": ["sorry, no list"]})
    plan = make_plan(bb, hints="", llm=llm)
    assert len(plan) == 1 and plan[0].description == "do the thing"


def test_replan_renumbers_from_one():
    from tests.conftest import FakeLLM
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({"planner": ["1. retry login\n2. fetch again"]})
    plan = replan(bb, failure="login failed", llm=llm)
    assert [sg.id for sg in plan] == [1, 2]
    assert "login failed" in llm.calls[0][1][0]["content"]
