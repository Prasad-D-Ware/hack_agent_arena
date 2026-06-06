from executor import run
from state import Blackboard, Subgoal
from tests.conftest import FakeLLM, FakeWorld, FakeMem


def test_executor_completes_on_done_marker():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "log in")
    llm = FakeLLM({"executor": [
        "```python\nprint('hi')\n```",
        "SUBGOAL_DONE: logged in",
    ]})
    world = FakeWorld(outputs=["hi"])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "done" and result == "logged in"
    assert bb.interactions_used == 1          # one code execution before DONE


def test_executor_injects_retrieved_knowledge():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "search")
    llm = FakeLLM({"executor": ["SUBGOAL_DONE: ok"]})
    mem = FakeMem(recall_value="API: spotify.search(query)")
    run(sg, bb, FakeWorld(), mem, llm=llm, max_steps=3, interaction_budget=40)
    assert "spotify.search" in llm.calls[0][1][0]["content"]


def test_executor_fails_on_repeated_error():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": [
        "```python\nbad\n```",
        "```python\nbad\n```",
    ]})
    world = FakeWorld(outputs=["KeyError: 'x'", "KeyError: 'x'"])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "failed" and "KeyError" in result


def test_executor_stops_at_global_budget():
    bb = Blackboard(task_instruction="t")
    bb.interactions_used = 40
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": ["```python\nx\n```"]})
    status, _ = run(sg, bb, FakeWorld(), FakeMem(), llm=llm,
                    max_steps=5, interaction_budget=40)
    assert status == "failed"


def test_executor_does_not_execute_empty_code():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": ["```python\n\n```", "SUBGOAL_DONE: ok"]})
    world = FakeWorld()
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=3, interaction_budget=40)
    assert status == "done"
    assert result == "ok"
    assert world.executed == []
    assert bb.interactions_used == 0


def test_executor_fails_after_consecutive_empty_replies():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    # two empty replies in a row → fail fast, don't burn all 10 steps
    llm = FakeLLM({"executor": ["<think>hmm</think>", "<think>still thinking</think>"]})
    world = FakeWorld()
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=10, interaction_budget=40)
    assert status == "failed"
    assert "no executable" in result
    assert world.executed == []


def test_executor_reprompts_truncated_code_then_fails():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    truncated = '```python\nresult = apis.spotify.show_song(access_token="eyJhbGci\n```'
    llm = FakeLLM({"executor": [truncated, truncated]})
    world = FakeWorld()
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=10, interaction_budget=40)
    assert status == "failed"
    assert "truncated" in result
    assert world.executed == []
