from orchestrator import solve, finalize
from state import Blackboard, Subgoal
from tests.conftest import FakeLLM, FakeWorld, FakeMem


def test_solve_runs_plan_then_completes():
    bb = Blackboard(task_instruction="play a song")
    # planner -> 1 subgoal; executor -> DONE; verifier subgoal PASS;
    # finalize executor -> completion code; verifier final PASS.
    llm = FakeLLM({
        "planner": ["1. play the song"],
        "executor": ["SUBGOAL_DONE: played",
                     "```python\napis.supervisor.complete_task(answer=None)\n```"],
        "verifier": ["PASS", "PASS"],
    })
    world = FakeWorld()
    mem = FakeMem()
    solve(world, bb, mem, llm=llm)
    assert world.task_completed() is True
    assert mem.remembered and mem.remembered[0][1] is True   # recorded as success


def test_solve_retries_subgoal_then_replans():
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({
        "planner": ["1. do x", "1. do x differently"],   # initial, then replan
        "executor": [
            "```python\nbad\n```", "```python\nbad\n```",   # subgoal attempt 1 -> repeated err
            "```python\nbad\n```", "```python\nbad\n```",   # subgoal attempt 2 (retry) -> repeated err
            "SUBGOAL_DONE: ok",                              # replanned subgoal succeeds
            "```python\napis.supervisor.complete_task(answer=None)\n```",
        ],
        "verifier": ["PASS", "PASS"],
    })
    world = FakeWorld(outputs=["KeyError: 'x'"] * 4)
    solve(world, bb, mem=FakeMem(), llm=llm,
          interaction_budget=40, max_replans=2, max_subgoal_retries=1)
    assert bb.replans_used == 1
    assert world.task_completed() is True


def test_finalize_reprompts_when_verifier_rejects():
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({
        "executor": ["```python\napis.supervisor.complete_task(answer='wrong')\n```",
                     "```python\napis.supervisor.complete_task(answer=42)\n```"],
        "verifier": ["FAIL: must be int", "PASS"],
    })
    world = FakeWorld()
    ok = finalize(world, bb, FakeMem(), llm=llm, max_finalize=3, interaction_budget=40)
    assert ok is True
    assert world.task_completed() is True
