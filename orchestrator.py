"""Orchestrator: drives Planner -> [Executor -> Verifier] -> finalize."""
from config import (MAX_FINALIZE, MAX_INTERACTIONS, MAX_REPLANS,
                    MAX_SUBGOAL_RETRIES)
from executor import run as run_executor
from llm import call_llm
from parsing import extract_code
from planner import make_plan, replan
from prompts import EXECUTOR_SYSTEM
from state import Blackboard
from verifier import verify_final, verify_subgoal


def solve(world, state: Blackboard, mem, llm=call_llm,
          interaction_budget: int = MAX_INTERACTIONS,
          max_replans: int = MAX_REPLANS,
          max_subgoal_retries: int = MAX_SUBGOAL_RETRIES) -> None:
    hints = mem.recall(state.task_instruction, kind="memory")
    state.plan = make_plan(state, hints, llm=llm)

    i = 0
    while (i < len(state.plan)
           and state.interactions_used < interaction_budget
           and not world.task_completed()):
        sg = state.plan[i]
        sg.status = "active"
        status, result = run_executor(sg, state, world, mem, llm=llm,
                                       interaction_budget=interaction_budget)
        sg.result = result

        if status == "done":
            ok, feedback = verify_subgoal(sg, state, llm=llm)
            if ok:
                sg.status = "done"
                i += 1
                continue
            result = feedback   # verifier overrides into a failure

        # failure path
        sg.attempts += 1
        if sg.attempts <= max_subgoal_retries:
            sg.status = "pending"
            continue            # retry the same subgoal
        if state.replans_used < max_replans:
            state.replans_used += 1
            state.plan = state.plan[:i] + replan(state, result, llm=llm)
            continue            # fresh remaining plan from index i
        sg.status = "failed"
        i += 1                  # give up on this subgoal, move on

    finalize(world, state, mem, llm=llm, interaction_budget=interaction_budget)
    mem.remember_episode(state.task_instruction, state, world.task_completed())


def finalize(world, state: Blackboard, mem, llm=call_llm,
             max_finalize: int = MAX_FINALIZE,
             interaction_budget: int = MAX_INTERACTIONS) -> bool:
    if world.task_completed():
        return True
    messages = [{"role": "user", "content": state.render_for("finalize")
                 + "\n\nAll planned work is done. Reply with ONE python code block that "
                 "calls apis.supervisor.complete_task(answer=<the answer, or None if the "
                 "task is not a question)."}]
    for _ in range(max_finalize):
        if state.interactions_used >= interaction_budget:
            break
        reply = llm("executor", messages, system=EXECUTOR_SYSTEM)
        code = extract_code(reply)
        ok, feedback = verify_final(state, code, llm=llm)
        if not ok:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": f"Verifier rejected completion: {feedback}\n"
                                        "Fix and resend ONLY the completion code block."})
            continue
        output = world.execute(code)
        state.add_step(-1, code, output)
        if world.task_completed():
            return True
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user",
                         "content": f"Execution output:\n{output}\n"
                                    "If not complete, fix and resend the completion code."})
    return world.task_completed()
