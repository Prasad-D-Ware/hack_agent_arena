"""Orchestrator: drives Planner -> [Executor -> Verifier] -> finalize."""
import os

from config import (MAX_FINALIZE, MAX_INTERACTIONS, MAX_REPLANS,
                    MAX_SUBGOAL_RETRIES)
from executor import run as run_executor
from llm import call_llm
from parsing import extract_code
from planner import make_plan, replan
from prompts import EXECUTOR_SYSTEM
from state import Blackboard
from verifier import verify_final, verify_subgoal


def _verbose() -> bool:
    return os.environ.get("AGENT_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _short(text, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def solve(world, state: Blackboard, mem, llm=call_llm,
          interaction_budget: int = MAX_INTERACTIONS,
          max_replans: int = MAX_REPLANS,
          max_subgoal_retries: int = MAX_SUBGOAL_RETRIES) -> None:
    hints = mem.recall(state.task_instruction, kind="memory")
    state.plan = make_plan(state, hints, llm=llm)
    work_budget = max(1, interaction_budget - MAX_FINALIZE)
    if _verbose():
        print(f"  [plan] {len(state.plan)} subgoals", flush=True)
        for sg in state.plan:
            print(f"  [plan] {sg.id}. {_short(sg.description)}", flush=True)

    i = 0
    while (i < len(state.plan)
           and state.interactions_used < work_budget
           and not world.task_completed()):
        sg = state.plan[i]
        sg.status = "active"
        if _verbose():
            print(f"  [orch] start subgoal {sg.id}: {_short(sg.description)}", flush=True)
        status, result = run_executor(sg, state, world, mem, llm=llm,
                                       interaction_budget=work_budget)
        sg.result = result
        if _verbose():
            print(f"  [orch] subgoal {sg.id} executor status={status}: {_short(result)}", flush=True)

        if status == "done":
            ok, feedback = verify_subgoal(sg, state, llm=llm)
            if _verbose():
                verdict = "accepted" if ok else "rejected"
                print(f"  [verify] subgoal {sg.id} {verdict}: {_short(feedback)}", flush=True)
            if ok:
                sg.status = "done"
                i += 1
                continue
            result = feedback   # verifier overrides into a failure

        # failure path
        sg.attempts += 1
        if sg.attempts <= max_subgoal_retries:
            sg.status = "pending"
            if _verbose():
                print(f"  [orch] retrying subgoal {sg.id}", flush=True)
            continue            # retry the same subgoal
        if state.replans_used < max_replans:
            state.replans_used += 1
            if _verbose():
                print(f"  [orch] replanning after subgoal {sg.id} failure", flush=True)
            state.plan = state.plan[:i] + replan(state, result, llm=llm)
            continue            # fresh remaining plan from index i
        sg.status = "failed"
        i += 1                  # give up on this subgoal, move on

    if _verbose() and state.interactions_used >= work_budget and not world.task_completed():
        print("  [orch] reserved remaining budget for finalization", flush=True)
    finalize(world, state, mem, llm=llm, interaction_budget=interaction_budget)
    mem.remember_episode(state.task_instruction, state, world.task_completed())


def finalize(world, state: Blackboard, mem, llm=call_llm,
             max_finalize: int = MAX_FINALIZE,
             interaction_budget: int = MAX_INTERACTIONS) -> bool:
    if world.task_completed():
        return True
    if _verbose():
        print("  [finalize] attempting complete_task", flush=True)
    messages = [{"role": "user", "content": state.render_for("finalize")
                 + "\n\nAll planned work is done. Reply with ONE python code block that "
                 "calls apis.supervisor.complete_task(answer=<the answer, or None if the "
                 "task is not a question)."}]
    for _ in range(max_finalize):
        if state.interactions_used >= interaction_budget:
            break
        reply = llm("executor", messages, system=EXECUTOR_SYSTEM)
        code = extract_code(reply)
        if not code or "complete_task" not in code:
            if _verbose():
                print("  [finalize] missing complete_task call; reprompting", flush=True)
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Reply with exactly one python code block that calls "
                           "apis.supervisor.complete_task(answer=<final answer or None>).",
            })
            continue
        ok, feedback = verify_final(state, code, llm=llm)
        if _verbose():
            verdict = "accepted" if ok else "rejected"
            print(f"  [finalize] verifier {verdict}: {_short(feedback)}", flush=True)
        if not ok:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": f"Verifier rejected completion: {feedback}\n"
                                        "Fix and resend ONLY the completion code block."})
            continue
        output = world.execute(code)
        state.add_step(-1, code, output)
        if _verbose():
            print(f"  [finalize] output: {_short(output)}", flush=True)
        if world.task_completed():
            return True
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user",
                         "content": f"Execution output:\n{output}\n"
                                    "If not complete, fix and resend the completion code."})
    return world.task_completed()
