"""Executor role: a tight ReAct loop that achieves ONE subgoal."""
from config import MAX_INTERACTIONS, MAX_SUBGOAL_STEPS
from llm import call_llm
from parsing import extract_code
from prompts import EXECUTOR_SYSTEM
from state import Blackboard, Subgoal

DONE_MARKER = "SUBGOAL_DONE"


def run(subgoal: Subgoal, state: Blackboard, world, mem, llm=call_llm,
        max_steps: int = MAX_SUBGOAL_STEPS,
        interaction_budget: int = MAX_INTERACTIONS) -> tuple[str, str]:
    """Returns ("done", result) or ("failed", reason)."""
    retrieved = mem.recall(subgoal.description, kind="knowledge")
    seed = state.render_for("executor", subgoal)
    if retrieved:
        seed += "\n\nRETRIEVED API KNOWLEDGE:\n" + retrieved
    seed += (f"\n\nWork on the current subgoal. When achieved, reply exactly:\n"
             f"{DONE_MARKER}: <one-line result>")
    messages = [{"role": "user", "content": seed}]

    for _ in range(max_steps):
        if state.interactions_used >= interaction_budget:
            return ("failed", "global budget exhausted")
        reply = llm("executor", messages, system=EXECUTOR_SYSTEM)
        if DONE_MARKER in reply:
            result = reply.split(DONE_MARKER, 1)[1].lstrip(": ").strip()
            return ("done", result or "done")
        code = extract_code(reply)
        output = world.execute(code)
        state.add_step(subgoal.id, code, output)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Execution output:\n{output}"})
        if world.task_completed():
            return ("done", "task completed")
        if state.has_repeated_error(subgoal.id):
            sigs = state.recent_error_signatures(subgoal.id, 2)
            return ("failed", sigs[-1] if sigs else "repeated error")
    return ("failed", "max steps reached")
