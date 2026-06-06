"""Planner role: decompose the task into subgoals; replan on failure."""
import re

from llm import call_llm
from prompts import PLANNER_SYSTEM
from state import Blackboard, Subgoal

_ITEM = re.compile(r"^(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


def _parse_plan(text: str) -> list[Subgoal]:
    out: list[Subgoal] = []
    for line in (text or "").splitlines():
        m = _ITEM.match(line.strip())
        if m:
            out.append(Subgoal(id=len(out) + 1, description=m.group(1).strip()))
    return out


def make_plan(state: Blackboard, hints: str = "", llm=call_llm) -> list[Subgoal]:
    user = state.render_for("planner")
    if hints:
        user += "\n\nPRIOR EXPERIENCE (may help, may be irrelevant):\n" + hints
    user += "\n\nBreak this task into a short ordered, numbered list of subgoals."
    text = llm("planner", [{"role": "user", "content": user}], system=PLANNER_SYSTEM)
    plan = _parse_plan(text)
    return plan or [Subgoal(id=1, description=state.task_instruction)]


def replan(state: Blackboard, failure: str, llm=call_llm) -> list[Subgoal]:
    user = (state.render_for("planner")
            + f"\n\nThe current approach failed: {failure}\n"
            "Provide a REVISED ordered, numbered list of the remaining subgoals.")
    text = llm("planner", [{"role": "user", "content": user}], system=PLANNER_SYSTEM)
    plan = _parse_plan(text) or [Subgoal(id=1, description=state.task_instruction)]
    for i, sg in enumerate(plan, 1):
        sg.id = i
    return plan
