"""Verifier role: gate each subgoal and the final completion."""
from llm import call_llm
from prompts import VERIFIER_SYSTEM
from state import Blackboard, Subgoal


def _parse_verdict(text: str) -> tuple[bool, str]:
    t = (text or "").strip()
    up = t.upper()
    if up.startswith("PASS"):
        return True, ""
    if up.startswith("FAIL"):
        return False, t[4:].lstrip(": ").strip() or "verifier failed"
    return False, t[:200] or "verifier gave no verdict"


def verify_subgoal(subgoal: Subgoal, state: Blackboard, llm=call_llm) -> tuple[bool, str]:
    user = (state.render_for("verifier", subgoal)
            + f"\n\nReported subgoal result: {subgoal.result}\n\n"
            "Was this subgoal actually achieved? Reply 'PASS' or 'FAIL: <reason>'.")
    return _parse_verdict(llm("verifier", [{"role": "user", "content": user}],
                              system=VERIFIER_SYSTEM))


def verify_final(state: Blackboard, code: str, llm=call_llm) -> tuple[bool, str]:
    user = (state.render_for("verifier")
            + f"\n\nProposed completion code:\n{code}\n\n"
            "Will this correctly complete the task with the right answer/side-effects? "
            "Reply 'PASS' or 'FAIL: <reason>'.")
    return _parse_verdict(llm("verifier", [{"role": "user", "content": user}],
                              system=VERIFIER_SYSTEM))
