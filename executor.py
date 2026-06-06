"""Executor role: a tight ReAct loop that achieves ONE subgoal."""
import json
import os
import re

from config import MAX_INTERACTIONS, MAX_SUBGOAL_STEPS
from llm import call_llm
from parsing import extract_code, is_truncated
from prompts import EXECUTOR_SYSTEM
from state import Blackboard, Subgoal

DONE_MARKER = "SUBGOAL_DONE"
_LOGIN_CALL = re.compile(r"apis\.([a-zA-Z_][\w]*)\.login\s*\(")
_STALE_TOKEN = re.compile(r"(401|token.*expir|expir.*token|unauthorized|invalid.*token|token.*invalid)", re.I)


def _verbose() -> bool:
    return os.environ.get("AGENT_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _short(text, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _capture_login_token(code: str, output: str, state: Blackboard) -> None:
    """Persist app access tokens printed by login calls for later subgoals."""
    match = _LOGIN_CALL.search(code or "")
    if not match:
        return
    try:
        data = json.loads(str(output))
    except (TypeError, json.JSONDecodeError):
        return
    token = data.get("access_token") if isinstance(data, dict) else None
    if token:
        state.credentials[match.group(1).lower()] = token


def run(subgoal: Subgoal, state: Blackboard, world, mem, llm=call_llm,
        max_steps: int = MAX_SUBGOAL_STEPS,
        interaction_budget: int = MAX_INTERACTIONS) -> tuple[str, str]:
    """Returns ("done", result) or ("failed", reason)."""
    retrieved = mem.recall(subgoal.description, kind="knowledge")
    seed = state.render_for("executor", subgoal)
    if retrieved:
        seed += "\n\nRETRIEVED API KNOWLEDGE:\n" + retrieved
    if subgoal.last_feedback:
        seed += f"\n\nPREVIOUS ATTEMPT FAILED — verifier reason: {subgoal.last_feedback}\nFix this specific issue before proceeding."
    seed += (f"\n\nWork on the current subgoal. When achieved, reply exactly:\n"
             f"{DONE_MARKER}: <one-line result>")
    messages = [{"role": "user", "content": seed}]
    _empty_streak = 0
    _unsupported_done_streak = 0
    has_successful_evidence = False

    for step in range(1, max_steps + 1):
        if state.interactions_used >= interaction_budget:
            if _verbose():
                print("  [exec] global budget exhausted", flush=True)
            return ("failed", "global budget exhausted")
        if _verbose():
            print(f"  [exec] subgoal {subgoal.id} step {step}/{max_steps}: {_short(subgoal.description, 120)}",
                  flush=True)
        reply = llm("executor", messages, system=EXECUTOR_SYSTEM)
        if DONE_MARKER in reply:
            result = reply.split(DONE_MARKER, 1)[1].lstrip(": ").strip()
            if not has_successful_evidence:
                _unsupported_done_streak += 1
                if _unsupported_done_streak >= 2:
                    if _verbose():
                        print(f"  [exec] unsupported {DONE_MARKER}; failing subgoal", flush=True)
                    return ("failed", f"{DONE_MARKER} without fresh successful evidence")
                if _verbose():
                    print(f"  [exec] unsupported {DONE_MARKER}; requesting evidence", flush=True)
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": f"Do not reply with {DONE_MARKER} yet. First run exactly one "
                               "python code block that prints fresh evidence for this subgoal. "
                               "For retrieval, filtering, aggregation, or sorting tasks, print "
                               "structured rows with the fields used to decide the result.",
                })
                continue
            if _verbose():
                print(f"  [exec] subgoal {subgoal.id} done: {_short(result or 'done')}", flush=True)
            return ("done", result or "done")
        code = extract_code(reply)
        if not code or is_truncated(code):
            _empty_streak += 1
            if not code:
                reason = "no executable Python"
            else:
                reason = "truncated code (response cut off mid-statement)"
            if _empty_streak >= 2:
                if _verbose():
                    print(f"  [exec] {_empty_streak} consecutive unusable replies ({reason}); failing subgoal",
                          flush=True)
                return ("failed", f"model returned {reason}")
            if _verbose():
                print(f"  [exec] unusable reply ({reason}); reprompting", flush=True)
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Your reply contained no executable Python or was cut off. "
                           "Send exactly one complete python code block that prints useful evidence, "
                           "or reply with "
                           f"{DONE_MARKER}: <one-line result> if this subgoal is complete.",
            })
            continue
        _empty_streak = 0
        if _verbose():
            print(f"  [exec] code: {_short(code)}", flush=True)
        output = world.execute(code)
        if _verbose():
            print(f"  [exec] output: {_short(output)}", flush=True)
        sig = state.add_step(subgoal.id, code, output)
        if sig is None:
            _capture_login_token(code, output, state)
        elif _STALE_TOKEN.search(str(output)):
            # evict expired tokens so the next step re-authenticates
            app_ref = re.search(r"apis\.([a-zA-Z_][\w]*)\.", code or "")
            if app_ref:
                state.credentials.pop(app_ref.group(1).lower(), None)
        if sig is None and str(output).strip() and str(output).strip() != "Execution successful.":
            has_successful_evidence = True
            _unsupported_done_streak = 0
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Execution output:\n{output}"})
        if world.task_completed():
            if _verbose():
                print("  [exec] task completed during subgoal", flush=True)
            return ("done", "task completed")
        if str(output).strip() == "Execution successful.":
            messages.append({
                "role": "user",
                "content": "The code executed but printed no observable result. In the next "
                           "step, print the relevant value/evidence, or reply with "
                           f"{DONE_MARKER}: <one-line result> if this subgoal is complete.",
            })
        if sig is not None or "Response status code" in str(output):
            messages.append({
                "role": "user",
                "content": "The API call failed. Before retrying the same API, inspect its "
                           "documentation with apis.api_docs.show_api_doc(app_name=..., "
                           "api_name=...) and then call it using only documented parameters.",
            })
        if state.has_repeated_error(subgoal.id):
            sigs = state.recent_error_signatures(subgoal.id, 2)
            if _verbose():
                print(f"  [exec] repeated error: {_short(sigs[-1] if sigs else 'repeated error')}", flush=True)
            return ("failed", sigs[-1] if sigs else "repeated error")
    if _verbose():
        print(f"  [exec] subgoal {subgoal.id} failed: max steps reached", flush=True)
    return ("failed", "max steps reached")
