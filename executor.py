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
# Detects unevaluated Python f-string expressions inside a claimed result
_UNEVALUATED_FSTRING = re.compile(r"\{[a-zA-Z_]\w*(\([^)]*\))?\}")
# Detects a pagination loop that fetched 0 items (likely missing page_limit kwarg)
_PAGINATION_ZERO = re.compile(r"(?:total|count|emails?|items?|drafts?)[^\n]*:\s*0\b", re.I)
_LOGIN_CALL = re.compile(r"apis\.([a-zA-Z_][\w]*)\.login\s*\(")
_STALE_TOKEN = re.compile(r"(status code is 401|token.*expir|expir.*token|unauthorized|invalid.*token|token.*invalid)", re.I)
# Detects password value being assigned directly as an access token (common misuse)
_PASSWORD_AS_TOKEN = re.compile(
    r"""(?:tok|token|access_token)\s*=\s*(?:next\s*\([^)]*\[['"]\s*password\s*['"]\]|[^#\n]*\[['"]\s*password\s*['"]\])""",
    re.I,
)


def _verbose() -> bool:
    return os.environ.get("AGENT_VERBOSE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _short(text, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _capture_login_token(code: str, output: str, state: Blackboard) -> None:
    """Persist app access tokens returned by any login call for later subgoals.

    Captures tokens even when login is called mid-subgoal for a secondary app,
    not just as a dedicated login subgoal.
    """
    match = _LOGIN_CALL.search(code or "")
    if not match:
        return
    app = match.group(1).lower()
    # Try to parse the raw output as JSON first
    try:
        data = json.loads(str(output))
    except (TypeError, json.JSONDecodeError):
        data = None
    token = data.get("access_token") if isinstance(data, dict) else None
    if token:
        state.credentials[app] = token
        return
    # Fallback: scan any line in the output for {"access_token": "..."} patterns
    _TOKEN_FIELD = re.compile(r'"access_token"\s*:\s*"([^"]+)"')
    for line in str(output).splitlines():
        m = _TOKEN_FIELD.search(line)
        if m:
            state.credentials[app] = m.group(1)
            return


def run(subgoal: Subgoal, state: Blackboard, world, mem, llm=call_llm,
        max_steps: int = MAX_SUBGOAL_STEPS,
        interaction_budget: int = MAX_INTERACTIONS) -> tuple[str, str]:
    """Returns ("done", result) or ("failed", reason)."""
    retrieved = mem.recall(subgoal.description, kind="knowledge")
    seed = state.render_for("executor", subgoal)
    if retrieved:
        seed += "\n\nRETRIEVED API KNOWLEDGE:\n" + retrieved
    if subgoal.last_feedback:
        seed += (f"\n\nPREVIOUS ATTEMPT FAILED — verifier reason: {subgoal.last_feedback}\n"
                 "Fix this specific issue before proceeding. IMPORTANT: each retry starts "
                 "a fresh Python environment — variables from prior attempts do not exist. "
                 "Re-fetch all data you need (lists, IDs, tokens) from the API before acting.")
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
            if _UNEVALUATED_FSTRING.search(result):
                if _verbose():
                    print(f"  [exec] unevaluated f-string in {DONE_MARKER}; requesting print", flush=True)
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": f"Your {DONE_MARKER} result contains an unevaluated Python "
                               "expression (e.g. {len(x)}). Run a code block that prints "
                               "the actual value, then reply with the numeric result.",
                })
                continue
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
        if _PASSWORD_AS_TOKEN.search(code or ""):
            if _verbose():
                print("  [exec] password-as-token pattern detected; reprompting", flush=True)
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Your code assigns p[\"password\"] (the login credential) directly "
                           "as an access token. That is wrong. You must call the app's login "
                           "API first and then read [\"access_token\"] from the response:\n"
                           "  result = apis.<app>.login(username=\"<email>\", password=password)\n"
                           "  tok    = result[\"access_token\"]\n"
                           "Never pass a password string to an API as a token.",
            })
            continue
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
        if (sig is None
                and str(output).strip()
                and str(output).strip() != "Execution successful."
                and "Response status code" not in str(output)):
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
        if _PAGINATION_ZERO.search(str(output)):
            messages.append({
                "role": "user",
                "content": (
                    "The loop collected 0 items even though the account is not empty. "
                    "The API call inside the loop is missing page_index= and/or page_limit= "
                    "as explicit keyword arguments — they must be IN the call, not just outer "
                    "variables. Use exactly this skeleton:\n"
                    "  PAGE_LIMIT = 20\n"
                    "  items = []\n"
                    "  page = 0\n"
                    "  while True:\n"
                    "      batch = apis.<app>.<api>(access_token=tok,\n"
                    "                               page_index=page,\n"
                    "                               page_limit=PAGE_LIMIT)\n"
                    "      items.extend(batch)\n"
                    "      if len(batch) < PAGE_LIMIT:\n"
                    "          break\n"
                    "      page += 1\n"
                    "  print(f'collected {len(items)} in {page+1} pages')"
                ),
            })
        if sig is not None:
            messages.append({
                "role": "user",
                "content": "The API call failed. Before retrying the same API, inspect its "
                           "documentation with apis.api_docs.show_api_doc(app_name=..., "
                           "api_name=...) and then call it using only documented parameters. "
                           "If the error says the resource does not exist (e.g. 409), fetch "
                           "the current list first to get valid IDs before acting on them.",
            })
        if state.has_repeated_error(subgoal.id):
            sigs = state.recent_error_signatures(subgoal.id, 2)
            if _verbose():
                print(f"  [exec] repeated error: {_short(sigs[-1] if sigs else 'repeated error')}", flush=True)
            return ("failed", sigs[-1] if sigs else "repeated error")
    if _verbose():
        print(f"  [exec] subgoal {subgoal.id} failed: max steps reached", flush=True)
    return ("failed", "max steps reached")
