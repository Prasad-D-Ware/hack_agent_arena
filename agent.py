"""
://agent_arena — AppWorld starter agent (ReAct code agent).

This is a WORKING template you can hack on. The loop and every AppWorld API
call below were verified against appworld==0.1.3. Your job is to make the agent
smarter: better prompting, planning, error recovery, retrieval, etc.

How AppWorld works (the rules your agent plays by):
  - Each task gives you a natural-language instruction from your "supervisor".
  - You act by writing PYTHON code. The env runs it and returns whatever you
    print(). A preloaded object `apis` is your only interface to the 9 apps.
  - Discover APIs at runtime:
        apis.api_docs.show_app_descriptions()
        apis.api_docs.show_api_descriptions(app_name='spotify')
        apis.api_docs.show_api_doc(app_name='spotify', api_name='login')
  - Get credentials to log into apps:
        apis.supervisor.show_account_passwords()
    (most app APIs need an access_token returned by that app's `login`).
  - Finish with:
        apis.supervisor.complete_task(answer=<answer or None>)
    Pass `answer` only when the task asks a question; otherwise leave it None.

Run:
  export ANTHROPIC_API_KEY=sk-...             # or put it in .env
  export APPWORLD_EXPERIMENT=team_<yourname>   # your unique team id
  export APPWORLD_DATASET=dev                  # dev while building; switch to the
                                               # official split at submission time
  python agent.py
"""

import os
import re

try:  # optional: load ANTHROPIC_API_KEY etc. from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from appworld import AppWorld, load_task_ids
import anthropic

# ---- config ---------------------------------------------------------------
MODEL = os.environ.get("MODEL", "claude-opus-4-8")          # or claude-sonnet-4-6
DATASET = os.environ.get("APPWORLD_DATASET", "dev")          # dev | test_normal | test_challenge
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "team_demo")
MAX_INTERACTIONS = int(os.environ.get("MAX_INTERACTIONS", "30"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))            # 0 = all tasks in split

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are an autonomous coding agent operating inside AppWorld.
You complete the supervisor's task by writing Python code that the environment executes.

RULES:
- Reply with EXACTLY ONE Python code block per turn, nothing else:
  ```python
  # your code
  ```
- A preloaded object `apis` is the ONLY way to interact with the apps. Whatever
  you print() is returned to you as the next observation.
- You do NOT know the APIs in advance. Discover them at runtime:
    print(apis.api_docs.show_app_descriptions())
    print(apis.api_docs.show_api_descriptions(app_name='<app>'))
    print(apis.api_docs.show_api_doc(app_name='<app>', api_name='<api>'))
- To act on the supervisor's accounts, get credentials and log in:
    print(apis.supervisor.show_account_passwords())
    # then call that app's login API to get an access_token, and pass it onward.
- Work in small steps: inspect results before the next action. Never invent API
  names or fields — look them up first.
- When and ONLY when the task is fully done, call:
    apis.supervisor.complete_task(answer=<answer>)   # answer=None if not a question
"""


def call_llm(messages: list[dict]) -> str:
    resp = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=messages,
        max_tokens=1500,
        temperature=0.0,
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def solve(world: AppWorld) -> None:
    messages = [{
        "role": "user",
        "content": (
            f"Supervisor: {world.task.supervisor}\n\n"
            f"Task: {world.task.instruction}\n\n"
            "Begin. Remember: one python code block per turn."
        ),
    }]
    for step in range(MAX_INTERACTIONS):
        reply = call_llm(messages)
        code = extract_code(reply)
        output = world.execute(code)
        print(f"  step {step+1}: ran {len(code)} chars -> {str(output)[:120]!r}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Execution output:\n{output}"})
        if world.task_completed():
            print("  ✓ task_completed")
            return
    print("  ✗ hit MAX_INTERACTIONS without completion")


def main() -> None:
    task_ids = load_task_ids(DATASET)
    if MAX_TASKS:
        task_ids = task_ids[:MAX_TASKS]
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} '{DATASET}' tasks with {MODEL}")
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        with AppWorld(task_id=task_id, experiment_name=EXPERIMENT) as world:
            try:
                solve(world)
            except Exception as e:  # never let one task kill the whole run
                print(f"  ! error: {e}")
    print(f"\nDone. Outputs in ./experiments/outputs/{EXPERIMENT}/")
    print("Hand that folder to the organizers (or zip and submit per instructions).")


if __name__ == "__main__":
    main()
