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
  export OPENROUTER_API_KEY=sk-or-...          # or put it in .env
  export APPWORLD_EXPERIMENT=team_<yourname>   # your unique team id
  export APPWORLD_DATASET=dev                  # dev while building; switch to the
                                               # official split at submission time
  python agent.py

🐉 Bonus — HydraDB context layer (optional, off by default):
  export USE_HYDRA=1 HYDRA_DB_API_KEY=...       # see hydradb.py for what it does
  The agent then remembers what worked across tasks and retrieves relevant past
  experience + API docs before each task. Disabled => the loop runs unchanged.
"""

import os
import re

try:  # optional: load OPENROUTER_API_KEY etc. from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from appworld import AppWorld, load_task_ids
from openai import OpenAI

from hydradb import HydraMemory  # 🐉 optional HydraDB context layer (no-op unless USE_HYDRA=1)

# ---- config ---------------------------------------------------------------
# Model-agnostic via OpenRouter: use any "provider/model" slug, e.g.
#   anthropic/claude-opus-4  openai/gpt-4o  google/gemini-2.5-pro  meta-llama/llama-3.3-70b-instruct
MODEL = os.environ.get("MODEL", "anthropic/claude-opus-4")
DATASET = os.environ.get("APPWORLD_DATASET", "dev")          # dev | test_normal | test_challenge
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "team_demo")
MAX_INTERACTIONS = int(os.environ.get("MAX_INTERACTIONS", "30"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))            # 0 = all tasks in split

# OpenRouter is OpenAI-compatible: point the OpenAI SDK at its endpoint.
# Override OPENROUTER_BASE_URL to use any other OpenAI-compatible host
# (e.g. a local Ollama/litellm server at http://localhost:11434/v1).
client = OpenAI(
    base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

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
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        max_tokens=1500,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def solve(world: AppWorld, mem: HydraMemory) -> None:
    # --- HydraDB hooks live ONLY at the edges; the reasoning loop is untouched. ---
    mem.ingest_api_docs(world)                       # B) seed API-doc knowledge (once per run)
    recalled = mem.recall(world.task.instruction)    # A+B) retrieve context for THIS task

    intro = (
        f"Supervisor: {world.task.supervisor}\n\n"
        f"Task: {world.task.instruction}\n\n"
    )
    if recalled:
        intro += recalled + "\n\n"
    intro += "Begin. Remember: one python code block per turn."
    messages = [{"role": "user", "content": intro}]

    # --- reasoning loop: identical to the starter (no HydraDB inside it) ---
    completed = False
    for step in range(MAX_INTERACTIONS):
        reply = call_llm(messages)
        code = extract_code(reply)
        output = world.execute(code)
        print(f"  step {step+1}: ran {len(code)} chars -> {str(output)[:120]!r}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Execution output:\n{output}"})
        if world.task_completed():
            print("  ✓ task_completed")
            completed = True
            break
    else:
        print("  ✗ hit MAX_INTERACTIONS without completion")

    # A) remember the episode after the loop, from the messages it produced.
    mem.remember_task(world.task.instruction, messages, completed)


def main() -> None:
    task_ids = load_task_ids(DATASET)
    if MAX_TASKS:
        task_ids = task_ids[:MAX_TASKS]
    mem = HydraMemory()  # 🐉 shared across tasks so memory accumulates over the run
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} '{DATASET}' tasks with {MODEL}")
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        with AppWorld(task_id=task_id, experiment_name=EXPERIMENT) as world:
            try:
                solve(world, mem)
            except Exception as e:  # never let one task kill the whole run
                print(f"  ! error: {e}")
    print(f"\nDone. Outputs in ./experiments/outputs/{EXPERIMENT}/")
    print("Hand that folder to the organizers (or zip and submit per instructions).")


if __name__ == "__main__":
    main()
