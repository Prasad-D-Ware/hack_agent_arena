"""
://agent_arena — AppWorld agent (Planner -> Executor -> Verifier).

Entry point only: loads tasks, opens each AppWorld task, builds a Blackboard,
and hands off to orchestrator.solve(). Reasoning lives in planner/executor/
verifier/orchestrator; memory + API-doc retrieval live in hydradb (HydraMemory,
no-op unless USE_HYDRA=1).

Run:
  export OPENROUTER_API_KEY=sk-or-...          # or put it in .env
  export APPWORLD_EXPERIMENT=team_<yourname>
  export APPWORLD_DATASET=dev                  # dev while building
  python agent.py

🐉 Bonus — HydraDB context layer (optional, off by default):
  export USE_HYDRA=1 HYDRA_DB_API_KEY=...        # enable it for the run
  python bootstrap_docs.py                       # optional explicit ensure
  With it enabled the Planner recalls relevant past episodes per task and the
  Executor retrieves relevant API docs per subgoal. API-doc knowledge is checked
  at startup and only missing docs are ingested. Disabled => recall() is a no-op
  and the loop runs unchanged. See hydradb.py / bootstrap_docs.py.
"""
import json
import os

try:  # optional: load OPENROUTER_API_KEY etc. from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from appworld import AppWorld, load_task_ids

from hydradb import DEFAULT_API_DOCS_ARTIFACT, HydraMemory
from orchestrator import solve
from state import Blackboard

DATASET = os.environ.get("APPWORLD_DATASET", "dev")
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "team_demo")
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))   # 0 = all tasks in split
MODEL = os.environ.get("MODEL", "meta-llama/llama-3.3-70b-instruct:free")
API_DOCS_ARTIFACT = os.environ.get("API_DOCS_ARTIFACT", DEFAULT_API_DOCS_ARTIFACT)


def ensure_hydra_knowledge(mem: HydraMemory) -> None:
    """Prepare Hydra API-doc knowledge once at startup without duplicate ingest."""
    if not mem.on:
        return
    if not os.path.exists(API_DOCS_ARTIFACT):
        print(f"  [hydra] API-doc artifact missing ({API_DOCS_ARTIFACT}); retrieval will use existing DB only")
        return
    try:
        with open(API_DOCS_ARTIFACT) as f:
            api_docs = json.load(f)
        count, ids = mem.ensure_api_docs(api_docs)
    except Exception as e:
        print(f"  [hydra] API-doc ensure failed; continuing without bootstrap: {e}")
        return
    if count:
        print(f"  [hydra] submitted {count} missing API docs; retrieval may improve as indexing finishes")
        mem.wait_until_indexed(ids, timeout=60)


def main() -> None:
    task_ids = load_task_ids(DATASET)
    if MAX_TASKS:
        task_ids = task_ids[:MAX_TASKS]
    mem = HydraMemory()   # shared across tasks so episodic memory accumulates
    ensure_hydra_knowledge(mem)
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} '{DATASET}' tasks with {MODEL}")
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        with AppWorld(task_id=task_id, experiment_name=EXPERIMENT) as world:
            try:
                state = Blackboard(
                    task_instruction=world.task.instruction,
                    supervisor=world.task.supervisor,
                )
                solve(world, state, mem)
                print("  ✓ completed" if world.task_completed()
                      else "  ✗ ended without completion")
            except Exception as e:   # never let one task kill the whole run
                print(f"  ! error: {e}")
    print(f"\nDone. Outputs in ./experiments/outputs/{EXPERIMENT}/")
    print("Self-evaluate:  appworld evaluate $APPWORLD_EXPERIMENT $APPWORLD_DATASET")


if __name__ == "__main__":
    main()
