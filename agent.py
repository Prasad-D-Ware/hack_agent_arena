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
"""
import os

try:  # optional: load OPENROUTER_API_KEY etc. from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from appworld import AppWorld, load_task_ids

from hydradb import HydraMemory
from orchestrator import solve
from state import Blackboard

DATASET = os.environ.get("APPWORLD_DATASET", "dev")
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "team_demo")
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))   # 0 = all tasks in split
MODEL = os.environ.get("MODEL", "anthropic/claude-opus-4")


def main() -> None:
    task_ids = load_task_ids(DATASET)
    if MAX_TASKS:
        task_ids = task_ids[:MAX_TASKS]
    mem = HydraMemory()   # shared across tasks so episodic memory accumulates
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} '{DATASET}' tasks with {MODEL}")
    for i, task_id in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {task_id}")
        with AppWorld(task_id=task_id, experiment_name=EXPERIMENT) as world:
            try:
                mem.ingest_api_docs(world)   # one-time API-doc knowledge seed
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
