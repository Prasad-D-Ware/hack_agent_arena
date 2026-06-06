# Project Context

## Overview

This repository is for `://agent_arena`, where teams build an autonomous AI agent for the AppWorld benchmark. AppWorld simulates everyday apps such as Spotify, Gmail, Venmo, Amazon, Splitwise, Phone, File System, Simple Note, Todoist, plus `supervisor` and `api_docs`. The agent receives a natural-language task and completes it by executing Python code against AppWorld APIs.

The project has moved from a single ReAct loop to a Planner -> Executor -> Verifier architecture over a shared Blackboard. The goal is still Task Goal Completion (TGC): the percentage of tasks fully completed according to AppWorld's database-state evaluator.

## Current Architecture

`agent.py` is now a thin entrypoint:

1. Loads `.env` through `python-dotenv`.
2. Loads task IDs for `APPWORLD_DATASET`.
3. Creates one shared `HydraMemory`.
4. Opens each task with `AppWorld`.
5. Builds a `Blackboard` from the task instruction and supervisor.
6. Calls `orchestrator.solve(world, state, mem)`.

The reasoning flow lives outside `agent.py`:

- `planner.py`: creates an ordered subgoal plan and replans after failures.
- `executor.py`: runs a tight ReAct loop for one subgoal, retrieves HydraDB API knowledge per subgoal, executes code, records steps, and stops on `SUBGOAL_DONE`.
- `verifier.py`: asks an LLM to gate subgoal results and final completion code with `PASS` / `FAIL`.
- `orchestrator.py`: coordinates planning, execution, verification, retry/replan budgets, final `complete_task`, and episodic memory storage.
- `state.py`: defines `Subgoal`, `StepRecord`, `error_signature`, and `Blackboard`.
- `llm.py`: wraps OpenRouter/OpenAI-compatible chat completions with role-based model routing, retry/backoff, and optional fallback model.
- `prompts.py`: role system prompts for Planner, Executor, and Verifier.
- `parsing.py`: extracts a single Python code block from LLM output.
- `config.py`: env-driven budgets and loop limits.

## HydraDB And API Docs

HydraDB is optional and fail-safe during agent runs. If `USE_HYDRA` is not enabled, unconfigured, unavailable, or erroring, runtime recall and memory writes no-op.

The API-doc path changed:

- `assets/api_docs.json`: committed local snapshot of the AppWorld API docs. Current metadata reports AppWorld `0.1.3.post1`, 11 apps, and 457 APIs.
- `dump_docs.py`: regenerates `assets/api_docs.json` from a live AppWorld sandbox. Use only when refreshing the API-doc artifact after AppWorld changes.
- `bootstrap_docs.py`: one-time offline HydraDB bootstrap. It reads `assets/api_docs.json` and ingests one knowledge document per API. This does not run inside `agent.py`.
- `hydradb.py`: supports offline API-doc ingestion, indexing status checks, runtime `recall(query, kind="memory"|"knowledge"|"all")`, and structured `remember_episode(...)`.

Runtime retrieval:

- Planner recalls `kind="memory"` once per task for relevant past episodes.
- Executor recalls `kind="knowledge"` per subgoal for relevant API docs.
- After each task, the orchestrator stores a structured episode from the Blackboard.

## Important Files

- `agent.py`: entrypoint and AppWorld task loop.
- `orchestrator.py`: Planner -> Executor -> Verifier control flow.
- `planner.py`: task decomposition and replanning.
- `executor.py`: subgoal-level code execution loop.
- `verifier.py`: subgoal and final completion checks.
- `state.py`: Blackboard and execution records.
- `llm.py`: OpenRouter/OpenAI-compatible LLM client.
- `prompts.py`: role prompts.
- `hydradb.py`: optional context layer and offline API-doc ingestion support.
- `bootstrap_docs.py`: HydraDB API-doc bootstrap.
- `dump_docs.py`: API-doc artifact generator.
- `assets/api_docs.json`: local API-doc artifact.
- `tests/`: pytest coverage for parser, state, LLM wrapper, planner, executor, verifier, orchestrator, and HydraDB wrapper.
- `docs/APPWORLD_TASK_COOKBOOK.md`: docs-only task-pattern cookbook.
- `Plan.md`: teammate implementation plan/reference.
- `logs.txt`: untracked local runtime log; contains sensitive run output and should not be committed.

## Configuration

Core run env vars:

- `OPENROUTER_API_KEY`: required for OpenRouter-hosted models.
- `OPENROUTER_BASE_URL`: optional OpenAI-compatible endpoint, default `https://openrouter.ai/api/v1`.
- `MODEL`: default in code is `meta-llama/llama-3.3-70b-instruct:free`.
- `MODEL_PLANNER`, `MODEL_EXECUTOR`, `MODEL_VERIFIER`: optional per-role overrides.
- `MODEL_FALLBACK`: optional fallback model used near the end of LLM retry attempts.
- `APPWORLD_EXPERIMENT`: output folder name, default `team_demo`.
- `APPWORLD_DATASET`: default `dev`.
- `MAX_TASKS`: `0` means all tasks in the split.

Budget env vars from `config.py`:

- `MAX_INTERACTIONS`: default `40`.
- `MAX_SUBGOAL_STEPS`: default `10`.
- `MAX_REPLANS`: default `2`.
- `MAX_SUBGOAL_RETRIES`: default `1`.
- `MAX_FINALIZE`: default `3`.

HydraDB env vars:

- `USE_HYDRA=1`: enables runtime HydraDB recall/memory.
- `HYDRA_DB_API_KEY`: required for bootstrapping and runtime HydraDB.
- `HYDRA_TENANT_ID`: default `appworld_agent`.
- `HYDRA_MAX_RESULTS`: default `12`, clamped to HydraDB API range.
- `HYDRA_CHUNK_CHARS`: default `1000`.
- `HYDRA_READY_TIMEOUT`: default `30`.

## How To Run

Initial setup, if needed:

```bash
bash setup.sh
source .venv/bin/activate
```

Install dev test dependency, if not already installed:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Add your OpenRouter key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-...
```

Smoke-test imports and tests:

```bash
source .venv/bin/activate
python -c "import agent; print('agent ok')"
python -m pytest -q
```

Run one dev task without HydraDB:

```bash
source .venv/bin/activate
export APPWORLD_EXPERIMENT=team_smoke
export APPWORLD_DATASET=dev
export MAX_TASKS=1
python agent.py
```

Run a small dev subset:

```bash
source .venv/bin/activate
export APPWORLD_EXPERIMENT=team_dev_probe
export APPWORLD_DATASET=dev
export MAX_TASKS=5
python agent.py
```

Evaluate a complete split run:

```bash
appworld evaluate $APPWORLD_EXPERIMENT $APPWORLD_DATASET
```

For full dev:

```bash
source .venv/bin/activate
export APPWORLD_EXPERIMENT=team_dev_full
export APPWORLD_DATASET=dev
export MAX_TASKS=0
python agent.py
appworld evaluate $APPWORLD_EXPERIMENT dev
```

For the official split when ready:

```bash
source .venv/bin/activate
export APPWORLD_EXPERIMENT=team_<yourname>
export APPWORLD_DATASET=test_normal
export MAX_TASKS=0
python agent.py
appworld evaluate $APPWORLD_EXPERIMENT test_normal
```

## HydraDB Run Flow

To use HydraDB API-doc retrieval and episodic memory:

1. Bootstrap API docs once, offline:

```bash
source .venv/bin/activate
export HYDRA_DB_API_KEY=...
python bootstrap_docs.py
```

2. Enable HydraDB for the agent run:

```bash
export USE_HYDRA=1
export HYDRA_DB_API_KEY=...
export APPWORLD_EXPERIMENT=team_hydra_smoke
export APPWORLD_DATASET=dev
export MAX_TASKS=1
python agent.py
```

Re-run `bootstrap_docs.py` only when the API-doc artifact changes or the Hydra tenant needs to be reseeded.

To regenerate the local API-doc artifact:

```bash
source .venv/bin/activate
python dump_docs.py
```

That opens one AppWorld sandbox and writes `assets/api_docs.json`.

## Verification Status

Current local checks:

```text
python -m pytest -q  -> 43 passed
python -c "import agent; print('agent ok')" -> agent ok
```

No live agent smoke run was executed during this context update because that would call the configured LLM/API.

## Current Local State

As of this update:

- Tracked implementation now includes the Planner/Executor/Verifier modules, HydraDB bootstrap flow, API-doc artifact, and tests.
- Local untracked artifacts include `PROJECT_CONTEXT.md`, `Plan.md`, `docs/`, and `logs.txt`.
- Ignored local artifacts still include `.env`, `.venv/`, `data/`, and `experiments/`.
- `logs.txt` contains sensitive task output/tokens and should not be committed.

## Near-Term Improvement Areas

Avoid duplicating the teammate architecture work. Good next lanes are:

- Build run/evaluation tooling that can summarize only present task outputs.
- Add failure-analysis scripts for wrong answer vs wrong side effect vs missing outputs.
- Turn `docs/APPWORLD_TASK_COOKBOOK.md` into retrieval snippets after the core architecture stabilizes.
- Add deterministic helper patterns for pagination and answer semantics after measuring the new architecture.
- Improve observability around per-role LLM calls, verifier failures, and replans without changing the role interfaces.
