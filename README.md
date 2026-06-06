# HydraDB Agent Arena

Autonomous AppWorld agent for `://agent_arena`. The agent reads a supervisor
task, writes Python code against AppWorld APIs, verifies its answer or mutation,
and completes the task with `apis.supervisor.complete_task(...)`.

## What We Added

- Provider/model routing: `MODEL`, `ROUTER_MODEL`, `SOLVER_MODEL`,
  `ERROR_MODEL`, `VERIFIER_MODEL`, and `DISTILLER_MODEL` accept
  `provider/model` strings. The default target is OpenRouter Llama 3.3 70B:
  `openrouter/meta-llama/llama-3.3-70b-instruct`.
- OpenAI-compatible backends: OpenRouter, Groq, OpenAI, Anthropic, and the
  legacy Azure Llama backend are routed from one agent loop.
- HydraDB memory: successful and failed trajectories can be distilled into
  reusable skills/lessons with `scripts/distill_skills.py`. Runtime recall
  injects verified memory hints while treating live AppWorld API docs as the
  source of truth.
- Verified-only memory by default: in-run captures are off unless explicitly
  enabled, which avoids poisoning future tasks with unverified completions.
- Solver prompt improvements: the agent now emphasizes schema inspection,
  credential/login patterns, lowercase app namespaces, exact answer shape, and
  concise evidence before completion.
- Completion verifier: answer tasks are checked before submission for empty or
  guessed answers, missing evidence, suspicious counts, bad top-N shape, and
  ignored errors.
- Pagination, inspection, and ranking safeguards: prompts tell the solver to
  exhaust paginated list APIs, print sample keys before reading fields, fetch
  authoritative detail records for ranking metrics, and deduplicate entities.
- Relative-date handling: the AppWorld task datetime is injected and must be used
  for "today", "this year", "last month", and similar relative dates.
- Mutation completeness: payment/message/create/update tasks are prompted to
  include exact amounts, recipients, titles, notes, descriptions, dates, and
  follow-up messages before completion.
- **ACE evolving playbook** (`strategy_ledger.py`): an Agentic Context
  Engineering (arXiv:2510.04618) "playbook" of itemized, GENERAL strategy /
  API-pattern / gotcha bullets that is injected into the solver context each
  task. Backed by the local cache **and HydraDB**, grown offline from execution
  feedback (no ground-truth labels). Toggle with `ENABLE_PLAYBOOK` (default on).

## Setup

Use Python 3.11.

```bash
bash setup.sh
source .venv/bin/activate
cp .env.example .env
```

Add your OpenRouter key to `.env`:

```bash
OPENROUTER_API_KEY=...
MODEL=openrouter/meta-llama/llama-3.3-70b-instruct
```

If OpenRouter changes the slug, set:

```bash
OPENROUTER_LLAMA_MODEL=meta-llama/llama-3.3-70b-instruct
```

Other supported examples:

```bash
MODEL=groq/llama-3.3-70b-versatile
MODEL=openai/gpt-5.4-mini
MODEL=anthropic/claude-opus-4-8
MODEL=azure-llama
```

## Run And Evaluate

Short development smoke test:

```bash
export APPWORLD_EXPERIMENT=team_openrouter_smoke
export APPWORLD_DATASET=dev
export MAX_TASKS=3
python agent.py
appworld evaluate team_openrouter_smoke dev
```

Official organizer eval set:

```bash
mkdir -p data/datasets
cp eval/agent_arena_eval.txt data/datasets/agent_arena_eval.txt
export APPWORLD_EXPERIMENT=team_<name>
export APPWORLD_DATASET=agent_arena_eval
export MAX_TASKS=0
python agent.py
appworld evaluate team_<name> agent_arena_eval
```

See `EVAL.md` and `SUBMISSION.md` for the organizer-provided eval set and
submission format.

## HydraDB Memory Workflow

Enable HydraDB by adding a key to `.env`:

```bash
HYDRA_DB_API_KEY=...
ENABLE_MEMORY=1
MEMORY_VERIFIED_ONLY=1
```

After a run, distill verified trajectories into the local memory cache and
HydraDB:

```bash
python scripts/distill_skills.py --experiment team_<name> --dataset dev
```

Use `--no-llm` for deterministic extraction, or `--dry-run` to inspect what would
be stored. The script scrubs secrets before persisting skills.

## ACE Playbook (Agentic Context Engineering)

The playbook is an evolving **context** rather than a per-task skill recall. It
follows the ACE paper (arXiv:2510.04618): contexts are comprehensive, itemized
playbooks of reusable bullets that accumulate strategies, domain concepts, API
patterns, and common failure modes — never task-specific answers.

Roles map onto the existing system:

- **Generator** — the ReAct loop in `agent.py` solves a task and emits a trace.
- **Reflector** — `scripts/grow_ledger.py` distills concrete, GENERAL bullets
  from each trajectory using **execution feedback only** (completion + recurring
  errors); no ground-truth labels are required (`--use-eval` is optional).
- **Curator** — `StrategyLedger.absorb`: a lightweight, non-LLM merge with
  string/semantic de-duplication ("grow-and-refine"); `StrategyLedger.consolidate`
  consolidates near-duplicates and caps each section.

Each bullet carries metadata (`bullet_id`, `helpful`/`harmful` counters,
`section`, `apps`, `source`). A defensive `is_reusable` guard rejects any bullet
that references a concrete `task_id`, embeds a literal `complete_task` answer, or
otherwise looks task-specific, so the playbook can never leak memorized answers.

Storage (HydraDB bonus): every bullet is mirrored into HydraDB `knowledge`
(`pb_<id>.md`) for cross-run persistence and semantic recall, alongside the
deterministic local cache at `memory/playbook.jsonl`. At inference the top-K
most relevant bullets (apps-aware, then general) are rendered into the solver
context within a token budget; set `PLAYBOOK_HYDRA_TOPK>0` to additionally fold
in HydraDB-recalled bullets at inject time.

Offline warmup and inspection:

```bash
# seed the general defaults + push to HydraDB
python scripts/grow_ledger.py --seed --sync-hydra

# reflect over a finished dev run's trajectories (Reflector + Curator)
python scripts/grow_ledger.py --experiment team_<dev_run> --dataset dev

# consolidate near-duplicates (grow-and-refine)
python strategy_ledger.py --refine --refine-threshold 0.45 --per-section-cap 8

# inspect what gets injected for a given app mix
python strategy_ledger.py --show --apps spotify,phone
```

Relevant env flags: `ENABLE_PLAYBOOK` (default 1), `PLAYBOOK_AUTOSEED`,
`PLAYBOOK_MAX_BULLETS`, `PLAYBOOK_CHAR_BUDGET`, `PLAYBOOK_HYDRA_TOPK`,
`PLAYBOOK_DEDUP_THRESHOLD`.

## Repo Hygiene

Do not commit `.env`, API keys, `memory/`, `memory_backup_*`, probe scripts,
terminal logs, caches, or local AppWorld outputs unless they are the exact
official submission artifacts requested by organizers.
