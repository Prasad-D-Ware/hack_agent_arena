# 🏁 Final Evaluation Set — `agent_arena_eval` (10 tasks)

This is the **official set you'll be ranked on.** It's a fixed, stratified random
subset of AppWorld `test_challenge` — **3 easy / 3 medium / 4 hard**, 10 distinct
scenarios. Same set for every team.

**Integrity:** verify it's unchanged — the SHA-256 of these 10 IDs must equal the
committed hash:
```
3fd1d4fc28a1cc9570d1cfeac5f27fae48c9f866bb0fb71822887925c265e4b8
```

## 1. Install the set
After `git pull`, copy it into your AppWorld data dir:
```bash
mkdir -p data/datasets
cp eval/agent_arena_eval.txt data/datasets/agent_arena_eval.txt
```
(Not pulling? Create `data/datasets/agent_arena_eval.txt` with the 10 IDs at the bottom of this file.)

## 2. Run your agent on it
```bash
source .venv/bin/activate
export APPWORLD_EXPERIMENT=team_<yourname>     # your unique team id
export APPWORLD_DATASET=agent_arena_eval MAX_TASKS=0
python agent.py
```
⏱️ Budget ~15–25 min — these are the hard challenge tasks.

## 3. Check your own score
```bash
appworld evaluate $APPWORLD_EXPERIMENT agent_arena_eval
```
Primary metric = **TGC** (Task Goal Completion). SGC breaks ties.

## 4. Submit before the deadline
See [`SUBMISSION.md`](SUBMISSION.md). Zip and send your output folder:
```
experiments/outputs/team_<yourname>/
```
It must contain `evaluations/agent_arena_eval.json` and the `tasks/<id>/dbs/` folders.

> Rules: build a **general** agent. No hardcoding answers to specific `task_id`s —
> submitted code is reviewed and such entries are disqualified.

---
### The 10 task IDs
```
5e27cd7_2
ba46d91_2
dbc0276_3
20c1328_3
9871968_2
c1091c7_2
18670a5_3
f6be291_1
23d431c_3
8d42650_3
```
