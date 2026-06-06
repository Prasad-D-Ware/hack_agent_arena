# ://agent_arena

Build an **autonomous AI agent** that completes everyday-app tasks in
[AppWorld](https://appworld.dev). You are ranked by **Task Goal Completion (TGC)** —
the percentage of tasks your agent fully completes.

## What AppWorld is
A simulated world of **9 apps** (Spotify, Gmail, Venmo, Amazon, Splitwise, Phone,
File System, Simple Note, + `supervisor`/`api_docs`), **457 APIs**, and ~100
simulated people. Your agent reads a natural-language instruction from its
"supervisor" and acts by **writing Python code** that calls the apps' APIs.

## 1. Setup (~3 min) — needs Python 3.11
```bash
git clone git@github.com:interface4agi/hack_agent_arena.git
cd hack_agent_arena
bash setup.sh                 # installs uv+py3.11, appworld + data, creates .env; verifies
source .venv/bin/activate
```
Then add one LLM provider key to **`.env`**. OpenRouter is the default:
```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
```
Or use Groq's OpenAI-compatible endpoint:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
MODEL=llama-3.3-70b-versatile
```
Pick any model with the `MODEL` env var. For OpenRouter, use `provider/model` slugs,
e.g. `anthropic/claude-opus-4`, `openai/gpt-4o`, `google/gemini-2.5-pro`,
`meta-llama/llama-3.3-70b-instruct`. For Groq, use Groq model ids such as
`llama-3.3-70b-versatile`.
> **No key? You can run a local model.** AppWorld itself needs no API key — you can
> explore tasks (`appworld play`) and hand-solve them fully offline. Only the agent's
> "brain" needs a model. The starter uses the OpenAI-compatible API, so `call_llm`
> works with OpenRouter, Groq, or a local compatible host: point the provider base
> URL at a local [Ollama](https://ollama.com) or `litellm`
> server (e.g. `http://localhost:11434/v1`). Small local models score well below
> frontier models on AppWorld, but they're great for building and debugging your
> agent loop for free.

## 2. Smoke-test the starter agent (2 tasks)
```bash
export APPWORLD_EXPERIMENT=team_<yourname>     # your UNIQUE team id
export APPWORLD_DATASET=dev MAX_TASKS=2
python agent.py
```
`agent.py` is a working ReAct code agent — read it, then make it smarter
(planning, error recovery, better prompts, retrieval over `apis.api_docs`, …).

Explore a task world by hand: `appworld play`

## 3. The rules your agent plays by
- One Python code block per turn; whatever you `print()` comes back as the next observation.
- Discover APIs at runtime:
  `apis.api_docs.show_api_descriptions(app_name='spotify')`, then
  `apis.api_docs.show_api_doc(app_name='spotify', api_name='login')`.
- Get credentials: `apis.supervisor.show_account_passwords()`, then log into each app.
- Finish a task: `apis.supervisor.complete_task(answer=<answer or None>)`.

## 4. Submit (at each checkpoint)
1. Run your agent on the **official split** the organizers announce
   (default `test_normal`, 168 tasks):
   ```bash
   export APPWORLD_DATASET=test_normal MAX_TASKS=0
   python agent.py
   ```
2. Self-evaluate:
   ```bash
   appworld evaluate $APPWORLD_EXPERIMENT test_normal
   ```
3. Zip and submit your whole output folder:
   `experiments/outputs/$APPWORLD_EXPERIMENT/`
   It must include `evaluations/test_normal.json` and the `tasks/<id>/dbs/` folders.

## Scoring
- **TGC** (primary) — % of tasks fully completed. **SGC** (scenario goal completion) breaks ties.
- 🐉 **Bonus:** teams that integrate **HydraDB** into their agent's architecture
  earn extra credit (ask organizers for details).
- Reference baseline on `test_normal`: ReAct + GPT-4o ≈ **48.8 TGC**. Beat it.

## 🐉 HydraDB integration (bonus, optional)
[`hydradb.py`](hydradb.py) wires [HydraDB](https://hydradb.com) — a graph-native
context layer for agents — into the agent in two ways, both at the **edges** of
the loop (the ReAct reasoning loop itself is never touched):
- **API-doc knowledge (B):** the 457 AppWorld API docs are a static snapshot
  committed at [`assets/api_docs.json`](assets/api_docs.json). Hydra setup uses
  stable per-API ids and checks first, so [`bootstrap_docs.py`](bootstrap_docs.py)
  and agent startup ingest only missing `knowledge` records. During subgoals the
  agent *queries* them — RAG over the 457 APIs — so it skips runtime discovery.
- **Episodic memory (A):** after each task the agent stores the episode
  (`context.ingest(type="memory")`); before each task it retrieves the most
  relevant past experience (`query`) and injects it into the seed prompt — so it
  stops repeating mistakes across tasks.

It's **off by default** and fully fail-safe — if disabled, unconfigured, or
erroring, every run-time call is a no-op and the agent behaves exactly as before.

```bash
pip install "hydradb-sdk>=2,<3"                 # already in requirements.txt

# 1) Optional explicit ensure: checks first, ingests only missing API docs
HYDRA_DB_API_KEY=... python bootstrap_docs.py    # idempotent; waits for new indexing

# 2) enable HydraDB for the run; agent.py also checks/ensures API docs at startup
export USE_HYDRA=1 HYDRA_DB_API_KEY=...           # key from https://app.hydradb.com
python agent.py
```
Tunables: `HYDRA_TENANT_ID`, `HYDRA_MAX_RESULTS`, `HYDRA_CHUNK_CHARS`,
`HYDRA_READY_TIMEOUT` (see `.env.example`).

---
Built for **://agent_arena** · benchmark: [AppWorld](https://github.com/StonyBrookNLP/appworld) (ACL'24 Best Resource Paper)
