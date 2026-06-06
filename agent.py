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
  export OPENROUTER_API_KEY=<your-openrouter-key>  # or put it in .env
  export APPWORLD_EXPERIMENT=team_<yourname>   # your unique team id
  export APPWORLD_DATASET=dev                  # dev while building; switch to the
                                               # official split at submission time
  python agent.py
"""

import json
import os
import re

try:  # optional: load OPENAI_API_KEY etc. from a local .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from appworld import AppWorld, load_task_ids
import anthropic
from openai import OpenAI

try:  # self-improving memory layer (HydraDB + local skill cache)
    from skill_store import SkillNote as Skill, SkillRepository as SkillMemory
    _MEMORY_IMPORT_ERROR = None
except Exception as _exc:  # never let memory import break the agent
    Skill = None  # type: ignore
    SkillMemory = None  # type: ignore
    _MEMORY_IMPORT_ERROR = _exc

try:  # ACE-style evolving playbook (general strategies, HydraDB-backed)
    from strategy_ledger import StrategyLedger as Playbook, seed_ledger as seed_playbook
    _PLAYBOOK_IMPORT_ERROR = None
except Exception as _exc:  # never let the playbook break the agent
    Playbook = None  # type: ignore
    seed_playbook = None  # type: ignore
    _PLAYBOOK_IMPORT_ERROR = _exc

try:  # local API-doc snapshot retriever (extra, app-filtered hint source)
    from apidoc_index import ApiDocIndex as ApiDocStore
    _APIDOCS_IMPORT_ERROR = None
except Exception as _exc:  # never let the api-doc add-on break the agent
    ApiDocStore = None  # type: ignore
    _APIDOCS_IMPORT_ERROR = _exc

# ---- config ---------------------------------------------------------------
# MODEL and role model env vars accept provider/model or provider:model strings:
#   openrouter/meta-llama/llama-3.3-70b-instruct
#   groq/llama-3.3-70b-versatile
#   openai/gpt-5.4-mini
#   anthropic/claude-opus-4-8
OPENROUTER_LLAMA_MODEL = os.environ.get(
    "OPENROUTER_LLAMA_MODEL",
    "meta-llama/llama-3.3-70b-instruct",
)
MODEL = os.environ.get("MODEL", f"openrouter/{OPENROUTER_LLAMA_MODEL}")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", MODEL)
SOLVER_MODEL = os.environ.get("SOLVER_MODEL", MODEL)
ERROR_MODEL = os.environ.get("ERROR_MODEL", SOLVER_MODEL)
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", SOLVER_MODEL)
DATASET = os.environ.get("APPWORLD_DATASET", "dev")          # dev | test_normal | test_challenge
EXPERIMENT = os.environ.get("APPWORLD_EXPERIMENT", "team_demo")
MAX_INTERACTIONS = int(os.environ.get("MAX_INTERACTIONS", "30"))
MAX_MODEL_TURNS = int(os.environ.get("MAX_MODEL_TURNS", str(MAX_INTERACTIONS * 2)))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))            # 0 = all tasks in split
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "3000"))
ROUTER_MAX_TOKENS = int(os.environ.get("ROUTER_MAX_TOKENS", "600"))
VERIFIER_MAX_TOKENS = int(os.environ.get("VERIFIER_MAX_TOKENS", "500"))
OUTPUT_PREVIEW_CHARS = int(os.environ.get("OUTPUT_PREVIEW_CHARS", "500"))
OBSERVATION_CHARS = int(os.environ.get("OBSERVATION_CHARS", "6000"))
VERIFIER_CONTEXT_CHARS = int(os.environ.get("VERIFIER_CONTEXT_CHARS", "3500"))
ENABLE_COMPLETION_VERIFIER = os.environ.get("ENABLE_COMPLETION_VERIFIER", "1") != "0"
MAX_VERIFIER_REJECTIONS = int(os.environ.get("MAX_VERIFIER_REJECTIONS", "2"))
DISTILLER_MODEL = os.environ.get("DISTILLER_MODEL", MODEL)

# ---- self-improving memory (HydraDB + local skills) -----------------------
ENABLE_MEMORY = os.environ.get("ENABLE_MEMORY", "1") != "0"
# Within-run capture is OFF by default: unverified "completed" runs are not yet
# known-correct, and storing their raw code poisons later tasks in the same run.
# The verified path is scripts/distill_skills.py, which labels pass/fail
# with the AppWorld evaluator AFTER a run and ingests only trustworthy skills.
ENABLE_MEMORY_CAPTURE = os.environ.get("ENABLE_MEMORY_CAPTURE", "0") != "0"
MEMORY_TOP_K = int(os.environ.get("MEMORY_TOP_K", "3"))
MEMORY_CHAR_BUDGET = int(os.environ.get("MEMORY_CHAR_BUDGET", "900"))

# ---- ACE playbook (evolving general strategy bullets) ---------------------
# The playbook is the ACE context: itemized GENERAL strategies/gotchas injected
# into the solver each task. It is seeded with the lessons in this system prompt
# and grown offline by scripts/grow_ledger.py (Reflector + Curator). Toggle
# with ENABLE_PLAYBOOK for clean A/B comparisons.
ENABLE_PLAYBOOK = os.environ.get("ENABLE_PLAYBOOK", "1") != "0"
PLAYBOOK_AUTOSEED = os.environ.get("PLAYBOOK_AUTOSEED", "1") != "0"
PLAYBOOK_MAX_BULLETS = int(os.environ.get("PLAYBOOK_MAX_BULLETS", "40"))
PLAYBOOK_CHAR_BUDGET = int(os.environ.get("PLAYBOOK_CHAR_BUDGET", "6000"))
# pull a couple of semantically-relevant bullets from HydraDB at inject time so
# HydraDB is genuinely in the retrieval path (the bonus integration).
PLAYBOOK_HYDRA_TOPK = int(os.environ.get("PLAYBOOK_HYDRA_TOPK", "0"))

# ---- local API-doc snapshot (extra retrieval source) ----------------------
# Injects authoritative API signatures for the task's likely apps from the
# committed assets/api_docs.json snapshot, so the solver starts with real
# schemas instead of spending turns rediscovering them. Advisory only.
ENABLE_API_DOCS = os.environ.get("ENABLE_API_DOCS", "1") != "0"
API_DOCS_TOP_K = int(os.environ.get("API_DOCS_TOP_K", "6"))
API_DOCS_CHAR_BUDGET = int(os.environ.get("API_DOCS_CHAR_BUDGET", "2500"))

_CLIENTS = {}


def split_model(model: str | None) -> tuple[str, str]:
    """Resolve provider/model or provider:model into a concrete client + model."""
    model = (model or MODEL).strip()
    if model == "azure-llama":
        return "azure-llama", os.environ.get("AZURE_LLAMA_MODEL", "llama-3.3-70b-instruct")
    if model == "openrouter-llama":
        return "openrouter", OPENROUTER_LLAMA_MODEL
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider.strip().lower(), model_name.strip()
    if ":" in model:
        provider, model_name = model.split(":", 1)
        return provider.strip().lower(), model_name.strip()

    lower = model.lower()
    if lower.startswith(("llama", "meta-llama", "mixtral", "gemma", "qwen", "deepseek")):
        return os.environ.get("LLM_PROVIDER", "openrouter").lower(), model
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai", model
    if lower.startswith("claude"):
        return "anthropic", model

    # Keep OpenRouter as the default when the provider is omitted.
    return os.environ.get("LLM_PROVIDER", "openrouter").lower(), model


def get_client(provider: str):
    if provider in _CLIENTS:
        return _CLIENTS[provider]
    if provider == "groq":
        _CLIENTS[provider] = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
    elif provider == "openrouter":
        headers = {
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://github.com/interface4agi/hack_agent_arena"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "HydraDB Agent Arena"),
        }
        _CLIENTS[provider] = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            default_headers=headers,
            timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "120")),
            max_retries=int(os.environ.get("OPENROUTER_MAX_RETRIES", "3")),
        )
    elif provider == "azure-llama":
        azure_key = os.environ.get("AZURE_LLAMA_API_KEY")
        if os.environ.get("AZURE_LLAMA_AUTH", "").lower() == "entra" or not azure_key:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            azure_key = get_bearer_token_provider(
                DefaultAzureCredential(),
                os.environ.get("AZURE_LLAMA_TOKEN_SCOPE", "https://ai.azure.com/.default"),
            )
        base_url = (os.environ.get("AZURE_LLAMA_BASE_URL") or "").rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url = f"{base_url}/openai/v1"
        _CLIENTS[provider] = OpenAI(
            api_key=azure_key,
            base_url=base_url,
        )
    elif provider == "openai":
        _CLIENTS[provider] = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif provider == "anthropic":
        _CLIENTS[provider] = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}' for model '{MODEL}'")
    return _CLIENTS[provider]


def _init_memory():
    if not ENABLE_MEMORY or SkillMemory is None:
        if _MEMORY_IMPORT_ERROR is not None:
            print(f"  [memory] disabled (import error: {_MEMORY_IMPORT_ERROR})")
        return None
    try:
        return SkillMemory()
    except Exception as exc:
        print(f"  [memory] disabled (init error: {exc})")
        return None


MEMORY = _init_memory()


def _init_playbook():
    if not ENABLE_PLAYBOOK or Playbook is None:
        if _PLAYBOOK_IMPORT_ERROR is not None:
            print(f"  [playbook] disabled (import error: {_PLAYBOOK_IMPORT_ERROR})")
        return None
    try:
        pb = Playbook()
        if PLAYBOOK_AUTOSEED and len(pb) == 0 and seed_playbook is not None:
            report = seed_playbook(pb)
            print(f"  [playbook] seeded defaults: {report}")
        return pb
    except Exception as exc:
        print(f"  [playbook] disabled (init error: {exc})")
        return None


PLAYBOOK = _init_playbook()


def _init_apidocs():
    if not ENABLE_API_DOCS or ApiDocStore is None:
        if _APIDOCS_IMPORT_ERROR is not None:
            print(f"  [apidocs] disabled (import error: {_APIDOCS_IMPORT_ERROR})")
        return None
    try:
        store = ApiDocStore()
        if not store.available:
            print("  [apidocs] disabled (snapshot assets/api_docs.json not found)")
            return None
        return store
    except Exception as exc:
        print(f"  [apidocs] disabled (init error: {exc})")
        return None


APIDOCS = _init_apidocs()

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
- App login pattern: the login API usually wants username=<supervisor email>
  and password=<password from show_account_passwords() for that app's
  account_name>. The password record normally does NOT include an email field.
  Check the login doc's username description: some apps (e.g. the phone app)
  use the supervisor's phone_number as the username instead of the email.
- Be efficient. One turn may contain many safe read/API calls, loops, filtering,
  calculations, and the final completion call. Do not waste one turn per tiny
  lookup when calls can be bundled in one Python block.
- Never invent API names or fields. If you need an API schema, fetch the docs
  for all likely APIs in the same block, then use the exact parameter names.
- Prefer this workflow:
  1. Infer the relevant app(s) from the task.
  2. In the first useful block, get supervisor credentials and relevant API docs.
  3. Login to needed apps.
  4. Fetch enough data with pagination / page_limit where available.
  5. Compute the requested result or perform the requested mutation.
  6. Verify if cheap, then call complete_task.
- Print compact summaries, not giant raw dumps. Use Python to filter and show
  only the data needed for the next decision.
- For read-only question tasks, call complete_task(answer=...) as soon as you
  have the answer. For action tasks, call complete_task(answer=None) only after
  the required app state has been changed.
- Guard against false completion:
  - If the task asks for a list, count, name, title, amount, date, or any other
    answer, never call complete_task with an empty string/list unless the task
    explicitly asks for an empty result.
  - If an expected field is missing (for example play_count, genre, status,
    amount), inspect the API doc and print representative samples before
    deciding.
  - Before complete_task(answer=...), assign the exact answer string to a
    variable named final_answer and print {"final_answer": final_answer,
    "evidence": ...}. Then pass that same final_answer.
  - For top-N/ranking tasks, deduplicate entities, verify the sort key exists,
    and ensure you have at least N candidates when the data appears to contain
    enough candidates.
- If a call fails, immediately inspect that API's doc and retry with corrected
  names/types in the next block. Recover from errors; never give up or guess
  around an error.
- Inspect before you compute (CRITICAL):
  - Before aggregating, counting, or sorting a collection, print len(...) and
    the keys of one sample item, e.g. print(len(items), items[0].keys()).
  - NEVER call .get("field") on a guessed field name. Confirm the exact field
    exists in printed output or the API doc first. Closely related APIs often
    use different field names (a list endpoint vs a single-item detail endpoint
    frequently differ).
- Ranking / aggregation discipline (prevents wrong top-N and counts):
  - Build the set of UNIQUE entity ids (e.g. song_id) across every in-scope
    source first, deduplicating across song library, albums, and playlists.
  - Fetch each unique entity's AUTHORITATIVE detail record exactly once (e.g.
    show_song(song_id=...)) and read the ranking/filter metric (play_count,
    genre, amount, price, date) from THAT record.
  - Never read the ranking metric from list/library entries — they frequently
    omit it (play_count is usually only on the detail record). Never create
    more than one row per entity; one entity = one record = one metric value.
  - Re-read the task scope literally (which collections count) and the metric
    ("most played", "highest rated", "largest") before sorting.
- Always paginate to exhaustion. List endpoints are paged; a single default
  page is almost never the full data. Use a helper such as:
    def fetch_all(fn, **kw):
        out, page = [], 0
        while True:
            batch = fn(page_index=page, page_limit=20, **kw)  # page_limit max is usually 20
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 20:
                break
            page += 1
        return out
- Action / mutation completeness (payments, messages, creating records):
  - Before ANY mutating call, read that API's doc in an EARLIER turn and use the
    EXACT parameter names it lists. Do NOT fetch the doc and call the API in the
    same block — you cannot adapt to the real names that way, and a guessed
    keyword is silently ignored (e.g. a payment "note" is usually the parameter
    named `description`, not `note`).
  - Pass EVERY detail the task specifies. If it names a memo/note/description,
    an exact amount, an exact message text, a recipient, a date, or a title,
    map each to the correct documented parameter and include it. Optional-looking
    fields like description are graded — never omit them.
  - Extract values precisely from the source (the conversation, bill, email).
    Do not approximate or guess an amount; read the exact number from the data.
  - After the mutation, re-read the task and confirm each required field was set
    with the exact requested value (re-fetch the created record if possible)
    before completing.
- Relative dates: treat the environment "now" given in the task context as the
  current date. Compute "today / this year / last year / this month / recent"
  from THAT date, never from real-world wall-clock time. If no current date was
  provided, look for a clock/now source before assuming.
- Accuracy beats speed. Bundle safe read calls when convenient, but take as many
  turns as you need to inspect schemas, expand albums/playlists into their
  songs, and double-check the result. A confident wrong answer scores zero.
- Output discipline: respond with EXACTLY one Python code block, no prose,
  headings, or English outside Python comments/strings. Never call exit(),
  quit(), or raise to stop; if blocked, print the missing facts and continue.
- Self-check immediately before completing:
  - Assign the exact answer to final_answer and print {"final_answer": ...,
    "evidence": ...} in the SAME block you complete in.
  - Assert the shape: counts must be digit strings derived from printed data;
    top-N answers must contain exactly N deduplicated items; never submit an
    empty string/list or a placeholder like "done" for a question task.
- When and ONLY when the task is fully done, call:
    apis.supervisor.complete_task(answer=<answer>)   # answer=None if not a question
"""

ROUTER_PROMPT = """Classify the AppWorld task for the solver.
Return compact RAW JSON only. Do not use markdown fences. Use keys:
apps: likely app names,
task_type: one of answer|mutation|mixed,
risk: low|medium|high,
needs_answer: boolean,
needs_ranking: boolean,
notes: short tactical advice.
"""

VERIFIER_PROMPT = """You verify AppWorld completion code before it runs.
Return exactly one line:
APPROVE
or
REVISE: <short reason>

You receive recent execution context (printed observations). Use it as the
source of truth. APPROVE only when the context provides evidence that the answer
was actually DERIVED from retrieved data. REVISE when any of these hold:
- The answer is empty/placeholder for a question task ("done", "Task already
  completed", "Final answer provided", etc.).
- The final_answer value does NOT appear in (or follow from) the printed
  evidence — i.e. it looks guessed rather than computed.
- A COUNT answer is suspicious: it equals a default page size (like 5) or was
  computed without paginating all pages, or the printed collection lengths do
  not support it.
- A TOP-N / ranking answer does not contain exactly N deduplicated items, or
  the sort key was never confirmed in the data.
- The code reads a guessed field (e.g. .get("song_ids") on a detail object that
  actually exposes "songs") so the aggregation likely under-collected data.
- It mutates unrelated data, or ignores an error/empty result it should handle.

Do not require perfection. If the context clearly shows the data was retrieved,
paginated, and the answer derived from it, APPROVE even if the final block only
prints/submits the already-derived answer. Never falsely reject by claiming
there was no evidence when the context contains it.
"""

APPWORLD_SCHEMA_PROMPT = """

APPWORLD SCHEMA PATTERNS (common, but ALWAYS confirm field names from printed
output before relying on them; a wrong field guess silently returns nothing):
- App namespaces are lowercase: use apis.spotify, apis.venmo, apis.phone, etc.
  NEVER use apis.Spotify or capitalized app names in code.
- Supervisor password records use key "account_name", not "app_name".
  Example: next(x for x in apis.supervisor.show_account_passwords()
                if x["account_name"] == "spotify")["password"]
- For app login, use username=<supervisor email from the task prompt> and the
  password from show_account_passwords(). Password records usually do not
  contain an email/username.
- App descriptions from api_docs use key "name", not "app_name".
- Many AppWorld list APIs return a Python list directly. Do not call .get() on
  a list. Inspect sample types if unsure.
- LIBRARY/list endpoints and DETAIL endpoints DIFFER. Print the keys of a
  sample from each before reading fields. In Spotify specifically:
  - page_limit must be <= 20 (paginate to get everything).
  - show_song_library / show_album_library / show_playlist_library return LISTS.
  - song library items use "song_id".
  - album LIBRARY items (show_album_library) include "album_id" and a
    "song_ids" list; playlist LIBRARY items (show_playlist_library) include
    "playlist_id" and a "song_ids" list.
  - BUT album/playlist DETAIL responses (show_album / show_playlist) instead
    return a "songs" list of {"id","title",...} objects and do NOT contain
    "song_ids". To expand an album/playlist into its tracks, either read
    "song_ids" from the LIBRARY item or read "songs" from the DETAIL item —
    print one to see which you have.
  - show_song(song_id=...) does not need access_token.
  - show_playlist(playlist_id=..., access_token=...) needs playlist_id.
- Never submit generic placeholders. If the task asks for a count/list/title,
  final_answer must be exactly that count/list/title string, not "done",
  "Task already completed", or explanatory prose.
"""

LLAMA_STYLE_PROMPT = """

LLAMA/AZURE STYLE (you tend to hallucinate field/API names — counter it):
- Work in deliberate steps, not one giant guess. A good sequence is:
  (1) fetch credentials + the docs for every API you might call,
  (2) log in,
  (3) fetch and PAGINATE the relevant lists, then print len() and one sample's
      keys for each list,
  (4) only after seeing the real keys, expand/compute,
  (5) print {"final_answer": ..., "evidence": ...} and complete.
- Be literal and schema-safe: use ONLY field names you have seen in printed API
  docs or printed samples. If you are unsure a field exists, print a sample
  first. Do not assume "song_ids" vs "songs" — check.
- If a retrieved skill or your own memory conflicts with the live docs/sample,
  the live docs/sample win.
- Before complete_task(answer=...), assert the answer shape in code:
  counts are digit strings; top-N lists have exactly N items. If the result
  looks suspiciously small (e.g. equals a default page size), you probably did
  not paginate — go back and fetch all pages.
"""

HAIKU_STYLE_PROMPT = """

HAIKU STYLE:
- No narration outside the Python code block.
- Batch work aggressively. Do not spend turns printing one doc at a time.
- If you already have enough evidence, complete immediately.
"""


def prompt_for_model(system_prompt: str, model: str) -> str:
    """Attach model-specific prompt profile without changing AppWorld rules."""
    prompt = system_prompt
    if system_prompt == SYSTEM_PROMPT:
        prompt += APPWORLD_SCHEMA_PROMPT
        provider, model_name = split_model(model)
        normalized = f"{provider}/{model_name}".lower()
        if "llama" in normalized or "azure-llama" in normalized:
            prompt += LLAMA_STYLE_PROMPT
        if "haiku" in normalized:
            prompt += HAIKU_STYLE_PROMPT
    return prompt


def call_llm(
    messages: list[dict],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = SOLVER_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> str:
    provider, model_name = split_model(model)
    client = get_client(provider)
    final_system_prompt = prompt_for_model(system_prompt, model)
    if provider in {"groq", "openai", "azure-llama", "openrouter"}:
        request = {
            "model": model_name,
            "messages": [{"role": "system", "content": final_system_prompt}, *messages],
        }
        if provider == "openai" and model_name.startswith("gpt-5"):
            request["max_completion_tokens"] = max_tokens
        else:
            request["max_tokens"] = max_tokens
            request["temperature"] = 0.0
        resp = client.chat.completions.create(**request)
        return resp.choices[0].message.content or ""

    if provider == "anthropic":
        resp = client.messages.create(
            model=model_name,
            system=final_system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    raise ValueError(f"Unsupported LLM provider '{provider}'")


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    # AppWorld's rule is exactly one Python block. Executing prose as Python
    # burns turns on syntax errors, especially with smaller models.
    return ""


def preview(value: object) -> str:
    text = str(value).replace("\r", "")
    if len(text) <= OUTPUT_PREVIEW_CHARS:
        return repr(text)
    return repr(text[:OUTPUT_PREVIEW_CHARS] + "... <truncated>")


def prompt_observation(value: object) -> str:
    """Bound execution output before feeding it back into the LLM context."""
    text = str(value).replace("\r", "")
    if len(text) <= OBSERVATION_CHARS:
        return text
    return text[:OBSERVATION_CHARS] + "\n... <observation truncated; ask for focused samples if needed>"


def describe_model(model: str) -> str:
    provider, model_name = split_model(model)
    return f"{provider}/{model_name}"


def classify_task(world: AppWorld) -> str:
    message = {
        "role": "user",
        "content": f"Supervisor: {world.task.supervisor}\nTask: {world.task.instruction}",
    }
    try:
        return call_llm(
            [message],
            system_prompt=ROUTER_PROMPT,
            model=ROUTER_MODEL,
            max_tokens=ROUTER_MAX_TOKENS,
        ).strip()
    except Exception as e:
        return f'{{"notes":"router failed: {e}"}}'


def parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def needs_completion_verification(code: str) -> bool:
    if not ENABLE_COMPLETION_VERIFIER:
        return False
    if "complete_task" not in code:
        return False
    return "answer=None" not in code.replace(" ", "")


def recent_context(messages: list[dict]) -> str:
    chunks = []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        if "Execution output:" not in content:
            continue
        chunks.append(content)
        joined = "\n\n".join(reversed(chunks))
        if len(joined) >= VERIFIER_CONTEXT_CHARS:
            return joined[-VERIFIER_CONTEXT_CHARS:]
    return "\n\n".join(reversed(chunks))[-VERIFIER_CONTEXT_CHARS:]


def verify_completion(
    world: AppWorld,
    task_profile: str,
    code: str,
    messages: list[dict] | None = None,
) -> str:
    context = recent_context(messages or [])
    message = {
        "role": "user",
        "content": (
            f"Task: {world.task.instruction}\n\n"
            f"Task profile: {task_profile}\n\n"
            f"Recent execution context:\n{context or '(none)'}\n\n"
            f"Code about to execute:\n```python\n{code}\n```"
        ),
    }
    try:
        verdict = call_llm(
            [message],
            system_prompt=VERIFIER_PROMPT,
            model=VERIFIER_MODEL,
            max_tokens=VERIFIER_MAX_TOKENS,
        ).strip()
        return verdict or "APPROVE"
    except Exception as e:
        return f"REVISE: verifier failed: {e}"


_APP_CALL_RE = re.compile(r"apis\.([a-z_][a-z0-9_]*)\.", re.I)
_NON_APP_NAMESPACES = {"api_docs", "supervisor"}


def apps_from_profile(task_profile: str) -> list[str]:
    data = parse_json_object(task_profile)
    apps = data.get("apps") or []
    if isinstance(apps, str):
        apps = [apps]
    return [str(a).lower() for a in apps if a]


def apps_from_code(code: str) -> list[str]:
    seen: list[str] = []
    for m in _APP_CALL_RE.findall(code or ""):
        ns = m.lower()
        if ns not in _NON_APP_NAMESPACES and ns not in seen:
            seen.append(ns)
    return seen


def retrieve_skills(instruction: str, task_profile: str) -> str:
    """Return a compact, advisory 'Retrieved skills' block (or empty string)."""
    if MEMORY is None:
        return ""
    apps_hint = apps_from_profile(task_profile)
    query = f"{instruction}\nApps: {', '.join(apps_hint)}"
    try:
        blocks = MEMORY.recall(query, k=MEMORY_TOP_K, apps_hint=apps_hint)
    except Exception as exc:
        print(f"  [memory] recall error: {exc}")
        return ""
    if not blocks:
        return ""
    trimmed = []
    for i, b in enumerate(blocks, 1):
        text = b.strip()
        if len(text) > MEMORY_CHAR_BUDGET:
            text = text[:MEMORY_CHAR_BUDGET] + " ... <truncated>"
        trimmed.append(f"[skill {i}]\n{text}")
    print(f"  recall: {len(trimmed)} skill(s) (hydra={MEMORY.hydra_enabled})")
    return (
        "Retrieved skills from past runs (HINTS ONLY, not ground truth — always "
        "verify against live API docs and this task's exact wording):\n\n"
        + "\n\n".join(trimmed)
    )


def retrieve_api_docs(instruction: str, task_profile: str) -> str:
    """Return an app-filtered API-reference block from the local snapshot."""
    if APIDOCS is None:
        return ""
    apps_hint = apps_from_profile(task_profile)
    if not apps_hint:
        return ""
    try:
        block = APIDOCS.retrieve(
            apps_hint,
            instruction,
            k=API_DOCS_TOP_K,
            char_budget=API_DOCS_CHAR_BUDGET,
        )
    except Exception as exc:
        print(f"  [apidocs] retrieve error: {exc}")
        return ""
    if block:
        print(f"  apidocs: {len(block)} chars for apps={apps_hint}")
    return block


def retrieve_playbook(instruction: str, task_profile: str) -> str:
    """Render the ACE playbook (general strategies) for the solver context."""
    if PLAYBOOK is None:
        return ""
    apps_hint = apps_from_profile(task_profile)
    try:
        block = PLAYBOOK.render(
            apps_hint=apps_hint,
            max_bullets=PLAYBOOK_MAX_BULLETS,
            char_budget=PLAYBOOK_CHAR_BUDGET,
        )
    except Exception as exc:
        print(f"  [playbook] render error: {exc}")
        return ""
    if not block:
        return ""
    # optionally fold in a few HydraDB-recalled bullets (keeps HydraDB in the
    # live retrieval path); deduped by HydraDB content, advisory only.
    if PLAYBOOK_HYDRA_TOPK > 0:
        try:
            query = f"{instruction}\nApps: {', '.join(apps_hint)}"
            extra = PLAYBOOK.recall_hydra(query, k=PLAYBOOK_HYDRA_TOPK)
            if extra:
                block += "\n\n## related (HydraDB recall)\n" + "\n".join(
                    f"- {e.strip()[:300]}" for e in extra
                )
        except Exception:
            pass
    print(f"  playbook: {len(PLAYBOOK)} bullet(s) (hydra={PLAYBOOK.hydra_enabled})")
    return block


def capture_skill(world: AppWorld, messages: list[dict], task_profile: str, completed: bool) -> None:
    """Lightweight within-run capture so similar later tasks can reuse the flow."""
    if MEMORY is None or Skill is None or not ENABLE_MEMORY_CAPTURE:
        return
    try:
        instruction = str(world.task.instruction)
        # find the last assistant code block that actually completed the task
        final_code = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                code = extract_code(msg.get("content", ""))
                if "complete_task" in code:
                    final_code = code
                    break
        apps = apps_from_code(final_code) or apps_from_profile(task_profile)
        is_answer = "answer=None" not in (final_code.replace(" ", "")) and "complete_task" in final_code
        skill = Skill(
            title=f"{', '.join(apps) or 'appworld'}: {instruction[:60]}",
            apps=apps,
            task_type="answer" if is_answer else "mutation",
            when_to_use=instruction[:200],
            procedure=(
                "Reference solving code from a prior run (verify before reuse):\n"
                f"```python\n{final_code[:1500]}\n```"
            ),
            outcome="completed" if completed else "fail",
            source_task_id=getattr(world.task, "id", ""),
            source_experiment=EXPERIMENT,
            tags=apps,
        )
        MEMORY.store(skill)
    except Exception as exc:
        print(f"  [memory] capture error: {exc}")


def error_signature(output: str) -> str | None:
    """Stable signature for a failed execution so we can detect repeat loops."""
    text = str(output)
    if "Execution failed" not in text and "status code is" not in text:
        return None
    # prefer the concrete exception / validation line
    m = re.search(r"(Exception|Error|message):.*", text)
    sig = (m.group(0) if m else text).strip()
    return re.sub(r"\s+", " ", sig)[:200]


def solve(world: AppWorld) -> None:
    task_profile = classify_task(world)
    print(f"  route: {preview(task_profile)}")
    apidocs_block = retrieve_api_docs(str(world.task.instruction), task_profile)
    playbook_block = retrieve_playbook(str(world.task.instruction), task_profile)
    skills_block = retrieve_skills(str(world.task.instruction), task_profile)
    env_now = getattr(world.task, "datetime", None)
    messages = [{
        "role": "user",
        "content": (
            f"Supervisor: {world.task.supervisor}\n\n"
            + (f"Current date/time in this environment (use this as 'now' for "
               f"any relative dates like today/this year/last month): {env_now}\n\n"
               if env_now else "")
            + f"Task: {world.task.instruction}\n\n"
            f"Task profile: {task_profile}\n\n"
            + (apidocs_block + "\n\n" if apidocs_block else "")
            + (playbook_block + "\n\n" if playbook_block else "")
            + (skills_block + "\n\n" if skills_block else "")
            + "Begin. One Python code block per turn. First discover docs and "
            "inspect real data shapes, paginate all lists, then compute and "
            "verify your answer before completing. Correctness over speed."
        ),
    }]
    last_output = ""
    executed_steps = 0
    model_turns = 0
    verifier_rejections = 0
    last_error_sig = None
    error_repeats = 0
    while executed_steps < MAX_INTERACTIONS and model_turns < MAX_MODEL_TURNS:
        model_turns += 1
        step_model = ERROR_MODEL if "Execution failed" in last_output else SOLVER_MODEL
        reply = call_llm(messages, model=step_model)
        code = extract_code(reply)
        if not code.strip():
            print(f"  model turn {model_turns}: no Python code block; retrying")
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response did not contain a Python code block. "
                    "Reply with exactly one fenced ```python code block and no prose."
                ),
            })
            continue
        # Verify completions, but never let the verifier deadlock the task:
        # after MAX_VERIFIER_REJECTIONS pushbacks we let the code execute.
        if needs_completion_verification(code) and verifier_rejections < MAX_VERIFIER_REJECTIONS:
            verdict = verify_completion(world, task_profile, code, messages)
            print(f"  verifier: {preview(verdict)}")
            if not verdict.upper().startswith("APPROVE"):
                verifier_rejections += 1
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": (
                        "Do not execute that completion yet. The verifier said:\n"
                        f"{verdict}\n\nRevise with one Python code block. Gather "
                        "more evidence or fix the final answer."
                    ),
                })
                last_output = verdict
                continue
        output = world.execute(code)
        last_output = str(output)
        executed_steps += 1
        print(f"  step {executed_steps}: ran {len(code)} chars -> {preview(output)}")
        messages.append({"role": "assistant", "content": reply})
        remaining = MAX_INTERACTIONS - executed_steps
        next_instruction = f"Execution output:\n{prompt_observation(output)}"
        # loop-breaker: detect the same error repeating and force a new approach
        sig = error_signature(output)
        if sig and sig == last_error_sig:
            error_repeats += 1
        elif sig:
            last_error_sig = sig
            error_repeats = 1
        else:
            last_error_sig = None
            error_repeats = 0
        if error_repeats >= 2:
            next_instruction += (
                f"\n\nSTOP: this same error has occurred {error_repeats} times. "
                "Do not repeat the same call again. Re-read the relevant API doc's "
                "parameter descriptions exactly and change your APPROACH or the "
                "VALUE you pass (not just the parameter name). For login, the "
                "username field's value may be the supervisor's phone number "
                "string rather than the email. In the next block, first print the "
                "exact API doc and the values you intend to pass, then call it once."
            )
        if remaining <= 6 and not world.task_completed():
            next_instruction += (
                f"\n\nOnly {remaining} turn(s) remain. If you have enough evidence, "
                "finish in the next Python block with final_answer, compact evidence, "
                "and complete_task. Do not print more docs unless absolutely necessary."
            )
        messages.append({"role": "user", "content": next_instruction})
        if world.task_completed():
            print("  OK task_completed")
            capture_skill(world, messages, task_profile, completed=True)
            return
    print(
        "  FAILED hit limit without completion "
        f"(executed_steps={executed_steps}, model_turns={model_turns})"
    )
    capture_skill(world, messages, task_profile, completed=False)


def main() -> None:
    task_ids = load_task_ids(DATASET)
    if MAX_TASKS:
        task_ids = task_ids[:MAX_TASKS]
    print(f"Running '{EXPERIMENT}' on {len(task_ids)} '{DATASET}' tasks")
    print(
        "Models: "
        f"router={describe_model(ROUTER_MODEL)}, solver={describe_model(SOLVER_MODEL)}, "
        f"error={describe_model(ERROR_MODEL)}, verifier={describe_model(VERIFIER_MODEL)}"
    )
    if MEMORY is not None:
        print(f"Memory: enabled (hydra={MEMORY.hydra_enabled}, top_k={MEMORY_TOP_K})")
    else:
        print("Memory: disabled")
    if PLAYBOOK is not None:
        print(f"Playbook: enabled ({len(PLAYBOOK)} bullets, hydra={PLAYBOOK.hydra_enabled})")
    else:
        print("Playbook: disabled")
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
