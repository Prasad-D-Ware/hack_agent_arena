# AppWorld Agent Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive single-agent ReAct loop in `agent.py` with a Planner → Executor → Verifier architecture over a shared Blackboard, wired to the merged HydraDB memory/retrieval layer, to raise AppWorld TGC.

**Architecture:** A `solve(world, state, mem)` orchestrator drives a Planner (decompose into subgoals), an Executor (tight ReAct loop per subgoal with HydraDB API-doc retrieval), and a Verifier (gate each subgoal and the final `complete_task`). All roles read/write a structured `Blackboard` instead of a growing transcript. Error recovery, budgets, and OpenRouter resilience are built in. Extension points are left for Approach-C techniques (best-of-N, self-consistency, Reflexion).

**Tech Stack:** Python 3.11, `appworld==0.1.3`, `openai` SDK against OpenRouter, `hydradb-sdk` (optional, no-op when `USE_HYDRA!=1`), `pytest` for tests.

**Source of truth:** `docs/superpowers/specs/2026-06-06-agent-architecture-design.md`

---

## File Structure

New modules (all at repo root, alongside `agent.py` / `hydradb.py`):

| File | Responsibility |
|------|----------------|
| `config.py` | Budget/limit constants read from env (single place). |
| `parsing.py` | `extract_code(text)` — pull the one python block from an LLM reply. |
| `state.py` | `Subgoal`, `StepRecord`, `error_signature`, `Blackboard` (+ `render_for`, `add_step`, repeated-error detection). |
| `llm.py` | `call_llm(role, messages, ...)` — OpenRouter client, per-role model routing, retry/backoff, fallback. |
| `prompts.py` | `PLANNER_SYSTEM`, `EXECUTOR_SYSTEM`, `VERIFIER_SYSTEM` strings. |
| `planner.py` | `make_plan`, `replan`, `_parse_plan`. |
| `executor.py` | `run(subgoal, state, world, mem, ...)` — per-subgoal ReAct loop. |
| `verifier.py` | `verify_subgoal`, `verify_final`, `_parse_verdict`. |
| `orchestrator.py` | `solve`, `finalize` — drives the whole task. |
| `hydradb.py` (modify) | add `kind` arg to `recall`; add `remember_episode`. |
| `agent.py` (modify) | `main()` builds `Blackboard` + calls `solve`; keeps env/CLI contract. |

Tests under `tests/`: `conftest.py` (fakes) + one test file per module.

**Conventions used throughout:**
- Every role function takes an injectable `llm=call_llm` so tests pass a `FakeLLM`.
- `world` is any object with `.execute(code) -> str` and `.task_completed() -> bool`.
- `mem` is any object with `.recall(query, kind=...) -> str`, `.ingest_api_docs(world)`, `.remember_episode(instruction, state, success)`.
- Run tests with the venv active: `source .venv/bin/activate` then `python -m pytest`.

---

### Task 1: Test scaffolding & shared fakes

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Create the dev requirements file**

`requirements-dev.txt`:
```
pytest>=8,<9
```

- [ ] **Step 2: Create pytest config**

`pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Create the test package marker**

`tests/__init__.py`:
```python
```
(empty file)

- [ ] **Step 4: Create shared fakes in conftest**

`tests/conftest.py`:
```python
"""Shared test doubles for the agent architecture."""
from types import SimpleNamespace


class FakeLLM:
    """Callable matching call_llm(role, messages, system=None, **kw).

    `responses` is either a list (consumed in order, any role) or a dict
    role -> list (consumed per role). Records every call for assertions.
    """

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, role, messages, system=None, **kw):
        self.calls.append((role, messages, system))
        q = self.responses.get(role) if isinstance(self.responses, dict) else self.responses
        if isinstance(q, list):
            return q.pop(0)
        return q


class FakeWorld:
    """world.execute(code) returns scripted outputs in order; any code
    containing 'complete_task' marks the task completed."""

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.executed = []
        self._completed = False

    def execute(self, code):
        self.executed.append(code)
        if "complete_task" in code:
            self._completed = True
            return "Task marked complete."
        return self.outputs.pop(0) if self.outputs else "ok"

    def task_completed(self):
        return self._completed


class FakeMem:
    """Stand-in for HydraMemory."""

    def __init__(self, recall_value=""):
        self.recall_value = recall_value
        self.remembered = []
        self.ingested = 0

    def recall(self, query, kind="all"):
        return self.recall_value

    def ingest_api_docs(self, world):
        self.ingested += 1

    def remember_episode(self, instruction, state, success):
        self.remembered.append((instruction, success))


def make_chat_resp(content):
    """Build an object shaped like an OpenAI chat completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
```

- [ ] **Step 5: Verify the fakes import cleanly**

Run: `source .venv/bin/activate && python -c "import tests.conftest as c; print(c.FakeLLM, c.FakeWorld, c.FakeMem)"`
Expected: prints the three class objects, no error.

- [ ] **Step 6: Commit**

```bash
git add tests/__init__.py tests/conftest.py pytest.ini requirements-dev.txt
git commit -m "test: add pytest scaffolding and shared fakes"
```

---

### Task 2: `config.py` and `parsing.py`

**Files:**
- Create: `config.py`
- Create: `parsing.py`
- Create: `tests/test_parsing.py`

- [ ] **Step 1: Write the failing test for extract_code**

`tests/test_parsing.py`:
```python
from parsing import extract_code


def test_extracts_fenced_python_block():
    text = "Here you go:\n```python\nprint(1)\n```\nthanks"
    assert extract_code(text) == "print(1)"


def test_extracts_unlabeled_fence():
    text = "```\nx = 2\n```"
    assert extract_code(text) == "x = 2"


def test_no_fence_returns_stripped_text():
    assert extract_code("  print(3)  ") == "print(3)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parsing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'parsing'`.

- [ ] **Step 3: Implement config.py and parsing.py**

`config.py`:
```python
"""Budget/limit constants for the agent loop (override via env)."""
import os


def _int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


MAX_INTERACTIONS = _int("MAX_INTERACTIONS", "40")     # global LLM-execute turns per task
MAX_SUBGOAL_STEPS = _int("MAX_SUBGOAL_STEPS", "10")   # executor turns per subgoal
MAX_REPLANS = _int("MAX_REPLANS", "2")                # planner replans per task
MAX_SUBGOAL_RETRIES = _int("MAX_SUBGOAL_RETRIES", "1")  # re-run a failed subgoal before replanning
MAX_FINALIZE = _int("MAX_FINALIZE", "3")              # complete_task attempts
```

`parsing.py`:
```python
"""Pull the single python code block out of an LLM reply."""
import re

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = _FENCE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_parsing.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add config.py parsing.py tests/test_parsing.py
git commit -m "feat: add config constants and extract_code parser"
```

---

### Task 3: `state.py` — dataclasses & `error_signature`

**Files:**
- Create: `state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
from state import Subgoal, StepRecord, error_signature


def test_subgoal_defaults():
    sg = Subgoal(id=1, description="log into spotify")
    assert sg.status == "pending" and sg.result is None and sg.attempts == 0


def test_steprecord_holds_fields():
    s = StepRecord(subgoal_id=2, code="print(1)", output="1", error_signature=None)
    assert s.subgoal_id == 2 and s.code == "print(1)"


def test_error_signature_detects_and_normalizes():
    out = "Traceback (most recent call last):\n  ...\nValueError: amount 1234 invalid"
    assert error_signature(out) == "ValueError: amount # invalid"


def test_error_signature_picks_last_error_line():
    out = "KeyError: 'a'\nsome text\nTypeError: bad"
    assert error_signature(out) == "TypeError: bad"


def test_error_signature_none_for_plain_output():
    assert error_signature("hello world") is None
    assert error_signature("") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'state'`.

- [ ] **Step 3: Implement the dataclasses and error_signature**

`state.py`:
```python
"""The Blackboard: structured task state shared by all roles."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

_ERR_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning)):\s*(.*)$")


@dataclass
class Subgoal:
    id: int
    description: str
    status: Literal["pending", "active", "done", "failed"] = "pending"
    result: str | None = None
    attempts: int = 0


@dataclass
class StepRecord:
    subgoal_id: int
    code: str
    output: str
    error_signature: str | None = None


def error_signature(output: str) -> str | None:
    """Return a normalized 'ExcType: msg' for the LAST error line, else None."""
    if not output:
        return None
    match = None
    for line in str(output).splitlines():
        m = _ERR_LINE.match(line.strip())
        if m:
            match = m
    if not match:
        return None
    exc, msg = match.group(1), match.group(2)
    msg = re.sub(r"0x[0-9a-fA-F]+", "#", msg)   # addresses
    msg = re.sub(r"\d+", "#", msg)              # numbers/ids
    return f"{exc}: {msg[:80]}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_state.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add Subgoal/StepRecord dataclasses and error_signature"
```

---

### Task 4: `state.py` — `Blackboard` step tracking & repeated-error detection

**Files:**
- Modify: `state.py`
- Modify: `tests/test_state.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_state.py`:
```python
from state import Blackboard


def test_add_step_records_and_counts():
    bb = Blackboard(task_instruction="t")
    sig = bb.add_step(1, "print(1)", "1")
    assert sig is None
    assert bb.interactions_used == 1 and len(bb.steps) == 1


def test_add_step_returns_error_signature():
    bb = Blackboard(task_instruction="t")
    sig = bb.add_step(1, "x", "ValueError: nope 5")
    assert sig == "ValueError: nope #"
    assert bb.steps[-1].error_signature == "ValueError: nope #"


def test_has_repeated_error_true_on_same_signature_twice():
    bb = Blackboard(task_instruction="t")
    bb.add_step(1, "x", "KeyError: 'a'")
    bb.add_step(1, "x", "KeyError: 'a'")
    assert bb.has_repeated_error(1) is True


def test_has_repeated_error_false_for_different_subgoals():
    bb = Blackboard(task_instruction="t")
    bb.add_step(1, "x", "KeyError: 'a'")
    bb.add_step(2, "x", "KeyError: 'a'")
    assert bb.has_repeated_error(1) is False


def test_recent_error_signatures_filters_by_subgoal():
    bb = Blackboard(task_instruction="t")
    bb.add_step(1, "x", "KeyError: 'a'")
    bb.add_step(1, "y", "ok")
    assert bb.recent_error_signatures(1, 2) == ["KeyError: 'a'"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_state.py -q`
Expected: FAIL — `ImportError: cannot import name 'Blackboard'`.

- [ ] **Step 3: Implement the Blackboard class (append to `state.py`)**

```python
@dataclass
class Blackboard:
    task_instruction: str
    supervisor: str = ""
    proposed_answer: Any = None
    plan: list[Subgoal] = field(default_factory=list)
    credentials: dict[str, str] = field(default_factory=dict)   # app -> token (write-once)
    api_cache: dict[str, str] = field(default_factory=dict)     # "app.api" -> doc
    results: dict[str, Any] = field(default_factory=dict)       # carried-forward scratch
    steps: list[StepRecord] = field(default_factory=list)
    interactions_used: int = 0
    replans_used: int = 0

    def add_step(self, subgoal_id: int, code: str, output: str) -> str | None:
        sig = error_signature(output)
        self.steps.append(StepRecord(subgoal_id, code, str(output), sig))
        self.interactions_used += 1
        return sig

    def recent_error_signatures(self, subgoal_id: int, n: int = 2) -> list[str]:
        sigs = [s.error_signature for s in self.steps
                if s.subgoal_id == subgoal_id and s.error_signature]
        return sigs[-n:]

    def has_repeated_error(self, subgoal_id: int, n: int = 2) -> bool:
        sigs = self.recent_error_signatures(subgoal_id, n)
        return len(sigs) >= n and len(set(sigs)) == 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_state.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add Blackboard step tracking and repeated-error detection"
```

---

### Task 5: `state.py` — `render_for` compact views

**Files:**
- Modify: `state.py`
- Modify: `tests/test_state.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_state.py`:
```python
def test_render_for_includes_task_and_plan():
    bb = Blackboard(task_instruction="buy milk", supervisor="Alice")
    bb.plan = [Subgoal(1, "open amazon", status="done"),
               Subgoal(2, "add milk", status="active")]
    text = bb.render_for("planner")
    assert "buy milk" in text and "open amazon" in text and "add milk" in text
    assert "[x]" in text and "[>]" in text


def test_render_for_executor_truncates_long_output():
    bb = Blackboard(task_instruction="t")
    bb.add_step(1, "print(x)", "Z" * 1000)
    text = bb.render_for("executor", Subgoal(1, "do thing"))
    assert "do thing" in text
    assert "Z" * 1000 not in text          # long output is truncated
    assert "…" in text


def test_render_for_lists_logged_in_apps():
    bb = Blackboard(task_instruction="t")
    bb.credentials = {"spotify": "tok", "gmail": "tok2"}
    text = bb.render_for("executor", Subgoal(1, "x"))
    assert "spotify" in text and "gmail" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_state.py -q`
Expected: FAIL — `AttributeError: 'Blackboard' object has no attribute 'render_for'`.

- [ ] **Step 3: Implement `render_for` (add method to `Blackboard` in `state.py`)**

```python
    _STATUS_MARK = {"done": "x", "active": ">", "failed": "!", "pending": " "}

    def render_for(self, role: str, subgoal: "Subgoal | None" = None) -> str:
        lines = [f"TASK: {self.task_instruction}"]
        if self.supervisor:
            lines.append(f"SUPERVISOR: {self.supervisor}")
        if self.credentials:
            lines.append("LOGGED IN: " + ", ".join(sorted(self.credentials)))
        if self.results:
            lines.append("KNOWN RESULTS: "
                         + "; ".join(f"{k}={str(v)[:60]}" for k, v in self.results.items()))
        if self.plan:
            lines.append("PLAN:")
            for sg in self.plan:
                mark = self._STATUS_MARK.get(sg.status, " ")
                lines.append(f"  [{mark}] {sg.id}. {sg.description}")
        if subgoal is not None:
            lines.append(f"CURRENT SUBGOAL: {subgoal.description}")
        if role in ("executor", "finalize", "verifier") and self.steps:
            lines.append("RECENT STEPS:")
            for s in self.steps[-3:]:
                out = s.output if len(s.output) <= 200 else s.output[:200] + "…"
                lines.append(f"  $ {s.code[:120]}\n    -> {out}")
        return "\n".join(lines)
```

Note: `_STATUS_MARK` is a class attribute — place it at the top of the `Blackboard` body (above the methods), not inside a method.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_state.py -q`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add Blackboard.render_for compact role views"
```

---

### Task 6: `llm.py` — OpenRouter call with routing, retry, fallback

**Files:**
- Create: `llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:
```python
import llm
from tests.conftest import make_chat_resp


class FakeChatClient:
    def __init__(self, fail_times=0, content="hi"):
        self.fail_times = fail_times
        self.content = content
        self.calls = 0
        self.last_kwargs = None

        class _Completions:
            def create(_inner, **kw):
                self.calls += 1
                if self.calls <= self.fail_times:
                    raise RuntimeError("transient 503")
                self.last_kwargs = kw
                return make_chat_resp(self.content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_returns_content_on_success():
    c = FakeChatClient(content="hello")
    out = llm.call_llm("planner", [{"role": "user", "content": "x"}],
                       client=c, sleep_fn=lambda s: None)
    assert out == "hello"
    assert c.calls == 1


def test_retries_then_succeeds():
    c = FakeChatClient(fail_times=2, content="ok")
    out = llm.call_llm("executor", [{"role": "user", "content": "x"}],
                       client=c, sleep_fn=lambda s: None)
    assert out == "ok"
    assert c.calls == 3


def test_per_role_model_routing(monkeypatch):
    monkeypatch.setenv("MODEL_PLANNER", "test/planner-model")
    c = FakeChatClient(content="ok")
    llm.call_llm("planner", [{"role": "user", "content": "x"}],
                 client=c, sleep_fn=lambda s: None)
    assert c.last_kwargs["model"] == "test/planner-model"


def test_system_prompt_prepended():
    c = FakeChatClient(content="ok")
    llm.call_llm("executor", [{"role": "user", "content": "x"}],
                 system="SYS", client=c, sleep_fn=lambda s: None)
    assert c.last_kwargs["messages"][0] == {"role": "system", "content": "SYS"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm'`.

- [ ] **Step 3: Implement `llm.py`**

```python
"""OpenRouter LLM access with per-role model routing and retry/backoff."""
import os
import time

from openai import OpenAI

_DEFAULT_CLIENT = None


def _client() -> OpenAI:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    return _DEFAULT_CLIENT


def _model_for(role: str) -> str:
    """MODEL_<ROLE> overrides MODEL; MODEL defaults to a strong slug."""
    default = os.environ.get("MODEL", "anthropic/claude-opus-4")
    return os.environ.get(f"MODEL_{role.upper()}", default)


def call_llm(role, messages, system=None, *, client=None, sleep_fn=time.sleep,
             max_tokens=1500, temperature=0.0, max_retries=4) -> str:
    client = client or _client()
    model = _model_for(role)
    fallback = os.environ.get("MODEL_FALLBACK")
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # transient: 429/5xx/network
            last_err = e
            if fallback and attempt == max_retries - 2:
                model = fallback   # last-ditch: swap to the fallback model
            sleep_fn(min(2 ** attempt, 8))
    raise last_err
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat: add llm.call_llm with role routing and retry/backoff"
```

---

### Task 7: `prompts.py` — role system prompts

**Files:**
- Create: `prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/test_prompts.py`:
```python
import prompts


def test_prompts_exist_and_are_nonempty():
    for name in ("PLANNER_SYSTEM", "EXECUTOR_SYSTEM", "VERIFIER_SYSTEM"):
        val = getattr(prompts, name)
        assert isinstance(val, str) and len(val) > 50


def test_executor_prompt_states_one_block_rule():
    assert "one" in prompts.EXECUTOR_SYSTEM.lower()
    assert "complete_task" in prompts.EXECUTOR_SYSTEM


def test_verifier_prompt_demands_pass_fail():
    assert "PASS" in prompts.VERIFIER_SYSTEM and "FAIL" in prompts.VERIFIER_SYSTEM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'prompts'`.

- [ ] **Step 3: Implement `prompts.py`**

```python
"""System prompts for each agent role in the AppWorld loop."""

PLANNER_SYSTEM = """You are the PLANNER for an autonomous agent operating inside AppWorld
(9 everyday apps, 457 APIs). You receive a supervisor's natural-language task.

Decompose it into a SHORT ordered list of concrete, verifiable subgoals — the
minimum needed, each one a single coherent objective (e.g. "Log into Venmo and
list this month's transactions"). Prefer 2-6 subgoals. Do not write code. Do not
explain. Output ONLY a numbered list, one subgoal per line.
"""

EXECUTOR_SYSTEM = """You are the EXECUTOR for an autonomous agent inside AppWorld.
You achieve ONE subgoal at a time by writing Python that the environment runs.

RULES:
- Reply with EXACTLY ONE python code block per turn, nothing else:
  ```python
  # your code
  ```
- The preloaded object `apis` is the ONLY interface to the apps. Whatever you
  print() comes back as the next observation.
- Discover APIs at runtime; never invent API names or fields:
    print(apis.api_docs.show_api_descriptions(app_name='<app>'))
    print(apis.api_docs.show_api_doc(app_name='<app>', api_name='<api>'))
- Credentials: print(apis.supervisor.show_account_passwords()) then call the
  app's login API for an access_token. If the state says you are already logged
  into an app, reuse that token — do not log in again.
- Work in small steps and inspect results before the next action.
- Do NOT call apis.supervisor.complete_task here — finishing is handled separately.
- When (and only when) the current subgoal is achieved, reply with a single line:
    SUBGOAL_DONE: <one-line summary of the result>
"""

VERIFIER_SYSTEM = """You are the VERIFIER for an autonomous agent inside AppWorld.
Given the task, the plan, recent execution evidence, and either a subgoal result
or a proposed completion, judge STRICTLY whether it is actually correct and
complete (right data, right answer format, required side-effects performed).

Reply with EXACTLY one line:
  PASS
or
  FAIL: <short, specific reason and what to fix>
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompts.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add prompts.py tests/test_prompts.py
git commit -m "feat: add role system prompts"
```

---

### Task 8: `planner.py` — make_plan / replan / _parse_plan

**Files:**
- Create: `planner.py`
- Create: `tests/test_planner.py`

- [ ] **Step 1: Write the failing test**

`tests/test_planner.py`:
```python
from planner import make_plan, replan, _parse_plan
from state import Blackboard


def test_parse_plan_handles_numbered_and_bulleted():
    text = "1. log in\n2) list songs\n- like first\n\nnotes here"
    plan = _parse_plan(text)
    assert [sg.description for sg in plan] == ["log in", "list songs", "like first"]
    assert [sg.id for sg in plan] == [1, 2, 3]


def test_make_plan_uses_llm_output(make_llm=None):
    from tests.conftest import FakeLLM
    bb = Blackboard(task_instruction="play my discover weekly")
    llm = FakeLLM({"planner": ["1. log into spotify\n2. play discover weekly"]})
    plan = make_plan(bb, hints="", llm=llm)
    assert [sg.description for sg in plan] == ["log into spotify", "play discover weekly"]
    # the prompt should carry the task text
    assert "discover weekly" in llm.calls[0][1][0]["content"]


def test_make_plan_falls_back_to_single_subgoal_when_empty():
    from tests.conftest import FakeLLM
    bb = Blackboard(task_instruction="do the thing")
    llm = FakeLLM({"planner": ["sorry, no list"]})
    plan = make_plan(bb, hints="", llm=llm)
    assert len(plan) == 1 and plan[0].description == "do the thing"


def test_replan_renumbers_from_one():
    from tests.conftest import FakeLLM
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({"planner": ["1. retry login\n2. fetch again"]})
    plan = replan(bb, failure="login failed", llm=llm)
    assert [sg.id for sg in plan] == [1, 2]
    assert "login failed" in llm.calls[0][1][0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_planner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'planner'`.

- [ ] **Step 3: Implement `planner.py`**

```python
"""Planner role: decompose the task into subgoals; replan on failure."""
import re

from llm import call_llm
from prompts import PLANNER_SYSTEM
from state import Blackboard, Subgoal

_ITEM = re.compile(r"^(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


def _parse_plan(text: str) -> list[Subgoal]:
    out: list[Subgoal] = []
    for line in (text or "").splitlines():
        m = _ITEM.match(line.strip())
        if m:
            out.append(Subgoal(id=len(out) + 1, description=m.group(1).strip()))
    return out


def make_plan(state: Blackboard, hints: str = "", llm=call_llm) -> list[Subgoal]:
    user = state.render_for("planner")
    if hints:
        user += "\n\nPRIOR EXPERIENCE (may help, may be irrelevant):\n" + hints
    user += "\n\nBreak this task into a short ordered, numbered list of subgoals."
    text = llm("planner", [{"role": "user", "content": user}], system=PLANNER_SYSTEM)
    plan = _parse_plan(text)
    return plan or [Subgoal(id=1, description=state.task_instruction)]


def replan(state: Blackboard, failure: str, llm=call_llm) -> list[Subgoal]:
    user = (state.render_for("planner")
            + f"\n\nThe current approach failed: {failure}\n"
            "Provide a REVISED ordered, numbered list of the remaining subgoals.")
    text = llm("planner", [{"role": "user", "content": user}], system=PLANNER_SYSTEM)
    plan = _parse_plan(text) or [Subgoal(id=1, description=state.task_instruction)]
    for i, sg in enumerate(plan, 1):
        sg.id = i
    return plan
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_planner.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add planner.py tests/test_planner.py
git commit -m "feat: add planner make_plan/replan"
```

---

### Task 9: `verifier.py` — verdict parsing & checks

**Files:**
- Create: `verifier.py`
- Create: `tests/test_verifier.py`

- [ ] **Step 1: Write the failing test**

`tests/test_verifier.py`:
```python
from verifier import verify_subgoal, verify_final, _parse_verdict
from state import Blackboard, Subgoal
from tests.conftest import FakeLLM


def test_parse_verdict_pass():
    assert _parse_verdict("PASS") == (True, "")


def test_parse_verdict_fail_with_reason():
    ok, reason = _parse_verdict("FAIL: token missing")
    assert ok is False and reason == "token missing"


def test_parse_verdict_unknown_is_conservative_fail():
    ok, reason = _parse_verdict("hmm not sure")
    assert ok is False and "hmm" in reason


def test_verify_subgoal_passes():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "log in", result="got token")
    llm = FakeLLM({"verifier": ["PASS"]})
    assert verify_subgoal(sg, bb, llm=llm) == (True, "")


def test_verify_final_fail_feedback_reaches_caller():
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({"verifier": ["FAIL: answer should be a number"]})
    ok, reason = verify_final(bb, "apis.supervisor.complete_task(answer='x')", llm=llm)
    assert ok is False and "number" in reason
    # the completion code is shown to the verifier
    assert "complete_task" in llm.calls[0][1][0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verifier.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'verifier'`.

- [ ] **Step 3: Implement `verifier.py`**

```python
"""Verifier role: gate each subgoal and the final completion."""
from llm import call_llm
from prompts import VERIFIER_SYSTEM
from state import Blackboard, Subgoal


def _parse_verdict(text: str) -> tuple[bool, str]:
    t = (text or "").strip()
    up = t.upper()
    if up.startswith("PASS"):
        return True, ""
    if up.startswith("FAIL"):
        return False, t[4:].lstrip(": ").strip() or "verifier failed"
    return False, t[:200] or "verifier gave no verdict"


def verify_subgoal(subgoal: Subgoal, state: Blackboard, llm=call_llm) -> tuple[bool, str]:
    user = (state.render_for("verifier", subgoal)
            + f"\n\nReported subgoal result: {subgoal.result}\n\n"
            "Was this subgoal actually achieved? Reply 'PASS' or 'FAIL: <reason>'.")
    return _parse_verdict(llm("verifier", [{"role": "user", "content": user}],
                              system=VERIFIER_SYSTEM))


def verify_final(state: Blackboard, code: str, llm=call_llm) -> tuple[bool, str]:
    user = (state.render_for("verifier")
            + f"\n\nProposed completion code:\n{code}\n\n"
            "Will this correctly complete the task with the right answer/side-effects? "
            "Reply 'PASS' or 'FAIL: <reason>'.")
    return _parse_verdict(llm("verifier", [{"role": "user", "content": user}],
                              system=VERIFIER_SYSTEM))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_verifier.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add verifier.py tests/test_verifier.py
git commit -m "feat: add verifier subgoal/final gates"
```

---

### Task 10: `executor.py` — per-subgoal ReAct loop

**Files:**
- Create: `executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write the failing test**

`tests/test_executor.py`:
```python
from executor import run
from state import Blackboard, Subgoal
from tests.conftest import FakeLLM, FakeWorld, FakeMem


def test_executor_completes_on_done_marker():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "log in")
    llm = FakeLLM({"executor": [
        "```python\nprint('hi')\n```",
        "SUBGOAL_DONE: logged in",
    ]})
    world = FakeWorld(outputs=["hi"])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "done" and result == "logged in"
    assert bb.interactions_used == 1          # one code execution before DONE


def test_executor_injects_retrieved_knowledge():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "search")
    llm = FakeLLM({"executor": ["SUBGOAL_DONE: ok"]})
    mem = FakeMem(recall_value="API: spotify.search(query)")
    run(sg, bb, FakeWorld(), mem, llm=llm, max_steps=3, interaction_budget=40)
    assert "spotify.search" in llm.calls[0][1][0]["content"]


def test_executor_fails_on_repeated_error():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": [
        "```python\nbad\n```",
        "```python\nbad\n```",
    ]})
    world = FakeWorld(outputs=["KeyError: 'x'", "KeyError: 'x'"])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "failed" and "KeyError" in result


def test_executor_stops_at_global_budget():
    bb = Blackboard(task_instruction="t")
    bb.interactions_used = 40
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": ["```python\nx\n```"]})
    status, _ = run(sg, bb, FakeWorld(), FakeMem(), llm=llm,
                    max_steps=5, interaction_budget=40)
    assert status == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_executor.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'executor'`.

- [ ] **Step 3: Implement `executor.py`**

```python
"""Executor role: a tight ReAct loop that achieves ONE subgoal."""
from config import MAX_INTERACTIONS, MAX_SUBGOAL_STEPS
from llm import call_llm
from parsing import extract_code
from prompts import EXECUTOR_SYSTEM
from state import Blackboard, Subgoal

DONE_MARKER = "SUBGOAL_DONE"


def run(subgoal: Subgoal, state: Blackboard, world, mem, llm=call_llm,
        max_steps: int = MAX_SUBGOAL_STEPS,
        interaction_budget: int = MAX_INTERACTIONS) -> tuple[str, str]:
    """Returns ("done", result) or ("failed", reason)."""
    retrieved = mem.recall(subgoal.description, kind="knowledge")
    seed = state.render_for("executor", subgoal)
    if retrieved:
        seed += "\n\nRETRIEVED API KNOWLEDGE:\n" + retrieved
    seed += (f"\n\nWork on the current subgoal. When achieved, reply exactly:\n"
             f"{DONE_MARKER}: <one-line result>")
    messages = [{"role": "user", "content": seed}]

    for _ in range(max_steps):
        if state.interactions_used >= interaction_budget:
            return ("failed", "global budget exhausted")
        reply = llm("executor", messages, system=EXECUTOR_SYSTEM)
        if DONE_MARKER in reply:
            result = reply.split(DONE_MARKER, 1)[1].lstrip(": ").strip()
            return ("done", result or "done")
        code = extract_code(reply)
        output = world.execute(code)
        state.add_step(subgoal.id, code, output)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Execution output:\n{output}"})
        if world.task_completed():
            return ("done", "task completed")
        if state.has_repeated_error(subgoal.id):
            sigs = state.recent_error_signatures(subgoal.id, 1)
            return ("failed", sigs[0] if sigs else "repeated error")
    return ("failed", "max steps reached")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_executor.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add executor.py tests/test_executor.py
git commit -m "feat: add per-subgoal executor ReAct loop"
```

---

### Task 11: `orchestrator.py` — solve & finalize

**Files:**
- Create: `orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

`tests/test_orchestrator.py`:
```python
from orchestrator import solve, finalize
from state import Blackboard, Subgoal
from tests.conftest import FakeLLM, FakeWorld, FakeMem


def test_solve_runs_plan_then_completes():
    bb = Blackboard(task_instruction="play a song")
    # planner -> 1 subgoal; executor -> DONE; verifier subgoal PASS;
    # finalize executor -> completion code; verifier final PASS.
    llm = FakeLLM({
        "planner": ["1. play the song"],
        "executor": ["SUBGOAL_DONE: played",
                     "```python\napis.supervisor.complete_task(answer=None)\n```"],
        "verifier": ["PASS", "PASS"],
    })
    world = FakeWorld()
    mem = FakeMem()
    solve(world, bb, mem, llm=llm)
    assert world.task_completed() is True
    assert mem.remembered and mem.remembered[0][1] is True   # recorded as success


def test_solve_retries_subgoal_then_replans():
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({
        "planner": ["1. do x", "1. do x differently"],   # initial, then replan
        "executor": [
            "```python\nbad\n```", "```python\nbad\n```",   # subgoal attempt 1 -> repeated err
            "```python\nbad\n```", "```python\nbad\n```",   # subgoal attempt 2 (retry) -> repeated err
            "SUBGOAL_DONE: ok",                              # replanned subgoal succeeds
            "```python\napis.supervisor.complete_task(answer=None)\n```",
        ],
        "verifier": ["PASS", "PASS"],
    })
    world = FakeWorld(outputs=["KeyError: 'x'"] * 4)
    solve(world, bb, mem=FakeMem(), llm=llm,
          interaction_budget=40, max_replans=2, max_subgoal_retries=1)
    assert bb.replans_used == 1
    assert world.task_completed() is True


def test_finalize_reprompts_when_verifier_rejects():
    bb = Blackboard(task_instruction="t")
    llm = FakeLLM({
        "executor": ["```python\napis.supervisor.complete_task(answer='wrong')\n```",
                     "```python\napis.supervisor.complete_task(answer=42)\n```"],
        "verifier": ["FAIL: must be int", "PASS"],
    })
    world = FakeWorld()
    ok = finalize(world, bb, FakeMem(), llm=llm, max_finalize=3, interaction_budget=40)
    assert ok is True
    assert world.task_completed() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator'`.

- [ ] **Step 3: Implement `orchestrator.py`**

```python
"""Orchestrator: drives Planner -> [Executor -> Verifier] -> finalize."""
from config import (MAX_FINALIZE, MAX_INTERACTIONS, MAX_REPLANS,
                    MAX_SUBGOAL_RETRIES)
from executor import run as run_executor
from llm import call_llm
from parsing import extract_code
from planner import make_plan, replan
from prompts import EXECUTOR_SYSTEM
from state import Blackboard
from verifier import verify_final, verify_subgoal


def solve(world, state: Blackboard, mem, llm=call_llm,
          interaction_budget: int = MAX_INTERACTIONS,
          max_replans: int = MAX_REPLANS,
          max_subgoal_retries: int = MAX_SUBGOAL_RETRIES) -> None:
    hints = mem.recall(state.task_instruction, kind="memory")
    state.plan = make_plan(state, hints, llm=llm)

    i = 0
    while (i < len(state.plan)
           and state.interactions_used < interaction_budget
           and not world.task_completed()):
        sg = state.plan[i]
        sg.status = "active"
        status, result = run_executor(sg, state, world, mem, llm=llm,
                                       interaction_budget=interaction_budget)
        sg.result = result

        if status == "done":
            ok, feedback = verify_subgoal(sg, state, llm=llm)
            if ok:
                sg.status = "done"
                i += 1
                continue
            result = feedback   # verifier overrides into a failure

        # failure path
        sg.attempts += 1
        if sg.attempts <= max_subgoal_retries:
            sg.status = "pending"
            continue            # retry the same subgoal
        if state.replans_used < max_replans:
            state.replans_used += 1
            state.plan = state.plan[:i] + replan(state, result, llm=llm)
            continue            # fresh remaining plan from index i
        sg.status = "failed"
        i += 1                  # give up on this subgoal, move on

    finalize(world, state, mem, llm=llm, interaction_budget=interaction_budget)
    mem.remember_episode(state.task_instruction, state, world.task_completed())


def finalize(world, state: Blackboard, mem, llm=call_llm,
             max_finalize: int = MAX_FINALIZE,
             interaction_budget: int = MAX_INTERACTIONS) -> bool:
    if world.task_completed():
        return True
    messages = [{"role": "user", "content": state.render_for("finalize")
                 + "\n\nAll planned work is done. Reply with ONE python code block that "
                 "calls apis.supervisor.complete_task(answer=<the answer, or None if the "
                 "task is not a question)."}]
    for _ in range(max_finalize):
        if state.interactions_used >= interaction_budget:
            break
        reply = llm("executor", messages, system=EXECUTOR_SYSTEM)
        code = extract_code(reply)
        ok, feedback = verify_final(state, code, llm=llm)
        if not ok:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                             "content": f"Verifier rejected completion: {feedback}\n"
                                        "Fix and resend ONLY the completion code block."})
            continue
        output = world.execute(code)
        state.add_step(-1, code, output)
        if world.task_completed():
            return True
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user",
                         "content": f"Execution output:\n{output}\n"
                                    "If not complete, fix and resend the completion code."})
    return world.task_completed()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_orchestrator.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add orchestrator solve/finalize driving the role loop"
```

---

### Task 12: Extend `hydradb.py` — `kind` arg on recall + `remember_episode`

**Files:**
- Modify: `hydradb.py`
- Create: `tests/test_hydradb.py`

- [ ] **Step 1: Write the failing test**

`tests/test_hydradb.py`:
```python
from hydradb import HydraMemory
from state import Blackboard, Subgoal


class FakeCtx:
    def __init__(self):
        self.ingests = []

    def ingest(self, **kw):
        self.ingests.append(kw)


class FakeHydraClient:
    def __init__(self):
        self.queries = []
        self.context = FakeCtx()

    def query(self, **kw):
        self.queries.append(kw)
        return {"data": {"chunks": [{"chunk_content": "doc", "source_title": "spotify"}]}}


def test_recall_disabled_returns_empty():
    mem = HydraMemory()           # USE_HYDRA not set -> client is None
    assert mem.recall("anything") == ""
    assert mem.recall("anything", kind="memory") == ""


def test_recall_routes_kind_to_query_type():
    mem = HydraMemory()
    mem.client = FakeHydraClient()
    out = mem.recall("find a song", kind="knowledge")
    assert mem.client.queries[0]["type"] == "knowledge"
    assert "doc" in out


def test_remember_episode_ingests_memory_type():
    mem = HydraMemory()
    mem.client = FakeHydraClient()
    bb = Blackboard(task_instruction="play song")
    bb.plan = [Subgoal(1, "log in", status="done", result="token")]
    bb.add_step(1, "print(login())", "token=abc")
    mem.remember_episode("play song", bb, success=True)
    kw = mem.client.context.ingests[0]
    assert kw["type"] == "memory"
    assert "play song" in kw["memories"]


def test_remember_episode_disabled_is_noop():
    mem = HydraMemory()           # client is None
    bb = Blackboard(task_instruction="t")
    mem.remember_episode("t", bb, success=False)   # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hydradb.py -q`
Expected: FAIL — `TypeError: recall() got an unexpected keyword argument 'kind'`.

- [ ] **Step 3a: Update `recall` to accept `kind`**

In `hydradb.py`, change the `recall` signature and the `query(...)` call. Replace:
```python
    def recall(self, instruction: str) -> str:
```
with:
```python
    def recall(self, instruction: str, kind: str = "all") -> str:
```
and inside that method replace:
```python
                type="all",           # both past episodes and API-doc knowledge
```
with:
```python
                type=kind,            # "all" | "memory" | "knowledge"
```

- [ ] **Step 3b: Add `remember_episode` method**

Add this method to the `HydraMemory` class in `hydradb.py` (next to `remember_task`):
```python
    def remember_episode(self, instruction: str, state, success: bool) -> None:
        """Ingest a structured episode (plan + key steps) from the Blackboard.

        Higher-signal than the raw transcript: stores what the plan was, how each
        subgoal resolved, and the last few executed steps. No-op when disabled.
        """
        if not self.on:
            return
        try:
            plan_lines = [
                f"{sg.id}. [{sg.status}] {sg.description}"
                + (f" -> {sg.result}" if sg.result else "")
                for sg in getattr(state, "plan", [])
            ]
            step_lines = [
                f"$ {s.code[:300]}\n-> {str(s.output)[:300]}"
                for s in getattr(state, "steps", [])[-8:]
            ]
            text = (
                f"AppWorld task ({'SOLVED' if success else 'FAILED'}): {instruction}\n\n"
                "PLAN:\n" + "\n".join(plan_lines)
                + "\n\nKEY STEPS:\n" + "\n\n".join(step_lines)
            )
            self.client.context.ingest(
                type="memory",
                tenant_id=self.tenant_id,
                memories=json.dumps([{
                    "text": text,
                    "infer": False,
                    "metadata": {
                        "kind": "episode",
                        "success": "true" if success else "false",
                    },
                }]),
            )
        except Exception as e:
            print(f"  [hydra] remember_episode failed: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hydradb.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add hydradb.py tests/test_hydradb.py
git commit -m "feat: add kind routing to recall and remember_episode to HydraMemory"
```

---

### Task 13: Rewire `agent.py` to the orchestrator

**Files:**
- Modify: `agent.py`

- [ ] **Step 1: Replace the agent module body**

Overwrite `agent.py` with:
```python
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
```

- [ ] **Step 2: Verify the full test suite still passes**

Run: `python -m pytest -q`
Expected: PASS (all tests green, ~30+ passed).

- [ ] **Step 3: Verify agent imports without a live run**

Run: `python -c "import agent; print('agent ok')"`
Expected: prints `agent ok` (no AppWorld task is opened at import time).

- [ ] **Step 4: Smoke-test on 1 real dev task (requires OPENROUTER_API_KEY in .env)**

Run:
```bash
export APPWORLD_EXPERIMENT=team_smoke
export APPWORLD_DATASET=dev MAX_TASKS=1
python agent.py
```
Expected: it prints the plan progress and ends with `✓ completed` or `✗ ended without completion` — and does **not** crash. (A single task may or may not solve; we only need a clean end-to-end run here.)

- [ ] **Step 5: Commit**

```bash
git add agent.py
git commit -m "feat: rewire agent entrypoint to Planner/Executor/Verifier orchestrator"
```

---

## Post-Plan: measure & iterate (not part of the TDD tasks)

Once Task 13 is green, run the dev split and read the result before adding any Approach-C technique:
```bash
export APPWORLD_EXPERIMENT=team_<yourname> APPWORLD_DATASET=dev MAX_TASKS=0
python agent.py
appworld evaluate $APPWORLD_EXPERIMENT dev
```
Use the TGC/SGC numbers and the per-task console logs to decide where to invest next (Phase 2 context-compression / Phase 3 best-of-N, self-consistency, Reflexion) — each behind its own flag, each A/B'd against this baseline.

---

## Self-Review

**Spec coverage:**
- Module layout (state/llm/planner/executor/verifier/orchestrator + HydraDB seams) → Tasks 2–13. ✓
- Blackboard with credentials/api_cache/results/steps/budgets + render_for → Tasks 3–5. ✓
- Control/data flow Planner→[Executor→Verifier]→finalize→complete_task → Task 11. ✓
- Per-subgoal HydraDB knowledge retrieval + per-task episodic recall → executor (Task 10) + orchestrator (Task 11) + hydradb `kind` routing (Task 12). ✓
- Error handling: reflection-via-observation + repeated-error loop-break (Task 10), bounded replans/retries (Task 11), budget-aware finalize bailout (Task 11), complete_task gating via verify_final (Task 11), OpenRouter retry/fallback (Task 6). ✓
- Episode-aware remember → Task 12. ✓
- Testing/measurement → per-task pytest + Post-Plan section. ✓
- Existing safety nets (per-task try/except, HydraDB no-op) preserved → Task 13 + Task 12. ✓

**Deferred to post-plan (explicit non-goals in spec):** auth-error self-heal token invalidation, context summarization of old steps (Phase 2), best-of-N / self-consistency / Reflexion (Phase 3). The architecture leaves clean seams (executor candidate generation, planner strategy, episodic recall) but these are not implemented in Phase 1 — consistent with the spec's rollout. *Note:* auth self-heal is described in the spec's error-handling list; it is intentionally deferred to Phase 2 here to keep Phase 1 shippable. If you want it in Phase 1, add a task after Task 10.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `call_llm(role, messages, system=...)` used uniformly; `mem.recall(query, kind=...)`, `mem.remember_episode(instruction, state, success)`, `run(subgoal, state, world, mem, llm, max_steps, interaction_budget) -> (status, result)`, `verify_subgoal/verify_final -> (bool, str)`, `make_plan/replan -> list[Subgoal]`, `solve(world, state, mem, llm, interaction_budget, max_replans, max_subgoal_retries)`, `finalize(world, state, mem, llm, max_finalize, interaction_budget)` — signatures match across defining and calling tasks. ✓

