"""Parse AppWorld experiment outputs into reusable skills/lessons.

For each task in an experiment's output folder this script:
  1. loads the task instruction + supervisor via ``Task.load`` (no agent loop),
  2. parses ``logs/environment_io.md`` (code/output steps) and
     ``logs/api_calls.jsonl`` (apps used, API docs consulted, submitted answer),
  3. labels pass/fail using the AppWorld evaluator,
  4. distills a Hermes-style Markdown skill (pass) or lesson (fail) -- with an
     LLM when available, falling back to a deterministic heuristic, and
  5. stores it via ``SkillRepository`` (local cache + HydraDB).

Everything is scrubbed of secrets before it is persisted.

Usage:
  python scripts/distill_skills.py --experiment team_routing_dev_gpt55 --dataset dev
  python scripts/distill_skills.py --experiment X --dataset dev --no-llm --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# make the parent package importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from skill_store import SkillNote, SkillRepository, redact_secrets  # noqa: E402

OUTPUTS_ROOT = ROOT / "experiments" / "outputs"
_PY_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.S)
_APP_CALL_RE = re.compile(r"apis\.([a-z_][a-z0-9_]*)\.", re.I)

_NON_APP_NS = {"api_docs", "supervisor"}


# ---------------------------------------------------------------------------
# trace model + parsing
# ---------------------------------------------------------------------------
@dataclass
class RunTrace:
    task_id: str
    experiment: str
    instruction: str = ""
    supervisor: str = ""
    apps_used: list[str] = field(default_factory=list)
    apis_consulted: list[str] = field(default_factory=list)
    final_code: str = ""
    submitted_answer: Optional[str] = None
    completed: bool = False
    passed: Optional[bool] = None
    failure_reasons: list[str] = field(default_factory=list)
    difficulty: Optional[int] = None


def _read_task_meta(task_id: str) -> tuple[str, str, Optional[int]]:
    try:
        from appworld.task import Task
        task = Task.load(task_id=task_id, load_ground_truth=False)
        sup = task.supervisor
        sup_name = " ".join(
            str(p) for p in [getattr(sup, "first_name", ""), getattr(sup, "last_name", "")] if p
        ).strip()
        return str(task.instruction), sup_name, None
    except Exception as exc:
        print(f"  [warn] Task.load({task_id}) failed: {exc}")
        return "", "", None


def _scan_api_calls(path: Path) -> tuple[list[str], list[str], Optional[str]]:
    apps: list[str] = []
    apis: list[str] = []
    answer: Optional[str] = None
    if not path.exists():
        return apps, apis, answer
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        url = rec.get("url", "")
        data = rec.get("data", {}) or {}
        parts = [p for p in url.split("/") if p]
        if parts:
            ns = parts[0]
            if ns == "api_docs" and parts[-1] == "api_doc":
                doc = f"{data.get('app_name', '?')}.{data.get('api_name', '?')}"
                if doc not in apis:
                    apis.append(doc)
            elif ns not in _NON_APP_NS:
                if ns not in apps:
                    apps.append(ns)
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer = str(data["answer"])
    return apps, apis, answer


def _scan_env_io(path: Path) -> tuple[str, bool]:
    """Return (final_code, completed) from environment_io.md."""
    if not path.exists():
        return "", False
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = _PY_BLOCK_RE.findall(text)
    if not blocks:
        return "", False
    final = redact_secrets(blocks[-1].strip())
    completed = any("complete_task" in b for b in blocks)
    return final, completed


def build_trace(experiment: str, task_id: str) -> RunTrace:
    task_dir = OUTPUTS_ROOT / experiment / "tasks" / task_id
    instruction, supervisor, difficulty = _read_task_meta(task_id)
    apps, apis, answer = _scan_api_calls(task_dir / "logs" / "api_calls.jsonl")
    final_code, completed = _scan_env_io(task_dir / "logs" / "environment_io.md")
    return RunTrace(
        task_id=task_id,
        experiment=experiment,
        instruction=instruction,
        supervisor=supervisor,
        apps_used=apps,
        apis_consulted=apis,
        final_code=final_code,
        submitted_answer=answer,
        completed=completed,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# evaluation labelling
# ---------------------------------------------------------------------------
def label_outcomes(experiment: str, task_ids: list[str]) -> dict[str, dict]:
    """Return {task_id: {success, failures:[...]}} using the AppWorld evaluator.

    Reads an existing ``evaluations/<dataset>.json`` if present, otherwise runs
    the evaluator programmatically on the given task ids.
    """
    # 1. try any existing aggregate json (covers full-run evaluations)
    eval_dir = OUTPUTS_ROOT / experiment / "evaluations"
    if eval_dir.exists():
        for jf in sorted(eval_dir.glob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                individual = data.get("individual") or {}
                if individual and any(t in individual for t in task_ids):
                    return individual
            except Exception:
                continue
    # 2. evaluate programmatically
    try:
        from appworld import evaluate_tasks
        metrics = evaluate_tasks(
            task_ids=task_ids,
            experiment_name=experiment,
            suppress_errors=True,
            include_details=True,
            save_reports=True,
        )
        return metrics.get("individual") or {}
    except Exception as exc:
        print(f"  [warn] evaluate_tasks failed ({exc}); falling back to completion-only labels")
        return {}


def _apply_outcome(trace: RunTrace, eval_rec: Optional[dict]) -> None:
    if not eval_rec:
        trace.passed = None
        return
    trace.passed = bool(eval_rec.get("success"))
    trace.difficulty = eval_rec.get("difficulty", trace.difficulty)
    reasons: list[str] = []
    for f in eval_rec.get("failures", []) or []:
        req = f.get("requirement") if isinstance(f, dict) else str(f)
        if req:
            reasons.append(redact_secrets(str(req))[:200])
    trace.failure_reasons = reasons[:5]


# ---------------------------------------------------------------------------
# distillation
# ---------------------------------------------------------------------------
DISTILL_SYSTEM = """You distill one AppWorld agent run into a concise, reusable
skill for a coding agent. Output STRICT JSON only with keys:
title (short, app + action),
when_to_use (one sentence trigger),
procedure (numbered steps describing the exact API flow and computation),
pitfalls (array of short strings: wrong keys, pagination, dedupe, answer format),
verification (array of short strings: how to check the answer before submitting),
answer_shape (short description of the final answer format, or "" for mutations).
Be specific to the apps/APIs actually used. Never include passwords or tokens.
For FAILED runs, focus pitfalls on the likely root cause given the failure notes."""


def _heuristic_note(trace: RunTrace) -> SkillNote:
    outcome = "pass" if trace.passed else ("fail" if trace.passed is False else "completed")
    app_str = ", ".join(trace.apps_used) or "unknown"
    title = f"{app_str}: {(trace.instruction or trace.task_id)[:60]}"
    procedure_lines = [
        f"Apps: {app_str}.",
        f"APIs consulted: {', '.join(trace.apis_consulted[:12]) or 'n/a'}.",
    ]
    if trace.final_code:
        procedure_lines.append("Final solving code (reference):\n```python\n" + trace.final_code[:1500] + "\n```")
    pitfalls = []
    verification = []
    if trace.submitted_answer:
        verification.append(f"Prior successful answer shape: {trace.submitted_answer[:120]}")
    if outcome == "fail":
        pitfalls = trace.failure_reasons or ["Run completed but failed hidden evaluation; re-check answer derivation and required state changes."]
    return SkillNote(
        title=title,
        apps=trace.apps_used,
        task_type="answer" if trace.submitted_answer else "mutation",
        when_to_use=(trace.instruction or "")[:200],
        procedure="\n".join(procedure_lines),
        pitfalls=pitfalls,
        verification=verification,
        answer_shape=(trace.submitted_answer or "")[:120] if trace.submitted_answer else "",
        outcome=outcome,
        source_task_id=trace.task_id,
        source_experiment=trace.experiment,
        tags=trace.apps_used,
    )


def _llm_note(trace: RunTrace, model: str) -> Optional[SkillNote]:
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as exc:
        print(f"  [warn] OpenAI client unavailable ({exc}); using heuristic")
        return None
    outcome = "passed" if trace.passed else ("FAILED" if trace.passed is False else "completed (unknown pass/fail)")
    user = (
        f"Outcome: {outcome}\n"
        f"Apps used: {', '.join(trace.apps_used) or 'unknown'}\n"
        f"APIs consulted: {', '.join(trace.apis_consulted[:20]) or 'n/a'}\n"
        f"Task instruction: {trace.instruction}\n"
        f"Submitted answer: {trace.submitted_answer}\n"
        f"Failure notes: {'; '.join(trace.failure_reasons) or 'none'}\n\n"
        f"Final solving code (secrets already redacted):\n```python\n{trace.final_code[:4000]}\n```"
    )
    try:
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": DISTILL_SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        if model.startswith("gpt-5"):
            request["max_completion_tokens"] = 1200
        else:
            request["max_tokens"] = 1200
            request["temperature"] = 0.0
        resp = client.chat.completions.create(**request)
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
    except Exception as exc:
        print(f"  [warn] LLM distill failed ({exc}); using heuristic")
        return None

    task_outcome = "pass" if trace.passed else ("fail" if trace.passed is False else "completed")
    return SkillNote(
        title=str(data.get("title") or trace.task_id)[:120],
        apps=trace.apps_used,
        task_type="answer" if trace.submitted_answer else "mutation",
        when_to_use=str(data.get("when_to_use", ""))[:300],
        procedure=str(data.get("procedure", "")),
        pitfalls=[str(p) for p in (data.get("pitfalls") or [])][:8],
        verification=[str(v) for v in (data.get("verification") or [])][:8],
        answer_shape=str(data.get("answer_shape", ""))[:160],
        outcome=task_outcome,
        source_task_id=trace.task_id,
        source_experiment=trace.experiment,
        tags=trace.apps_used,
    )


def distill_note(trace: RunTrace, use_llm: bool, model: str) -> SkillNote:
    if use_llm:
        note = _llm_note(trace, model)
        if note is not None:
            return note
    return _heuristic_note(trace)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Distill AppWorld runs into HydraDB skills.")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--dataset", default=os.environ.get("APPWORLD_DATASET", "dev"))
    ap.add_argument("--limit", type=int, default=0, help="max tasks (0 = all)")
    ap.add_argument("--no-llm", action="store_true", help="use heuristic distiller only")
    ap.add_argument("--no-hydra", action="store_true", help="local store only")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluator labelling")
    ap.add_argument("--dry-run", action="store_true", help="print skills, do not store")
    ap.add_argument("--model", default=os.environ.get("DISTILLER_MODEL", os.environ.get("MODEL", "gpt-5.5")))
    args = ap.parse_args()

    exp_dir = OUTPUTS_ROOT / args.experiment / "tasks"
    if not exp_dir.exists():
        print(f"No task outputs at {exp_dir}")
        return 1
    task_ids = sorted([p.name for p in exp_dir.iterdir() if p.is_dir()])
    if args.limit:
        task_ids = task_ids[: args.limit]
    print(f"Distilling {len(task_ids)} task(s) from '{args.experiment}' (llm={not args.no_llm}, hydra={not args.no_hydra})")

    outcome_map: dict[str, dict] = {}
    if not args.no_eval:
        print("Labelling pass/fail via AppWorld evaluator...")
        outcome_map = label_outcomes(args.experiment, task_ids)

    repo = None if args.dry_run else SkillRepository(use_hydra=not args.no_hydra)
    if repo is not None:
        print(f"HydraDB enabled: {repo.hydra_enabled}")

    n_pass = n_fail = n_other = 0
    for task_id in task_ids:
        trace = build_trace(args.experiment, task_id)
        _apply_outcome(trace, outcome_map.get(task_id))
        note = distill_note(trace, use_llm=not args.no_llm, model=args.model)
        label = {True: "PASS", False: "FAIL"}.get(trace.passed, "????")
        if trace.passed is True:
            n_pass += 1
        elif trace.passed is False:
            n_fail += 1
        else:
            n_other += 1
        print(f"  [{label}] {task_id} -> skill '{note.title[:60]}' ({note.kind}, id={note.skill_id})")
        if repo is not None:
            repo.store(note)
        elif args.dry_run:
            print(note.to_markdown())

    print(f"\nDone. pass={n_pass} fail={n_fail} unknown={n_other}")
    if repo is not None:
        print(f"Skills written to {ROOT / 'memory'} (hydra={repo.hydra_enabled}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
