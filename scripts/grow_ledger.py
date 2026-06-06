"""Offline ACE warmup: grow the strategy ledger from AppWorld run trajectories.

This is the Reflector + Curator half of the ACE loop (the Generator is the agent
itself, run separately to produce the experiment outputs):

  1. parse each task's trace from an experiment's output folder,
  2. derive EXECUTION FEEDBACK only (completion, repeated errors, whether an
     answer was submitted) -- no ground-truth labels required, matching ACE's
     label-free setting,
  3. **Reflector**: an LLM distills a small delta of GENERAL, reusable entries
     (strategies / API patterns / failure modes) from the trace,
  4. **Curator**: StrategyLedger.absorb merges the delta with string/semantic
     de-duplication (grow-and-refine) into the local + HydraDB ledger.

Every candidate entry passes the ``is_reusable`` guard, so no task-specific
answer or task id can ever leak into the ledger.

Usage:
  python scripts/grow_ledger.py --seed --sync-hydra            # just seed
  python scripts/grow_ledger.py --experiment team_x --dataset dev
  python scripts/grow_ledger.py --experiment team_x --use-eval # add GT signal
  python scripts/grow_ledger.py --experiment team_x --no-llm   # heuristic only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from strategy_ledger import StrategyLedger, LedgerEntry, seed_ledger, is_reusable  # noqa: E402
from scripts.distill_skills import build_trace, label_outcomes  # noqa: E402

OUTPUTS_ROOT = ROOT / "experiments" / "outputs"
_PY_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.S)

_VALID_SECTIONS = {
    "discovery", "auth", "pagination", "inspection", "ranking", "mutation",
    "dates", "answer_format", "recovery", "efficiency", "general",
    "spotify", "venmo", "phone", "amazon", "gmail", "todoist", "splitwise",
    "simple_note", "file_system",
}


# ---------------------------------------------------------------------------
# execution-feedback extraction (label-free signal)
# ---------------------------------------------------------------------------
def _distinct_errors(env_io_path: Path, limit: int = 6) -> list[str]:
    """Return distinct, normalized error lines observed during execution."""
    if not env_io_path.exists():
        return []
    text = env_io_path.read_text(encoding="utf-8", errors="ignore")
    sigs: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if "Execution failed" in line or re.search(r"(Exception|Error|message):", line):
            sig = re.sub(r"\s+", " ", line.strip())[:200]
            key = sig.lower()
            if key not in seen:
                seen.add(key)
                sigs.append(sig)
        if len(sigs) >= limit:
            break
    return sigs


def _count_blocks(env_io_path: Path) -> int:
    if not env_io_path.exists():
        return 0
    text = env_io_path.read_text(encoding="utf-8", errors="ignore")
    return len(_PY_BLOCK_RE.findall(text))


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------
REFLECTOR_SYSTEM = """You are the REFLECTOR in an Agentic Context Engineering loop
for the AppWorld coding agent. You read ONE task's execution trace plus its
execution feedback (did it complete, what errors recurred) and distill concrete,
REUSABLE lessons that would help solve FUTURE, DIFFERENT AppWorld tasks.

Output STRICT JSON ONLY: a JSON array of bullet objects, each:
  {"section": <one of: discovery, auth, pagination, inspection, ranking,
     mutation, dates, answer_format, recovery, efficiency, general, or an app
     name like spotify/venmo/phone/amazon/gmail>,
   "apps": [<app names this applies to, or empty for app-agnostic>],
   "text": <one concise, general, imperative sentence>}

HARD RULES (a violation makes the bullet useless and will be discarded):
- GENERAL ONLY. Each bullet must apply across many tasks. Capture API patterns,
  exact field-name gotchas, pagination/auth rules, ranking/dedup discipline,
  mutation completeness, date handling, answer formatting, and error-recovery
  tactics.
- NEVER include this task's specific answer, numbers, names, emails, phone
  numbers, ids, or task id. NEVER write a memorized result. If a lesson can only
  be stated with task-specific values, OMIT it.
- Prefer lessons backed by the trace: if an error recurred, write the rule that
  avoids it; if a non-obvious field/parameter name mattered, state it.
- 1 to 5 bullets. Be specific and actionable, not vague. No duplicates of each
  other. If the trace teaches nothing new and general, return [].
"""


def _heuristic_entries(trace, errors: list[str]) -> list[LedgerEntry]:
    out: list[LedgerEntry] = []
    apps = trace.apps_used or []
    if errors:
        # turn the most common error class into a generic recovery reminder
        joined = " ".join(errors).lower()
        if "login" in joined or "credential" in joined or "unauthor" in joined:
            out.append(LedgerEntry(
                text=("If login/auth fails, re-read the login API doc and try the "
                      "alternate username value (email vs phone_number) before retrying."),
                section="auth", apps=apps, source=f"reflector:{trace.experiment}"))
        if "keyerror" in joined or "has no key" in joined or "field" in joined:
            out.append(LedgerEntry(
                text=("KeyError/missing-field errors mean the field name was guessed; "
                      "print one sample's keys and use only confirmed field names."),
                section="inspection", apps=apps, source=f"reflector:{trace.experiment}"))
    return out


def _llm_entries(trace, errors: list[str], completed: bool, passed: Optional[bool],
                 model: str) -> list[LedgerEntry]:
    try:
        from agent import call_llm
    except Exception as exc:
        print(f"  [warn] cannot import call_llm ({exc}); heuristic only")
        return []
    feedback = []
    feedback.append(f"completed_run: {completed}")
    if passed is not None:
        feedback.append(f"evaluator_success: {passed}")
    feedback.append(f"num_distinct_errors: {len(errors)}")
    user = (
        f"Apps used: {', '.join(trace.apps_used) or 'unknown'}\n"
        f"APIs consulted: {', '.join(trace.apis_consulted[:20]) or 'n/a'}\n"
        f"Execution feedback: {'; '.join(feedback)}\n"
        f"Recurring errors observed:\n" + ("\n".join(f"- {e}" for e in errors) or "- none") + "\n\n"
        f"Task instruction (for context only -- do NOT encode its answer):\n{trace.instruction}\n\n"
        f"Final solving code (secrets redacted; reference only):\n```python\n{trace.final_code[:3500]}\n```\n\n"
        "Return the JSON array of general bullets now."
    )
    try:
        raw = call_llm(
            [{"role": "user", "content": user}],
            system_prompt=REFLECTOR_SYSTEM,
            model=model,
            max_tokens=900,
        )
    except Exception as exc:
        print(f"  [warn] reflector LLM failed ({exc}); heuristic only")
        return []
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    entries: list[LedgerEntry] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ok, _ = is_reusable(text)
        if not ok:
            continue
        section = str(item.get("section", "general")).strip().lower()
        if section not in _VALID_SECTIONS:
            section = "general"
        apps = item.get("apps") or []
        if isinstance(apps, str):
            apps = [apps]
        entries.append(LedgerEntry(
            text=text, section=section,
            apps=[str(a).lower() for a in apps if a],
            source=f"reflector:{trace.experiment}",
        ))
    return entries[:5]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Grow the ACE strategy ledger from run trajectories.")
    ap.add_argument("--experiment", default="", help="experiment output folder to reflect on")
    ap.add_argument("--dataset", default=os.environ.get("APPWORLD_DATASET", "dev"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", action="store_true", help="ensure default general entries exist")
    ap.add_argument("--sync-hydra", action="store_true", help="push whole ledger to HydraDB")
    ap.add_argument("--no-hydra", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="heuristic reflector only")
    ap.add_argument("--use-eval", action="store_true",
                    help="also use evaluator pass/fail as a feedback signal")
    ap.add_argument("--model", default=os.environ.get("DISTILLER_MODEL",
                                                      os.environ.get("MODEL", "gpt-5.5")))
    args = ap.parse_args()

    ledger = StrategyLedger(use_hydra=not args.no_hydra)
    print(f"Ledger: {len(ledger)} entry(ies) | hydra={ledger.hydra_enabled}")

    if args.seed:
        report = seed_ledger(ledger)
        print(f"Seed merge: {report}")

    if args.experiment:
        exp_dir = OUTPUTS_ROOT / args.experiment / "tasks"
        if not exp_dir.exists():
            print(f"No task outputs at {exp_dir}")
        else:
            task_ids = sorted([p.name for p in exp_dir.iterdir() if p.is_dir()])
            if args.limit:
                task_ids = task_ids[: args.limit]
            print(f"Reflecting on {len(task_ids)} task(s) from '{args.experiment}' "
                  f"(llm={not args.no_llm}, model={args.model})")

            outcome_map: dict[str, dict] = {}
            if args.use_eval:
                print("Labelling pass/fail via evaluator (extra feedback signal)...")
                outcome_map = label_outcomes(args.experiment, task_ids)

            totals = {"added": 0, "merged": 0, "rejected": 0}
            for task_id in task_ids:
                trace = build_trace(args.experiment, task_id)
                env_io = OUTPUTS_ROOT / args.experiment / "tasks" / task_id / "logs" / "environment_io.md"
                errors = _distinct_errors(env_io)
                passed = bool(outcome_map[task_id].get("success")) if task_id in outcome_map else None
                if args.no_llm:
                    cands = _heuristic_entries(trace, errors)
                else:
                    cands = _llm_entries(trace, errors, trace.completed, passed, args.model)
                    if not cands:
                        cands = _heuristic_entries(trace, errors)
                report = ledger.absorb(cands)
                for k in totals:
                    totals[k] += report[k]
                print(f"  {task_id}: {len(cands)} candidate(s) -> "
                      f"+{report['added']} ~{report['merged']} x{report['rejected']} "
                      f"(completed={trace.completed}, errors={len(errors)}"
                      + (f", passed={passed}" if passed is not None else "") + ")")
            print(f"\nReflection totals: {totals} | ledger now {len(ledger)} entry(ies)")

    if args.sync_hydra:
        ok = ledger.push_all_to_hydra()
        print(f"Synced all entries to HydraDB: {ok}")

    print(f"Done. Ledger has {len(ledger)} entry(ies) at {ledger and 'memory/playbook.jsonl'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
