"""The Blackboard: structured task state shared by all roles."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

_ERR_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning)):\s*(.*)$")
# Matches HTTP-level API errors like "Response status code is 409: {\"message\":\"...\"}"
_HTTP_ERR = re.compile(r"Response status code is (\d+):\s*(.*)", re.I)


@dataclass
class Subgoal:
    id: int
    description: str
    status: Literal["pending", "active", "done", "failed"] = "pending"
    result: str | None = None
    attempts: int = 0
    last_feedback: str | None = None   # verifier rejection reason for next retry


@dataclass
class StepRecord:
    subgoal_id: int
    code: str
    output: str
    error_signature: str | None = None


def error_signature(output: str) -> str | None:
    """Return a normalized error signature for the LAST error line, else None.

    Handles both Python exceptions (ExcType: msg) and HTTP API errors
    (Response status code is NNN: ...) so repeated-error detection fires
    for both kinds of failures.
    """
    if not output:
        return None
    py_match = None
    http_match = None
    for line in str(output).splitlines():
        stripped = line.strip()
        m = _ERR_LINE.match(stripped)
        if m:
            py_match = m
        h = _HTTP_ERR.search(stripped)
        if h:
            http_match = h
    if py_match:
        exc, msg = py_match.group(1), py_match.group(2)
        msg = re.sub(r"0x[0-9a-fA-F]+", "#", msg)
        msg = re.sub(r"\d+", "#", msg)
        return f"{exc}: {msg[:80]}"
    if http_match:
        status = http_match.group(1)
        # Normalize the message body: strip IDs/numbers so the same HTTP error
        # on different resource IDs hashes to the same signature.
        body = re.sub(r"\d+", "#", http_match.group(2))
        return f"HTTP {status}: {body[:80]}"
    return None


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

    _STATUS_MARK = {"done": "x", "active": ">", "failed": "!", "pending": " "}

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

    def render_for(self, role: str, subgoal: "Subgoal | None" = None) -> str:
        lines = [f"TASK: {self.task_instruction}"]
        if self.supervisor:
            lines.append(f"SUPERVISOR: {self.supervisor}")
        if self.credentials:
            lines.append("LOGGED IN: " + ", ".join(sorted(self.credentials)))
            if role in ("executor", "finalize"):
                lines.append("TOKENS:")
                for app, token in sorted(self.credentials.items()):
                    lines.append(f"  {app}_access_token={token}")
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
            # Show up to 5 most-recent steps; for the active subgoal show all its steps
            # so the executor never loses sight of what it already tried this round.
            subgoal_id = subgoal.id if subgoal is not None else None
            if subgoal_id is not None:
                sg_steps = [s for s in self.steps if s.subgoal_id == subgoal_id]
                other_steps = [s for s in self.steps[-5:] if s.subgoal_id != subgoal_id]
                shown = other_steps + sg_steps
            else:
                shown = self.steps[-5:]
            lines.append("RECENT STEPS:")
            for s in shown:
                out = s.output if len(s.output) <= 200 else s.output[:200] + "…"
                lines.append(f"  $ {s.code[:120]}\n    -> {out}")
        return "\n".join(lines)
