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
