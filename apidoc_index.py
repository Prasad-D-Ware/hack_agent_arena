"""Local API-doc retrieval over the committed AppWorld snapshot.

The reference agent discovers APIs at runtime via ``apis.api_docs.show_api_doc``.
This module adds the committed ``assets/api_docs.json`` snapshot (every app + API
+ full signature, produced by ``snapshot_api_docs.py``) as an *extra, up-front*
retrieval source: given the apps a task is likely to touch, it injects the most
relevant API signatures into the solver's first message so the model starts with
authoritative schemas instead of spending turns rediscovering them.

Design rules:
* Purely local + deterministic (no HydraDB, no network). The HydraDB ``knowledge``
  recall space stays dedicated to skills + the ACE strategy ledger.
* Advisory, not ground truth — the prompt still tells the solver to verify against
  live docs. The snapshot can lag the installed AppWorld version.
* Never breaks a run: a missing/corrupt artifact or any error yields an empty
  block, exactly like the skill/ledger retrievers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "assets" / "api_docs.json"

# Same tokenizer shape as skill_store._keyword_set (kept local to avoid a hard
# import dependency between the two add-on modules).
_NOISE_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "my",
    "me", "is", "are", "be", "that", "this", "it", "as", "at", "by", "from",
    "i", "you", "your", "their", "them", "then", "if", "all", "any", "into",
    "list", "show", "get", "give", "tell", "find", "please", "want", "need",
}


def _keyword_set(text: str) -> set[str]:
    found = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
    return {w for w in found if len(w) > 2 and w not in _NOISE_WORDS}


class ApiDocIndex:
    """Loads ``assets/api_docs.json`` and serves compact, app-filtered signatures."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path else _SNAPSHOT_PATH
        # app -> {api_name -> doc dict}
        self._catalog: dict[str, dict[str, dict]] = {}
        self._read_snapshot()

    def _read_snapshot(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            apps = data.get("apps", {}) if isinstance(data, dict) else {}
            for app, info in apps.items():
                api_map = (info or {}).get("apis", {}) or {}
                if api_map:
                    self._catalog[str(app).lower()] = api_map
        except Exception:
            # missing or corrupt artifact -> empty index (retrieve() returns "")
            self._catalog = {}

    @property
    def available(self) -> bool:
        return bool(self._catalog)

    @property
    def apps(self) -> list[str]:
        return sorted(self._catalog)

    @staticmethod
    def _one_line_sig(doc: dict) -> str:
        """One-line ``apis.app.api(p*, q)`` signature; ``*`` marks required params."""
        app = doc.get("app_name", "")
        name = doc.get("api_name", "")
        parts = []
        for p in doc.get("parameters", []) or []:
            pname = p.get("name", "")
            if not pname:
                continue
            parts.append(f"{pname}*" if p.get("required") else pname)
        return f"apis.{app}.{name}({', '.join(parts)})"

    def _format_entry(self, doc: dict, char_budget: int) -> str:
        sig = self._one_line_sig(doc)
        desc = (doc.get("description") or "").strip()
        line = f"- {sig}" + (f" — {desc}" if desc else "")
        return line[:char_budget]

    def retrieve(
        self,
        apps_hint: Iterable[str],
        instruction: str,
        k: int = 6,
        char_budget: int = 2500,
    ) -> str:
        """Return a compact 'API reference' block for the apps a task touches.

        Filters to ``apps_hint`` (lowercased), ranks each app's APIs by token
        overlap of name+description+param-names against the instruction, and
        returns up to ``k`` signatures within ``char_budget``. Empty string when
        the index is unavailable or no hinted app is known.
        """
        if not self._catalog:
            return ""
        wanted = [a.lower() for a in apps_hint if a]
        # restrict to apps we actually have docs for
        wanted = [a for a in wanted if a in self._catalog]
        if not wanted:
            return ""

        q = _keyword_set(instruction)
        ranked: list[tuple[float, str, dict]] = []
        for app in wanted:
            for api_name, doc in self._catalog[app].items():
                if not isinstance(doc, dict):
                    continue
                param_names = " ".join(
                    p.get("name", "") for p in doc.get("parameters", []) or []
                )
                hay = _keyword_set(f"{api_name} {doc.get('description', '')} {param_names}")
                overlap = len(q & hay)
                # tiny boost so the API name matching a query word ranks high
                if q & _keyword_set(api_name):
                    overlap += 1
                ranked.append((float(overlap), app, doc))

        # rank by relevance, then keep deterministic order for ties
        ranked.sort(key=lambda x: (-x[0], x[1], x[2].get("api_name", "")))
        top = ranked[:k]
        if not top:
            return ""

        lines: list[str] = []
        used = 0
        for _score, _app, doc in top:
            entry = self._format_entry(doc, char_budget)
            if used + len(entry) + 1 > char_budget:
                break
            lines.append(entry)
            used += len(entry) + 1
        if not lines:
            return ""

        header = (
            "API reference for likely apps (from the local snapshot — authoritative "
            "signatures, but VERIFY against live docs via apis.api_docs.show_api_doc "
            "before relying on response field names):"
        )
        return header + "\n" + "\n".join(lines)


if __name__ == "__main__":
    import sys

    index = ApiDocIndex()
    print(f"available={index.available} apps={index.apps}")
    apps = sys.argv[1].split(",") if len(sys.argv) > 1 else ["spotify"]
    instr = sys.argv[2] if len(sys.argv) > 2 else "top 5 songs by play count"
    print(index.retrieve(apps, instr, k=6, char_budget=2000))
