"""ACE-style evolving "strategy ledger" for the AppWorld agent.

This implements the context-engineering mechanism from *Agentic Context
Engineering* (ACE, arXiv:2510.04618). A ledger is a collection of itemized,
GENERAL strategy entries. Each entry is a small reusable unit (a strategy,
domain concept, API pattern, or common failure mode) carrying metadata:

* ``entry_id`` (stored as ``bullet_id``) - stable identifier,
* ``helpful`` / ``harmful`` counters,
* ``section`` + ``apps`` tags for fine-grained retrieval,
* ``source`` provenance.

The three ACE roles map onto our system as follows:

* **Generator** - the existing ReAct agent loop in ``agent.py`` that solves a
  task and produces an execution trajectory.
* **Reflector** - ``scripts/grow_ledger.py`` distills concrete, GENERAL lessons
  from each trajectory (successes AND errors), using execution feedback only (no
  ground-truth labels required).
* **Curator** - :meth:`StrategyLedger.absorb` below: a lightweight, non-LLM
  merge with semantic/string de-duplication ("grow-and-refine"). New entries are
  appended; near-duplicates increment counters instead of bloating the context.

CRITICAL SAFETY RULE: entries must be GENERAL (AppWorld-wide strategies, API
patterns, gotchas). They must NEVER contain task-specific answers or be keyed to
specific ``task_id``s. :func:`is_reusable` enforces this defensively so the
offline Reflector can never leak a memorized answer into the ledger.

The ledger is stored in two cooperating places, mirroring ``SkillRepository``:

* a deterministic on-disk JSONL cache (always available), and
* HydraDB ``knowledge`` (persistent + semantically recallable across runs),
  which earns the HydraDB bonus integration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

from skill_store import HydraBridge, redact_secrets, _keyword_set

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
LEDGER_ROOT = Path(os.environ.get("MEMORY_DIR", Path(__file__).resolve().parent / "memory"))
LEDGER_FILE = Path(os.environ.get("PLAYBOOK_PATH", LEDGER_ROOT / "playbook.jsonl"))

# grow-and-refine de-dup threshold (token Jaccard). Above this, two entries are
# considered the same lesson and merged (counters combined) instead of appended.
DEDUP_THRESHOLD = float(os.environ.get("PLAYBOOK_DEDUP_THRESHOLD", "0.82"))
# max entries rendered into the solver context per task.
MAX_ENTRIES_RENDERED = int(os.environ.get("PLAYBOOK_MAX_BULLETS", "40"))
# char budget for the rendered ledger block (Llama has long context, but keep
# it sane so the live docs/data still dominate the decision).
RENDER_CHAR_BUDGET = int(os.environ.get("PLAYBOOK_CHAR_BUDGET", "6000"))

# The 10 official eval task ids. The ledger must NEVER be keyed to these (or any)
# specific tasks; this list is used only as a defensive guard to reject any
# candidate entry that accidentally references a concrete eval task id.
_HELD_OUT_TASK_IDS = {
    "5e27cd7_2", "ba46d91_2", "dbc0276_3", "20c1328_3", "9871968_2",
    "c1091c7_2", "18670a5_3", "f6be291_1", "23d431c_3", "8d42650_3",
}
_TASK_ID_PATTERN = re.compile(r"\b[0-9a-f]{7}_\d\b")


# ---------------------------------------------------------------------------
# entry model
# ---------------------------------------------------------------------------
@dataclass
class LedgerEntry:
    text: str
    section: str = "general"           # auth | pagination | inspection | ranking |
                                       # mutation | dates | answer_format | <app> | general
    apps: list[str] = field(default_factory=list)
    helpful: int = 0
    harmful: int = 0
    source: str = "seed"               # seed | reflector:<exp> | <task_id>
    bullet_id: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        self.text = redact_secrets(re.sub(r"\s+", " ", (self.text or "").strip()))
        self.apps = [a.lower().strip() for a in self.apps if a]
        if not self.bullet_id:
            self.bullet_id = self._derive_id()
        if not self.created_at:
            self.created_at = time.time()

    def _derive_id(self) -> str:
        norm = re.sub(r"[^a-z0-9 ]", "", self.text.lower())
        norm = " ".join(sorted(_keyword_set(norm)))
        return hashlib.sha1((self.section + "|" + norm).encode("utf-8")).hexdigest()[:12]

    @property
    def score(self) -> int:
        return self.helpful - self.harmful

    def _format_line(self, idx: int) -> str:
        tag = f" [{', '.join(self.apps)}]" if self.apps else ""
        return f"{idx}. ({self.bullet_id}){tag} {self.text}"

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# generality guard
# ---------------------------------------------------------------------------
def is_reusable(text: str) -> tuple[bool, str]:
    """Return (ok, reason). Reject anything that smells task-specific.

    An entry is rejected if it references a concrete task id, embeds a literal
    submitted answer, or is too short/long to be a reusable strategy. This keeps
    the ledger a set of GENERAL strategies and prevents answer leakage.
    """
    t = (text or "").strip()
    low = t.lower()
    if len(t) < 15:
        return False, "too short"
    if len(t) > 600:
        return False, "too long (likely task-specific dump)"
    for tid in _HELD_OUT_TASK_IDS:
        if tid in low:
            return False, f"references eval task id {tid}"
    if _TASK_ID_PATTERN.search(low):
        return False, "references a concrete task id"
    # literal memorized answer patterns
    if re.search(r"complete_task\s*\(\s*answer\s*=\s*[\"'][^\"')]+[\"']", low):
        return False, "embeds a literal complete_task answer"
    # an entry that is mostly a person's specific contact/number is task-specific
    if re.search(r"\b\d{10}\b", t):
        return False, "embeds a specific phone number"
    return True, ""


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
class StrategyLedger:
    def __init__(self, use_hydra: bool = True) -> None:
        LEDGER_ROOT.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, LedgerEntry] = {}
        self.bridge = HydraBridge() if use_hydra else HydraBridge(api_key=None)
        self._read()

    # -- persistence --------------------------------------------------------
    @property
    def hydra_enabled(self) -> bool:
        return self.bridge.enabled

    def __len__(self) -> int:
        return len(self._entries)

    def _read(self) -> None:
        if not LEDGER_FILE.exists():
            return
        for raw in LEDGER_FILE.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                entry = LedgerEntry(**rec)
                self._entries[entry.bullet_id] = entry
            except Exception:
                continue

    def _persist(self) -> None:
        with LEDGER_FILE.open("w", encoding="utf-8") as fh:
            for entry in self._entries.values():
                fh.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")

    # -- curator: grow-and-refine merge ------------------------------------
    def _nearest_duplicate(self, cand: LedgerEntry) -> Optional[LedgerEntry]:
        if cand.bullet_id in self._entries:
            return self._entries[cand.bullet_id]
        cand_tokens = _keyword_set(cand.text)
        if not cand_tokens:
            return None
        best: Optional[LedgerEntry] = None
        best_sim = 0.0
        for entry in self._entries.values():
            entry_tokens = _keyword_set(entry.text)
            if not entry_tokens:
                continue
            inter = len(cand_tokens & entry_tokens)
            union = len(cand_tokens | entry_tokens)
            sim = inter / union if union else 0.0
            if sim > best_sim:
                best_sim, best = sim, entry
        if best is not None and best_sim >= DEDUP_THRESHOLD:
            return best
        return None

    def absorb(self, candidates: Iterable[LedgerEntry], flush: bool = True) -> dict:
        """Curator step: merge new entries in, de-duplicating against existing.

        Returns a small report dict {added, merged, rejected}. New ids are
        appended; near-duplicates fold their counters into the existing entry.
        """
        added = merged = rejected = 0
        fresh: list[LedgerEntry] = []
        for cand in candidates:
            ok, _reason = is_reusable(cand.text)
            if not ok:
                rejected += 1
                continue
            dup = self._nearest_duplicate(cand)
            if dup is not None:
                dup.helpful += max(cand.helpful, 0)
                dup.harmful += max(cand.harmful, 0)
                for a in cand.apps:
                    if a not in dup.apps:
                        dup.apps.append(a)
                merged += 1
            else:
                self._entries[cand.bullet_id] = cand
                fresh.append(cand)
                added += 1
        if flush:
            self._persist()
            self._mirror(fresh)
        return {"added": added, "merged": merged, "rejected": rejected, "total": len(self._entries)}

    def consolidate(self, threshold: float = 0.55, per_section_cap: int = 0, flush: bool = True) -> dict:
        """Grow-and-refine: consolidate near-duplicate entries already stored.

        Uses a looser similarity rule than ingest-time dedup (token Jaccard OR
        strong containment of the shorter entry) to collapse equivalent lessons
        that were phrased differently by the Reflector. The survivor keeps the
        clearest (longest) wording within a cluster and absorbs counters/apps.
        Seed entries are preferred as survivors to keep curated wording.
        """
        entries = list(self._entries.values())
        # process longer/seed entries first so they become cluster survivors
        entries.sort(key=lambda e: (0 if e.source == "seed" else 1, -len(e.text)))
        survivors: list[LedgerEntry] = []
        removed = 0
        for entry in entries:
            entry_tok = _keyword_set(entry.text)
            absorbed = False
            for survivor in survivors:
                if survivor.section != entry.section and not (set(survivor.apps) & set(entry.apps)):
                    # only merge across sections when apps overlap; else keep
                    pass
                survivor_tok = _keyword_set(survivor.text)
                if not entry_tok or not survivor_tok:
                    continue
                inter = len(entry_tok & survivor_tok)
                union = len(entry_tok | survivor_tok)
                jac = inter / union if union else 0.0
                contain = inter / min(len(entry_tok), len(survivor_tok))
                if jac >= threshold or contain >= 0.85:
                    survivor.helpful += entry.helpful
                    survivor.harmful += entry.harmful
                    for a in entry.apps:
                        if a not in survivor.apps:
                            survivor.apps.append(a)
                    absorbed = True
                    removed += 1
                    break
            if not absorbed:
                survivors.append(entry)
        capped = 0
        if per_section_cap > 0:
            by_section: dict[str, list[LedgerEntry]] = {}
            for survivor in survivors:
                by_section.setdefault(survivor.section, []).append(survivor)
            kept: list[LedgerEntry] = []
            for _sec, items in by_section.items():
                # keep seeds first, then highest score, then most informative (longest)
                items.sort(key=lambda e: (0 if e.source == "seed" else 1, -e.score, -len(e.text)))
                kept.extend(items[:per_section_cap])
                capped += max(0, len(items) - per_section_cap)
            survivors = kept
        self._entries = {s.bullet_id: s for s in survivors}
        if flush:
            self._persist()
        return {"removed": removed, "capped": capped, "total": len(self._entries)}

    def add_entry(self, text: str, section: str = "general", apps: Iterable[str] = (),
                  source: str = "seed", helpful: int = 0, harmful: int = 0) -> Optional[LedgerEntry]:
        entry = LedgerEntry(text=text, section=section, apps=list(apps),
                            source=source, helpful=helpful, harmful=harmful)
        report = self.absorb([entry])
        return entry if report["added"] else None

    # -- HydraDB mirror (bonus) --------------------------------------------
    def _mirror(self, entries: list[LedgerEntry]) -> None:
        if not entries or not self.bridge.enabled:
            return
        docs = []
        for entry in entries:
            md = (
                f"# playbook:{entry.section}\n\n- apps: {', '.join(entry.apps) or 'general'}\n"
                f"- source: {entry.source}\n\n{entry.text}\n"
            )
            docs.append((f"pb_{entry.bullet_id}", md, {
                "kind": "playbook",
                "section": entry.section,
                "apps": ",".join(entry.apps),
            }))
        try:
            self.bridge.store_documents(docs)
        except Exception:
            pass

    def push_all_to_hydra(self) -> bool:
        """Push the entire local ledger to HydraDB (used after seeding)."""
        if not self.bridge.enabled:
            return False
        self._mirror(list(self._entries.values()))
        return True

    # -- retrieval / rendering ---------------------------------------------
    def _rank(self, apps_hint: Iterable[str]) -> list[LedgerEntry]:
        apps = {a.lower() for a in apps_hint}

        def sort_key(entry: LedgerEntry) -> tuple:
            app_match = 1 if (apps & set(entry.apps)) else 0
            general = 1 if not entry.apps else 0
            return (app_match, general, entry.score, -entry.created_at)

        return sorted(self._entries.values(), key=sort_key, reverse=True)

    def render(self, apps_hint: Iterable[str] = (), max_bullets: int = MAX_ENTRIES_RENDERED,
               char_budget: int = RENDER_CHAR_BUDGET) -> str:
        """Render a compact, grouped ledger block for the solver context."""
        if not self._entries:
            return ""
        ranked = self._rank(apps_hint)[:max_bullets]
        # regroup by section but preserve the relevance ranking inside groups
        by_section: dict[str, list[LedgerEntry]] = {}
        order: list[str] = []
        for entry in ranked:
            if entry.section not in by_section:
                by_section[entry.section] = []
                order.append(entry.section)
            by_section[entry.section].append(entry)

        lines = [
            "AppWorld PLAYBOOK (general strategies & gotchas learned from past "
            "runs; HINTS ONLY -- the live API docs and this task's printed data "
            "always win over any bullet):",
        ]
        idx = 1
        used = len(lines[0])
        for section in order:
            header = f"\n## {section}"
            lines.append(header)
            used += len(header)
            for entry in by_section[section]:
                line = entry._format_line(idx)
                if used + len(line) > char_budget:
                    return "\n".join(lines).strip()
                lines.append(line)
                used += len(line)
                idx += 1
        return "\n".join(lines).strip()

    def recall_hydra(self, query: str, k: int = 4) -> list[str]:
        """Semantic recall of ledger entries from HydraDB (bonus path)."""
        try:
            hits = self.bridge.search(query, k=k)
        except Exception:
            return []
        out = []
        for h in hits:
            t = (h.get("text") or "").strip()
            if t:
                out.append(t)
        return out


# ---------------------------------------------------------------------------
# seed entries: the general lessons already encoded in the system prompt,
# itemized as a starting ledger (offline warmup baseline). GENERAL ONLY.
# ---------------------------------------------------------------------------
SEED_ENTRIES: list[tuple[str, str, list[str]]] = [
    # (text, section, apps)
    ("Discover APIs at runtime before using them: show_app_descriptions(), then "
     "show_api_descriptions(app_name=...), then show_api_doc(app_name=..., api_name=...). "
     "Never invent API or field names.", "discovery", []),
    ("Fetch the docs for every API you might call in an EARLIER turn, then call the "
     "API in a LATER turn using the exact documented parameter names. A guessed "
     "keyword is silently ignored, not an error.", "discovery", []),
    ("App namespaces are lowercase (apis.spotify, apis.venmo, apis.phone). Never use "
     "capitalized app names.", "discovery", []),
    ("Login pattern: username is usually the supervisor's email and the password "
     "comes from apis.supervisor.show_account_passwords() matched on key "
     "'account_name'. Password records do NOT contain an email field.", "auth", []),
    ("Some apps use the supervisor's phone_number (not email) as the login username "
     "-- always read the login doc's username description first. The phone app "
     "commonly uses phone_number.", "auth", ["phone"]),
    ("List/library endpoints are paginated. Always paginate to exhaustion with "
     "page_limit (max is usually 20); a single default page is almost never the full "
     "data. Stop when a batch returns fewer than the page_limit.", "pagination", []),
    ("Before aggregating/counting/sorting a collection, print len(items) and "
     "items[0].keys() to confirm the real shape. Never call .get('field') on a guessed "
     "field name.", "inspection", []),
    ("LIBRARY (list) endpoints and DETAIL endpoints return DIFFERENT field names. "
     "Print a sample from each before reading fields; a wrong field guess silently "
     "returns nothing.", "inspection", []),
    ("Spotify: show_song_library / show_album_library / show_playlist_library return "
     "LISTS. Album/playlist LIBRARY items expose a 'song_ids' list; album/playlist "
     "DETAIL responses (show_album / show_playlist) instead expose a 'songs' list of "
     "objects and have NO 'song_ids'. Check which you have before expanding.",
     "spotify", ["spotify"]),
    ("For top-N / ranking / 'most played' tasks: build the set of UNIQUE entity ids "
     "across every in-scope source first (dedupe across song/album/playlist), then "
     "fetch each entity's AUTHORITATIVE detail record once and read the metric "
     "(e.g. play_count) from THAT record. List entries often omit the metric.",
     "ranking", []),
    ("One entity = one record = one metric value. Never create more than one row per "
     "entity when ranking or counting.", "ranking", []),
    ("Re-read the task scope literally (which collections count) and the exact metric "
     "('most played' vs 'highest rated' vs 'largest') before sorting.", "ranking", []),
    ("Mutation completeness: pass EVERY detail the task specifies -- a memo/note, an "
     "exact amount, exact message text, a recipient, a date, a title -- each mapped to "
     "its correct documented parameter. Optional-looking fields like 'description' are "
     "graded; never omit them. A payment 'note' is often the parameter 'description', "
     "not 'note'.", "mutation", []),
    ("Extract values precisely from the source (conversation, bill, email). Never "
     "approximate or guess an amount; read the exact number from the data.", "mutation", []),
    ("After a mutation, re-fetch the created/updated record if possible and confirm "
     "each required field was set to the exact requested value before completing.",
     "mutation", []),
    ("Relative dates ('today', 'this year', 'last month', 'recent') must be computed "
     "from the environment's current date/time provided in the task context, NOT from "
     "real-world wall-clock time.", "dates", []),
    ("For a question task, the final answer must be exactly the requested "
     "count/list/title/amount -- never a placeholder like 'done', 'Task already "
     "completed', or explanatory prose, and never an empty string/list unless the task "
     "explicitly asks for empty.", "answer_format", []),
    ("Counts must be digit strings derived from printed data. Top-N answers must "
     "contain exactly N deduplicated items. If a count equals a default page size "
     "(e.g. 5 or 20), you probably did not paginate -- fetch all pages.",
     "answer_format", []),
    ("Before completing, assign the exact answer to a variable final_answer and print "
     "{'final_answer': final_answer, 'evidence': ...} in the SAME block, then pass that "
     "same value to complete_task.", "answer_format", []),
    ("Call complete_task(answer=...) for question tasks as soon as the answer is "
     "derived; call complete_task(answer=None) for action/mutation tasks only after "
     "the required app state has actually been changed.", "answer_format", []),
    ("On error, immediately inspect that API's doc and retry with corrected "
     "names/types/values. If the SAME error repeats, change your APPROACH or the VALUE "
     "you pass (not just the parameter name) -- e.g. for login, the username value may "
     "need to be the phone number string rather than the email.", "recovery", []),
    ("Bundle safe read/API calls, filtering, and computation into one Python block to "
     "save turns, but take as many turns as needed to inspect schemas and verify. A "
     "confident wrong answer scores zero -- accuracy beats speed.", "efficiency", []),
    ("Respond with EXACTLY one fenced python code block per turn and no prose outside "
     "it. Never call exit()/quit() or raise to stop; if blocked, print the missing "
     "facts and continue.", "efficiency", []),
    ("Print compact summaries (lengths, sample keys, the few fields you need), not "
     "giant raw dumps, to keep the working context focused.", "efficiency", []),
]


def seed_ledger(ledger: StrategyLedger) -> dict:
    """Populate a ledger with the GENERAL default entries. Idempotent."""
    cands = [LedgerEntry(text=t, section=s, apps=a, source="seed") for t, s, a in SEED_ENTRIES]
    return ledger.absorb(cands)


# ---------------------------------------------------------------------------
# CLI: python strategy_ledger.py [--seed] [--show] [--apps spotify,phone]
# ---------------------------------------------------------------------------
def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect / seed the ACE strategy ledger.")
    ap.add_argument("--seed", action="store_true", help="seed default general entries")
    ap.add_argument("--refine", action="store_true", help="consolidate near-duplicate entries")
    ap.add_argument("--refine-threshold", type=float, default=0.55)
    ap.add_argument("--per-section-cap", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="render the ledger")
    ap.add_argument("--apps", default="", help="comma-separated apps hint for render")
    ap.add_argument("--no-hydra", action="store_true")
    ap.add_argument("--sync-hydra", action="store_true", help="push whole ledger to HydraDB")
    args = ap.parse_args()

    ledger = StrategyLedger(use_hydra=not args.no_hydra)
    print(f"Ledger loaded: {len(ledger)} entry(ies) | hydra={ledger.hydra_enabled}")
    if args.seed:
        report = seed_ledger(ledger)
        print(f"Seed merge: {report}")
    if args.refine:
        report = ledger.consolidate(threshold=args.refine_threshold, per_section_cap=args.per_section_cap)
        print(f"Refine: {report}")
    if args.sync_hydra:
        ok = ledger.push_all_to_hydra()
        print(f"Synced all entries to HydraDB: {ok}")
    if args.show or not (args.seed or args.sync_hydra):
        apps = [a for a in args.apps.split(",") if a]
        print("\n" + ledger.render(apps_hint=apps))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
