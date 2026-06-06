"""Self-improving skill store for the AppWorld agent.

Two cooperating backends:

* ``LocalSkillCache`` - instant, deterministic on-disk cache of distilled
  skills/lessons (Markdown + a JSONL index). Always available, even with no
  network or API key. This is what makes within-run skill reuse work.
* ``HydraBridge`` - thin, defensive wrapper over the HydraDB SDK that mirrors
  skills into HydraDB ``knowledge`` and lessons into HydraDB ``memory`` and does
  semantic/graph recall. Degrades to a no-op if the SDK or key is missing or any
  call fails, so the agent never breaks.

``SkillRepository`` is the facade the agent and scripts use: it writes to both
and recalls from both, merging + de-duplicating results.

Design rules (unchanged):
* Skills are HINTS, AppWorld API docs are the source of truth.
* Never store secrets (passwords, access tokens, JWTs) in any backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # load .env so the module works standalone and inside scripts
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
STORE_ROOT = Path(os.environ.get("MEMORY_DIR", Path(__file__).resolve().parent / "memory"))
CARDS_DIR = STORE_ROOT / "skills"
INDEX_FILE = STORE_ROOT / "index.jsonl"

HYDRA_TENANT = os.environ.get("HYDRA_TENANT_ID", "appworld_agentathon")
HYDRA_SUBTENANT = os.environ.get("HYDRA_SUB_TENANT", "dev")
HYDRA_KEY = (
    os.environ.get("HYDRA_DB_API_KEY")
    or os.environ.get("HYDRADB_API_KEY")
    or os.environ.get("HYDRA_API_KEY")
)
# When set, recall returns only eval-verified knowledge (outcome pass/fail),
# never unverified in-run "completed" captures. Prevents memory poisoning.
VERIFIED_RECALL_ONLY = os.environ.get("MEMORY_VERIFIED_ONLY", "1") != "0"
HYDRA_QUERY_MODE = os.environ.get("HYDRA_RECALL_MODE", "fast")  # fast | accurate
HYDRA_REQUEST_TIMEOUT = float(os.environ.get("HYDRA_TIMEOUT", "20"))
HYDRA_PROVISION_WAIT = float(os.environ.get("HYDRA_TENANT_WAIT", "180"))  # max wait for async provisioning
HYDRA_POLL_SECONDS = float(os.environ.get("HYDRA_POLL_INTERVAL", "5"))

_NOISE_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "my",
    "me", "is", "are", "be", "that", "this", "it", "as", "at", "by", "from",
    "i", "you", "your", "their", "them", "then", "if", "all", "any", "into",
    "list", "show", "get", "give", "tell", "find", "please", "want", "need",
}


# ---------------------------------------------------------------------------
# secret scrubbing
# ---------------------------------------------------------------------------
_JWT_TOKEN_RE = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)([\"']?(?:password|passwd|pwd|access[_-]?token|token|api[_-]?key|secret|"
    r"client[_-]?secret|authorization|bearer)[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,}\)]+)"
)


def redact_secrets(text: str) -> str:
    """Redact credentials so nothing sensitive is ever persisted."""
    if not text:
        return text
    text = _JWT_TOKEN_RE.sub("<redacted-token>", text)
    text = _SECRET_ASSIGN_RE.sub(r"\1<redacted>", text)
    return text


def _keyword_set(text: str) -> set[str]:
    found = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())
    return {w for w in found if len(w) > 2 and w not in _NOISE_WORDS}


# ---------------------------------------------------------------------------
# skill model
# ---------------------------------------------------------------------------
@dataclass
class SkillNote:
    title: str
    apps: list[str] = field(default_factory=list)
    task_type: str = "answer"          # answer | mutation | mixed
    when_to_use: str = ""
    procedure: str = ""                # numbered API flow / approach
    pitfalls: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    answer_shape: str = ""
    outcome: str = "pass"              # pass | fail | completed
    source_task_id: str = ""
    source_experiment: str = ""
    tags: list[str] = field(default_factory=list)
    skill_id: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.skill_id:
            self.skill_id = self._derive_id()
        if not self.created_at:
            self.created_at = time.time()
        # scrub everything that can hold free text
        self.procedure = redact_secrets(self.procedure)
        self.when_to_use = redact_secrets(self.when_to_use)
        self.pitfalls = [redact_secrets(p) for p in self.pitfalls]
        self.verification = [redact_secrets(v) for v in self.verification]

    def _derive_id(self) -> str:
        key_words = sorted(_keyword_set(self.title))[:6]
        sig = "|".join(sorted(self.apps)) + "|" + self.task_type + "|" + "_".join(key_words)
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]

    @property
    def kind(self) -> str:
        return "lesson" if self.outcome == "fail" else "skill"

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        lines.append(f"- apps: {', '.join(self.apps) or 'unknown'}")
        lines.append(f"- task_type: {self.task_type}")
        lines.append(f"- outcome: {self.outcome}")
        if self.source_task_id:
            lines.append(f"- source_task: {self.source_task_id}")
        lines.append("")
        if self.when_to_use:
            lines += ["## When to use", self.when_to_use, ""]
        if self.procedure:
            lines += ["## Procedure", self.procedure, ""]
        if self.pitfalls:
            lines += ["## Pitfalls"] + [f"- {p}" for p in self.pitfalls] + [""]
        if self.verification:
            lines += ["## Verification"] + [f"- {v}" for v in self.verification] + [""]
        if self.answer_shape:
            lines += ["## Answer shape", self.answer_shape, ""]
        return "\n".join(lines).strip() + "\n"

    def _index_text(self) -> str:
        return " ".join(
            [self.title, " ".join(self.apps), self.task_type, self.when_to_use,
             " ".join(self.tags), self.procedure]
        )


# ---------------------------------------------------------------------------
# local cache
# ---------------------------------------------------------------------------
class LocalSkillCache:
    def __init__(self) -> None:
        CARDS_DIR.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict] = {}
        self._read_index()

    def _read_index(self) -> None:
        if not INDEX_FILE.exists():
            return
        for raw in INDEX_FILE.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                self._records[rec["skill_id"]] = rec
            except Exception:
                continue

    def _persist(self) -> None:
        with INDEX_FILE.open("w", encoding="utf-8") as fh:
            for rec in self._records.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def upsert(self, note: SkillNote) -> dict:
        md = note.to_markdown()
        (CARDS_DIR / f"{note.skill_id}.md").write_text(md, encoding="utf-8")
        rec = {
            "skill_id": note.skill_id,
            "title": note.title,
            "apps": note.apps,
            "task_type": note.task_type,
            "outcome": note.outcome,
            "tags": note.tags,
            "source_task_id": note.source_task_id,
            "source_experiment": note.source_experiment,
            "created_at": note.created_at,
            "search": note._index_text().lower(),
            "markdown": md,
        }
        self._records[note.skill_id] = rec  # upsert
        self._persist()
        return rec

    def lookup(self, query: str, k: int = 3, apps_hint: Iterable[str] = ()) -> list[dict]:
        q_tokens = _keyword_set(query)
        hint = {a.lower() for a in apps_hint}
        ranked: list[tuple[float, dict]] = []
        for rec in self._records.values():
            # verified-only mode: skip unverified in-run captures entirely
            if VERIFIED_RECALL_ONLY and rec.get("outcome") not in {"pass", "fail"}:
                continue
            rec_tokens = _keyword_set(rec.get("search", ""))
            if not rec_tokens:
                continue
            rec_apps = {a.lower() for a in rec.get("apps", [])}
            overlap = len(q_tokens & rec_tokens)
            app_match = bool(hint & rec_apps)
            if overlap == 0 and not app_match:
                continue
            score = float(overlap)
            if app_match:
                score += 2.0
            # prefer eval-verified knowledge over unverified in-run captures
            outcome = rec.get("outcome")
            if outcome == "pass":
                score += 1.5   # verified-correct recipe
            elif outcome == "fail":
                score += 0.75  # verified failure -> valuable warning
            # outcome == "completed" (unverified capture): no boost
            ranked.append((score, rec))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _, rec in ranked[:k]]

    def entries(self) -> list[dict]:
        return list(self._records.values())


# ---------------------------------------------------------------------------
# HydraDB bridge (defensive)
# ---------------------------------------------------------------------------
class HydraBridge:
    def __init__(
        self,
        tenant_id: str = HYDRA_TENANT,
        sub_tenant_id: str = HYDRA_SUBTENANT,
        api_key: Optional[str] = HYDRA_KEY,
    ) -> None:
        self.tenant_id = tenant_id
        self.sub_tenant_id = sub_tenant_id
        self.api_key = api_key
        self._handle = None
        self._tenant_ready = False
        self._noted = False
        self.enabled = bool(api_key)
        if self.enabled:
            try:
                from hydra_db import HydraDB  # noqa: F401
            except Exception as exc:  # SDK not installed
                self._note(f"hydra_db import failed ({exc}); HydraDB disabled")
                self.enabled = False

    # -- helpers ------------------------------------------------------------
    def _note(self, msg: str) -> None:
        # rate-limited but not permanently muted, so repeated failures stay visible
        self._note_count = getattr(self, "_note_count", 0) + 1
        if self._note_count <= 8:
            print(f"  [hydra] {msg}")

    @staticmethod
    def _looks_like_provisioning(exc: Exception) -> bool:
        s = str(exc).upper()
        return "TENANT_NOT_FOUND" in s or "NOT_READY" in s or "NOT PROVISIONED" in s

    def _client_handle(self):
        if self._handle is None:
            from hydra_db import HydraDB
            self._handle = HydraDB(token=self.api_key, timeout=HYDRA_REQUEST_TIMEOUT)
        return self._handle

    def _ingestion_ready(self, handle) -> bool:
        try:
            resp = handle.tenants.status(tenant_id=self.tenant_id)
            infra = getattr(getattr(resp, "data", None), "infra", None)
            return bool(getattr(infra, "ready_for_ingestion", False))
        except Exception:
            return False

    def _ensure_tenant(self) -> bool:
        if not self.enabled:
            return False
        if self._tenant_ready:
            return True
        try:
            handle = self._client_handle()
            try:
                handle.tenants.create(tenant_id=self.tenant_id, is_embeddings_tenant=False)
            except Exception as exc:
                # already-exists is fine; anything else we log once
                if "exist" not in str(exc).lower() and "409" not in str(exc):
                    self._note(f"tenant create note: {exc}")
            # tenant provisioning is asynchronous; poll until infra is ready
            deadline = time.time() + HYDRA_PROVISION_WAIT
            while time.time() < deadline:
                if self._ingestion_ready(handle):
                    self._tenant_ready = True
                    return True
                time.sleep(HYDRA_POLL_SECONDS)
            self._note(
                f"tenant '{self.tenant_id}' not ready after {HYDRA_PROVISION_WAIT}s; "
                "skipping HydraDB this run (it will likely be ready next run)"
            )
            return False
        except Exception as exc:
            self._note(f"ensure_tenant failed ({exc}); HydraDB disabled")
            self.enabled = False
            return False

    # -- writes -------------------------------------------------------------
    def _write_with_retry(self, action) -> bool:
        try:
            action()
            return True
        except Exception as exc:
            if self._looks_like_provisioning(exc):
                # status said ready but the collection lagged; wait once and retry
                self._tenant_ready = False
                if self._ensure_tenant():
                    try:
                        action()
                        return True
                    except Exception as exc2:
                        self._note(f"ingest retry failed ({exc2})")
                        return False
            self._note(f"ingest failed ({exc})")
            return False

    def store_skill(self, note: SkillNote) -> bool:
        if not self._ensure_tenant():
            return False
        handle = self._client_handle()
        md = note.to_markdown().encode("utf-8")
        # one metadata object per document, as a JSON array
        meta = json.dumps([{
            "apps": ",".join(note.apps),
            "task_type": note.task_type,
            "outcome": note.outcome,
            "source_task_id": note.source_task_id,
            "tags": ",".join(note.tags),
        }])

        def _push():
            handle.context.ingest(
                tenant_id=self.tenant_id,
                type="knowledge",
                upsert=True,
                documents=[(f"{note.skill_id}.md", md, "text/markdown")],
                document_metadata=meta,
            )

        return self._write_with_retry(_push)

    def store_documents(self, docs: list[tuple[str, str, dict]]) -> bool:
        """Ingest arbitrary knowledge documents (used by the strategy ledger).

        ``docs`` is a list of ``(doc_id, markdown_text, metadata_dict)``. Each
        doc is mirrored into HydraDB ``knowledge`` so the ledger persists and is
        semantically recallable across runs (the HydraDB bonus story).
        """
        if not docs:
            return True
        if not self._ensure_tenant():
            return False
        handle = self._client_handle()
        documents = [
            (f"{doc_id}.md", redact_secrets(text).encode("utf-8"), "text/markdown")
            for doc_id, text, _meta in docs
        ]
        meta = json.dumps([m for _id, _t, m in docs])

        def _push():
            handle.context.ingest(
                tenant_id=self.tenant_id,
                type="knowledge",
                upsert=True,
                documents=documents,
                document_metadata=meta,
            )

        return self._write_with_retry(_push)

    def store_lesson(self, note: SkillNote) -> bool:
        if not self._ensure_tenant():
            return False
        handle = self._client_handle()
        text = f"{note.title}\n{note.to_markdown()}"
        memories = json.dumps([{"text": redact_secrets(text), "infer": False}])

        def _push():
            handle.context.ingest(
                tenant_id=self.tenant_id,
                type="memory",
                sub_tenant_id=self.sub_tenant_id,
                memories=memories,
            )

        return self._write_with_retry(_push)

    # -- reads --------------------------------------------------------------
    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.enabled:
            return []
        if not self._ensure_tenant():
            return []
        results: list[dict] = []
        try:
            handle = self._client_handle()
            resp = handle.query(
                tenant_id=self.tenant_id,
                query=query,
                max_results=k,
                mode=HYDRA_QUERY_MODE,
                graph_context=True,
            )
            chunks = getattr(getattr(resp, "data", None), "chunks", None) or []
            for ch in chunks:
                content = getattr(ch, "chunk_content", None)
                if not content:
                    continue
                results.append({
                    "text": content,
                    "title": getattr(ch, "source_title", "") or "",
                    "score": getattr(ch, "relevancy_score", None),
                    "source": "hydra",
                })
        except Exception as exc:
            self._note(f"recall failed ({exc})")
        return results


# ---------------------------------------------------------------------------
# facade
# ---------------------------------------------------------------------------
class SkillRepository:
    def __init__(self, use_hydra: bool = True) -> None:
        self.cache = LocalSkillCache()
        self.bridge = HydraBridge() if use_hydra else HydraBridge(api_key=None)

    @property
    def hydra_enabled(self) -> bool:
        return self.bridge.enabled

    def store(self, note: SkillNote) -> None:
        self.cache.upsert(note)
        if note.outcome == "fail":
            self.bridge.store_lesson(note)
        else:
            self.bridge.store_skill(note)

    @staticmethod
    def _dedup_key(text: str) -> str:
        norm = re.sub(r"\s+", " ", (text or "").strip().lower())[:160]
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()

    def recall(self, query: str, k: int = 3, apps_hint: Iterable[str] = ()) -> list[str]:
        """Return up to k formatted skill blocks (Hydra-first, then local)."""
        blocks: list[str] = []
        seen: set[str] = set()

        for hit in self.bridge.search(query, k=k):
            text = hit.get("text", "").strip()
            if not text:
                continue
            key = self._dedup_key(text)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(text)

        if len(blocks) < k:
            for rec in self.cache.lookup(query, k=k, apps_hint=apps_hint):
                md = rec.get("markdown", "").strip()
                if not md:
                    continue
                key = self._dedup_key(md)
                if key in seen:
                    continue
                seen.add(key)
                blocks.append(md)
                if len(blocks) >= k:
                    break
        return blocks[:k]


# ---------------------------------------------------------------------------
# self-test CLI: python skill_store.py --selftest
# ---------------------------------------------------------------------------
def _self_test() -> int:
    print(f"HYDRA key present: {bool(HYDRA_KEY)} | tenant={HYDRA_TENANT} | sub={HYDRA_SUBTENANT}")
    repo = SkillRepository()
    print(f"Hydra enabled: {repo.hydra_enabled}")
    demo = SkillNote(
        title="selftest spotify top-n genre ranking",
        apps=["spotify"],
        task_type="answer",
        when_to_use="A connectivity self-test skill; safe to delete.",
        procedure="1. login 2. gather library 3. sort by play_count 4. answer top N",
        pitfalls=["This is a self-test record."],
        verification=["Ignore in production."],
        source_task_id="selftest",
        tags=["selftest"],
    )
    repo.store(demo)
    print(f"Stored skill {demo.skill_id} locally + (hydra={repo.hydra_enabled}).")
    if repo.hydra_enabled:
        print("Waiting for HydraDB indexing (best-effort, up to 30s)...")
        time.sleep(5)
    hits = repo.recall("spotify top songs by play count", k=3, apps_hint=["spotify"])
    print(f"Recalled {len(hits)} block(s):")
    for i, h in enumerate(hits, 1):
        print(f"--- block {i} ---")
        print(h[:300])
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_self_test())
    print("skill_store: import this module, or run with --selftest")
