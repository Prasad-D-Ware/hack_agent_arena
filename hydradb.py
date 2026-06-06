"""
HydraDB integration for the AppWorld agent — the 🐉 bonus.

HydraDB (https://hydradb.com) is a graph-native *context layer* for AI agents.
It stores two kinds of context, both retrieved through one `query()` call:

  • memory    — experiential, per-agent records ("what worked on past tasks")
  • knowledge — documents/facts ("the AppWorld API docs")

We use BOTH:

  A) Episodic memory  — after each task we ingest(type="memory") a compact
     summary of the trajectory; before the next task we query(type="memory") and
     inject the most relevant past experience into the prompt, so the agent stops
     rediscovering the same APIs and repeating mistakes from scratch.

  B) API-doc knowledge — once per run we ingest(type="knowledge") the AppWorld
     per-app API descriptions, then query(type="knowledge") retrieves only the
     relevant ones per task (RAG over the 457 APIs).

NOTE on async indexing: HydraDB ingest is asynchronous (202 Accepted). The API
docs and the first task's memory won't appear in recall() until they are indexed
(typically seconds to a minute). In practice tasks take long enough that by the
time the NEXT task runs the prior memory is indexed. No polling is added because
it would stall the run; the benefit accumulates from task 2 onward.

Everything here is OPTIONAL and defensive: if HydraDB is disabled (USE_HYDRA!=1),
unconfigured (no HYDRA_DB_API_KEY), uninstalled, or erroring, every method
degrades to a no-op and the agent runs exactly as it did before.

SDK: pip install "hydradb-sdk>=2,<3"  |  from hydra_db import HydraDB
Calls used: client.tenants.create/status, client.context.ingest, client.query.
Response envelopes wrap the payload under `.data`; we unwrap defensively because
field shapes can vary by SDK minor version.
"""

import io
import json
import os
import time


def _enabled() -> bool:
    flag = os.environ.get("USE_HYDRA", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(os.environ.get("HYDRA_DB_API_KEY"))


def _loads_lenient(raw):
    """Parse JSON from world.execute() output, tolerating surrounding text.

    AppWorld returns whatever the code printed; if anything wraps our JSON line
    we still recover it by slicing between the outermost braces.
    """
    s = str(raw).strip()
    try:
        return json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def _unwrap(res):
    """Return the payload dict/object from a HydraDB response envelope."""
    data = getattr(res, "data", None)
    if data is None:
        if isinstance(res, dict):
            data = res.get("data", res)
        else:
            data = res
    return data or {}


def _get(obj, key, default=None):
    """Read `key` off a dict OR an attr-style object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class HydraMemory:
    """Thin, fail-safe wrapper around the HydraDB SDK for the agent loop."""

    def __init__(self) -> None:
        self.client = None
        self.tenant_id = os.environ.get("HYDRA_TENANT_ID", "appworld_agent")
        self.max_results = max(5, min(50, int(os.environ.get("HYDRA_MAX_RESULTS", "5"))))  # API range 5..50
        self.chunk_chars = int(os.environ.get("HYDRA_CHUNK_CHARS", "1000"))  # cap per recalled chunk
        self._docs_done = False     # flipped to True only after a successful ingest
        self._docs_attempts = 0     # bounded retries so a persistent B failure can't tax every task
        self._docs_max_attempts = int(os.environ.get("HYDRA_DOCS_MAX_ATTEMPTS", "3"))

        if not _enabled():
            return
        try:
            from hydra_db import HydraDB

            self.client = HydraDB(token=os.environ["HYDRA_DB_API_KEY"])
            self._ensure_tenant()
            print(f"  [hydra] enabled (tenant={self.tenant_id})")
        except Exception as e:  # missing pkg, bad key, network — never fatal
            print(f"  [hydra] disabled (init failed: {e})")
            self.client = None

    @property
    def on(self) -> bool:
        return self.client is not None

    # -- setup ----------------------------------------------------------------
    def _ensure_tenant(self) -> None:
        """Create the tenant (idempotent) and best-effort wait until ready."""
        try:
            self.client.tenants.create(tenant_id=self.tenant_id)
        except Exception:
            pass  # "already exists" is the common case — safe to ignore

        timeout = int(os.environ.get("HYDRA_READY_TIMEOUT", "30"))
        for _ in range(timeout):
            try:
                st = _unwrap(self.client.tenants.status(tenant_id=self.tenant_id))
                infra = _get(st, "infra", {})
                if _get(infra, "ready_for_ingestion", True):
                    return
            except Exception:
                return  # unknown shape or network hiccup — don't block the run
            time.sleep(1)

    # -- A) episodic memory ---------------------------------------------------
    def remember_task(self, instruction: str, messages: list, success: bool) -> None:
        """Ingest the task transcript as an episodic memory.

        Reads the loop's existing `messages` list (skipping the seed) so that
        NOTHING needs to be captured inside the reasoning loop itself. messages[0]
        is the seed prompt (which may contain prior recall) and is skipped so we
        never fold retrieved context back into a new memory.
        """
        if not self.on or len(messages) <= 1:
            return  # only the seed present → no steps ran, nothing to learn
        try:
            turns = []
            for m in messages[1:]:
                role = m.get("role", "?")
                content = str(m.get("content", ""))[:1200]  # cap each turn's size
                turns.append(f"[{role}] {content}")
            text = (
                f"AppWorld task ({'SOLVED' if success else 'FAILED'}): {instruction}\n\n"
                + "\n\n".join(turns)
            )
            self.client.context.ingest(
                type="memory",
                tenant_id=self.tenant_id,
                memories=json.dumps([{
                    "text": text,
                    "infer": False,   # store verbatim; already a structured episode
                    "metadata": {
                        "kind": "episode",
                        "success": "true" if success else "false",  # strings only
                    },
                }]),
            )
        except Exception as e:
            print(f"  [hydra] remember failed: {e}")

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

    # -- B) API-doc knowledge -------------------------------------------------
    def ingest_api_docs(self, world) -> None:
        """One-time: pull AppWorld per-app API descriptions and store as knowledge.

        Executed inside the first live task's sandbox so `apis` is available.
        `_docs_done` is only set to True on success; if the execute or ingest
        fails, the next task retries (idempotent upsert via stable `id`s), up to
        `_docs_max_attempts` so a persistent failure can't tax every task.
        """
        if not self.on or self._docs_done or self._docs_attempts >= self._docs_max_attempts:
            return
        self._docs_attempts += 1
        try:
            # Run inside the AppWorld sandbox to access the `apis` object.
            # Per-app try/except inside the code block ensures one bad app
            # cannot silently drop all others.
            dump = "\n".join([
                "import json",
                "apps = apis.api_docs.show_app_descriptions()",
                # show_app_descriptions() returns a list of dicts OR a dict keyed by name
                "if isinstance(apps, dict):",
                "    names = list(apps.keys())",
                "elif isinstance(apps, list):",
                "    names = [a.get('name') or a.get('app_name') for a in apps if isinstance(a, dict)]",
                "else:",
                "    names = []",
                "out = {}",
                "for n in names:",
                "    if not n: continue",
                "    try:",
                "        out[n] = apis.api_docs.show_api_descriptions(app_name=n)",
                "    except Exception as ex:",
                "        out[n] = f'[error: {ex}]'",
                "print(json.dumps(out, default=str))",
            ])
            raw = world.execute(dump)
            descriptions = _loads_lenient(raw)  # tolerate any wrapper text around the JSON
            if not isinstance(descriptions, dict):
                raise ValueError(f"unexpected shape from show_app_descriptions: {type(descriptions)}")

            valid = [
                (app, text) for app, text in descriptions.items()
                if text and not str(text).startswith("[error:")
            ]
            if not valid:
                print("  [hydra] api-doc ingest skipped (no valid app descriptions returned)")
                return
            # Use HydraDB's DOCUMENTED knowledge path: documents (file-like) +
            # document_metadata, aligned by position. One in-memory text "file"
            # per app so retrieval returns per-app chunks; stable ids make a
            # re-ingest an idempotent upsert.
            documents, document_metadata = [], []
            for app, text in valid:
                body = f"AppWorld API descriptions for app '{app}':\n{text}".encode("utf-8")
                documents.append((f"apidoc_{app}.txt", io.BytesIO(body), "text/plain"))
                document_metadata.append({"id": f"apidoc_{app}", "metadata": {"kind": "api_doc", "app": app}})
            self.client.context.ingest(
                type="knowledge",
                tenant_id=self.tenant_id,
                documents=documents,
                document_metadata=json.dumps(document_metadata),
            )
            self._docs_done = True  # only mark done after successful ingest
            print(f"  [hydra] ingested API docs for {len(documents)} apps")
        except Exception as e:
            print(f"  [hydra] api-doc ingest failed: {e}")
            # _docs_done stays False → next task will retry

    # -- retrieval ------------------------------------------------------------
    def recall(self, instruction: str, kind: str = "all") -> str:
        """Query memory + knowledge for this task; return a prompt-ready string.

        Returns "" when disabled, on error, or when nothing relevant is found —
        the caller must handle the empty-string case gracefully.
        """
        if not self.on:
            return ""
        try:
            res = self.client.query(
                tenant_id=self.tenant_id,
                query=instruction,
                type=kind,            # "all" | "memory" | "knowledge"
                query_by="hybrid",    # semantic + BM25
                mode="thinking",      # multi-pass expansion
                max_results=self.max_results,
            )
            chunks = _get(_unwrap(res), "chunks", []) or []
            lines = []
            for c in chunks:
                text = _get(c, "chunk_content") or _get(c, "content") or ""
                title = _get(c, "source_title") or _get(c, "source_type") or ""
                text = str(text).strip()
                if text:
                    if len(text) > self.chunk_chars:  # keep the injected context bounded
                        text = text[:self.chunk_chars] + "…"
                    lines.append(f"- ({title}) {text}")
            if not lines:
                return ""
            return (
                "Relevant past experience and API knowledge retrieved from HydraDB "
                "(use it to skip rediscovery and avoid earlier mistakes):\n"
                + "\n".join(lines)
            )
        except Exception as e:
            print(f"  [hydra] recall failed: {e}")
            return ""
