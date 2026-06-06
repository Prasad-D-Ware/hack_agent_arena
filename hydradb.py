"""
HydraDB integration for the AppWorld agent — the 🐉 bonus.

Two roles, both optional and fail-safe:

  A) Episodic memory  — after each task, remember the episode; before each task,
     recall the most relevant past experience into the prompt. Lives at the EDGES
     of solve(); the reasoning loop is never touched.

  B) API-doc knowledge — the 457 AppWorld API docs (a static snapshot committed at
     assets/api_docs.json) are ensured once with stable ids. bootstrap_docs.py can
     do that explicitly, and agent.py can do it at startup when Hydra is enabled.
     Both paths check first and ingest only missing records. During subgoals the
     agent only QUERIES them via recall().

Run-time behaviour is defensive: if HydraDB is disabled (USE_HYDRA!=1),
unconfigured (no HYDRA_DB_API_KEY), uninstalled, or erroring, recall(),
remember_task(), and runtime API-doc ensure degrade safely so the agent can keep
running. The standalone bootstrap still fails loudly because it is an operator
setup command.

SDK: pip install "hydradb-sdk>=2,<3"  |  from hydra_db import HydraDB
Calls: client.tenants.create/status, client.context.ingest/status, client.query.
Envelopes wrap payloads under `.data`; we unwrap defensively.
"""

import io
import json
import os
import time

DEFAULT_API_DOCS_ARTIFACT = "assets/api_docs.json"


def _enabled() -> bool:
    flag = os.environ.get("USE_HYDRA", "0").strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(os.environ.get("HYDRA_DB_API_KEY"))


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


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _status_values(st):
    """Extract a list of lowercased indexing-status strings from a status response.

    Tolerates list, dict-keyed-by-id, {statuses:[...]}, and single-object shapes.
    """
    data = _unwrap(st)
    if isinstance(data, dict):
        if isinstance(data.get("statuses"), list):
            items = data["statuses"]
        elif data and all(isinstance(v, dict) for v in data.values()):
            items = list(data.values())
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []
    out = []
    for it in items:
        s = _get(it, "indexing_status") or _get(it, "status")
        if s:
            out.append(str(s).lower())
    return out


def _status_by_id(st) -> dict:
    """Extract {document_id: status} from a HydraDB status response if present."""
    data = _unwrap(st)
    items = []
    if isinstance(data, dict):
        if isinstance(data.get("statuses"), list):
            items = data["statuses"]
        elif data and all(isinstance(v, dict) for v in data.values()):
            for key, value in data.items():
                item = dict(value)
                item.setdefault("id", key)
                items.append(item)
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    out = {}
    for item in items:
        doc_id = (_get(item, "id") or _get(item, "document_id")
                  or _get(item, "source_id") or _get(item, "file_id"))
        status = _get(item, "indexing_status") or _get(item, "status")
        if doc_id and status:
            out[str(doc_id)] = str(status).lower()
    return out


def _looks_indexed(status: str) -> bool:
    return str(status).lower() in {"completed", "complete", "indexed", "ready", "succeeded", "success"}


def _format_api_doc(app: str, api: str, doc) -> str:
    """Render one structured AppWorld API doc into retrieval-friendly text.

    Keeps the semantic signal (names + descriptions) for hybrid search AND the
    exact signature (params/types/required, returns) the agent needs to call it.
    Falls back to a plain dump for any unexpected shape.
    """
    if not isinstance(doc, dict):
        return f"{app}.{api}\n{doc}"
    method, path = doc.get("method", ""), doc.get("path", "")
    header = f"{app}.{api}" + (f"  ({method} {path})".rstrip() if (method or path) else "")
    lines = [header]
    if doc.get("description"):
        lines.append(f"Description: {doc['description']}")
    params = doc.get("parameters") or []
    if params:
        lines.append("Parameters:")
        for p in params:
            if not isinstance(p, dict):
                lines.append(f"  - {p}")
                continue
            req = "required" if p.get("required") else "optional"
            piece = f"  - {p.get('name', '?')} ({p.get('type', '?')}, {req})"
            if p.get("description"):
                piece += f": {p['description']}"
            if p.get("default") is not None:
                piece += f" [default: {p['default']}]"
            lines.append(piece)
    else:
        lines.append("Parameters: none")
    rs = doc.get("response_schemas")
    if isinstance(rs, dict):
        if "success" in rs:
            lines.append(f"Returns (success): {json.dumps(rs['success'])}")
        if "failure" in rs:
            lines.append(f"Returns (failure): {json.dumps(rs['failure'])}")
    return "\n".join(lines)


def _flatten_api_docs(api_docs: dict) -> list:
    """Flatten the artifact ({meta, apps:{app:{apis:{api:doc}}}}) into per-API records.

    Accepts the full artifact OR just its `apps` map. Skips errored entries.
    """
    apps = api_docs.get("apps", api_docs) if isinstance(api_docs, dict) else {}
    out = []
    for app, v in apps.items():
        apis = (v.get("apis", {}) if isinstance(v, dict) else {}) or {}
        for api, doc in apis.items():
            if isinstance(doc, dict) and list(doc.keys()) == ["error"]:
                continue
            out.append({
                "id": f"apidoc_{app}_{api}",
                "app": app,
                "api": api,
                "text": _format_api_doc(app, api, doc),
            })
    return out


class HydraMemory:
    """Fail-safe wrapper around the HydraDB SDK for both the agent loop and the bootstrap."""

    def __init__(self, force_enable: bool = False) -> None:
        self.client = None
        self.tenant_id = os.environ.get("HYDRA_TENANT_ID", "appworld_agent")
        self.max_results = max(5, min(50, int(os.environ.get("HYDRA_MAX_RESULTS", "12"))))  # API range 5..50
        self.chunk_chars = int(os.environ.get("HYDRA_CHUNK_CHARS", "1000"))

        # The agent run gates on USE_HYDRA; the offline bootstrap passes
        # force_enable=True so it only needs HYDRA_DB_API_KEY.
        enabled = _enabled() or (force_enable and bool(os.environ.get("HYDRA_DB_API_KEY")))
        if not enabled:
            return
        try:
            from hydra_db import HydraDB

            self.client = HydraDB(token=os.environ["HYDRA_DB_API_KEY"])
            self._ensure_tenant()
            print(f"  [hydra] enabled (tenant={self.tenant_id})")
        except Exception as e:  # missing pkg, bad key, network — never fatal for the run
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
                if _get(_get(st, "infra", {}), "ready_for_ingestion", True):
                    return
            except Exception:
                return  # unknown shape or network hiccup — don't block
            time.sleep(1)

    # -- B) knowledge ingestion / ensure -------------------------------------
    def api_doc_records(self, api_docs: dict) -> list:
        return _flatten_api_docs(api_docs)

    def api_doc_ingestion_status(self, api_docs: dict, batch_size: int = 100) -> dict:
        """Best-effort check for which API-doc records are already indexed.

        Returns counts plus missing ids. If HydraDB cannot expose per-id status,
        falls back to a few knowledge queries for known API names. A query-probe
        hit is enough to skip re-ingestion because API docs use stable ids.
        """
        flat = self.api_doc_records(api_docs)
        ids = [d["id"] for d in flat]
        result = {
            "expected": len(ids),
            "indexed": 0,
            "pending": 0,
            "failed": 0,
            "missing": ids[:],
            "unknown": not self.on,
            "source": "disabled" if not self.on else "status",
            "probe_hits": 0,
            "probe_checked": 0,
        }
        if not self.on or not ids:
            return result

        seen = {}
        status_supported = True
        for batch in _chunked(ids, batch_size):
            try:
                st = self.client.context.status(tenant_id=self.tenant_id, ids=batch)
            except Exception:
                status_supported = False
                break
            seen.update(_status_by_id(st))
        if status_supported and seen:
            _FAIL = {"failed", "failure", "error", "errored"}
            indexed = {doc_id for doc_id, status in seen.items() if _looks_indexed(status)}
            failed = {doc_id for doc_id, status in seen.items() if str(status).lower() in _FAIL}
            pending = set(seen) - indexed - failed
            missing = [doc_id for doc_id in ids if doc_id not in indexed and doc_id not in pending]
            result.update({
                "indexed": len(indexed),
                "pending": len(pending),
                "failed": len(failed),
                "missing": missing,
                "unknown": False,
                "source": "status",
            })
            return result

        probe = self._api_doc_query_probe(flat)
        result.update({
            "probe_hits": probe["hits"],
            "probe_checked": probe["checked"],
        })
        if probe["ok"]:
            result.update({
                "indexed": len(ids),
                "pending": 0,
                "failed": 0,
                "missing": [],
                "unknown": False,
                "source": "query_probe",
            })
        else:
            result.update({
                "unknown": False,
                "source": "query_probe",
            })
        return result

    def _api_doc_query_probe(self, flat: list) -> dict:
        if not self.on or not flat:
            return {"ok": False, "hits": 0, "checked": 0}
        by_app = {}
        for record in flat:
            by_app.setdefault(record["app"], record)
        picks = list(by_app.values())
        hits = 0
        for d in picks:
            needle = f"{d['app']}.{d['api']}".lower()
            try:
                res = self.client.query(
                    tenant_id=self.tenant_id,
                    query=needle,
                    type="knowledge",
                    query_by="hybrid",
                    mode="thinking",
                    max_results=5,
                    graph_context=False,
                )
            except Exception:
                return {"ok": False, "hits": hits, "checked": len(picks)}
            chunks = _get(_unwrap(res), "chunks", []) or []
            haystack = " ".join(
                str(_get(c, "chunk_content") or _get(c, "content") or "")
                + " "
                + str(_get(c, "source_title") or _get(c, "source_type") or "")
                for c in chunks
            ).lower()
            if needle in haystack or d["id"].lower() in haystack:
                hits += 1
        return {"ok": hits == len(picks), "hits": hits, "checked": len(picks)}

    def ensure_api_docs(self, api_docs: dict, batch_size: int = 50,
                        ingest_on_probe_miss: bool = True) -> tuple:
        """Ensure API docs are present, ingesting only records missing from HydraDB."""
        if not self.on:
            return 0, []
        flat = self.api_doc_records(api_docs)
        if not flat:
            return 0, []
        status = self.api_doc_ingestion_status(api_docs)
        expected = status["expected"]
        if expected and not status["missing"] and not status["failed"]:
            if status["source"] == "query_probe":
                print("  [hydra] API docs appear indexed "
                      f"(query probe {status['probe_hits']}/{status['probe_checked']}); skipping ingest")
            else:
                print(f"  [hydra] API docs already indexed ({expected}/{expected}, via {status['source']}); skipping ingest")
            return 0, [d["id"] for d in flat]
        if status["source"] == "query_probe" and not ingest_on_probe_miss:
            print("  [hydra] API docs could not be verified by query probe; skipping auto-ingest")
            print("  [hydra] run `python bootstrap_docs.py` explicitly if you want to repair/reseed knowledge")
            return 0, []
        if status["failed"]:
            print(f"  [hydra] {status['failed']} API docs previously failed indexing; re-submitting missing/failed docs")
        missing = set(status["missing"])
        to_ingest = [d for d in flat if d["id"] in missing] if missing else flat
        if not to_ingest:
            return 0, [d["id"] for d in flat]
        count, ids = self._ingest_api_doc_records(to_ingest, batch_size=batch_size)
        return count, ids

    def ingest_api_docs(self, api_docs: dict, batch_size: int = 50) -> tuple:
        """Ingest the API-doc artifact as one knowledge document per API.

        Returns (count, ids). RAISES on a hard failure so the bootstrap can report
        it — this is setup, not the run, so it must NOT silently no-op. Uses
        HydraDB's documented documents+document_metadata path; stable ids make a
        re-ingest an idempotent upsert.
        """
        if not self.on:
            raise RuntimeError("HydraDB not initialized; cannot ingest (need HYDRA_DB_API_KEY)")
        flat = _flatten_api_docs(api_docs)
        if not flat:
            return 0, []
        return self._ingest_api_doc_records(flat, batch_size=batch_size)

    def _ingest_api_doc_records(self, records: list, batch_size: int = 50) -> tuple:
        for batch in _chunked(records, batch_size):
            documents, document_metadata = [], []
            for d in batch:
                body = d["text"].encode("utf-8")
                documents.append((f"{d['id']}.txt", io.BytesIO(body), "text/plain"))
                document_metadata.append({
                    "id": d["id"],
                    "metadata": {"kind": "api_doc", "app": d["app"], "api": d["api"]},
                })
            self.client.context.ingest(
                type="knowledge",
                tenant_id=self.tenant_id,
                documents=documents,
                document_metadata=json.dumps(document_metadata),
            )
        return len(records), [d["id"] for d in records]

    def wait_until_indexed(self, ids, timeout: int = 900, poll: int = 5, sample: int = 8) -> bool:
        """Poll context.status on a sample of ids until all 'completed' (or timeout).

        Defensive: if the status shape can't be read, returns True after a short
        grace rather than blocking forever — the run degrades safely regardless.
        """
        if not self.on or not ids:
            return True
        sample_ids = list(ids)[:sample]
        waited, unknown = 0, 0
        while waited < timeout:
            try:
                st = self.client.context.status(tenant_id=self.tenant_id, ids=sample_ids)
                statuses = _status_values(st)
            except Exception as e:
                print(f"  [hydra] status check failed: {e}")
                return True  # don't block the operator; indexing usually still completes
            if statuses:
                if all(_looks_indexed(s) for s in statuses):
                    return True
                if any(s in {"failed", "failure", "error", "errored"} for s in statuses):
                    print(f"  [hydra] indexing reported failure: {statuses}")
                    return False
                unknown = 0
            else:
                unknown += 1
                if unknown >= 3:
                    print("  [hydra] could not read indexing status; assuming in-progress")
                    return True
            time.sleep(poll)
            waited += poll
        return False

    # -- A) episodic memory ---------------------------------------------------
    def remember_task(self, instruction: str, messages: list, success: bool) -> None:
        """Ingest the task transcript as an episodic memory.

        Reads the loop's existing `messages` list (skipping the seed) so nothing
        is captured inside the reasoning loop. messages[0] is the seed (which may
        contain prior recall) and is skipped so retrieved context is never folded
        back into a new memory.
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
                    "infer": False,
                    "metadata": {"kind": "episode", "success": "true" if success else "false"},
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

    # -- retrieval ------------------------------------------------------------
    def recall(self, instruction: str, kind: str = "all") -> str:
        """Query memory + knowledge for this task; return a prompt-ready string.

        `kind` targets the store: "memory" (past episodes), "knowledge" (API
        docs), or "all" (both). The Planner recalls "memory" per task; the
        Executor recalls "knowledge" per subgoal. Returns "" when disabled, on
        error, or when nothing relevant is found — callers handle the empty case.
        """
        if not self.on:
            return ""
        try:
            res = self.client.query(
                tenant_id=self.tenant_id,
                query=instruction,
                type=kind,             # "all" | "memory" | "knowledge"
                query_by="hybrid",     # semantic + BM25
                mode="thinking",       # multi-pass expansion
                max_results=self.max_results,
                graph_context=True,    # pull related APIs (e.g. the login a call needs)
            )
            chunks = _get(_unwrap(res), "chunks", []) or []
            lines = []
            for c in chunks:
                text = _get(c, "chunk_content") or _get(c, "content") or ""
                title = _get(c, "source_title") or _get(c, "source_type") or ""
                text = str(text).strip()
                if text:
                    if len(text) > self.chunk_chars:  # keep injected context bounded
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
