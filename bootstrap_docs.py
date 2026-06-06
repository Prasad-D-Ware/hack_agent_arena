#!/usr/bin/env python3
"""
🐉 One-time, OFFLINE HydraDB knowledge bootstrap.

Reads the committed AppWorld API-doc snapshot (assets/api_docs.json) and ingests
all 457 APIs into HydraDB as one knowledge document per API, then waits until
they are indexed. This is SETUP — it runs once, before any agent run, and never
touches agent.py or the reasoning loop. At run time the agent only QUERIES.

It is idempotent (stable per-API ids upsert), so re-running is safe.

Usage:
  export HYDRA_DB_API_KEY=sk-...            # key from https://app.hydradb.com
  python bootstrap_docs.py [path/to/api_docs.json]   # default: assets/api_docs.json

Notes:
  - Only HYDRA_DB_API_KEY is required here (not USE_HYDRA — that flag gates the
    RUN). Optional: HYDRA_TENANT_ID (default appworld_agent).
  - To then USE the knowledge base during a run: export USE_HYDRA=1 and run agent.py.

Exit codes: 0 ok · 1 artifact missing/unreadable · 2 HydraDB not configured · 3 ingest failed.
"""

import json
import os
import sys

from hydradb import HydraMemory

DEFAULT_ARTIFACT = "assets/api_docs.json"


def main() -> int:
    artifact_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARTIFACT

    # 1) load the artifact
    if not os.path.exists(artifact_path):
        print(f"ERROR: artifact not found: {artifact_path}")
        print("Generate it first (dump the AppWorld API docs), then re-run.")
        return 1
    try:
        with open(artifact_path) as f:
            api_docs = json.load(f)
    except Exception as e:
        print(f"ERROR: could not read/parse {artifact_path}: {e}")
        return 1
    meta = api_docs.get("meta", {}) if isinstance(api_docs, dict) else {}

    # 2) connect to HydraDB (force_enable: only the key is needed for setup)
    mem = HydraMemory(force_enable=True)
    if not mem.on:
        print("ERROR: HydraDB is not configured. Set HYDRA_DB_API_KEY (and install "
              "the SDK: pip install \"hydradb-sdk>=2,<3\").")
        return 2

    # 3) ingest
    print(f"Ingesting {meta.get('api_count', '?')} APIs across {meta.get('app_count', '?')} "
          f"apps from {artifact_path} into tenant '{mem.tenant_id}' ...")
    try:
        count, ids = mem.ingest_api_docs(api_docs)
    except Exception as e:
        print(f"ERROR: ingest failed: {e}")
        return 3
    if count == 0:
        print("ERROR: no API docs found in the artifact (nothing ingested).")
        return 3
    print(f"  submitted {count} per-API knowledge docs (async indexing started)")

    # 4) wait for indexing so the very first agent run can already retrieve them
    print("  waiting for indexing to complete ...")
    if mem.wait_until_indexed(ids):
        print("  ✓ indexed")
    else:
        print("  ! indexing not confirmed before timeout — it may still be in "
              "progress; check the tenant or just proceed (the run degrades safely).")

    print("\nDone. Knowledge base is ready.")
    print("To use it during a run:  export USE_HYDRA=1  &&  python agent.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
