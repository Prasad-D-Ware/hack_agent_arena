#!/usr/bin/env python3
"""
🐉 Regenerate the AppWorld API-doc snapshot at assets/api_docs.json.

Opens ONE AppWorld sandbox purely to reach `apis.api_docs`, enumerates every app
and every API, captures each API's full doc (signature + params + response
schemas), and writes a single committed artifact. Run this only when you need to
refresh the snapshot (e.g. after an AppWorld version bump) — it is upstream of,
and independent from, both the agent run and the HydraDB bootstrap.

Requires AppWorld installed with data downloaded (see setup.sh). No HydraDB.

Usage:
  python snapshot_api_docs.py [output_path]      # default: assets/api_docs.json
"""

import json
import os
import sys

import appworld as aw
from appworld import AppWorld, load_task_ids

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "assets/api_docs.json"


def main() -> int:
    task_id = load_task_ids("dev")[0]  # any task; we only need the sandbox's `apis`
    with AppWorld(task_id=task_id, experiment_name="snapshot_api_docs") as world:
        app_descriptions = json.loads(world.execute(
            "import json; print(json.dumps(apis.api_docs.show_app_descriptions()))"
        ).strip())

        app_catalog = {}
        for entry in app_descriptions:
            app = entry["name"]
            # Enumerate this app's APIs and fetch each full doc INSIDE the sandbox,
            # one app at a time so no single execute() output gets too large.
            code = (
                "import json\n"
                "out = {}\n"
                f"for d in apis.api_docs.show_api_descriptions(app_name={app!r}):\n"
                "    n = d['name']\n"
                "    try:\n"
                f"        out[n] = apis.api_docs.show_api_doc(app_name={app!r}, api_name=n)\n"
                "    except Exception as e:\n"
                "        out[n] = {'error': str(e)}\n"
                "print(json.dumps(out))"
            )
            api_docs = json.loads(world.execute(code).strip())
            app_catalog[app] = {
                "description": entry.get("description", ""),
                "api_count": len(api_docs),
                "apis": api_docs,
            }
            print(f"  {app}: {len(api_docs)} apis")

    app_count = len(app_catalog)
    api_count = sum(v["api_count"] for v in app_catalog.values())
    artifact = {
        "meta": {
            "source": "AppWorld api_docs (show_app_descriptions / show_api_descriptions / show_api_doc)",
            "appworld_version": getattr(aw, "__version__", "unknown"),
            "app_count": app_count,
            "api_count": api_count,
        },
        "apps": app_catalog,
    }

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"\nWROTE {OUT_PATH}  ->  {app_count} apps, {api_count} apis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
