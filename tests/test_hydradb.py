from hydradb import HydraMemory
from state import Blackboard, Subgoal


class FakeCtx:
    def __init__(self):
        self.ingests = []
        self.statuses = {}

    def ingest(self, **kw):
        self.ingests.append(kw)

    def status(self, **kw):
        return {
            "data": {
                "statuses": [
                    {"id": doc_id, "indexing_status": self.statuses[doc_id]}
                    for doc_id in kw.get("ids", [])
                    if doc_id in self.statuses
                ]
            }
        }


class FakeHydraClient:
    def __init__(self):
        self.queries = []
        self.context = FakeCtx()

    def query(self, **kw):
        self.queries.append(kw)
        return {"data": {"chunks": [{"chunk_content": "doc", "source_title": "spotify"}]}}


def test_recall_disabled_returns_empty():
    mem = HydraMemory()           # USE_HYDRA not set -> client is None
    assert mem.recall("anything") == ""
    assert mem.recall("anything", kind="memory") == ""


def test_recall_routes_kind_to_query_type():
    mem = HydraMemory()
    mem.client = FakeHydraClient()
    out = mem.recall("find a song", kind="knowledge")
    assert mem.client.queries[0]["type"] == "knowledge"
    assert "doc" in out


def test_remember_episode_ingests_memory_type():
    mem = HydraMemory()
    mem.client = FakeHydraClient()
    bb = Blackboard(task_instruction="play song")
    bb.plan = [Subgoal(1, "log in", status="done", result="token")]
    bb.add_step(1, "print(login())", "token=abc")
    mem.remember_episode("play song", bb, success=True)
    kw = mem.client.context.ingests[0]
    assert kw["type"] == "memory"
    assert "play song" in kw["memories"]


def test_remember_episode_disabled_is_noop():
    mem = HydraMemory()           # client is None
    bb = Blackboard(task_instruction="t")
    mem.remember_episode("t", bb, success=False)   # must not raise


def _sample_api_docs():
    return {
        "apps": {
            "spotify": {
                "apis": {
                    "login": {"description": "Log in", "parameters": []},
                    "play_song": {"description": "Play a song", "parameters": []},
                }
            }
        }
    }


def test_ensure_api_docs_skips_when_already_indexed():
    mem = HydraMemory()
    mem.client = FakeHydraClient()
    mem.client.context.statuses = {
        "apidoc_spotify_login": "completed",
        "apidoc_spotify_play_song": "completed",
    }
    count, ids = mem.ensure_api_docs(_sample_api_docs())
    assert count == 0
    assert ids == ["apidoc_spotify_login", "apidoc_spotify_play_song"]
    assert mem.client.context.ingests == []


def test_ensure_api_docs_ingests_only_missing_records():
    mem = HydraMemory()
    mem.client = FakeHydraClient()
    mem.client.context.statuses = {
        "apidoc_spotify_login": "completed",
    }
    count, ids = mem.ensure_api_docs(_sample_api_docs())
    assert count == 1
    assert ids == ["apidoc_spotify_play_song"]
    assert len(mem.client.context.ingests) == 1
    ingest = mem.client.context.ingests[0]
    assert ingest["type"] == "knowledge"
    assert len(ingest["documents"]) == 1
    assert ingest["documents"][0][0] == "apidoc_spotify_play_song.txt"
