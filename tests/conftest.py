"""Shared test doubles for the agent architecture."""
from types import SimpleNamespace


class FakeLLM:
    """Callable matching call_llm(role, messages, system=None, **kw).

    `responses` is either a list (consumed in order, any role) or a dict
    role -> list (consumed per role). Records every call for assertions.
    """

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, role, messages, system=None, **kw):
        self.calls.append((role, messages, system))
        q = self.responses.get(role) if isinstance(self.responses, dict) else self.responses
        if isinstance(q, list):
            return q.pop(0)
        return q


class FakeWorld:
    """world.execute(code) returns scripted outputs in order; any code
    containing 'complete_task' marks the task completed."""

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.executed = []
        self._completed = False

    def execute(self, code):
        self.executed.append(code)
        if "complete_task" in code:
            self._completed = True
            return "Task marked complete."
        return self.outputs.pop(0) if self.outputs else "ok"

    def task_completed(self):
        return self._completed


class FakeMem:
    """Stand-in for HydraMemory."""

    def __init__(self, recall_value=""):
        self.recall_value = recall_value
        self.remembered = []
        self.ingested = 0

    def recall(self, query, kind="all"):
        return self.recall_value

    def ingest_api_docs(self, world):
        self.ingested += 1

    def remember_episode(self, instruction, state, success):
        self.remembered.append((instruction, success))


def make_chat_resp(content):
    """Build an object shaped like an OpenAI chat completion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
