import llm
from tests.conftest import make_chat_resp


class FakeChatClient:
    def __init__(self, fail_times=0, content="hi"):
        self.fail_times = fail_times
        self.content = content
        self.calls = 0
        self.last_kwargs = None

        class _Completions:
            def create(_inner, **kw):
                self.calls += 1
                if self.calls <= self.fail_times:
                    raise RuntimeError("transient 503")
                self.last_kwargs = kw
                return make_chat_resp(self.content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_returns_content_on_success():
    c = FakeChatClient(content="hello")
    out = llm.call_llm("planner", [{"role": "user", "content": "x"}],
                       client=c, sleep_fn=lambda s: None)
    assert out == "hello"
    assert c.calls == 1


def test_retries_then_succeeds():
    c = FakeChatClient(fail_times=2, content="ok")
    out = llm.call_llm("executor", [{"role": "user", "content": "x"}],
                       client=c, sleep_fn=lambda s: None)
    assert out == "ok"
    assert c.calls == 3


def test_per_role_model_routing(monkeypatch):
    monkeypatch.setenv("MODEL_PLANNER", "test/planner-model")
    c = FakeChatClient(content="ok")
    llm.call_llm("planner", [{"role": "user", "content": "x"}],
                 client=c, sleep_fn=lambda s: None)
    assert c.last_kwargs["model"] == "test/planner-model"


def test_system_prompt_prepended():
    c = FakeChatClient(content="ok")
    llm.call_llm("executor", [{"role": "user", "content": "x"}],
                 system="SYS", client=c, sleep_fn=lambda s: None)
    assert c.last_kwargs["messages"][0] == {"role": "system", "content": "SYS"}
