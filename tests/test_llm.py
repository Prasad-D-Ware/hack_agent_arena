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


def test_switches_to_fallback_after_first_error(monkeypatch):
    monkeypatch.setenv("MODEL", "primary-model")
    monkeypatch.setenv("MODEL_FALLBACK", "fallback-model")
    c = FakeChatClient(fail_times=1, content="ok")
    out = llm.call_llm("executor", [{"role": "user", "content": "x"}],
                       client=c, sleep_fn=lambda s: None)
    assert out == "ok"
    assert c.last_kwargs["model"] == "fallback-model"


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


def test_groq_provider_default_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("MODEL", raising=False)
    c = FakeChatClient(content="ok")
    llm.call_llm("planner", [{"role": "user", "content": "x"}],
                 client=c, sleep_fn=lambda s: None)
    assert c.last_kwargs["model"] == "llama-3.3-70b-versatile"


def test_groq_provider_client_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_DEFAULT_CLIENT", None)
    monkeypatch.setattr(llm, "_DEFAULT_CLIENT_CONFIG", None)

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    llm._client()

    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["api_key"] == "test-key"
