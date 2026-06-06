from executor import run
from state import Blackboard, Subgoal
from tests.conftest import FakeLLM, FakeWorld, FakeMem


def test_executor_completes_on_done_marker():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "log in")
    llm = FakeLLM({"executor": [
        "```python\nprint('hi')\n```",
        "SUBGOAL_DONE: logged in",
    ]})
    world = FakeWorld(outputs=["hi"])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "done" and result == "logged in"
    assert bb.interactions_used == 1          # one code execution before DONE


def test_executor_injects_retrieved_knowledge():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "search")
    llm = FakeLLM({"executor": ["```python\nprint('evidence')\n```", "SUBGOAL_DONE: ok"]})
    mem = FakeMem(recall_value="API: spotify.search(query)")
    run(sg, bb, FakeWorld(outputs=["evidence"]), mem, llm=llm,
        max_steps=3, interaction_budget=40)
    assert "spotify.search" in llm.calls[0][1][0]["content"]


def test_executor_captures_login_token_for_later_subgoals():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "log into amazon")
    llm = FakeLLM({"executor": [
        "```python\nprint(apis.amazon.login(username='u', password='p'))\n```",
        "SUBGOAL_DONE: logged in",
    ]})
    world = FakeWorld(outputs=['{"access_token": "tok123", "token_type": "Bearer"}'])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "done"
    assert result == "logged in"
    assert bb.credentials["amazon"] == "tok123"


def test_executor_prompts_for_api_docs_after_api_failure():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "retrieve orders")
    llm = FakeLLM({"executor": [
        "```python\nprint(apis.amazon.show_orders(sort_by='-created_at'))\n```",
        "```python\nprint(apis.api_docs.show_api_doc(app_name='amazon', api_name='show_orders'))\n```",
        "SUBGOAL_DONE: inspected docs",
    ]})
    world = FakeWorld(outputs=[
        'Execution failed. Traceback:\nException: Response status code is 422: bad',
        '{"api_name": "show_orders"}',
    ])
    run(sg, bb, world, FakeMem(), llm=llm, max_steps=5, interaction_budget=40)
    second_call_messages = llm.calls[1][1]
    assert any("inspect its documentation" in m["content"] for m in second_call_messages)


def test_executor_fails_on_repeated_error():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": [
        "```python\nbad\n```",
        "```python\nbad\n```",
    ]})
    world = FakeWorld(outputs=["KeyError: 'x'", "KeyError: 'x'"])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "failed" and "KeyError" in result


def test_executor_stops_at_global_budget():
    bb = Blackboard(task_instruction="t")
    bb.interactions_used = 40
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": ["```python\nx\n```"]})
    status, _ = run(sg, bb, FakeWorld(), FakeMem(), llm=llm,
                    max_steps=5, interaction_budget=40)
    assert status == "failed"


def test_executor_does_not_execute_empty_code():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    llm = FakeLLM({"executor": ["```python\n\n```", "SUBGOAL_DONE: ok", "SUBGOAL_DONE: ok"]})
    world = FakeWorld()
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=3, interaction_budget=40)
    assert status == "failed"
    assert "fresh successful evidence" in result
    assert world.executed == []
    assert bb.interactions_used == 0


def test_executor_reprompts_done_without_fresh_evidence():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "retrieve rows")
    llm = FakeLLM({"executor": [
        "SUBGOAL_DONE: Retrieved 50 rows",
        "```python\nprint('[{\"id\": 1}]')\n```",
        "SUBGOAL_DONE: Retrieved 1 row",
    ]})
    world = FakeWorld(outputs=['[{"id": 1}]'])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "done"
    assert result == "Retrieved 1 row"
    assert world.executed == ["print('[{\"id\": 1}]')"]


def test_executor_fails_repeated_done_without_fresh_evidence():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "retrieve rows")
    llm = FakeLLM({"executor": [
        "SUBGOAL_DONE: Retrieved 50 rows",
        "SUBGOAL_DONE: Retrieved 50 rows",
    ]})
    status, result = run(sg, bb, FakeWorld(), FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "failed"
    assert "fresh successful evidence" in result


def test_executor_fails_after_consecutive_empty_replies():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    # two empty replies in a row → fail fast, don't burn all 10 steps
    llm = FakeLLM({"executor": ["<think>hmm</think>", "<think>still thinking</think>"]})
    world = FakeWorld()
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=10, interaction_budget=40)
    assert status == "failed"
    assert "no executable" in result
    assert world.executed == []


def test_executor_reprompts_truncated_code_then_fails():
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "do")
    truncated = '```python\nresult = apis.spotify.show_song(access_token="eyJhbGci\n```'
    llm = FakeLLM({"executor": [truncated, truncated]})
    world = FakeWorld()
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=10, interaction_budget=40)
    assert status == "failed"
    assert "truncated" in result
    assert world.executed == []


# --- Regression tests for run-log failure modes ---

def test_executor_exits_early_on_repeated_http_409():
    # Regression: before the fix, HTTP 409 errors had sig=None so has_repeated_error
    # never fired and the executor burned all 10 steps retrying stale draft IDs.
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "delete empty drafts")
    out_409 = 'Response status code is 409: {"message":"The draft with id 27 does not exist."}'
    llm = FakeLLM({"executor": [
        "```python\napis.gmail.delete_draft(draft_id=27, access_token='tok')\n```",
        "```python\napis.gmail.delete_draft(draft_id=85, access_token='tok')\n```",
        "```python\napis.gmail.delete_draft(draft_id=209, access_token='tok')\n```",
    ]})
    world = FakeWorld(outputs=[out_409, out_409, out_409])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=10, interaction_budget=40)
    assert status == "failed"
    # Must exit after 2 identical HTTP 409s, not run all 10 steps
    assert bb.interactions_used == 2
    assert "HTTP 409" in result


def test_executor_http_error_does_not_count_as_evidence():
    # Regression: HTTP errors previously had sig=None which caused them to set
    # has_successful_evidence=True, letting the executor claim SUBGOAL_DONE.
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "delete draft")
    out_409 = 'Response status code is 409: {"message":"The draft with id 27 does not exist."}'
    llm = FakeLLM({"executor": [
        "```python\napis.gmail.delete_draft(draft_id=27, access_token='tok')\n```",
        # Now claims done — must be blocked because 409 is not evidence of success
        "SUBGOAL_DONE: Deleted all empty drafts",
        "SUBGOAL_DONE: Deleted all empty drafts",
    ]})
    world = FakeWorld(outputs=[out_409])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "failed"
    assert "fresh successful evidence" in result


def test_executor_evicts_stale_token_on_401():
    # Regression: 401 responses must evict the cached token so the executor
    # re-authenticates rather than retrying with an expired token.
    bb = Blackboard(task_instruction="t")
    bb.credentials["gmail"] = "expired-token"
    sg = Subgoal(1, "delete draft")
    out_401 = 'Response status code is 401: {"message":"Your access token is missing, invalid or expired."}'
    llm = FakeLLM({"executor": [
        "```python\napis.gmail.delete_draft(draft_id=27, access_token='expired-token')\n```",
        "SUBGOAL_DONE: done",
        "SUBGOAL_DONE: done",
    ]})
    world = FakeWorld(outputs=[out_401])
    run(sg, bb, world, FakeMem(), llm=llm, max_steps=5, interaction_budget=40)
    # Token must have been evicted after the 401
    assert "gmail" not in bb.credentials


def test_executor_rejects_unevaluated_fstring_in_done_result():
    # Regression: model wrote SUBGOAL_DONE: Retrieved {len(all_drafts)} emails
    # instead of evaluating the expression. Must be caught and reprompted.
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "retrieve drafts")
    llm = FakeLLM({"executor": [
        "```python\nprint(len(drafts))\n```",
        "SUBGOAL_DONE: Retrieved {len(all_drafts)} draft emails in {page_index} pages",
        "```python\nprint(17)\n```",
        "SUBGOAL_DONE: Retrieved 17 draft emails",
    ]})
    world = FakeWorld(outputs=["17", "17"])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=6, interaction_budget=40)
    assert status == "done"
    assert result == "Retrieved 17 draft emails"
    # The unevaluated f-string reply must have triggered a reprompt
    reprompt_msgs = llm.calls[2][1]
    assert any("unevaluated" in m["content"].lower() or "f-string" in m["content"].lower()
               or "actual value" in m["content"].lower()
               for m in reprompt_msgs)


def test_executor_warns_on_pagination_returning_zero():
    # Regression: pagination loop with missing page_limit kwarg returns 0 items.
    # Executor must be told the loop is broken, not that the list is empty.
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "retrieve all drafts")
    llm = FakeLLM({"executor": [
        "```python\nprint('Total count: 0, Page count: 0')\n```",
        "SUBGOAL_DONE: done",
        "SUBGOAL_DONE: done",
    ]})
    world = FakeWorld(outputs=["Total count: 0, Page count: 0"])
    run(sg, bb, world, FakeMem(), llm=llm, max_steps=5, interaction_budget=40)
    second_call_messages = llm.calls[1][1]
    assert any("page_limit" in m["content"].lower() or "pagination" in m["content"].lower()
               for m in second_call_messages)


def test_executor_retry_seed_warns_about_fresh_environment():
    # Regression: on subgoal retry, executor tried to reuse all_drafts variable
    # from prior attempt, which doesn't exist in the fresh Python env.
    # The retry seed must warn that variables must be re-fetched.
    from state import Blackboard, Subgoal
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "delete drafts")
    sg.last_feedback = "Deleted count mismatch"
    llm = FakeLLM({"executor": [
        "```python\nprint('ok')\n```",
        "SUBGOAL_DONE: done",
    ]})
    world = FakeWorld(outputs=["ok"])
    run(sg, bb, world, FakeMem(), llm=llm, max_steps=5, interaction_budget=40)
    seed = llm.calls[0][1][0]["content"]
    assert "fresh Python environment" in seed or "Re-fetch" in seed


def test_executor_api_failure_reprompt_mentions_fetch_fresh_ids():
    # Regression: when a 409 "does not exist" error fires, the reprompt must
    # tell the executor to fetch the current list before retrying.
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "delete draft")
    out_409 = 'Response status code is 409: {"message":"The draft with id 27 does not exist."}'
    llm = FakeLLM({"executor": [
        "```python\napis.gmail.delete_draft(draft_id=27, access_token='tok')\n```",
        "SUBGOAL_DONE: done",
        "SUBGOAL_DONE: done",
    ]})
    world = FakeWorld(outputs=[out_409])
    run(sg, bb, world, FakeMem(), llm=llm, max_steps=5, interaction_budget=40)
    # Second LLM call must include guidance to fetch current IDs
    second_call_messages = llm.calls[1][1]
    assert any("fetch" in m["content"].lower() or "current list" in m["content"].lower()
               for m in second_call_messages)


def test_executor_rejects_password_as_token_pattern():
    # Regression: executor wrote `tok = next(p["password"] for p in ...)` using
    # the raw password string as an access token instead of calling login first.
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "fetch orders")
    # Code assigns p["password"] directly to tok — should be intercepted
    bad_code = (
        '```python\n'
        'passwords = apis.supervisor.show_account_passwords()\n'
        'tok = next(p["password"] for p in passwords if p["account_name"] == "amazon")\n'
        'print(apis.amazon.show_orders(access_token=tok))\n'
        '```'
    )
    llm = FakeLLM({"executor": [
        bad_code,
        "```python\nprint(apis.amazon.login(username='u', password='pw'))\n```",
        "SUBGOAL_DONE: done",
    ]})
    world = FakeWorld(outputs=['{"access_token": "real-tok"}'])
    run(sg, bb, world, FakeMem(), llm=llm, max_steps=5, interaction_budget=40)
    # The bad code must never have been executed
    assert world.executed == ["print(apis.amazon.login(username='u', password='pw'))"]
    # The reprompt must have mentioned the mistake
    second_call_messages = llm.calls[1][1]
    assert any("password" in m["content"].lower() and "access_token" in m["content"].lower()
               for m in second_call_messages)


def test_executor_captures_token_from_mid_subgoal_login():
    # Regression: secondary app login happening mid-subgoal (not as a dedicated
    # login subgoal) must still persist the token to state.credentials for
    # subsequent subgoals to reuse without re-authenticating.
    from state import Blackboard, Subgoal
    bb = Blackboard(task_instruction="t")
    sg = Subgoal(1, "log into splitwise and list expenses")
    login_output = '{"access_token": "sw-tok-xyz", "token_type": "Bearer"}'
    llm = FakeLLM({"executor": [
        "```python\nprint(apis.splitwise.login(username='u', password='p'))\n```",
        "SUBGOAL_DONE: logged in to splitwise",
    ]})
    world = FakeWorld(outputs=[login_output])
    status, result = run(sg, bb, world, FakeMem(), llm=llm,
                         max_steps=5, interaction_budget=40)
    assert status == "done"
    assert bb.credentials.get("splitwise") == "sw-tok-xyz"
