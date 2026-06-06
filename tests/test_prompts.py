import prompts


def test_prompts_exist_and_are_nonempty():
    for name in ("PLANNER_SYSTEM", "EXECUTOR_SYSTEM", "VERIFIER_SYSTEM"):
        val = getattr(prompts, name)
        assert isinstance(val, str) and len(val) > 50


def test_executor_prompt_states_one_block_rule():
    assert "one" in prompts.EXECUTOR_SYSTEM.lower()
    assert "complete_task" in prompts.EXECUTOR_SYSTEM


def test_executor_prompt_uses_case_insensitive_app_credential_lookup():
    prompt = prompts.EXECUTOR_SYSTEM.lower()
    assert "case-insensitive app-name lookup" in prompt
    assert 'p["account_name"].lower()' in prompts.EXECUTOR_SYSTEM
    assert "never match on the user's\n  email address" in prompts.EXECUTOR_SYSTEM


def test_executor_prompt_reuses_tokens_and_requires_docs_after_failures():
    prompt = prompts.EXECUTOR_SYSTEM
    assert "TOKENS" in prompt
    assert "do not copy from old outputs, truncate it, or log in\n  again" in prompt
    assert "After any API failure" in prompt
    assert "show_api_doc" in prompt


def test_executor_prompt_requires_complete_pagination():
    prompt = prompts.EXECUTOR_SYSTEM
    assert "handle pagination completely" in prompt
    assert "Do not mark a\n  retrieval subgoal done after one page" in prompt
    assert "page count, total item count, and stopping condition" in prompt


def test_verifier_prompt_demands_pass_fail():
    assert "PASS" in prompts.VERIFIER_SYSTEM and "FAIL" in prompts.VERIFIER_SYSTEM
