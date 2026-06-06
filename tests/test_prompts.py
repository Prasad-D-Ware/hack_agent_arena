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
    # Concrete template must be present so the model can follow by example
    assert "page_index=page" in prompt
    assert "page_limit=PAGE_LIMIT" in prompt
    assert "len(batch) < PAGE_LIMIT" in prompt
    assert "show_app_descriptions" in prompt   # unknown-app discovery rule


def test_verifier_prompt_demands_pass_fail():
    assert "PASS" in prompts.VERIFIER_SYSTEM and "FAIL" in prompts.VERIFIER_SYSTEM


def test_executor_prompt_login_requires_keyword_args():
    prompt = prompts.EXECUTOR_SYSTEM
    # Must show login with both username= and password= as keyword args
    assert 'login(username=' in prompt
    assert 'password=password' in prompt
    # Must warn that p["password"] is never the token
    assert 'p["password"]' in prompt
    assert '["access_token"]' in prompt


def test_executor_prompt_api_names_require_lookup():
    prompt = prompts.EXECUTOR_SYSTEM
    # Must warn that API names must be looked up, not guessed
    assert "show_api_descriptions" in prompt
    assert "app_name=" in prompt


def test_executor_prompt_422_guidance():
    prompt = prompts.EXECUTOR_SYSTEM
    # Must explain that 422 means wrong param name/type/structure
    assert "422" in prompt


def test_executor_prompt_no_repaginate_on_retry():
    prompt = prompts.EXECUTOR_SYSTEM
    # Must warn not to re-run the full pagination loop on every retry
    assert "re-paginate" in prompt or "once" in prompt.lower()
