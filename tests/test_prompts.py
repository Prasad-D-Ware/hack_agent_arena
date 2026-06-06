import prompts


def test_prompts_exist_and_are_nonempty():
    for name in ("PLANNER_SYSTEM", "EXECUTOR_SYSTEM", "VERIFIER_SYSTEM"):
        val = getattr(prompts, name)
        assert isinstance(val, str) and len(val) > 50


def test_executor_prompt_states_one_block_rule():
    assert "one" in prompts.EXECUTOR_SYSTEM.lower()
    assert "complete_task" in prompts.EXECUTOR_SYSTEM


def test_verifier_prompt_demands_pass_fail():
    assert "PASS" in prompts.VERIFIER_SYSTEM and "FAIL" in prompts.VERIFIER_SYSTEM
