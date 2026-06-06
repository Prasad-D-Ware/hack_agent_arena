"""Budget/limit constants for the agent loop (override via env)."""
import os


def _int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


MAX_INTERACTIONS = _int("MAX_INTERACTIONS", "40")     # global LLM-execute turns per task
MAX_SUBGOAL_STEPS = _int("MAX_SUBGOAL_STEPS", "10")   # executor turns per subgoal
MAX_REPLANS = _int("MAX_REPLANS", "2")                # planner replans per task
MAX_SUBGOAL_RETRIES = _int("MAX_SUBGOAL_RETRIES", "1")  # re-run a failed subgoal before replanning
MAX_FINALIZE = _int("MAX_FINALIZE", "3")              # complete_task attempts
