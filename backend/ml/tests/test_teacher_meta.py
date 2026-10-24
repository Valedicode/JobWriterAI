"""teacher_meta tells a frozen teacher from a trial, and two teachers apart.

Run from backend/:  python -m pytest ml/tests
"""
from ml.labeling.teacher_meta import differences, unfrozen

FROZEN = {"tag": "teacher-v1", "commit": "abc", "dirty": False, "route": "openrouter", "providers": ["deepseek"],
          "models": {"extract": "deepseek/deepseek-v4.1-flash-1017", "judge": "deepseek/deepseek-v4.1-flash-1017"},
          "embedding_route": "openai", "embedding_model": "text-embedding-3-small", "reasoning": False,
          "dataset": {"id": "d", "revision": "r", "split": "train"}, "started": "2026-10-24T10:00:00+00:00"}


def test_a_clean_pinned_tagged_checkout_is_frozen():
    assert unfrozen(FROZEN) == []


def test_each_reason_for_a_trial_is_named():
    meta = {**FROZEN, "tag": None, "dirty": True, "providers": [],
            "models": {"extract": "deepseek/deepseek-v4.1-flash", "judge": "deepseek/deepseek-v4.1-flash-1017"}}
    assert unfrozen(meta) == [
        "HEAD is not on a teacher-* tag",
        "backend/app or backend/ml has uncommitted changes",
        "LLM_PROVIDER is not set, so OpenRouter may switch hosts",
        "deepseek/deepseek-v4.1-flash is not a versioned model id",
    ]


def test_a_direct_openai_route_needs_no_provider_pin():
    assert unfrozen({**FROZEN, "route": "openai", "providers": [],
                     "models": {"extract": "gpt-6-luna-0901", "judge": "gpt-6-luna-0901"}}) == []


def test_only_teacher_defining_fields_count_as_a_different_teacher():
    later_run = {**FROZEN, "started": "2026-10-25T08:00:00+00:00", "packages": {"openai": "9"}}
    assert differences(FROZEN, later_run) == []
    judge = {**FROZEN["models"], "judge": "mistralai/mistral-large-4-0"}
    assert differences(FROZEN, {**FROZEN, "models": judge}) == ["models"]
    assert differences(FROZEN, {**FROZEN, "dataset": {**FROZEN["dataset"], "split": "test"}}) == ["dataset"]
