from judgekit.hashing import (
    JUDGE_CONFIG_HASH_FIELDS,
    case_content_json,
    dataset_version,
    judge_config_version_hash,
)
from judgekit.models import CaseRecord, JudgeConfigRecord, JudgeParams, JudgePricing

# Golden digest for the fixture below. Updating this is a deliberate recipe
# change, never a casual fix.
GOLDEN_HASH = "sha256:9f3a35486ad7de02730db990d26ce56a895e6a1e75ae9f8410586075cee4ebe8"

REC1 = CaseRecord(
    id="c1",
    input="What is the capital of France?",
    reference="Paris",
    human_label=1,
    metadata={"difficulty": "easy"},
)
REC2 = CaseRecord(id="c2", input="2+2?")


def test_golden_hash() -> None:
    assert dataset_version([REC1, REC2]) == GOLDEN_HASH


def test_order_insensitivity() -> None:
    assert dataset_version([REC2, REC1]) == GOLDEN_HASH


def test_omitted_vs_explicit_null_hash_identically() -> None:
    omitted = CaseRecord(id="c2", input="2+2?")
    explicit = CaseRecord(id="c2", input="2+2?", reference=None, human_label=None, metadata={})
    assert case_content_json(omitted) == case_content_json(explicit)


def test_metadata_key_order_does_not_affect_hash() -> None:
    a = CaseRecord(id="c1", input="q", metadata={"a": 1, "b": 2})
    b = CaseRecord(id="c1", input="q", metadata={"b": 2, "a": 1})
    assert case_content_json(a) == case_content_json(b)


def test_unicode_is_stable_and_distinct() -> None:
    unicode_record = CaseRecord(id="c1", input="Grüße, 世界")
    ascii_record = CaseRecord(id="c1", input="Greetings, world")

    first = dataset_version([unicode_record])
    second = dataset_version([unicode_record])
    assert first == second
    assert first != dataset_version([ascii_record])


def test_content_change_changes_hash() -> None:
    changed = CaseRecord(
        id="c1",
        input="What is the capital of France?",
        reference="Paris",
        human_label=2,
        metadata={"difficulty": "easy"},
    )
    assert dataset_version([changed, REC2]) != GOLDEN_HASH


def test_bool_and_int_values_hash_differently() -> None:
    bool_record = CaseRecord(id="c1", input="q", human_label=True, metadata={"x": True})
    int_record = CaseRecord(id="c1", input="q", human_label=1, metadata={"x": 1})
    assert dataset_version([bool_record]) != dataset_version([int_record])


def test_case_content_json_exact_serialization() -> None:
    assert case_content_json(REC2) == (
        '{"human_label":null,"id":"c2","input":"2+2?","metadata":{},"reference":null}'
    )


# Golden digest for the fixture below. Updating this is a deliberate recipe
# change, never a casual fix.
JUDGE_GOLDEN_HASH = "sha256:4775a1362e9ec8997893c096e84fdecefcf5062d9397631bf4934afc70996433"

JUDGE_REC = JudgeConfigRecord(
    id="example-judge",
    provider="anthropic",
    model="test-model",
    rubric="Rate this support reply.\n\n$input\n\nReply with JSON.",
    labels=("good", "bad"),
    params=JudgeParams(temperature=0.0, max_tokens=256, top_p=None, stop=()),
)


def test_judge_config_golden_hash() -> None:
    assert judge_config_version_hash(JUDGE_REC) == JUDGE_GOLDEN_HASH


def test_judge_config_hash_fields_frozen() -> None:
    assert JUDGE_CONFIG_HASH_FIELDS == ("id", "labels", "model", "params", "rubric")


def test_judge_config_ops_fields_do_not_affect_hash() -> None:
    without_ops = JudgeConfigRecord(
        id="example-judge",
        provider="openai-compatible",
        model="test-model",
        rubric="Rate this support reply.\n\n$input\n\nReply with JSON.",
        labels=("good", "bad"),
        params=JudgeParams(temperature=0.0, max_tokens=256, top_p=None, stop=()),
        base_url=None,
        api_key_env=None,
        pricing=None,
        timeout_s=60.0,
        max_retries=3,
        max_label_attempts=3,
    )
    with_ops = JudgeConfigRecord(
        id="example-judge",
        provider="openai-compatible",
        model="test-model",
        rubric="Rate this support reply.\n\n$input\n\nReply with JSON.",
        labels=("good", "bad"),
        params=JudgeParams(temperature=0.0, max_tokens=256, top_p=None, stop=()),
        base_url="https://example.com/v1",
        api_key_env="JUDGE_API_KEY",
        pricing=JudgePricing(input_per_mtok=3.0, output_per_mtok=15.0),
        timeout_s=120.0,
        max_retries=5,
        max_label_attempts=1,
    )
    assert judge_config_version_hash(without_ops) == JUDGE_GOLDEN_HASH
    assert judge_config_version_hash(with_ops) == JUDGE_GOLDEN_HASH


def test_judge_config_id_change_changes_hash() -> None:
    changed = JUDGE_REC.model_copy(update={"id": "other-judge"})
    assert judge_config_version_hash(changed) != JUDGE_GOLDEN_HASH


def test_judge_config_label_change_changes_hash() -> None:
    changed = JUDGE_REC.model_copy(update={"labels": ("good", "neutral")})
    assert judge_config_version_hash(changed) != JUDGE_GOLDEN_HASH


def test_judge_config_model_change_changes_hash() -> None:
    changed = JUDGE_REC.model_copy(update={"model": "other-model"})
    assert judge_config_version_hash(changed) != JUDGE_GOLDEN_HASH


def test_judge_config_rubric_change_changes_hash() -> None:
    changed = JUDGE_REC.model_copy(update={"rubric": "Different rubric.\n\n$input"})
    assert judge_config_version_hash(changed) != JUDGE_GOLDEN_HASH


def test_judge_config_temperature_change_changes_hash() -> None:
    changed = JUDGE_REC.model_copy(update={"params": JudgeParams(temperature=0.5)})
    assert judge_config_version_hash(changed) != JUDGE_GOLDEN_HASH


def test_judge_config_default_params_materialize() -> None:
    explicit = JudgeConfigRecord(
        id="example-judge",
        provider="anthropic",
        model="test-model",
        rubric="Rate this support reply.\n\n$input\n\nReply with JSON.",
        labels=("good", "bad"),
        params=JudgeParams(),
    )
    implicit = JudgeConfigRecord(
        id="example-judge",
        provider="anthropic",
        model="test-model",
        rubric="Rate this support reply.\n\n$input\n\nReply with JSON.",
        labels=("good", "bad"),
    )
    assert judge_config_version_hash(explicit) == JUDGE_GOLDEN_HASH
    assert judge_config_version_hash(implicit) == JUDGE_GOLDEN_HASH
