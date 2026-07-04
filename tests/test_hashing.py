from judgekit.hashing import case_content_json, dataset_version
from judgekit.models import CaseRecord

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
