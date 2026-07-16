import hashlib
import json
from pathlib import Path

import pytest

from judgekit import hashing
from judgekit.dataset import load_dataset
from judgekit.errors import BudgetExceededError, JudgeError, ProviderError
from judgekit.hashing import judge_config_version_hash
from judgekit.judge import _RETRY_SUFFIX, execute_judge, write_judge_artifact
from judgekit.models import (
    Case,
    JudgeConfig,
    JudgeConfigRecord,
    JudgePricing,
    JudgeRunManifest,
    JudgeVerdict,
)
from judgekit.providers import ProviderRequest, ProviderResponse


class FakeProvider:
    """A scripted Provider double that records every request it receives."""

    provider_id = "fake"

    def __init__(self, script: list[ProviderResponse | Exception]) -> None:
        self._script = list(script)
        self.requests: list[ProviderRequest] = []

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if not self._script:
            raise AssertionError("FakeProvider called with no script left")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _config(
    *,
    labels: tuple[object, ...] = ("good", "bad"),
    rubric: str = "Rate this reply.\n\n$input",
    pricing: JudgePricing | None = None,
    max_label_attempts: int = 3,
) -> JudgeConfig:
    fields: dict[str, object] = {
        "id": "test-judge",
        "provider": "anthropic",
        "model": "test-model",
        "rubric": rubric,
        "labels": list(labels),
        "max_label_attempts": max_label_attempts,
    }
    if pricing is not None:
        fields["pricing"] = pricing
    record = JudgeConfigRecord(**fields)  # type: ignore[arg-type]
    version_hash = judge_config_version_hash(record)
    return JudgeConfig(**record.model_dump(), version_hash=version_hash)


def _write_dataset(tmp_path: Path, name: str, records: list[dict[str, object]]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _label_response(
    label: object, *, input_tokens: int = 10, output_tokens: int = 5
) -> ProviderResponse:
    return ProviderResponse(
        text=json.dumps({"label": label}), input_tokens=input_tokens, output_tokens=output_tokens
    )


def _cache_entry_path(cache_dir: Path, config: JudgeConfig, case: Case) -> Path:
    case_sha = hashlib.sha256(hashing.case_content_json(case).encode("utf-8")).hexdigest()
    return cache_dir / config.version_hash.split(":", 1)[1] / f"{case_sha}.json"


def test_happy_path_three_cases(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [
                {"id": "c1", "input": "in1"},
                {"id": "c2", "input": "in2"},
                {"id": "c3", "input": "in3"},
            ],
        )
    )
    config = _config()
    provider = FakeProvider(
        [
            _label_response("good", input_tokens=10, output_tokens=5),
            _label_response("bad", input_tokens=20, output_tokens=6),
            _label_response("good", input_tokens=30, output_tokens=7),
        ]
    )
    cache_dir = tmp_path / "cache"

    artifact = execute_judge(
        dataset,
        config,
        provider,
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
    )

    assert [v.case_id for v in artifact.verdicts] == ["c1", "c2", "c3"]
    assert [v.label for v in artifact.verdicts] == ["good", "bad", "good"]
    for verdict in artifact.verdicts:
        assert verdict.cached is False
        assert verdict.n_attempts == 1
        assert verdict.cost == 0.0

    entry_dir = cache_dir / config.version_hash.split(":", 1)[1]
    entry_files = list(entry_dir.glob("*.json"))
    assert len(entry_files) == 3

    totals = artifact.manifest.totals
    assert totals.n_cases == 3
    assert totals.n_cached == 0
    assert totals.n_live == 3
    assert totals.input_tokens == 10 + 20 + 30
    assert totals.output_tokens == 5 + 6 + 7
    assert totals.cost == 0.0


def test_warm_rerun_uses_cache_and_calls_provider_never(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [{"id": "c1", "input": "in1"}, {"id": "c2", "input": "in2"}],
        )
    )
    config = _config()
    cache_dir = tmp_path / "cache"
    first = execute_judge(
        dataset,
        config,
        FakeProvider(
            [_label_response("good", input_tokens=10, output_tokens=5), _label_response("bad")]
        ),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
    )

    second = execute_judge(
        dataset,
        config,
        FakeProvider([]),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
    )

    assert [v.label for v in second.verdicts] == [v.label for v in first.verdicts]
    assert all(v.cached is True for v in second.verdicts)
    assert second.manifest.totals.cost == 0.0
    assert second.manifest.totals.input_tokens == 0
    assert second.manifest.totals.output_tokens == 0
    assert second.verdicts[0].input_tokens == first.verdicts[0].input_tokens
    assert second.verdicts[0].output_tokens == first.verdicts[0].output_tokens


def test_exact_cost_arithmetic(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [{"id": "c1", "input": "in1"}, {"id": "c2", "input": "in2"}],
        )
    )
    pricing = JudgePricing(input_per_mtok=1.0, output_per_mtok=2.0)
    config = _config(pricing=pricing)
    provider = FakeProvider(
        [
            _label_response("good", input_tokens=1000, output_tokens=500),
            _label_response("good", input_tokens=1000, output_tokens=500),
        ]
    )

    artifact = execute_judge(
        dataset,
        config,
        provider,
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache",
    )

    expected_case_cost = (1000 * 1.0 + 500 * 2.0) / 1_000_000
    assert artifact.verdicts[0].cost == expected_case_cost
    assert artifact.verdicts[1].cost == expected_case_cost
    assert artifact.manifest.totals.cost == expected_case_cost * 2


def test_budget_stop_mid_run(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [
                {"id": "c1", "input": "in1"},
                {"id": "c2", "input": "in2"},
                {"id": "c3", "input": "in3"},
            ],
        )
    )
    pricing = JudgePricing(input_per_mtok=2.0, output_per_mtok=0.0)
    config = _config(pricing=pricing)
    provider = FakeProvider([_label_response("good", input_tokens=1000, output_tokens=0)])
    cache_dir = tmp_path / "cache"

    with pytest.raises(BudgetExceededError) as exc_info:
        execute_judge(
            dataset,
            config,
            provider,
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=cache_dir,
            max_cost=0.001,
        )

    assert "2 case(s) unresolved" in str(exc_info.value)
    entry_dir = cache_dir / config.version_hash.split(":", 1)[1]
    assert len(list(entry_dir.glob("*.json"))) == 1


def test_max_cost_zero_cold_cache_never_calls_provider(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "in1"}]))
    config = _config(pricing=JudgePricing(input_per_mtok=1.0, output_per_mtok=1.0))
    provider = FakeProvider([])

    with pytest.raises(BudgetExceededError):
        execute_judge(
            dataset,
            config,
            provider,
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=tmp_path / "cache",
            max_cost=0.0,
        )

    assert provider.requests == []


def test_max_cost_zero_warm_cache_completes_offline(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "in1"}]))
    config = _config(pricing=JudgePricing(input_per_mtok=1.0, output_per_mtok=1.0))
    cache_dir = tmp_path / "cache"
    execute_judge(
        dataset,
        config,
        FakeProvider([_label_response("good")]),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
    )

    artifact = execute_judge(
        dataset,
        config,
        FakeProvider([]),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
        max_cost=0.0,
    )

    assert artifact.verdicts[0].cached is True


def test_max_cost_without_pricing_raises_judge_error(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "in1"}]))
    config = _config()
    provider = FakeProvider([])

    with pytest.raises(JudgeError):
        execute_judge(
            dataset,
            config,
            provider,
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=tmp_path / "cache",
            max_cost=1.0,
        )

    assert provider.requests == []


def test_retry_then_success(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "in1"}]))
    config = _config()
    provider = FakeProvider(
        [
            ProviderResponse(text="this is not json", input_tokens=10, output_tokens=5),
            _label_response("good", input_tokens=20, output_tokens=6),
        ]
    )

    artifact = execute_judge(
        dataset,
        config,
        provider,
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache",
    )

    verdict = artifact.verdicts[0]
    assert verdict.n_attempts == 2
    assert verdict.input_tokens == 10 + 20
    assert verdict.output_tokens == 5 + 6
    assert not provider.requests[0].prompt.endswith(_RETRY_SUFFIX)
    assert provider.requests[1].prompt.endswith(_RETRY_SUFFIX)


def test_retries_exhausted_no_cache_for_failed_case(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [{"id": "c1", "input": "in1"}, {"id": "c2", "input": "in2"}],
        )
    )
    config = _config(max_label_attempts=2)
    provider = FakeProvider(
        [
            _label_response("good"),
            ProviderResponse(text="nope", input_tokens=1, output_tokens=1),
            ProviderResponse(text="still nope", input_tokens=1, output_tokens=1),
        ]
    )
    cache_dir = tmp_path / "cache"

    with pytest.raises(JudgeError) as exc_info:
        execute_judge(
            dataset,
            config,
            provider,
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=cache_dir,
        )

    message = str(exc_info.value)
    assert "c2" in message
    assert "2 attempt(s)" in message

    entry_dir = cache_dir / config.version_hash.split(":", 1)[1]
    entries = list(entry_dir.glob("*.json"))
    assert len(entries) == 1
    c1_case = load_dataset(tmp_path / "cases.jsonl").cases[0]
    assert _cache_entry_path(cache_dir, config, c1_case).exists()


def test_type_aware_labels(tmp_path: Path) -> None:
    config = _config(labels=(1, True), max_label_attempts=1)

    dataset_bool = load_dataset(
        _write_dataset(tmp_path, "bool.jsonl", [{"id": "c1", "input": "x"}])
    )
    bool_artifact = execute_judge(
        dataset_bool,
        config,
        FakeProvider([_label_response(True)]),
        dataset_path="bool.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache_bool",
    )
    assert bool_artifact.verdicts[0].label is True

    dataset_int = load_dataset(_write_dataset(tmp_path, "int.jsonl", [{"id": "c1", "input": "x"}]))
    int_artifact = execute_judge(
        dataset_int,
        config,
        FakeProvider([_label_response(1)]),
        dataset_path="int.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache_int",
    )
    assert int_artifact.verdicts[0].label == 1
    assert int_artifact.verdicts[0].label is not True

    dataset_str = load_dataset(_write_dataset(tmp_path, "str.jsonl", [{"id": "c1", "input": "x"}]))
    with pytest.raises(JudgeError):
        execute_judge(
            dataset_str,
            config,
            FakeProvider([_label_response("1")]),
            dataset_path="str.jsonl",
            judge_config_path="judge.json",
            cache_dir=tmp_path / "cache_str",
        )


def test_extra_keys_in_reply_are_tolerated(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "x"}]))
    config = _config()
    response = ProviderResponse(
        text=json.dumps({"label": "good", "reasoning": "solid reply"}),
        input_tokens=10,
        output_tokens=5,
    )

    artifact = execute_judge(
        dataset,
        config,
        FakeProvider([response]),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache",
    )

    assert artifact.verdicts[0].n_attempts == 1
    assert artifact.verdicts[0].label == "good"


def test_cache_poisoning_regression_same_id_different_content(tmp_path: Path) -> None:
    config = _config()
    cache_dir = tmp_path / "cache"
    dataset_a = load_dataset(
        _write_dataset(tmp_path, "a.jsonl", [{"id": "c1", "input": "content-x"}])
    )
    execute_judge(
        dataset_a,
        config,
        FakeProvider([_label_response("good")]),
        dataset_path="a.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
    )

    dataset_b = load_dataset(
        _write_dataset(tmp_path, "b.jsonl", [{"id": "c1", "input": "content-y"}])
    )
    provider_b = FakeProvider([_label_response("bad")])
    artifact_b = execute_judge(
        dataset_b,
        config,
        provider_b,
        dataset_path="b.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
    )

    assert len(provider_b.requests) == 1
    assert artifact_b.verdicts[0].label == "bad"


def test_corrupt_cache_entry_raises_judge_error(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "x"}]))
    config = _config()
    cache_dir = tmp_path / "cache"
    entry_path = _cache_entry_path(cache_dir, config, dataset.cases[0])
    entry_path.parent.mkdir(parents=True)
    entry_path.write_text("this is garbage, not json", encoding="utf-8")

    with pytest.raises(JudgeError) as exc_info:
        execute_judge(
            dataset,
            config,
            FakeProvider([]),
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=cache_dir,
        )

    message = str(exc_info.value)
    assert str(entry_path) in message
    assert "delete the file" in message


def test_cache_entry_with_out_of_config_label_raises_judge_error(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "x"}]))
    config = _config()
    cache_dir = tmp_path / "cache"
    entry_path = _cache_entry_path(cache_dir, config, dataset.cases[0])
    entry_path.parent.mkdir(parents=True)
    entry_path.write_text(
        json.dumps(
            {
                "label": "not-a-config-label",
                "response_text": "{}",
                "input_tokens": 1,
                "output_tokens": 1,
                "n_attempts": 1,
                "model": "test-model",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(JudgeError) as exc_info:
        execute_judge(
            dataset,
            config,
            FakeProvider([]),
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=cache_dir,
        )

    assert str(entry_path) in str(exc_info.value)


def test_reference_rubric_missing_reference_lists_all_ids(tmp_path: Path) -> None:
    config = _config(rubric="Rate $input against $reference.")
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [
                {"id": "c1", "input": "in1", "reference": "ref1"},
                {"id": "c2", "input": "in2"},
                {"id": "c3", "input": "in3"},
            ],
        )
    )

    with pytest.raises(JudgeError) as exc_info:
        execute_judge(
            dataset,
            config,
            FakeProvider([]),
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=tmp_path / "cache",
        )

    message = str(exc_info.value)
    assert "c2" in message
    assert "c3" in message


def test_reference_rubric_with_references_present(tmp_path: Path) -> None:
    config = _config(rubric="Rate $input against $reference.")
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [{"id": "c1", "input": "in1", "reference": "the-reference-text"}],
        )
    )
    provider = FakeProvider([_label_response("good")])

    execute_judge(
        dataset,
        config,
        provider,
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache",
    )

    assert "the-reference-text" in provider.requests[0].prompt


def test_prompt_contains_case_input_verbatim(tmp_path: Path) -> None:
    config = _config()
    dataset = load_dataset(
        _write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "the exact input text"}])
    )
    provider = FakeProvider([_label_response("good")])

    execute_judge(
        dataset,
        config,
        provider,
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache",
    )

    assert "the exact input text" in provider.requests[0].prompt


def test_provider_error_propagates_and_first_case_stays_cached(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [{"id": "c1", "input": "in1"}, {"id": "c2", "input": "in2"}],
        )
    )
    config = _config()
    cache_dir = tmp_path / "cache"
    provider = FakeProvider([_label_response("good"), ProviderError("boom")])

    with pytest.raises(ProviderError):
        execute_judge(
            dataset,
            config,
            provider,
            dataset_path="cases.jsonl",
            judge_config_path="judge.json",
            cache_dir=cache_dir,
        )

    assert _cache_entry_path(cache_dir, config, dataset.cases[0]).exists()


def test_write_judge_artifact_round_trip_and_refuses_overwrite(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [{"id": "c1", "input": "in1"}, {"id": "c2", "input": "in2"}],
        )
    )
    config = _config()
    artifact = execute_judge(
        dataset,
        config,
        FakeProvider([_label_response("good"), _label_response("bad")]),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache",
    )
    out_path = tmp_path / "nested" / "run.jsonl"

    write_judge_artifact(artifact, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + len(artifact.verdicts)
    manifest = JudgeRunManifest.model_validate(json.loads(lines[0]))
    assert manifest.kind == "judge_run"
    for line in lines[1:]:
        JudgeVerdict.model_validate(json.loads(line))

    with pytest.raises(JudgeError) as exc_info:
        write_judge_artifact(artifact, out_path)
    assert "refusing to overwrite" in str(exc_info.value)


def test_no_tmp_files_remain_after_run(tmp_path: Path) -> None:
    dataset = load_dataset(
        _write_dataset(
            tmp_path,
            "cases.jsonl",
            [{"id": "c1", "input": "in1"}, {"id": "c2", "input": "in2"}],
        )
    )
    config = _config()
    cache_dir = tmp_path / "cache"

    execute_judge(
        dataset,
        config,
        FakeProvider([_label_response("good"), _label_response("bad")]),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=cache_dir,
    )

    assert list(cache_dir.rglob("*.tmp")) == []


def test_manifest_integrity(tmp_path: Path) -> None:
    dataset = load_dataset(_write_dataset(tmp_path, "cases.jsonl", [{"id": "c1", "input": "in1"}]))
    config = _config(pricing=JudgePricing(input_per_mtok=1.0, output_per_mtok=1.0))

    artifact = execute_judge(
        dataset,
        config,
        FakeProvider([_label_response("good")]),
        dataset_path="cases.jsonl",
        judge_config_path="judge.json",
        cache_dir=tmp_path / "cache",
        max_cost=5.0,
    )

    manifest = artifact.manifest
    assert manifest.dataset_version == dataset.dataset_version
    assert manifest.judge_config_hash == config.version_hash
    assert manifest.provider == config.provider
    assert manifest.model == config.model
    assert manifest.max_cost == 5.0
