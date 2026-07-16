"""Judge config loading and the judge runner."""

import hashlib
import json
import math
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from string import Template

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from judgekit import hashing
from judgekit.errors import BudgetExceededError, JudgeError
from judgekit.models import (
    Dataset,
    JudgeConfig,
    JudgeConfigRecord,
    JudgeRunArtifact,
    JudgeRunManifest,
    JudgeTotals,
    JudgeVerdict,
    Label,
)
from judgekit.providers import Provider, ProviderRequest
from judgekit.stats import category


def _validation_detail(exc: ValidationError) -> str:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first["loc"])
    return f"{loc}: {first['msg']}" if loc else first["msg"]


def load_judge_config(path: Path) -> JudgeConfig:
    """Load a single-JSON-object judge config file into a version-hashed JudgeConfig."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise JudgeError(f"{path}: {exc}") from exc

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{path}: invalid JSON: {exc.msg}") from exc

    if not isinstance(value, dict):
        raise JudgeError(f"{path}: expected a JSON object, got {type(value).__name__}")

    try:
        record = JudgeConfigRecord.model_validate(value)
    except ValidationError as exc:
        raise JudgeError(f"{path}: {_validation_detail(exc)}") from exc

    dupes = _duplicate_label_categories(record.labels)
    if dupes:
        raise JudgeError(f"{path}: duplicate labels: {', '.join(dupes)}")

    version_hash = hashing.judge_config_version_hash(record)
    return JudgeConfig(**record.model_dump(), version_hash=version_hash)


def _duplicate_label_categories(labels: tuple[Label, ...]) -> list[str]:
    """Return the sorted category strings of labels that appear more than once."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for label in labels:
        cat = category(label)
        if cat in seen:
            dupes.add(cat)
        seen.add(cat)
    return sorted(dupes)


class _CacheEntry(BaseModel):
    """A durable cached verdict for one case content hash, keyed under a judge config's hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Label
    response_text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    n_attempts: int = Field(ge=1)
    model: str
    created_at: datetime


_RETRY_SUFFIX = (
    "Your previous reply was not a valid verdict. Reply with ONLY a JSON object of the form "
    '{"label": <label>}, choosing exactly one of the allowed labels.'
)


def _read_cache_entry(entry_path: Path) -> _CacheEntry:
    """Read and validate a cache entry, or raise JudgeError naming the file to re-spend."""
    try:
        text = entry_path.read_text(encoding="utf-8")
        value = json.loads(text)
        return _CacheEntry.model_validate(value)
    except OSError as exc:
        detail = str(exc)
    except json.JSONDecodeError as exc:
        detail = f"invalid JSON: {exc.msg}"
    except ValidationError as exc:
        detail = _validation_detail(exc)
    raise JudgeError(f"{entry_path}: corrupt cache entry ({detail}); delete the file to re-spend")


def execute_judge(
    dataset: Dataset,
    config: JudgeConfig,
    provider: Provider,
    *,
    dataset_path: str,
    judge_config_path: str,
    cache_dir: Path,
    max_cost: float | None = None,
) -> JudgeRunArtifact:
    """Run a judge config over every dataset case, caching verdicts and enforcing the budget."""
    if max_cost is not None and config.pricing is None:
        raise JudgeError("max_cost requires pricing in the judge config")

    allowed: dict[str, Label] = {category(label): label for label in config.labels}
    if len(allowed) != len(config.labels):
        raise JudgeError(f"judge config {config.id!r}: duplicate labels")

    template = Template(config.rubric)
    identifiers = template.get_identifiers()
    if "reference" in identifiers:
        missing = [case.id for case in dataset.cases if case.reference is None]
        if missing:
            ids = ", ".join(missing)
            raise JudgeError(f"rubric references $reference; missing reference for case ids: {ids}")

    judge_cache_dir = cache_dir / config.version_hash.split(":", 1)[1]

    verdicts: list[JudgeVerdict] = []
    spent = 0.0
    live_input_tokens = 0
    live_output_tokens = 0
    n_cached = 0
    n_live = 0

    for case in dataset.cases:
        case_sha = hashlib.sha256(hashing.case_content_json(case).encode("utf-8")).hexdigest()
        entry_path = judge_cache_dir / f"{case_sha}.json"

        if entry_path.exists():
            entry = _read_cache_entry(entry_path)
            entry_category = category(entry.label)
            if entry_category not in allowed:
                raise JudgeError(
                    f"{entry_path}: cached label {entry_category} is not in the config's "
                    "labels; delete the file to re-spend"
                )
            verdicts.append(
                JudgeVerdict(
                    case_id=case.id,
                    label=allowed[entry_category],
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    cost=0.0,
                    cached=True,
                    n_attempts=entry.n_attempts,
                )
            )
            n_cached += 1
            continue

        substitutions: dict[str, str] = {"input": case.input}
        if "reference" in identifiers:
            reference = case.reference
            assert reference is not None
            substitutions["reference"] = reference
        prompt = template.substitute(substitutions)

        case_input_tokens = 0
        case_output_tokens = 0
        case_cost = 0.0
        final_label: Label | None = None
        final_response_text = ""
        n_attempts_used = 0
        last_reason = ""

        for attempt in range(1, config.max_label_attempts + 1):
            if max_cost is not None and spent >= max_cost:
                remaining = len(dataset.cases) - len(verdicts)
                raise BudgetExceededError(
                    f"cost ceiling {max_cost} reached after spending {spent:.6f} "
                    f"with {remaining} case(s) unresolved"
                )

            attempt_prompt = prompt if attempt == 1 else prompt + "\n\n" + _RETRY_SUFFIX
            response = provider.complete(
                ProviderRequest(
                    model=config.model,
                    prompt=attempt_prompt,
                    temperature=config.params.temperature,
                    max_tokens=config.params.max_tokens,
                    top_p=config.params.top_p,
                    stop=config.params.stop,
                )
            )
            case_input_tokens += response.input_tokens
            case_output_tokens += response.output_tokens
            if config.pricing is not None:
                attempt_cost = (
                    response.input_tokens * config.pricing.input_per_mtok
                    + response.output_tokens * config.pricing.output_per_mtok
                ) / 1_000_000
            else:
                attempt_cost = 0.0
            case_cost += attempt_cost
            spent += attempt_cost

            try:
                parsed = json.loads(response.text)
            except json.JSONDecodeError:
                last_reason = "reply was not valid JSON"
                continue
            if not isinstance(parsed, dict) or "label" not in parsed:
                last_reason = 'reply was not a JSON object with a "label" key'
                continue

            raw_label = parsed["label"]
            if not isinstance(raw_label, str | int | float | bool):
                last_reason = f"label {raw_label!r} is not a valid label type"
                continue
            if isinstance(raw_label, float) and not math.isfinite(raw_label):
                last_reason = f"label {raw_label!r} is not finite"
                continue

            raw_category = category(raw_label)
            if raw_category not in allowed:
                last_reason = (
                    f"label {raw_category} is not among the allowed labels "
                    f"({', '.join(sorted(allowed))})"
                )
                continue

            final_label = allowed[raw_category]
            final_response_text = response.text
            n_attempts_used = attempt
            break

        if final_label is None:
            raise JudgeError(
                f"case {case.id!r}: no valid verdict after {config.max_label_attempts} "
                f"attempt(s); last problem: {last_reason}"
            )

        judge_cache_dir.mkdir(parents=True, exist_ok=True)
        entry = _CacheEntry(
            label=final_label,
            response_text=final_response_text,
            input_tokens=case_input_tokens,
            output_tokens=case_output_tokens,
            n_attempts=n_attempts_used,
            model=config.model,
            created_at=datetime.now(UTC),
        )
        tmp_path = entry_path.with_name(entry_path.name + ".tmp")
        tmp_path.write_text(entry.model_dump_json(), encoding="utf-8")
        os.replace(tmp_path, entry_path)

        verdicts.append(
            JudgeVerdict(
                case_id=case.id,
                label=final_label,
                input_tokens=case_input_tokens,
                output_tokens=case_output_tokens,
                cost=case_cost,
                cached=False,
                n_attempts=n_attempts_used,
            )
        )
        n_live += 1
        live_input_tokens += case_input_tokens
        live_output_tokens += case_output_tokens

    totals = JudgeTotals(
        n_cases=len(dataset.cases),
        n_cached=n_cached,
        n_live=n_live,
        input_tokens=live_input_tokens,
        output_tokens=live_output_tokens,
        cost=spent,
    )
    manifest = JudgeRunManifest(
        run_id=uuid.uuid4().hex,
        created_at=datetime.now(UTC),
        dataset_path=dataset_path,
        dataset_version=dataset.dataset_version,
        judge_config_path=judge_config_path,
        judge_config_id=config.id,
        judge_config_hash=config.version_hash,
        provider=config.provider,
        model=config.model,
        cache_dir=str(cache_dir),
        max_cost=max_cost,
        totals=totals,
    )
    return JudgeRunArtifact(manifest=manifest, verdicts=tuple(verdicts))


def write_judge_artifact(artifact: JudgeRunArtifact, out_path: Path) -> None:
    """Write a judge run as JSONL: a manifest line followed by one line per verdict."""
    if out_path.exists():
        raise JudgeError(f"{out_path}: artifact already exists; refusing to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [artifact.manifest.model_dump_json(), *(v.model_dump_json() for v in artifact.verdicts)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
