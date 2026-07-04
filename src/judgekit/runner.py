"""Orchestrates scoring a dataset against outputs and persisting the resulting artifact."""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from judgekit.errors import RunError
from judgekit.models import Dataset, RunArtifact, RunManifest, RunResult, RunSummary
from judgekit.scorers import Scorer


def execute_run(
    dataset: Dataset,
    outputs: Mapping[str, str],
    scorer: Scorer,
    *,
    dataset_path: str,
    outputs_path: str,
) -> RunArtifact:
    """Score every case in a dataset against outputs, producing a full run artifact."""
    if scorer.requires_reference:
        missing_refs = [case.id for case in dataset.cases if case.reference is None]
        if missing_refs:
            ids = ", ".join(missing_refs)
            raise RunError(
                f"scorer {scorer.scorer_id!r} requires a reference; missing for case ids: {ids}"
            )

    missing_outputs = [case.id for case in dataset.cases if case.id not in outputs]
    if missing_outputs:
        ids = ", ".join(missing_outputs)
        raise RunError(f"no output provided for case ids: {ids}")

    run_id = uuid.uuid4().hex
    created_at = datetime.now(UTC)

    results = tuple(
        RunResult(
            run_id=run_id,
            dataset_version=dataset.dataset_version,
            judge_config_id=None,
            scorer_id=scorer.scorer_id,
            case_id=case.id,
            score=scorer.score(case, outputs[case.id]),
            cost=0.0,
            cached=False,
        )
        for case in dataset.cases
    )

    n_cases = len(results)
    if n_cases == 0:
        raise RunError("cannot summarize a run with zero cases")
    scores = [result.score for result in results]
    summary = RunSummary(
        n_cases=n_cases,
        mean_score=sum(scores) / n_cases,
        pass_rate=sum(1 for score in scores if score == 1.0) / n_cases,
    )

    manifest = RunManifest(
        run_id=run_id,
        created_at=created_at,
        dataset_path=dataset_path,
        dataset_version=dataset.dataset_version,
        outputs_path=outputs_path,
        scorer_id=scorer.scorer_id,
        summary=summary,
    )
    return RunArtifact(manifest=manifest, results=results)


def write_artifact(artifact: RunArtifact, out_path: Path) -> None:
    """Write a run artifact as JSONL: a manifest line followed by one line per result."""
    if out_path.exists():
        raise RunError(f"{out_path}: artifact already exists; refusing to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [artifact.manifest.model_dump_json(), *(r.model_dump_json() for r in artifact.results)]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
