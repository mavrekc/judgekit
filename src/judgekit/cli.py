"""Command-line interface: score outputs against a dataset and calibrate raters."""

from pathlib import Path
from typing import Annotated, cast

import typer

from judgekit import __version__, stats
from judgekit.calibration import RaterRatings, execute_calibration, write_report
from judgekit.dataset import load_dataset, load_outputs, load_rater
from judgekit.errors import JudgekitError
from judgekit.judge import execute_judge, load_judge_config, write_judge_artifact
from judgekit.providers import get_provider
from judgekit.runner import execute_run, write_artifact
from judgekit.scorers import get_scorer

_LEVELS: tuple[str, ...] = ("nominal", "ordinal", "interval")


def _level_callback(value: str) -> str:
    if value not in _LEVELS:
        raise typer.BadParameter(f"must be one of: {', '.join(_LEVELS)}")
    return value


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"judgekit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """judgekit: evaluate your evaluators."""


@app.command()
def run(
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", exists=True, dir_okay=False, readable=True),
    ],
    outputs_path: Annotated[
        Path,
        typer.Option("--outputs", exists=True, dir_okay=False, readable=True),
    ],
    scorer: Annotated[str, typer.Option("--scorer")],
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Score a dataset's outputs with a scorer and write a run artifact."""
    try:
        dataset = load_dataset(dataset_path)
        outputs_map = load_outputs(outputs_path)
        scorer_obj = get_scorer(scorer)

        case_ids = {case.id for case in dataset.cases}
        n_extra = sum(1 for case_id in outputs_map if case_id not in case_ids)
        if n_extra:
            typer.echo(
                f"warning: {n_extra} output(s) have no matching case id and will be ignored",
                err=True,
            )

        artifact = execute_run(
            dataset,
            outputs_map,
            scorer_obj,
            dataset_path=str(dataset_path),
            outputs_path=str(outputs_path),
        )

        out_path = out if out is not None else Path("runs") / f"{artifact.manifest.run_id}.jsonl"
        write_artifact(artifact, out_path)

        summary = artifact.manifest.summary
        typer.echo(f"{'run_id':<16}{artifact.manifest.run_id}")
        typer.echo(f"{'dataset_version':<16}{artifact.manifest.dataset_version}")
        typer.echo(f"{'scorer':<16}{artifact.manifest.scorer_id}")
        typer.echo(f"{'n_cases':<16}{summary.n_cases}")
        typer.echo(f"{'mean_score':<16}{summary.mean_score:.4f}")
        typer.echo(f"{'pass_rate':<16}{summary.pass_rate:.4f}")
        typer.echo(f"{'artifact':<16}{out_path}")
    except JudgekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def judge(
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", exists=True, dir_okay=False, readable=True),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path(".judgekit/cache"),
    max_cost: Annotated[float | None, typer.Option("--max-cost")] = None,
) -> None:
    """Run a judge config over a dataset and write a verdict artifact."""
    try:
        dataset = load_dataset(dataset_path)
        config = load_judge_config(config_path)
        provider = get_provider(
            config.provider,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
        )

        artifact = execute_judge(
            dataset,
            config,
            provider,
            dataset_path=str(dataset_path),
            judge_config_path=str(config_path),
            cache_dir=cache_dir,
            max_cost=max_cost,
        )

        out_path = (
            out if out is not None else Path("runs") / f"judge-{artifact.manifest.run_id}.jsonl"
        )
        write_judge_artifact(artifact, out_path)

        totals = artifact.manifest.totals
        typer.echo(f"{'run_id':<16}{artifact.manifest.run_id}")
        typer.echo(f"{'dataset_version':<16}{artifact.manifest.dataset_version}")
        typer.echo(f"{'judge_config':<16}{config.id}")
        typer.echo(f"{'config_hash':<16}{artifact.manifest.judge_config_hash}")
        typer.echo(f"{'model':<16}{artifact.manifest.model}")
        typer.echo(f"{'n_cases':<16}{totals.n_cases}")
        typer.echo(f"{'n_cached':<16}{totals.n_cached}")
        typer.echo(f"{'n_live':<16}{totals.n_live}")
        typer.echo(f"{'input_tokens':<16}{totals.input_tokens}")
        typer.echo(f"{'output_tokens':<16}{totals.output_tokens}")
        typer.echo(f"{'cost':<16}{totals.cost:.6f}")
        typer.echo(f"{'artifact':<16}{out_path}")
    except JudgekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@app.command()
def calibrate(
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset", exists=True, dir_okay=False, readable=True),
    ],
    ratings_paths: Annotated[
        list[Path],
        typer.Option("--ratings", exists=True, dir_okay=False, readable=True),
    ],
    level: Annotated[str, typer.Option("--level", callback=_level_callback)] = "nominal",
    out: Annotated[Path | None, typer.Option("--out")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    resamples: Annotated[int, typer.Option("--resamples")] = 1000,
) -> None:
    """Compare rater labels against dataset human labels and write a calibration report."""
    try:
        dataset = load_dataset(dataset_path)
        case_ids = {case.id for case in dataset.cases}

        raters: list[RaterRatings] = []
        for path in ratings_paths:
            rater_id, labels = load_rater(path)
            n_unknown = sum(1 for case_id in labels if case_id not in case_ids)
            if n_unknown:
                typer.echo(
                    f"warning: {n_unknown} rating(s) from {rater_id} have no matching "
                    "case id and will be ignored",
                    err=True,
                )
            raters.append(RaterRatings(rater_id, str(path), labels))

        report = execute_calibration(
            dataset,
            raters,
            dataset_path=str(dataset_path),
            level=cast(stats.Level, level),
            seed=seed,
            n_resamples=resamples,
        )

        out_path = out if out is not None else Path("reports") / f"{report.report_id}.json"
        write_report(report, out_path)

        typer.echo(f"{'report_id':<16}{report.report_id}")
        typer.echo(f"{'dataset_version':<16}{report.dataset_version}")
        typer.echo(f"{'level':<16}{report.level}")
        typer.echo(f"{'n_cases':<16}{report.n_cases}")
        typer.echo(f"{'n_labeled':<16}{report.n_labeled}")
        if report.kappa is not None:
            typer.echo(
                f"{'kappa':<16}{report.kappa.value:.4f} "
                f"[{report.kappa.ci_low:.4f}, {report.kappa.ci_high:.4f}] "
                f"(n={report.kappa.n_used})"
            )
        typer.echo(
            f"{'alpha':<16}{report.alpha.value:.4f} "
            f"[{report.alpha.ci_low:.4f}, {report.alpha.ci_high:.4f}] "
            f"(n={report.alpha.n_used})"
        )
        typer.echo(f"{'artifact':<16}{out_path}")
    except JudgekitError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
