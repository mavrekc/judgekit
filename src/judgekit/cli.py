"""Command-line interface: run a scorer over a dataset and its outputs."""

from pathlib import Path
from typing import Annotated

import typer

from judgekit import __version__
from judgekit.dataset import load_dataset, load_outputs
from judgekit.errors import JudgekitError
from judgekit.runner import execute_run, write_artifact
from judgekit.scorers import get_scorer

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
