# judgekit

Evaluate your evaluators.

judgekit is a calibration and agent-trajectory evaluation platform whose headline metric is
judge-vs-human validity rather than judge-vs-judge agreement. Most eval tooling measures whether
an LLM judge is consistent with other judges. That is the wrong question: a judge can agree with
other judges while drifting well away from human judgment. judgekit treats a judge as an untrusted
measurement instrument until it has been calibrated against human labels, and it ships the
instruments needed to prove that calibration: a five-type bias battery, item-response-theory
analysis of eval items, and trajectory-aware agent scoring, all wired to a regression gate that
drops into CI.

## Status

The first release has shipped: the dataset schema, the deterministic scorers, and `judgekit run`.
Next up is the statistical core (Cohen's kappa and Krippendorff's alpha vs. human labels). See
Roadmap for the rest of the milestone plan.

## Architecture

```mermaid
flowchart LR
    DS[Dataset layer<br/>versioned JSONL cases] --> RN[Judge runner<br/>batched, cached, cost-tracked]
    SC[Scorer registry<br/>deterministic and judge] --> RN
    RN --> CAL[Calibration report<br/>kappa / alpha vs human]
    RN --> BIAS[Bias battery<br/>five bias types]
    RN --> IRT[IRT item analysis<br/>difficulty / discrimination]
    CAL --> GATE[Regression CI gate]
    BIAS --> GATE
    PROD[Production traces] -.feedback.-> DS
```

## Why this exists

- Judges can agree with one another while diverging from human judgment. Variance compression and
  surface-quality inflation are documented effects, not hypotheticals
  ([Reliability without Validity](https://arxiv.org/pdf/2606.19544)).
- In production bias tests, judge error exceeds 50% even when controlled-setting agreement looks
  like roughly 85%. Agreement is not validity
  ([LLM-as-judge reliability](https://www.adaline.ai/blog/llm-as-a-judge-reliability-bias)).
- The credible way to run evals is error-analysis-first and human-calibration-first
  ([evals FAQ, Husain and Shankar](https://hamel.dev/blog/posts/evals-faq/)). judgekit builds that
  discipline into the tooling instead of leaving it to convention.

## What the first release delivers

- A versioned JSONL dataset schema (input, reference, human label, metadata slices) validated by
  pydantic models, with a content-addressed `dataset_version` (a sha256 hash over the case set).
- Deterministic scorers (exact, regex, structured) behind a `judgekit run` command. Zero model
  calls; cost is recorded as 0.0 on every result.
- Append-only JSONL run artifacts: a manifest line (run id, dataset version, scorer, numeric
  summary) followed by one traced `RunResult` line per case.

## Quickstart

```bash
uv run judgekit run --dataset examples/exact/cases.jsonl --outputs examples/exact/outputs.jsonl --scorer exact
uv run judgekit run --dataset examples/regex/cases.jsonl --outputs examples/regex/outputs.jsonl --scorer regex
uv run judgekit run --dataset examples/structured/cases.jsonl --outputs examples/structured/outputs.jsonl --scorer structured
```

Each run writes its artifact to `runs/<run_id>.jsonl` by default (pass `--out` to choose the path)
and prints a summary, for example:

```
run_id          f256b44ca6014509a2cd031200804a87
dataset_version sha256:ca0bf0709f3afd4f9ef3d1ee2fcd5e32253624bdb5cbc5473caf6523b6b3f600
scorer          exact
n_cases         8
mean_score      0.7500
pass_rate       0.7500
artifact        runs/f256b44ca6014509a2cd031200804a87.jsonl
```

Exit codes: 0 = run completed (scores are measurements, not gates), 1 = data or validation errors
(dataset and outputs problems are reported with line numbers), 2 = usage errors.

## Roadmap

1. Dataset schema, deterministic scorers, and CLI. (shipped)
2. Statistical core: kappa and alpha with confidence intervals.
3. Judge runner: provider-agnostic interface, response caching, per-run cost tracking.
4. Calibration studies against human labels, with locked rubrics and an explicit "unknown" option.
5. Five-type bias battery: position, verbosity, self-preference, format, calibration drift.
6. Regression gate for CI.
7. Trajectory evaluator: outcome and policy scoring over agent runs, reported as pass@k and pass^k.
8. Trace ingestion, dashboard, and the trace-to-dataset feedback loop.

## Development

```bash
uv sync
make check   # lint, typecheck, test
```

Individual targets: `make lint`, `make format`, `make typecheck`, `make test`, `make docker-build`.

The CLI installs as the `judgekit` entry point; check it with `uv run judgekit --version`.

## License

Apache-2.0. See [LICENSE](LICENSE).
