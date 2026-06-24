# Phase 0 — Evaluation Harness

An **offline, read-only** harness that scores the parser's output against a
hand-annotated ground-truth corpus. It is the measurement foundation every later
phase (L1 structural understanding, L2 logical widgets, …) is gated on: nothing
ships "better" until this says so.

## Why it is production-safe

- Lives entirely in `src/evaluation/` and imports **nothing** from the live
  pipeline (`textract_pipeline`, `field_anchor_engine`, `lambda_worker`).
- No live-path code imports it; it is never called in the Lambda worker or the
  FastAPI request path.
- It only **reads** the `mappings.json` the pipeline already emits, so running it
  cannot change any production behaviour.
- Kill-switch `FORM_PARSER_EVAL_HARNESS_ENABLED` (default **OFF**) gates any
  future *inline* integration; it does not gate the offline CLI.

> The aggregate report is written to `eval_runs/<run>/benchmark_summary.json`.
> This is a **different path** from the pipeline's per-job
> `output/<job>/benchmark_summary.json`, so the two never collide.

## Components

| Module | Responsibility |
|---|---|
| `schema.py` | Ground-truth & prediction records; loaders; type taxonomy |
| `geometry.py` | Fraction-box IoU / containment (same `{x,y,width,height}` convention as `mappings.json`) |
| `matching.py` | Greedy IoU matching of predictions ↔ ground truth (no scipy) |
| `metrics.py` | Detection + failure-mode metrics; corpus aggregation |
| `visualize.py` | Per-page colour-coded diff images (Pillow) |
| `runner.py` | Corpus discovery → per-form eval → `benchmark_summary.json` |
| `scripts/evaluate.py` | CLI |

## Metrics contract

Headline (micro-averaged over the corpus, also per-form):

- **precision / recall / F1** — a prediction is a true positive when it matches a
  GT widget at IoU ≥ `--iou` (default 0.5); `--strict-type` additionally requires
  the widget type to agree.
- **mean IoU** over matched pairs.
- **type / family match rate** among matches (soft correctness without gating TP).

Failure-mode metrics (each maps to a listed production problem):

| Metric | Production problem |
|---|---|
| `duplicate_rate` | duplicate mappings |
| `fragmentation_rate`, `avg_fragments_per_group` | character boxes / checkbox matrices as independent fields |
| `non_fillable_false_positives` | non-interactive pages generating false positives |
| `repeat_group_recall` | repeated sections causing ambiguity |
| `fillability_ratio` | output has ~the right widget count |
| `per_type` P/R | wrong region/type detection, broken down by widget type |

## Usage

```bash
# Score the bundled corpus (predictions shipped per form):
venv/Scripts/python.exe scripts/evaluate.py --corpus benchmarks/corpus --out eval_runs/local

# Score a fresh pipeline output batch against the same ground truth:
python scripts/evaluate.py --corpus benchmarks/corpus --predictions-root output/ --out eval_runs/run42

# CI regression gate:
python scripts/evaluate.py --corpus benchmarks/corpus --fail-under-f1 0.80
```

Outputs:

```
eval_runs/<run>/
  benchmark_summary.json        # corpus rollup (config + pipeline-flag snapshot + per-form)
  <form_id>/
    evaluation.json             # per-form metrics
    diff_page_<n>.png           # GT vs prediction overlay (green=matched GT, red=FN, blue=matched pred, orange=FP)
```

## Tunables (CLI flags / env)

| CLI | Env | Default |
|---|---|---|
| `--iou` | `FORM_PARSER_EVAL_IOU` | 0.5 |
| `--duplicate-iou` | `FORM_PARSER_EVAL_DUP_IOU` | 0.7 |
| `--strict-type` | `FORM_PARSER_EVAL_STRICT_TYPE` | false |
| — | `FORM_PARSER_EVAL_HARNESS_ENABLED` | false (inline hook only) |

## Corpus

See `benchmarks/corpus/README.md`. One form per production failure mode; a
bundled `_example_synthetic` form ships predictions and is fully runnable so the
harness can be exercised with no AWS access.
