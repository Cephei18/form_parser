"""Evaluation CLI — score pipeline predictions against a ground-truth corpus.

Offline and read-only: it never calls Textract and never runs the model. It
compares the ``mappings.json`` the pipeline already emits against hand-annotated
ground truth, writing per-form metrics, per-page diff images, and a corpus
``benchmark_summary.json``.

Examples
--------
    # Evaluate the bundled corpus (predictions shipped under each form):
    venv/Scripts/python.exe scripts/evaluate.py --corpus benchmarks/corpus --out eval_runs/local

    # Score a fresh batch of pipeline outputs against the same ground truth:
    python scripts/evaluate.py --corpus benchmarks/corpus \
        --predictions-root output/ --out eval_runs/run42

    # CI gate: fail if micro-F1 regresses below a floor.
    python scripts/evaluate.py --corpus benchmarks/corpus --fail-under-f1 0.80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when invoked as a plain script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.flags import EvalConfig  # noqa: E402
from src.evaluation.runner import evaluate_corpus  # noqa: E402


def _build_config(args: argparse.Namespace) -> EvalConfig:
    base = EvalConfig()
    return EvalConfig(
        iou_threshold=args.iou if args.iou is not None else base.iou_threshold,
        duplicate_iou=args.duplicate_iou if args.duplicate_iou is not None else base.duplicate_iou,
        fragment_iou=base.fragment_iou,
        strict_type=args.strict_type or base.strict_type,
        canvas_width=base.canvas_width,
        canvas_height=base.canvas_height,
    )


def _print_report(summary: dict) -> None:
    agg = summary.get("aggregate", {})
    micro = agg.get("micro", {})
    corpus = summary.get("corpus", {})
    print("\n=== Evaluation summary ===")
    print(f"corpus       : {summary.get('corpus_dir')}")
    print(f"forms        : {corpus.get('evaluated')}/{corpus.get('form_count')} evaluated")
    if corpus.get("missing_predictions"):
        print(f"  no preds   : {', '.join(corpus['missing_predictions'])}")
    print(f"micro P/R/F1 : {micro.get('precision')} / {micro.get('recall')} / {micro.get('f1')}")
    print(f"macro F1     : {agg.get('macro_f1')}")
    print(f"mean IoU     : {micro.get('mean_iou')}")
    print(f"duplicate    : {agg.get('duplicate_rate')}")
    print(f"fragmentation: {agg.get('fragmentation_rate')}")
    print(f"non-fillable FP: {agg.get('non_fillable_false_positives')}")
    print("\nper form:")
    for m in summary.get("per_form", []):
        if m.get("status") != "evaluated":
            print(f"  - {m['form_id']:<28} [{m.get('status')}]")
            continue
        o = m["overall"]
        c = m["counts"]
        print(
            f"  - {m['form_id']:<28} F1={o['f1']:.3f}  P={o['precision']:.3f}  R={o['recall']:.3f}  "
            f"IoU={o['mean_iou']:.3f}  (tp{c['tp']}/fp{c['fp']}/fn{c['fn']})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score form-parser predictions against a ground-truth corpus.")
    parser.add_argument("--corpus", default="benchmarks/corpus", help="Corpus directory (default: benchmarks/corpus)")
    parser.add_argument("--out", default="eval_runs/latest", help="Output directory (default: eval_runs/latest)")
    parser.add_argument(
        "--predictions-root",
        default=None,
        help="Optional parallel tree of predictions: <root>/<form_dir>/mappings.json. "
        "When omitted, reads <form>/predictions/mappings.json from the corpus.",
    )
    parser.add_argument("--iou", type=float, default=None, help="Match IoU threshold (default 0.5)")
    parser.add_argument("--duplicate-iou", type=float, default=None, help="Duplicate-detection IoU (default 0.7)")
    parser.add_argument("--strict-type", action="store_true", help="Require widget type to match for a true positive")
    parser.add_argument("--fail-under-f1", type=float, default=None, help="Exit non-zero if micro-F1 < this (CI gate)")
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 2

    config = _build_config(args)
    summary = evaluate_corpus(corpus_dir, args.out, config, predictions_root=args.predictions_root)
    _print_report(summary)
    print(f"\nwrote {Path(args.out) / 'benchmark_summary.json'}")

    if args.fail_under_f1 is not None:
        f1 = summary.get("aggregate", {}).get("micro", {}).get("f1", 0.0)
        if f1 < args.fail_under_f1:
            print(f"\nFAIL: micro-F1 {f1} < threshold {args.fail_under_f1}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
