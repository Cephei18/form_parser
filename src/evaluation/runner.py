"""Orchestration: discover a corpus, evaluate each form, write artifacts.

Outputs (all under the chosen ``--out`` directory, never the live ``output/``):

    <out>/
      benchmark_summary.json      # corpus rollup (deliverable #5)
      <form_id>/
        evaluation.json           # per-form metrics
        diff_page_<n>.png         # per-page visual diff (deliverable #6)

Predictions are read from the pipeline's ``mappings.json`` — either shipped in
``<form>/predictions/mappings.json`` or supplied via a parallel
``--predictions-root/<form>/mappings.json`` tree. The harness never runs the
model; it only scores artifacts that already exist.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.evaluation.flags import EvalConfig
from src.evaluation.matching import match_widgets
from src.evaluation.metrics import aggregate_metrics, compute_form_metrics
from src.evaluation.schema import SCHEMA_VERSION, load_ground_truth, load_predictions
from src.evaluation.visualize import render_form_diffs

# Pipeline phase flags snapshotted into every summary so a result is always
# attributable to the configuration that produced the predictions.
_TRACKED_FLAGS = (
    "FORM_PARSER_MULTIPAGE",
    "FORM_PARSER_PRESERVE_PAGE_SIZE",
    "FORM_PARSER_GLOBAL_ASSIGNMENT_ENABLED",
    "FORM_PARSER_ANSWER_REGION_V2_ENABLED",
    "FORM_PARSER_TABLE_INTELLIGENCE_ENABLED",
    "FORM_PARSER_SEMANTICS_ENABLED",
    "FORM_PARSER_COMB_DETECTION_ENABLED",
    "FORM_PARSER_RADIO_GROUPING_ENABLED",
    "FORM_PARSER_CONFIDENCE_PIPELINE_ENABLED",
    "FORM_PARSER_WIDGET_REGISTRY_ENABLED",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _flag_snapshot() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _TRACKED_FLAGS}


def discover_forms(corpus_dir: Path) -> list[Path]:
    """Return form directories, honouring an optional ``manifest.json`` order."""
    manifest = corpus_dir / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        forms = [corpus_dir / entry["dir"] for entry in data.get("forms", []) if entry.get("dir")]
        if forms:
            return [f for f in forms if (f / "ground_truth.json").is_file()]
    # Auto-discover: any sub-directory holding a ground_truth.json.
    return sorted(p.parent for p in corpus_dir.glob("*/ground_truth.json"))


def _predictions_path(form_dir: Path, predictions_root: Path | None) -> Path | None:
    if predictions_root is not None:
        candidate = predictions_root / form_dir.name / "mappings.json"
        return candidate if candidate.is_file() else None
    candidate = form_dir / "predictions" / "mappings.json"
    return candidate if candidate.is_file() else None


def evaluate_form(
    form_dir: Path,
    out_dir: Path,
    config: EvalConfig,
    predictions_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one form. Returns its metrics dict (with a ``status`` field)."""
    form = load_ground_truth(form_dir / "ground_truth.json")
    form_out = out_dir / form.form_id

    pred_path = _predictions_path(form_dir, predictions_root)
    if pred_path is None:
        metrics = {
            "form_id": form.form_id,
            "doc_class": form.doc_class,
            "status": "no_predictions",
            "gt_widget_count": len(form.widgets),
        }
        _write_json(form_out / "evaluation.json", metrics)
        return metrics

    preds = load_predictions(pred_path)
    result = match_widgets(list(form.widgets), preds, config)
    metrics = compute_form_metrics(form, preds, result, config)
    metrics["status"] = "evaluated"
    metrics["predictions_path"] = str(pred_path)

    # Per-page diff images (background = shipped page rasters when present).
    pages_dir = form_dir / "pages"
    page_images = {
        int(p.stem.split("_")[-1]): p
        for p in pages_dir.glob("page_*.png")
        if p.stem.split("_")[-1].isdigit()
    } if pages_dir.is_dir() else {}
    metrics["diff_images"] = render_form_diffs(form, preds, result, form_out, config, page_images=page_images)

    _write_json(form_out / "evaluation.json", metrics)
    return metrics


def evaluate_corpus(
    corpus_dir: str | Path,
    out_dir: str | Path,
    config: EvalConfig | None = None,
    predictions_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every form in a corpus and write ``benchmark_summary.json``."""
    corpus_dir = Path(corpus_dir)
    out_dir = Path(out_dir)
    config = config or EvalConfig()
    predictions_root = Path(predictions_root) if predictions_root else None

    form_dirs = discover_forms(corpus_dir)
    per_form: list[dict[str, Any]] = []
    for form_dir in form_dirs:
        per_form.append(evaluate_form(form_dir, out_dir, config, predictions_root=predictions_root))

    evaluated = [m for m in per_form if m.get("status") == "evaluated"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "corpus_dir": str(corpus_dir),
        "config": config.to_dict(),
        "pipeline_flags": _flag_snapshot(),
        "corpus": {
            "form_count": len(per_form),
            "evaluated": len(evaluated),
            "missing_predictions": [m["form_id"] for m in per_form if m.get("status") == "no_predictions"],
        },
        "aggregate": aggregate_metrics(evaluated),
        "per_form": per_form,
    }
    _write_json(out_dir / "benchmark_summary.json", summary)
    return summary
