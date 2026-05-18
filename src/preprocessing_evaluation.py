from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_pipeline


PREPROCESS_ENV_KEYS = [
    "FORM_PARSER_PREPROCESSING_ENABLED",
    "FORM_PARSER_PREPROCESS_DENOISE",
    "FORM_PARSER_PREPROCESS_CONTRAST",
    "FORM_PARSER_PREPROCESS_SHARPEN",
    "FORM_PARSER_PREPROCESS_ADAPTIVE_THRESHOLD",
    "FORM_PARSER_PREPROCESS_SKEW",
    "FORM_PARSER_PREPROCESS_DPI_NORMALIZE",
]

STRUCTURAL_ENV_KEYS = [
    "FORM_PARSER_STRUCTURAL_REFINEMENT_ENABLED",
    "FORM_PARSER_FIELD_QUALITY_REFINEMENT_ENABLED",
    "FORM_PARSER_SECTION_GROUPING_ENABLED",
    "FORM_PARSER_OWNERSHIP_PROPAGATION_ENABLED",
    "FORM_PARSER_TABLE_AWARE_REFINEMENT_ENABLED",
]

ISOLATED_EXPERIMENTS: dict[str, dict[str, str]] = {
    "thresholding_only": {
        "FORM_PARSER_PREPROCESSING_ENABLED": "true",
        "FORM_PARSER_PREPROCESS_ADAPTIVE_THRESHOLD": "true",
    },
    "denoising_only": {
        "FORM_PARSER_PREPROCESSING_ENABLED": "true",
        "FORM_PARSER_PREPROCESS_DENOISE": "true",
    },
    "contrast_only": {
        "FORM_PARSER_PREPROCESSING_ENABLED": "true",
        "FORM_PARSER_PREPROCESS_CONTRAST": "true",
    },
    "sharpening_only": {
        "FORM_PARSER_PREPROCESSING_ENABLED": "true",
        "FORM_PARSER_PREPROCESS_SHARPEN": "true",
    },
    "skew_correction_only": {
        "FORM_PARSER_PREPROCESSING_ENABLED": "true",
        "FORM_PARSER_PREPROCESS_SKEW": "true",
    },
}


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    category: str
    source_image: Path
    reference_artifact_dir: Path | None = None
    notes: str = ""


def _quiet_pipeline_logs() -> None:
    for logger_name in [
        "form_parser.pipeline",
        "form_parser.mapping",
        "form_parser.pdf",
        "src.ocr",
        "src.api",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _normalize_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[EvalSample]]:
    manifest = _load_json(manifest_path, {})
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid preprocessing evaluation manifest: {manifest_path}")

    samples = []
    for entry in manifest.get("samples", []):
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("source_image"):
            continue
        samples.append(
            EvalSample(
                sample_id=str(entry["id"]),
                category=str(entry.get("category", "uncategorized")),
                source_image=_normalize_path(str(entry["source_image"])),
                reference_artifact_dir=_normalize_path(str(entry["reference_artifact_dir"]))
                if entry.get("reference_artifact_dir")
                else None,
                notes=str(entry.get("notes", "")),
            )
        )

    if not samples:
        raise ValueError(f"No usable samples found in {manifest_path}")
    return manifest, samples


@contextmanager
def _temporary_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _base_pipeline_env() -> dict[str, str]:
    env = {
        "FORM_PARSER_OCR_DIAGNOSTICS_ENABLED": "true",
        "FORM_PARSER_DYNAMIC_THRESHOLDS_ENABLED": "false",
        "FORM_PARSER_IMAGE_TABLE_FILTERING_ENABLED": "false",
        "FORM_PARSER_FALLBACK_FIELD_LINES_ENABLED": "false",
        "FORM_PARSER_CHECKBOX_DETECTION_ENABLED": "false",
    }
    for key in PREPROCESS_ENV_KEYS + STRUCTURAL_ENV_KEYS:
        env[key] = "false"
    return env


def _experiment_env(experiment_name: str, baseline_dir: Path | None = None) -> dict[str, str]:
    env = _base_pipeline_env()
    env.update(ISOLATED_EXPERIMENTS[experiment_name])
    if baseline_dir is not None:
        env["FORM_PARSER_BASELINE_DIR"] = str(baseline_dir)
    return env


def _prepare_artifact_copy(run_dir: Path, source_image: Path) -> None:
    target = run_dir / "source_image.png"
    if source_image.exists() and not target.exists():
        shutil.copy2(source_image, target)


def _confidence_summary(ocr_diagnostics: dict[str, Any]) -> dict[str, Any]:
    cleaned = ocr_diagnostics.get("cleaned_confidence_distribution")
    raw = ocr_diagnostics.get("raw_confidence_distribution")
    if isinstance(cleaned, dict):
        return cleaned
    return raw if isinstance(raw, dict) else {}


def _count_table_semantic_regions(diagnostics: dict[str, Any]) -> int:
    return sum(
        1
        for region in diagnostics.get("semantic_regions", []) or []
        if isinstance(region, dict) and region.get("type") == "table_like_region"
    )


def _count_inside_table_rejections(payload: Any) -> int:
    if isinstance(payload, dict):
        return sum(_count_inside_table_rejections(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(_count_inside_table_rejections(value) for value in payload)
    return 1 if payload == "inside_table_region" else 0


def _artifact_status(run_dir: Path) -> dict[str, bool]:
    names = [
        "source_image.png",
        "preprocessed.png",
        "preprocessing_diagnostics.json",
        "ocr_raw.json",
        "ocr_diagnostics.json",
        "result.json",
        "lines_detected.png",
        "layout_structure.json",
        "structural_refinement_diagnostics.json",
        "mappings.json",
        "mapping_diagnostics.json",
        "mapping.png",
        "debug_reasoning.png",
        "output.pdf",
        "before_after_comparison.json",
    ]
    return {name: (run_dir / name).exists() and (run_dir / name).stat().st_size > 0 for name in names}


def _collect_run_metrics(run_dir: Path) -> dict[str, Any]:
    result_items = _load_json(run_dir / "result.json", [])
    mappings = _load_json(run_dir / "mappings.json", [])
    benchmark = _load_json(run_dir / "benchmark_summary.json", {})
    diagnostics = _load_json(run_dir / "mapping_diagnostics.json", {})
    ocr_diagnostics = _load_json(run_dir / "ocr_diagnostics.json", {})
    comparison = _load_json(run_dir / "before_after_comparison.json", {})

    if not isinstance(result_items, list):
        result_items = []
    if not isinstance(mappings, list):
        mappings = []
    if not isinstance(benchmark, dict):
        benchmark = {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if not isinstance(ocr_diagnostics, dict):
        ocr_diagnostics = {}
    if not isinstance(comparison, dict):
        comparison = {}

    confidence_classes = benchmark.get("confidence_classes", {})
    if not isinstance(confidence_classes, dict):
        confidence_classes = {}

    comparison_delta = comparison.get("comparison", {})
    if not isinstance(comparison_delta, dict):
        comparison_delta = {}

    return {
        "run_dir": str(run_dir),
        "artifact_status": _artifact_status(run_dir),
        "ocr_text_count": len(result_items),
        "ocr_confidence_distribution": _confidence_summary(ocr_diagnostics),
        "raw_ocr_count": int(ocr_diagnostics.get("raw_item_count", len(result_items)) or 0),
        "low_confidence_count": int(
            (_confidence_summary(ocr_diagnostics).get("buckets", {}) or {}).get("0.00-0.25", 0)
        ),
        "field_candidate_count": int(diagnostics.get("field_candidate_count", 0) or 0),
        "fallback_field_candidate_count": int(diagnostics.get("fallback_field_candidate_count", 0) or 0),
        "mapping_count": len(mappings),
        "mapping_confidence": float(benchmark.get("average_candidate_score", 0.0) or 0.0),
        "missed_field_count": int(benchmark.get("unmatched_label_count", 0) or 0),
        "missed_fields": benchmark.get("unmatched_labels", []) or [],
        "false_positive_proxy": int(confidence_classes.get("ambiguous", 0) or 0)
        + int(confidence_classes.get("weak_match", 0) or 0),
        "confidence_classes": confidence_classes,
        "table_detection_behavior": {
            "image_table_region_count": int(diagnostics.get("image_table_region_count", 0) or 0),
            "semantic_table_region_count": _count_table_semantic_regions(diagnostics),
            "inside_table_rejection_count": _count_inside_table_rejections(mappings),
        },
        "comparison": comparison_delta,
    }


def _classify_result(delta: dict[str, Any], metrics: dict[str, Any]) -> tuple[list[str], list[str]]:
    gains = []
    regressions = []

    mapping_delta = int(delta.get("mapping_count_delta", 0) or 0)
    missed_delta = int(delta.get("unmatched_label_delta", 0) or 0)
    confidence_delta = float(delta.get("average_candidate_score_delta", 0.0) or 0.0)
    ocr_delta = int(delta.get("ocr_item_count_delta", 0) or 0)
    candidate_delta = int(delta.get("field_candidate_count_delta", 0) or 0)

    if mapping_delta > 0:
        gains.append("mapping_count_increased")
    if missed_delta < 0:
        gains.append("missed_fields_decreased")
    if confidence_delta >= 0.01:
        gains.append("mapping_confidence_improved")
    if ocr_delta > 0 and missed_delta <= 0:
        gains.append("ocr_text_count_increased_without_more_misses")

    if mapping_delta < 0:
        regressions.append("mapping_count_decreased")
    if missed_delta > 0:
        regressions.append("missed_fields_increased")
    if confidence_delta <= -0.02:
        regressions.append("mapping_confidence_dropped")
    if candidate_delta > max(10, int(metrics.get("field_candidate_count", 0) * 0.5)):
        regressions.append("field_candidates_became_unstable")
    if ocr_delta < -2:
        regressions.append("ocr_text_count_dropped")

    table_behavior = metrics.get("table_detection_behavior", {})
    if isinstance(table_behavior, dict) and int(table_behavior.get("inside_table_rejection_count", 0) or 0) > 0:
        regressions.append("table_rejection_behavior_changed")

    return gains, regressions


def _run_single_pipeline(source_image: Path, run_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    _prepare_artifact_copy(run_dir, source_image)
    stdout_path = run_dir / "pipeline_stdout.log"
    stderr_path = run_dir / "pipeline_stderr.log"
    with _temporary_env(env), stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_file, redirect_stdout(stdout_file), redirect_stderr(stderr_file):
        started = time.perf_counter()
        run_pipeline(source_image, run_dir)
    metrics = _collect_run_metrics(run_dir)
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    metrics["logs"] = {
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    return metrics


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Controlled Preprocessing Evaluation", ""]
    lines.append(f"Manifest: `{report['manifest_path']}`")
    lines.append(f"Output dir: `{report['output_dir']}`")
    lines.append("")
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append("| Experiment | Samples | Gains | Regressions | Avg Mapping Delta | Avg Confidence Delta | Recommendation |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for experiment in report["experiment_summary"]:
        lines.append(
            "| {name} | {samples} | {gains} | {regressions} | {mapping_delta:.2f} | {confidence_delta:.4f} | {recommendation} |".format(
                name=experiment["experiment"],
                samples=experiment["sample_count"],
                gains=experiment["gain_count"],
                regressions=experiment["regression_count"],
                mapping_delta=float(experiment["average_mapping_count_delta"]),
                confidence_delta=float(experiment["average_mapping_confidence_delta"]),
                recommendation=experiment["interaction_recommendation"],
            )
        )

    lines.append("")
    lines.append("## Per-Sample Results")
    lines.append("")
    lines.append("| Sample | Category | Experiment | Mapping Delta | Missed Delta | Candidate Delta | Confidence Delta | Result |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for sample in report["samples"]:
        for experiment in sample["experiments"]:
            delta = experiment["metrics"]["comparison"]
            result = "regression" if experiment["regressions"] else ("gain" if experiment["gains"] else "stable")
            lines.append(
                "| {sample_id} | {category} | {experiment} | {mapping_delta} | {missed_delta} | {candidate_delta} | {confidence_delta:.4f} | {result} |".format(
                    sample_id=sample["sample_id"],
                    category=sample["category"],
                    experiment=experiment["experiment"],
                    mapping_delta=int(delta.get("mapping_count_delta", 0) or 0),
                    missed_delta=int(delta.get("unmatched_label_delta", 0) or 0),
                    candidate_delta=int(delta.get("field_candidate_count_delta", 0) or 0),
                    confidence_delta=float(delta.get("average_candidate_score_delta", 0.0) or 0.0),
                    result=result,
                )
            )
    lines.append("")
    lines.append("Behavior-changing preprocessing flags remain disabled by default.")
    return "\n".join(lines)


def _aggregate_experiments(samples: list[dict[str, Any]], experiment_names: list[str]) -> list[dict[str, Any]]:
    aggregates = []
    for experiment_name in experiment_names:
        rows = [
            experiment
            for sample in samples
            for experiment in sample["experiments"]
            if experiment["experiment"] == experiment_name
        ]
        mapping_deltas = [float(row["metrics"]["comparison"].get("mapping_count_delta", 0) or 0) for row in rows]
        confidence_deltas = [
            float(row["metrics"]["comparison"].get("average_candidate_score_delta", 0.0) or 0.0)
            for row in rows
        ]
        gain_count = sum(1 for row in rows if row["gains"])
        regression_count = sum(1 for row in rows if row["regressions"])
        if regression_count == 0 and gain_count > 0:
            recommendation = "candidate_for_small_combinations"
        elif regression_count == 0:
            recommendation = "stable_no_clear_gain"
        else:
            recommendation = "do_not_combine_yet"

        aggregates.append(
            {
                "experiment": experiment_name,
                "sample_count": len(rows),
                "gain_count": gain_count,
                "regression_count": regression_count,
                "average_mapping_count_delta": round(mean(mapping_deltas), 4) if mapping_deltas else 0.0,
                "average_mapping_confidence_delta": round(mean(confidence_deltas), 4) if confidence_deltas else 0.0,
                "interaction_recommendation": recommendation,
            }
        )
    return aggregates


def run_preprocessing_evaluation(
    manifest_path: Path,
    output_dir: Path,
    experiments: list[str] | None = None,
    sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    _quiet_pipeline_logs()
    manifest, samples = _load_manifest(manifest_path)
    selected_experiments = experiments or list(ISOLATED_EXPERIMENTS)
    invalid = [experiment for experiment in selected_experiments if experiment not in ISOLATED_EXPERIMENTS]
    if invalid:
        raise ValueError(f"Unknown experiments: {', '.join(invalid)}")

    if sample_ids:
        samples = [sample for sample in samples if sample.sample_id in sample_ids]
    if not samples:
        raise ValueError("No selected samples to evaluate.")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_reports = []

    for sample in samples:
        if not sample.source_image.exists():
            raise FileNotFoundError(f"Sample source image does not exist: {sample.source_image}")

        sample_dir = output_dir / "samples" / sample.sample_id
        baseline_dir = sample_dir / "baseline"
        baseline_metrics = _run_single_pipeline(
            sample.source_image,
            baseline_dir,
            _base_pipeline_env(),
        )

        experiment_reports = []
        for experiment_name in selected_experiments:
            run_dir = sample_dir / experiment_name
            metrics = _run_single_pipeline(
                sample.source_image,
                run_dir,
                _experiment_env(experiment_name, baseline_dir=baseline_dir),
            )
            gains, regressions = _classify_result(metrics["comparison"], metrics)
            experiment_reports.append(
                {
                    "experiment": experiment_name,
                    "run_dir": str(run_dir),
                    "metrics": metrics,
                    "gains": gains,
                    "regressions": regressions,
                }
            )

        sample_reports.append(
            {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "source_image": str(sample.source_image),
                "reference_artifact_dir": str(sample.reference_artifact_dir) if sample.reference_artifact_dir else None,
                "baseline_run_dir": str(baseline_dir),
                "baseline_metrics": baseline_metrics,
                "experiments": experiment_reports,
                "notes": sample.notes,
            }
        )
        _write_json(sample_dir / "sample_report.json", sample_reports[-1])

    report = {
        "name": manifest.get("name", "preprocessing-evaluation"),
        "description": manifest.get("description", ""),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "experiment_design": {
            "production_defaults_changed": False,
            "isolated_experiments_only": True,
            "experiments": {name: ISOLATED_EXPERIMENTS[name] for name in selected_experiments},
        },
        "experiment_summary": _aggregate_experiments(sample_reports, selected_experiments),
        "samples": sample_reports,
    }

    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated preprocessing evaluations against representative forms.")
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "benchmarks" / "preprocessing_eval" / "manifest.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "preprocessing_eval"))
    parser.add_argument(
        "--experiments",
        default=",".join(ISOLATED_EXPERIMENTS),
        help="Comma-separated experiment names. Defaults to all isolated preprocessing stages.",
    )
    parser.add_argument(
        "--samples",
        default="",
        help="Optional comma-separated sample ids. Defaults to every manifest sample.",
    )
    args = parser.parse_args()

    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    sample_ids = {item.strip() for item in args.samples.split(",") if item.strip()} or None
    report = run_preprocessing_evaluation(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        experiments=experiments,
        sample_ids=sample_ids,
    )
    print(
        json.dumps(
            {
                "report_path": str(Path(args.output_dir) / "report.json"),
                "sample_count": len(report["samples"]),
                "experiments": [item["experiment"] for item in report["experiment_summary"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
