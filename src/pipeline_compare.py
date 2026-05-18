from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compare_mapping_summaries, summarize_mappings
from src.ocr_diagnostics import write_ocr_comparison


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _artifact_status(run_dir: Path) -> dict[str, bool]:
    artifact_names = [
        "result.json",
        "ocr_raw.json",
        "ocr_diagnostics.json",
        "lines_detected.png",
        "layout_structure.json",
        "structural_refinement_diagnostics.json",
        "mappings.json",
        "mapping_diagnostics.json",
        "mapping.png",
        "debug_reasoning.png",
        "output.pdf",
    ]
    return {name: (run_dir / name).exists() and (run_dir / name).stat().st_size > 0 for name in artifact_names}


def _run_summary(run_dir: Path) -> dict[str, Any]:
    result_items = _load_json(run_dir / "result.json", [])
    mappings = _load_json(run_dir / "mappings.json", [])
    diagnostics = _load_json(run_dir / "mapping_diagnostics.json", {})
    layout = _load_json(run_dir / "layout_structure.json", {})
    benchmark = _load_json(run_dir / "benchmark_summary.json", None)

    if not isinstance(result_items, list):
        result_items = []
    if not isinstance(mappings, list):
        mappings = []
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if not isinstance(layout, dict):
        layout = {}

    summary = benchmark if isinstance(benchmark, dict) else summarize_mappings(mappings, result_items)
    metrics = layout.get("metrics", {}) if isinstance(layout.get("metrics", {}), dict) else {}
    confidence_classes = summary.get("confidence_classes", {}) if isinstance(summary.get("confidence_classes", {}), dict) else {}
    page_priors = layout.get("page_priors", {}) if isinstance(layout.get("page_priors", {}), dict) else {}
    structural = diagnostics.get("structural_refinement", {}) if isinstance(diagnostics.get("structural_refinement", {}), dict) else {}
    field_quality = (
        structural.get("field_candidate_quality", {})
        if isinstance(structural.get("field_candidate_quality", {}), dict)
        else {}
    )

    return {
        "path": str(run_dir),
        "artifact_status": _artifact_status(run_dir),
        "ocr_item_count": len(result_items),
        "mapping_count": len(mappings),
        "field_candidate_count": int(diagnostics.get("field_candidate_count", metrics.get("field_line_count", 0)) or 0),
        "semantic_region_count": int(diagnostics.get("semantic_region_count", metrics.get("semantic_region_count", 0)) or 0),
        "excluded_region_count": int(diagnostics.get("excluded_region_count", 0) or 0),
        "section_count": int(page_priors.get("section_count", 0) or 0),
        "ownership_chain_count": int(page_priors.get("ownership_chain_count", 0) or 0),
        "table_structure_count": int(page_priors.get("table_structure_count", 0) or 0),
        "field_quality_removed_count": int(field_quality.get("removed_count", 0) or 0),
        "average_candidate_score": float(summary.get("average_candidate_score", 0.0) or 0.0),
        "confidence_classes": confidence_classes,
        "summary": summary,
    }


def compare_pipeline_runs(
    baseline_dir: Path,
    current_dir: Path,
    output_path: Path | None = None,
    image_path: Path | None = None,
) -> dict[str, Any]:
    baseline = _run_summary(baseline_dir)
    current = _run_summary(current_dir)
    mapping_comparison = compare_mapping_summaries(baseline["summary"], current["summary"])

    payload: dict[str, Any] = {
        "baseline_dir": str(baseline_dir),
        "current_dir": str(current_dir),
        "baseline": baseline,
        "current": current,
        "comparison": {
            **mapping_comparison,
            "ocr_item_count_delta": current["ocr_item_count"] - baseline["ocr_item_count"],
            "field_candidate_count_delta": current["field_candidate_count"] - baseline["field_candidate_count"],
            "semantic_region_count_delta": current["semantic_region_count"] - baseline["semantic_region_count"],
            "excluded_region_count_delta": current["excluded_region_count"] - baseline["excluded_region_count"],
            "section_count_delta": current["section_count"] - baseline["section_count"],
            "ownership_chain_count_delta": current["ownership_chain_count"] - baseline["ownership_chain_count"],
            "table_structure_count_delta": current["table_structure_count"] - baseline["table_structure_count"],
            "field_quality_removed_count_delta": current["field_quality_removed_count"] - baseline["field_quality_removed_count"],
        },
    }

    if (baseline_dir / "result.json").exists() and (current_dir / "result.json").exists():
        comparison_dir = (output_path.parent if output_path else current_dir) / "ocr_comparison"
        ocr_payload = write_ocr_comparison(
            baseline_path=baseline_dir / "result.json",
            current_path=current_dir / "result.json",
            output_dir=comparison_dir,
            image_path=image_path,
            name="before_after_ocr",
        )
        payload["ocr_comparison"] = {
            key: ocr_payload.get(key)
            for key in ["comparison_path", "overlay_path", "baseline_summary", "current_summary"]
        }

    if output_path is not None:
        _write_json(output_path, payload)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two pipeline run artifact directories.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--current-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--image", default=None)
    args = parser.parse_args()

    payload = compare_pipeline_runs(
        baseline_dir=Path(args.baseline_dir),
        current_dir=Path(args.current_dir),
        output_path=Path(args.output) if args.output else None,
        image_path=Path(args.image) if args.image else None,
    )
    print(json.dumps(payload["comparison"], indent=2))


if __name__ == "__main__":
    main()
