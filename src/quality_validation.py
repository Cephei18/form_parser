from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compare_mapping_summaries, summarize_mappings


@dataclass(frozen=True)
class CorpusSample:
    sample_id: str
    artifact_dir: Path
    source_image: Path | None = None
    notes: str = ""


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _normalize_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sample_from_manifest_entry(project_root: Path, entry: dict[str, Any]) -> CorpusSample:
    return CorpusSample(
        sample_id=str(entry["id"]),
        artifact_dir=_normalize_path(project_root, str(entry["artifact_dir"])),
        source_image=_normalize_path(project_root, str(entry["source_image"])) if entry.get("source_image") else None,
        notes=str(entry.get("notes", "")),
    )


def _find_source_image(artifact_dir: Path, explicit_source: Path | None = None) -> Path:
    if explicit_source is not None and explicit_source.exists():
        return explicit_source

    candidates = sorted(artifact_dir.glob("converted_*.png"))
    if candidates:
        return candidates[0]

    fallback = artifact_dir / "mapping.png"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"No source image found for corpus sample in {artifact_dir}")


def _ocr_overlay(image_path: Path, result_items: list[dict[str, Any]], output_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    for index, item in enumerate(result_items):
        bbox = item.get("bbox")
        if not bbox:
            continue
        try:
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
        except Exception:
            continue
        x1, y1, x2, y2 = int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))
        color = (110, 170, 255) if int(item.get("source_item_count", 1) or 1) == 1 else (0, 200, 180)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            image,
            f"{index}:{str(item.get('text', ''))[:32]}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def _first_existing_path(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _extract_labels(result_items: list[dict[str, Any]], limit: int = 8) -> list[str]:
    labels = []
    for item in result_items:
        text = str(item.get("text", "")).strip()
        if text:
            labels.append(text)
        if len(labels) >= limit:
            break
    return labels


def _extract_region_types(diagnostics: dict[str, Any], layout_structure: dict[str, Any] | None = None) -> list[str]:
    region_types = []
    for region in diagnostics.get("semantic_regions", []) or []:
        region_type = str(region.get("type", "")).strip()
        if region_type and region_type not in region_types:
            region_types.append(region_type)

    if isinstance(layout_structure, dict):
        for cluster in layout_structure.get("field_clusters", []) or []:
            cluster_type = str(cluster.get("type", "")).strip()
            if cluster_type and cluster_type not in region_types:
                region_types.append(cluster_type)
        for band in layout_structure.get("layout_bands", []) or []:
            band_type = str(band.get("type", "")).strip()
            if band_type and band_type not in region_types:
                region_types.append(band_type)
        for zone in layout_structure.get("ownership_regions", []) or []:
            zone_type = str(zone.get("type", "")).strip()
            if zone_type and zone_type not in region_types:
                region_types.append(zone_type)
    return region_types


def _extract_multiline_behavior(summary: dict[str, Any], diagnostics: dict[str, Any], layout_structure: dict[str, Any] | None = None) -> dict[str, Any]:
    region_types = _extract_region_types(diagnostics, layout_structure)
    multiline_regions = [region for region in region_types if "multiline" in region]
    return {
        "observed_multiline_mappings": int(summary.get("multiline_mapping_count", 0)),
        "observed_multiline_regions": multiline_regions,
        "expected_multiline_present": bool(multiline_regions or summary.get("multiline_mapping_count", 0)),
    }


def _build_sample_summary(sample: CorpusSample, project_root: Path, baseline_summary: dict[str, Any] | None) -> dict[str, Any]:
    artifact_dir = sample.artifact_dir
    result_path = artifact_dir / "result.json"
    mappings_path = artifact_dir / "mappings.json"
    diagnostics_path = artifact_dir / "mapping_diagnostics.json"
    layout_path = artifact_dir / "layout_structure.json"
    benchmark_path = _first_existing_path(artifact_dir / "benchmark_summary.json", project_root / "output" / "benchmark_summary.json")

    result_items = _load_json(result_path, []) or []
    mappings = _load_json(mappings_path, []) or []
    diagnostics = _load_json(diagnostics_path, {}) or {}
    layout_structure = _load_json(layout_path, {}) or {}
    benchmark_summary = _load_json(benchmark_path, None)

    if not isinstance(result_items, list):
        result_items = []
    if not isinstance(mappings, list):
        mappings = []
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if not isinstance(layout_structure, dict):
        layout_structure = {}
    if not isinstance(benchmark_summary, dict):
        benchmark_summary = None

    benchmark = benchmark_summary or summarize_mappings(mappings, result_items)
    benchmark.setdefault("mapping_count", len(mappings))

    layout_metrics = layout_structure.get("metrics", {}) if isinstance(layout_structure.get("metrics", {}), dict) else {}
    ocr_item_count = int(layout_metrics.get("ocr_count", len(result_items)))
    field_candidate_count = int(layout_metrics.get("field_line_count", diagnostics.get("field_candidate_count", 0)))
    merged_ocr_count = sum(1 for item in result_items if int(item.get("source_item_count", 1) or 1) > 1)
    source_items_merged = sum(int(item.get("source_item_count", 1) or 1) for item in result_items if int(item.get("source_item_count", 1) or 1) > 1)

    confidence_classes = benchmark.get("confidence_classes", {}) if isinstance(benchmark.get("confidence_classes", {}), dict) else {}
    strong_mapping_count = int(benchmark.get("strong_match_count", confidence_classes.get("strong_match", 0)))
    ambiguous_mapping_count = int(benchmark.get("ambiguous_mapping_count", confidence_classes.get("ambiguous", 0)))
    weak_mapping_count = int(benchmark.get("weak_match_count", confidence_classes.get("weak_match", 0)))
    unresolved_mapping_count = int(benchmark.get("unresolved_mapping_count", confidence_classes.get("unresolved", 0)))
    multiline_mapping_count = int(benchmark.get("multiline_mapping_count", 0))
    average_mapping_confidence = float(benchmark.get("average_candidate_score", 0.0))

    mapped_count = int(benchmark.get("mapping_count", len(mappings)))
    precision_proxy = round(strong_mapping_count / mapped_count, 4) if mapped_count else 0.0
    recall_proxy = round(mapped_count / ocr_item_count, 4) if ocr_item_count else 0.0
    false_positive_proxy = ambiguous_mapping_count + weak_mapping_count

    region_types = _extract_region_types(diagnostics, layout_structure)
    unresolved_labels = list(diagnostics.get("unresolved_labels", [])) if isinstance(diagnostics.get("unresolved_labels", []), list) else []
    page_height = float(layout_metrics.get("page_height", 0.0) or 0.0)
    top_band = page_height * 0.3 if page_height else 0.0
    top_band_unresolved = [
        item.get("text")
        for item in result_items
        if item.get("text") in unresolved_labels and isinstance(item.get("center"), list) and len(item["center"]) == 2 and (not top_band or float(item["center"][1]) <= top_band)
    ]

    comparison = None
    if baseline_summary is not None:
        comparison = compare_mapping_summaries(baseline_summary, benchmark)
        comparison["precision_proxy_delta"] = round(precision_proxy - float(baseline_summary.get("precision_proxy", 0.0)), 4)
        comparison["recall_proxy_delta"] = round(recall_proxy - float(baseline_summary.get("recall_proxy", 0.0)), 4)

    failure_categories = classify_failure_categories(
        {
            "ocr_item_count": ocr_item_count,
            "merged_ocr_count": merged_ocr_count,
            "source_items_merged": source_items_merged,
            "field_candidate_count": field_candidate_count,
            "mapped_count": mapped_count,
            "strong_mapping_count": strong_mapping_count,
            "ambiguous_mapping_count": ambiguous_mapping_count,
            "weak_mapping_count": weak_mapping_count,
            "unresolved_mapping_count": unresolved_mapping_count,
            "multiline_mapping_count": multiline_mapping_count,
            "average_mapping_confidence": average_mapping_confidence,
            "false_positive_proxy": false_positive_proxy,
            "precision_proxy": precision_proxy,
            "recall_proxy": recall_proxy,
            "region_types": region_types,
            "unresolved_labels": unresolved_labels,
            "top_band_unresolved": top_band_unresolved,
            "diagnostics": diagnostics,
            "layout_metrics": layout_metrics,
        },
        baseline_summary=baseline_summary,
    )

    ocr_overlay_path = artifact_dir / "ocr_overlay.png"
    source_image = _find_source_image(artifact_dir, sample.source_image)
    _ocr_overlay(source_image, result_items, ocr_overlay_path)

    benchmark.setdefault("precision_proxy", precision_proxy)
    benchmark.setdefault("recall_proxy", recall_proxy)
    benchmark.setdefault("false_positive_proxy", false_positive_proxy)

    benchmark_for_compare = {
        "mapping_count": mapped_count,
        "average_candidate_score": average_mapping_confidence,
        "unmatched_label_count": len(unresolved_labels),
        "strong_match_count": strong_mapping_count,
        "weak_match_count": weak_mapping_count,
        "ambiguous_mapping_count": ambiguous_mapping_count,
        "unresolved_mapping_count": unresolved_mapping_count,
    }

    return {
        "sample_id": sample.sample_id,
        "artifact_dir": str(artifact_dir),
        "source_image": str(source_image),
        "ocr_overlay_path": str(ocr_overlay_path),
        "candidate_field_overlay_path": str(_first_existing_path(artifact_dir / "lines_detected.png", artifact_dir / "mapping.png") or artifact_dir / "mapping.png"),
        "ownership_overlay_path": str(_first_existing_path(artifact_dir / "debug_reasoning.png", artifact_dir / "mapping.png") or artifact_dir / "mapping.png"),
        "mapping_overlay_path": str(_first_existing_path(artifact_dir / "mapping.png", artifact_dir / "debug_reasoning.png") or artifact_dir / "mapping.png"),
        "expected_major_labels": _extract_labels(result_items),
        "expected_field_regions": region_types,
        "expected_multiline_behavior": _extract_multiline_behavior(benchmark, diagnostics, layout_structure),
        "summary": {
            "ocr_item_count": ocr_item_count,
            "merged_ocr_count": merged_ocr_count,
            "source_items_merged": source_items_merged,
            "field_candidate_count": field_candidate_count,
            "mapping_count": mapped_count,
            "strong_mapping_count": strong_mapping_count,
            "ambiguous_mapping_count": ambiguous_mapping_count,
            "weak_mapping_count": weak_mapping_count,
            "unresolved_mapping_count": unresolved_mapping_count,
            "multiline_group_count": multiline_mapping_count,
            "false_positive_count": false_positive_proxy,
            "average_mapping_confidence": average_mapping_confidence,
            "precision_proxy": precision_proxy,
            "recall_proxy": recall_proxy,
            "region_types": region_types,
            "unresolved_labels": unresolved_labels,
            "top_band_unresolved": top_band_unresolved,
        },
        "comparison": comparison,
        "failure_categories": failure_categories,
        "notes": sample.notes,
        "benchmark_for_compare": benchmark_for_compare,
    }


def classify_failure_categories(summary: dict[str, Any], baseline_summary: dict[str, Any] | None = None) -> list[str]:
    categories: list[str] = []
    region_types = set(summary.get("region_types", []))
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary.get("diagnostics", {}), dict) else {}
    layout_metrics = summary.get("layout_metrics", {}) if isinstance(summary.get("layout_metrics", {}), dict) else {}
    unresolved_labels = summary.get("unresolved_labels", []) or []
    top_band_unresolved = summary.get("top_band_unresolved", []) or []

    ocr_item_count = int(summary.get("ocr_item_count", 0))
    merged_ocr_count = int(summary.get("merged_ocr_count", 0))
    source_items_merged = int(summary.get("source_items_merged", 0))
    field_candidate_count = int(summary.get("field_candidate_count", 0))
    mapped_count = int(summary.get("mapped_count", 0))
    strong_mapping_count = int(summary.get("strong_mapping_count", 0))
    ambiguous_mapping_count = int(summary.get("ambiguous_mapping_count", 0))
    weak_mapping_count = int(summary.get("weak_mapping_count", 0))
    unresolved_mapping_count = int(summary.get("unresolved_mapping_count", 0))
    multiline_mapping_count = int(summary.get("multiline_mapping_count", 0))
    average_mapping_confidence = float(summary.get("average_mapping_confidence", 0.0))
    precision_proxy = float(summary.get("precision_proxy", 0.0))
    recall_proxy = float(summary.get("recall_proxy", 0.0))
    false_positive_proxy = int(summary.get("false_positive_proxy", 0))

    if merged_ocr_count > 0 and source_items_merged >= max(merged_ocr_count * 2, 4):
        categories.append("token fragmentation")

    if multiline_mapping_count == 0 and any("multiline" in region for region in region_types):
        categories.append("multiline splitting")

    if average_mapping_confidence < 0.5 or (ambiguous_mapping_count >= max(1, strong_mapping_count * 2) and precision_proxy < 0.35):
        categories.append("confidence-threshold issues")

    if field_candidate_count >= 70 and mapped_count < max(ocr_item_count, 1) and false_positive_proxy >= weak_mapping_count:
        categories.append("false field-line detection")

    if any(region in {"table_like_region", "checkbox_region"} for region in region_types) and ambiguous_mapping_count >= max(3, mapped_count // 2):
        categories.append("graph segmentation weakness")

    if top_band_unresolved and any(len(label or "") <= 24 for label in top_band_unresolved):
        categories.append("header/title contamination")

    if unresolved_labels and precision_proxy < 0.4 and recall_proxy < 0.85:
        categories.append("over-aggressive filtering")

    if diagnostics.get("excluded_region_count", 0) and field_candidate_count > 0 and mapped_count < field_candidate_count * 0.35:
        categories.append("sparse-band exclusion error")

    if baseline_summary is not None:
        baseline_recall = float(baseline_summary.get("recall_proxy", 0.0))
        baseline_precision = float(baseline_summary.get("precision_proxy", 0.0))
        if recall_proxy + 0.08 < baseline_recall:
            categories.append("over-aggressive filtering")
        if precision_proxy + 0.05 < baseline_precision and ambiguous_mapping_count > 0:
            categories.append("precision/recall trade-off regression")

    return list(dict.fromkeys(categories))


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Quality Validation Report", ""]
    lines.append(f"Baseline sample: `{report.get('baseline_sample_id', 'n/a')}`")
    lines.append("")
    lines.append("## Corpus Summary")
    lines.append("")
    lines.append("| Sample | Mappings | Strong | Ambiguous | Recall Proxy | Precision Proxy | Avg Confidence | Categories |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for sample in report.get("samples", []):
        summary = sample.get("summary", {})
        lines.append(
            "| {sample_id} | {mapping_count} | {strong} | {ambiguous} | {recall:.3f} | {precision:.3f} | {confidence:.3f} | {categories} |".format(
                sample_id=sample.get("sample_id", "n/a"),
                mapping_count=summary.get("mapping_count", 0),
                strong=summary.get("strong_mapping_count", 0),
                ambiguous=summary.get("ambiguous_mapping_count", 0),
                recall=float(summary.get("recall_proxy", 0.0) or 0.0),
                precision=float(summary.get("precision_proxy", 0.0) or 0.0),
                confidence=float(summary.get("average_mapping_confidence", 0.0) or 0.0),
                categories=", ".join(sample.get("failure_categories", [])) or "none",
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Recall Proxy` = mappings / OCR items.")
    lines.append("- `Precision Proxy` = strong mappings / mappings.")
    lines.append("- `False Positive Count` is the conservative proxy `ambiguous + weak`.")
    lines.append("")
    return "\n".join(lines)


def _load_manifest(manifest_path: Path, project_root: Path) -> tuple[str, str, list[CorpusSample], str | None]:
    manifest = _load_json(manifest_path, {}) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid corpus manifest: {manifest_path}")

    samples = [
        _sample_from_manifest_entry(project_root, entry)
        for entry in manifest.get("samples", [])
        if isinstance(entry, dict) and entry.get("id") and entry.get("artifact_dir")
    ]
    return (
        str(manifest.get("name", "quality-corpus")),
        str(manifest.get("description", "")),
        samples,
        str(manifest.get("baseline_sample_id")) if manifest.get("baseline_sample_id") else None,
    )


def run_quality_validation(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    corpus_name, corpus_description, samples, baseline_sample_id = _load_manifest(manifest_path, project_root)
    baseline_summary = None
    sample_reports: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_output_dir = output_dir / "samples"
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        sample_report = _build_sample_summary(sample, project_root, baseline_summary)
        if sample.sample_id == baseline_sample_id:
            baseline_summary = sample_report["benchmark_for_compare"]
            sample_report["comparison"] = None
            sample_report["failure_categories"] = classify_failure_categories(sample_report["summary"])
        sample_reports.append(sample_report)
        _write_json(sample_output_dir / f"{sample.sample_id}.json", sample_report)

    if baseline_summary is None and sample_reports:
        baseline_summary = sample_reports[0]["benchmark_for_compare"]

    if baseline_summary is not None:
        for sample_report in sample_reports:
            if sample_report.get("comparison") is None and sample_report.get("sample_id") != baseline_sample_id:
                sample_report["comparison"] = compare_mapping_summaries(baseline_summary, sample_report["benchmark_for_compare"])

    corpus_summary = {
        "sample_count": len(sample_reports),
        "average_mapping_confidence": round(mean([float(sample["summary"]["average_mapping_confidence"]) for sample in sample_reports]) if sample_reports else 0.0, 4),
        "average_recall_proxy": round(mean([float(sample["summary"]["recall_proxy"]) for sample in sample_reports]) if sample_reports else 0.0, 4),
        "average_precision_proxy": round(mean([float(sample["summary"]["precision_proxy"]) for sample in sample_reports]) if sample_reports else 0.0, 4),
        "category_counts": {},
    }
    category_counts: dict[str, int] = {}
    for sample_report in sample_reports:
        for category in sample_report.get("failure_categories", []):
            category_counts[category] = category_counts.get(category, 0) + 1
    corpus_summary["category_counts"] = dict(sorted(category_counts.items(), key=lambda item: (-item[1], item[0])))

    report = {
        "name": corpus_name,
        "description": corpus_description,
        "baseline_sample_id": baseline_sample_id,
        "summary": corpus_summary,
        "samples": sample_reports,
    }

    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a multi-form quality validation report from existing run artifacts.")
    parser.add_argument("--manifest", default=str(project_root / "benchmarks" / "quality_corpus" / "manifest.json"))
    parser.add_argument("--output-dir", default=str(project_root / "output" / "quality_validation"))
    args = parser.parse_args()

    report = run_quality_validation(Path(args.manifest), Path(args.output_dir))
    print(json.dumps({"report_path": str(Path(args.output_dir) / "report.json"), "sample_count": report["summary"]["sample_count"]}, indent=2))


if __name__ == "__main__":
    main()