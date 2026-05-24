import json
import logging
import sys
import time
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("form_parser.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _stage(message: str) -> None:
    logger.info("[pipeline] %s", message)


def _write_json(path: Path, payload: Any, label: str) -> None:
    _stage(f"{label} save start: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"{label} save failed or produced empty file: {path}")
    _stage(f"{label} save end: {path} size={path.stat().st_size}")


def _load_optional_module(module_name: str) -> Any:
    return import_module(module_name)


def _extract_ocr_payload(image_path: str) -> dict[str, Any]:
    from src.ocr import extract_text_with_diagnostics

    payload = extract_text_with_diagnostics(image_path)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {
            "raw_items": payload,
            "items": payload,
            "diagnostics": {
                "raw_item_count": len(payload),
                "cleaned_item_count": len(payload),
                "dropped_item_count": 0,
                "merged_item_count": 0,
                "source_items_merged": len(payload),
                "raw_confidence_distribution": {"available": False},
                "cleaned_confidence_distribution": {"available": False},
                "text_cleanup_change_count": 0,
                "text_cleanup_examples": [],
            },
        }
    raise RuntimeError(f"OCR returned an unsupported payload type: {type(payload).__name__}")


def convert_pdf_first_page(input_file: Path, output_path: Path) -> Path:
    _stage(f"PDF input conversion start: {input_file}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pdf2image = _load_optional_module("pdf2image")

        _stage("pdf2image conversion start")
        pages = pdf2image.convert_from_path(str(input_file), first_page=1, last_page=1)
        if not pages:
            raise RuntimeError("PDF conversion produced no pages.")
        pages[0].save(output_path, "PNG")
        _stage(f"pdf2image conversion end: {output_path}")
        return output_path
    except Exception:
        logger.exception("[pipeline] pdf2image conversion failed; trying PyMuPDF fallback")
        try:
            fitz = _load_optional_module("fitz")

            _stage("PyMuPDF conversion start")
            doc = fitz.open(str(input_file))
            try:
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=150)
                pix.save(output_path)
                _stage(f"PyMuPDF conversion end: {output_path}")
                return output_path
            finally:
                doc.close()
        except Exception as exc:
            logger.exception("[pipeline] PDF conversion failed")
            raise RuntimeError(
                "Failed to convert PDF. Install pdf2image+Poppler or ensure PyMuPDF is available."
            ) from exc


def resolve_input_image(project_root: Path) -> Path:
    input_dir = project_root / "input"
    png_candidates = [
        input_dir / "form.png",
        input_dir / "image.png",
        input_dir / "image_2.png",
        input_dir / "imange.png",
    ]
    pdf_candidates = [
        input_dir / "form.pdf",
        input_dir / "image_2.pdf",
    ]

    for png_path in png_candidates:
        if png_path.exists():
            return png_path

    for pdf_path in pdf_candidates:
        if not pdf_path.exists():
            continue

        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        converted_path = output_dir / "form_page_1.png"

        return convert_pdf_first_page(pdf_path, converted_path)

    raise FileNotFoundError(
        "No input found. Add input/form.png, input/image.png, input/form.pdf, or input/image_2.pdf"
    )


def resolve_uploaded_input(input_file: Path, output_dir: Path) -> Path:
    suffix = input_file.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        return input_file

    if suffix != ".pdf":
        raise ValueError("Unsupported file type. Only PNG, JPG, JPEG, and PDF are accepted.")

    converted_path = output_dir / f"converted_{uuid.uuid4().hex}.png"

    return convert_pdf_first_page(input_file, converted_path)


def run_pipeline(image_path: Path, output_dir: Path) -> dict[str, Any]:
    from src.detect_fields import (
        build_field_filter_thresholds,
        deduplicate_detected_lines,
        detect_additional_field_candidates,
        detect_checkbox_mappings,
        detect_lines,
        detect_semantic_regions,
        detect_table_regions as detect_image_table_regions,
        draw_lines,
        fallback_field_lines_from_ocr,
        filter_field_lines,
        filter_lines_outside_table_regions,
        table_regions_to_semantic_regions,
    )
    from src.global_layout import infer_page_structure
    from src.mapping import draw_mapping, map_labels_to_fields
    from src.debug_visualize import create_debug_overlay
    from src.evaluation import evaluate_mapping_file
    from src.pipeline_compare import compare_pipeline_runs
    from src.pipeline_config import PipelineConfig
    from src.preprocessing import preprocess_image
    from src.structural_refinement import refine_field_candidates, refine_layout_structure
    from src.pdf_generator import create_pdf_with_fields
    from src.utils import get_center

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_started = time.perf_counter()
    config = PipelineConfig.from_env()
    _stage(f"start image={image_path} output_dir={output_dir}")
    _stage(f"config={config.to_dict()}")

    _stage("preprocessing start")
    preprocessing_result = preprocess_image(image_path, output_dir, config.preprocessing)
    preprocessing_diagnostics_path = output_dir / "preprocessing_diagnostics.json"
    _write_json(preprocessing_diagnostics_path, preprocessing_result.diagnostics, "preprocessing_diagnostics.json")
    image_path_str = str(preprocessing_result.working_path)
    _stage(f"preprocessing end working_image={preprocessing_result.working_path}")

    _stage("OCR start")
    ocr_payload = _extract_ocr_payload(image_path_str)
    data = ocr_payload["items"]
    raw_ocr_data = ocr_payload["raw_items"]
    _stage(f"OCR end count={len(data or [])}")

    ocr_raw_path = output_dir / "ocr_raw.json"
    ocr_diagnostics_path = output_dir / "ocr_diagnostics.json"
    if config.ocr_diagnostics_enabled:
        _write_json(ocr_raw_path, raw_ocr_data, "ocr_raw.json")
        _write_json(ocr_diagnostics_path, ocr_payload["diagnostics"], "ocr_diagnostics.json")

    result = []

    _stage("OCR normalization start")
    for index, item in enumerate(data or []):
        try:
            text = item["text"]
            bbox = item["bbox"]
            center = get_center(bbox)
            result.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "center": center,
                    "confidence": item.get("confidence"),
                }
            )
        except Exception:
            logger.exception("[pipeline] skipping malformed OCR item index=%s", index)
    _stage(f"OCR normalization end usable={len(result)}")

    result_path = output_dir / "result.json"
    _write_json(result_path, result, "result.json")

    _stage("line detection start")
    lines = detect_lines(image_path_str)
    _stage(f"line detection end count={len(lines)}")

    _stage("additional field candidate detection start")
    additional_lines = detect_additional_field_candidates(image_path_str)
    candidate_lines = deduplicate_detected_lines(lines + additional_lines)
    _stage(
        f"additional field candidate detection end count={len(additional_lines)} total={len(candidate_lines)}"
    )

    image_table_regions = []
    if config.image_table_filtering_enabled:
        _stage("image table region detection start")
        image_table_regions = detect_image_table_regions(image_path_str)
        candidate_lines = filter_lines_outside_table_regions(candidate_lines, image_table_regions)
        _stage(
            f"image table region detection end regions={len(image_table_regions)} remaining_candidates={len(candidate_lines)}"
        )

    _stage("semantic region classification start")
    semantic_regions = detect_semantic_regions(image_path_str, result, candidate_lines)
    semantic_regions = table_regions_to_semantic_regions(image_table_regions) + semantic_regions
    excluded_regions = [
        region
        for region in semantic_regions
        if region.get("type") in {
            "non_text_sparse_region",
            "non_text_candidate",
            "photo_region",
            "signature_area",
            "table_like_region",
            "checkbox_region",
        }
    ]
    _stage(
        f"semantic region classification end regions={len(semantic_regions)} excluded={len(excluded_regions)}"
    )

    _stage("field line filtering start")
    field_threshold_diagnostics = {
        "mode": "legacy_fixed",
        "legacy_fixed_thresholds": {
            "min_length": 100,
            "x_threshold": 300,
            "max_vertical_distance": 30,
        },
    }
    field_filter_kwargs = {}
    if config.dynamic_thresholds_enabled:
        threshold_payload = build_field_filter_thresholds(image_path_str, result, candidate_lines)
        field_threshold_diagnostics = {"mode": "page_relative", **threshold_payload["diagnostics"]}
        field_filter_kwargs = {
            "min_length": threshold_payload["min_length"],
            "x_threshold": threshold_payload["x_threshold"],
            "max_vertical_distance": threshold_payload["max_vertical_distance"],
        }

    filtered_lines = filter_field_lines(
        candidate_lines,
        result,
        excluded_regions=excluded_regions,
        **field_filter_kwargs,
    )
    fallback_lines = []
    if config.fallback_field_lines_enabled and not filtered_lines:
        _stage("fallback OCR-derived field candidate detection start")
        fallback_lines = fallback_field_lines_from_ocr(image_path_str, result)
        filtered_lines = filter_field_lines(
            deduplicate_detected_lines(candidate_lines + fallback_lines),
            result,
            excluded_regions=excluded_regions,
            **field_filter_kwargs,
        )
        _stage(f"fallback OCR-derived field candidate detection end count={len(fallback_lines)}")

    structural_refinement_diagnostics = {
        "enabled": config.structural_refinement.enabled,
        "field_candidate_quality": {
            "enabled": False,
            "input_count": len(filtered_lines),
            "output_count": len(filtered_lines),
            "removed_count": 0,
        },
        "layout_refinement": {
            "enabled": False,
        },
    }
    if config.structural_refinement.enabled and config.structural_refinement.field_quality_enabled:
        _stage("structural field candidate refinement start")
        field_refinement = refine_field_candidates(
            image_path_str,
            filtered_lines,
            result,
            semantic_regions,
            config.structural_refinement,
        )
        filtered_lines = field_refinement.lines
        structural_refinement_diagnostics["field_candidate_quality"] = field_refinement.diagnostics
        _stage(
            "structural field candidate refinement end input=%s output=%s removed=%s"
            % (
                field_refinement.diagnostics.get("input_count", 0),
                field_refinement.diagnostics.get("output_count", 0),
                field_refinement.diagnostics.get("removed_count", 0),
            )
        )
    _stage(f"field line filtering end count={len(filtered_lines)}")

    _stage("layout structure inference start")
    layout_structure = infer_page_structure(result, filtered_lines, semantic_regions)
    if config.structural_refinement.enabled:
        _stage("structural layout refinement start")
        layout_structure, layout_refinement_diagnostics = refine_layout_structure(
            layout_structure,
            result,
            filtered_lines,
            semantic_regions,
            config.structural_refinement,
            structural_refinement_diagnostics.get("field_candidate_quality"),
        )
        structural_refinement_diagnostics["layout_refinement"] = layout_refinement_diagnostics
        _stage(
            "structural layout refinement end sections=%s ownership_chains=%s tables=%s"
            % (
                layout_refinement_diagnostics.get("section_count", 0),
                layout_refinement_diagnostics.get("ownership_chain_count", 0),
                layout_refinement_diagnostics.get("table_structure_count", 0),
            )
        )
    _stage(
        "layout structure inference end rows=%s clusters=%s regions=%s"
        % (
            len(layout_structure.get("text_rows", [])),
            len(layout_structure.get("field_clusters", [])),
            len(semantic_regions),
        )
    )

    structural_refinement_path = output_dir / "structural_refinement_diagnostics.json"
    _write_json(
        structural_refinement_path,
        structural_refinement_diagnostics,
        "structural_refinement_diagnostics.json",
    )

    lines_output_path = output_dir / "lines_detected.png"
    _stage(f"lines image save start: {lines_output_path}")
    draw_lines(image_path_str, filtered_lines, str(lines_output_path))
    if not lines_output_path.exists() or lines_output_path.stat().st_size <= 0:
        raise RuntimeError(f"lines image save failed or produced empty file: {lines_output_path}")
    _stage(f"lines image save end: {lines_output_path} size={lines_output_path.stat().st_size}")

    _stage("mapping start")
    mappings = map_labels_to_fields(
        result,
        filtered_lines,
        layout_structure=layout_structure,
        semantic_regions=semantic_regions,
    )
    checkbox_mappings = []
    if config.checkbox_detection_enabled:
        _stage("checkbox mapping detection start")
        checkbox_mappings = detect_checkbox_mappings(image_path_str, result)
        mapped_labels = {mapping.get("label") for mapping in mappings or []}
        for checkbox_mapping in checkbox_mappings:
            if checkbox_mapping.get("label") in mapped_labels:
                continue
            mappings.append(checkbox_mapping)
            mapped_labels.add(checkbox_mapping.get("label"))
        _stage(f"checkbox mapping detection end count={len(checkbox_mappings)} total={len(mappings or [])}")
    _stage(f"mapping end count={len(mappings or [])}")

    mappings_path = output_dir / "mappings.json"
    _write_json(mappings_path, mappings, "mappings.json")

    layout_structure_path = output_dir / "layout_structure.json"
    _write_json(layout_structure_path, layout_structure, "layout_structure.json")

    diagnostics_path = output_dir / "mapping_diagnostics.json"
    ocr_source_items = sum(int(item.get("source_item_count", 1) or 1) for item in data or [])
    merged_ocr_items = sum(1 for item in data or [] if int(item.get("source_item_count", 1) or 1) > 1)
    effective_excluded_regions = layout_structure.get("excluded_regions", excluded_regions)
    diagnostics = {
        "pipeline_config": config.to_dict(),
        "preprocessing": preprocessing_result.diagnostics,
        "labels_processed": len(result),
        "mappings_selected": len(mappings or []),
        "ocr_summary": {
            "item_count": len(result),
            "merged_item_count": merged_ocr_items,
            "source_items_merged": ocr_source_items,
        },
        "ocr_diagnostics": ocr_payload["diagnostics"],
        "field_filter_thresholds": field_threshold_diagnostics,
        "field_candidate_count": len(filtered_lines),
        "fallback_field_candidate_count": len(fallback_lines),
        "image_table_region_count": len(image_table_regions),
        "checkbox_mapping_candidate_count": len(checkbox_mappings),
        "semantic_region_count": len(semantic_regions),
        "excluded_region_count": len(effective_excluded_regions or []),
        "structural_refinement": structural_refinement_diagnostics,
        "unresolved_labels": [
            item.get("text")
            for item in result
            if item.get("text") not in {mapping.get("label") for mapping in mappings or []}
        ],
        "mappings": mappings or [],
        "semantic_regions": semantic_regions,
        "image_table_regions": image_table_regions,
        "layout_structure": layout_structure,
        "artifact_paths": {
            "preprocessing_diagnostics": str(preprocessing_diagnostics_path),
            "ocr_raw": str(ocr_raw_path) if config.ocr_diagnostics_enabled else None,
            "ocr_diagnostics": str(ocr_diagnostics_path) if config.ocr_diagnostics_enabled else None,
            "structural_refinement_diagnostics": str(structural_refinement_path),
        },
    }
    _write_json(diagnostics_path, diagnostics, "mapping_diagnostics.json")

    benchmark_summary_path = output_dir / "benchmark_summary.json"
    _stage(f"benchmark summary save start: {benchmark_summary_path}")
    benchmark_summary = evaluate_mapping_file(
        mappings_path,
        result_path,
        baseline_path=output_dir / "mappings_rule.json",
    )
    _write_json(benchmark_summary_path, benchmark_summary, "benchmark_summary.json")

    mapping_image_path = output_dir / "mapping.png"
    _stage(f"mapping preview save start: {mapping_image_path}")
    draw_mapping(
        image_path_str,
        mappings,
        str(mapping_image_path),
        ocr_data=result,
        candidate_lines=filtered_lines,
        semantic_regions=semantic_regions,
        layout_structure=layout_structure,
    )
    if not mapping_image_path.exists() or mapping_image_path.stat().st_size <= 0:
        raise RuntimeError(f"mapping preview save failed or produced empty file: {mapping_image_path}")
    _stage(f"mapping preview save end: {mapping_image_path} size={mapping_image_path.stat().st_size}")

    debug_reasoning_path = None
    if config.debug_artifacts_enabled:
        debug_reasoning_path = output_dir / "debug_reasoning.png"
        _stage(f"debug overlay save start: {debug_reasoning_path}")
        create_debug_overlay(
            Path(__file__).resolve().parents[1],
            debug_reasoning_path,
            image_path=preprocessing_result.working_path,
            artifact_dir=output_dir,
        )
        if not debug_reasoning_path.exists() or debug_reasoning_path.stat().st_size <= 0:
            raise RuntimeError(f"debug overlay save failed or produced empty file: {debug_reasoning_path}")
        _stage(f"debug overlay save end: {debug_reasoning_path} size={debug_reasoning_path.stat().st_size}")
    else:
        _stage("debug overlay skipped: FORM_PARSER_DEBUG_ARTIFACTS_ENABLED=false")

    pdf_output_path = output_dir / "output.pdf"
    _stage(f"PDF generation start: {pdf_output_path}")
    try:
        create_pdf_with_fields(image_path_str, mappings or [], str(pdf_output_path))
    except Exception:
        logger.exception("[pipeline] PDF generation failed")
        raise
    _stage(f"PDF generation end: {pdf_output_path}")

    comparison_path = None
    if config.baseline_dir is not None and config.baseline_dir.exists():
        comparison_path = output_dir / "before_after_comparison.json"
        _stage(f"before/after comparison start baseline={config.baseline_dir}")
        compare_pipeline_runs(
            config.baseline_dir,
            output_dir,
            output_path=comparison_path,
            image_path=preprocessing_result.working_path,
        )
        _stage(f"before/after comparison end: {comparison_path}")

    _stage(f"end elapsed={time.perf_counter() - pipeline_started:.2f}s")

    return {
        "preprocessing_diagnostics_path": preprocessing_diagnostics_path,
        "ocr_raw_path": ocr_raw_path if config.ocr_diagnostics_enabled else None,
        "ocr_diagnostics_path": ocr_diagnostics_path if config.ocr_diagnostics_enabled else None,
        "result_path": result_path,
        "lines_output_path": lines_output_path,
        "mappings_path": mappings_path,
        "layout_structure_path": layout_structure_path,
        "structural_refinement_path": structural_refinement_path,
        "diagnostics_path": diagnostics_path,
        "mapping_image_path": mapping_image_path,
        "debug_reasoning_path": debug_reasoning_path,
        "pdf_output_path": pdf_output_path,
        "comparison_path": comparison_path,
        "lines_count": len(lines),
        "additional_lines_count": len(additional_lines),
        "image_table_regions_count": len(image_table_regions),
        "semantic_regions": semantic_regions,
        "filtered_lines_count": len(filtered_lines),
        "mappings": mappings,
    }


def run_default_pipeline() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    image_path = resolve_input_image(project_root)
    output_dir = project_root / "output"
    output = run_pipeline(image_path, output_dir)

    print(f"Saved JSON: {output['result_path']}")
    print(f"Detected {output['lines_count']} lines")
    print(f"Filtered to {output['filtered_lines_count']} candidate field lines")
    print(f"Saved lines image: {output['lines_output_path']}")
    print(f"Saved mappings: {output['mappings_path']}")
    print(f"Saved mapping image: {output['mapping_image_path']}")
    print(f"Saved fillable PDF: {output['pdf_output_path']}")

    for mapping in output["mappings"]:
        field_lines = mapping.get("field_lines", [])
        print(f'{mapping["label"]} -> {field_lines}')

    return output


if __name__ == "__main__":
    run_default_pipeline()
