import json
import logging
import sys
import time
import uuid
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


def convert_pdf_first_page(input_file: Path, output_path: Path) -> Path:
    _stage(f"PDF input conversion start: {input_file}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pdf2image import convert_from_path

        _stage("pdf2image conversion start")
        pages = convert_from_path(str(input_file), first_page=1, last_page=1)
        if not pages:
            raise RuntimeError("PDF conversion produced no pages.")
        pages[0].save(output_path, "PNG")
        _stage(f"pdf2image conversion end: {output_path}")
        return output_path
    except Exception:
        logger.exception("[pipeline] pdf2image conversion failed; trying PyMuPDF fallback")
        try:
            import fitz

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
    from src.detect_fields import detect_lines, draw_lines, filter_field_lines
    from src.mapping import draw_mapping, map_labels_to_fields
    from src.ocr import extract_text
    from src.pdf_generator import create_pdf_with_fields
    from src.utils import get_center

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path_str = str(image_path)

    pipeline_started = time.perf_counter()
    _stage(f"start image={image_path} output_dir={output_dir}")

    _stage("OCR start")
    data = extract_text(image_path_str)
    _stage(f"OCR end count={len(data or [])}")

    result = []

    _stage("OCR normalization start")
    for index, item in enumerate(data or []):
        try:
            text = item["text"]
            bbox = item["bbox"]
            center = get_center(bbox)
            result.append({"text": text, "bbox": bbox, "center": center})
        except Exception:
            logger.exception("[pipeline] skipping malformed OCR item index=%s", index)
    _stage(f"OCR normalization end usable={len(result)}")

    result_path = output_dir / "result.json"
    _write_json(result_path, result, "result.json")

    _stage("line detection start")
    lines = detect_lines(image_path_str)
    _stage(f"line detection end count={len(lines)}")

    _stage("field line filtering start")
    filtered_lines = filter_field_lines(lines, result)
    _stage(f"field line filtering end count={len(filtered_lines)}")

    lines_output_path = output_dir / "lines_detected.png"
    _stage(f"lines image save start: {lines_output_path}")
    draw_lines(image_path_str, filtered_lines, str(lines_output_path))
    if not lines_output_path.exists() or lines_output_path.stat().st_size <= 0:
        raise RuntimeError(f"lines image save failed or produced empty file: {lines_output_path}")
    _stage(f"lines image save end: {lines_output_path} size={lines_output_path.stat().st_size}")

    _stage("mapping start")
    mappings = map_labels_to_fields(result, filtered_lines)
    _stage(f"mapping end count={len(mappings or [])}")

    mappings_path = output_dir / "mappings.json"
    _write_json(mappings_path, mappings, "mappings.json")

    mapping_image_path = output_dir / "mapping.png"
    _stage(f"mapping preview save start: {mapping_image_path}")
    draw_mapping(image_path_str, mappings, str(mapping_image_path))
    if not mapping_image_path.exists() or mapping_image_path.stat().st_size <= 0:
        raise RuntimeError(f"mapping preview save failed or produced empty file: {mapping_image_path}")
    _stage(f"mapping preview save end: {mapping_image_path} size={mapping_image_path.stat().st_size}")

    pdf_output_path = output_dir / "output.pdf"
    _stage(f"PDF generation start: {pdf_output_path}")
    try:
        create_pdf_with_fields(image_path_str, mappings or [], str(pdf_output_path))
    except Exception:
        logger.exception("[pipeline] PDF generation failed")
        raise
    _stage(f"PDF generation end: {pdf_output_path}")

    _stage(f"end elapsed={time.perf_counter() - pipeline_started:.2f}s")

    return {
        "result_path": result_path,
        "lines_output_path": lines_output_path,
        "mappings_path": mappings_path,
        "mapping_image_path": mapping_image_path,
        "pdf_output_path": pdf_output_path,
        "lines_count": len(lines),
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
