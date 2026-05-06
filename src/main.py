import json
import uuid
from pathlib import Path
from typing import Any


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

        try:
            from pdf2image import convert_from_path

            pages = convert_from_path(str(pdf_path), first_page=1, last_page=1)
        except Exception as exc:
            raise RuntimeError(
                "Failed to convert input/form.pdf. On Windows, install Poppler and add it to PATH."
            ) from exc

        if not pages:
            raise RuntimeError("PDF conversion produced no pages.")

        pages[0].save(converted_path, "PNG")
        return converted_path

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

    try:
        from pdf2image import convert_from_path

        pages = convert_from_path(str(input_file), first_page=1, last_page=1)
    except Exception as exc:
        raise RuntimeError(
            "Failed to convert uploaded PDF. On Windows, install Poppler and add it to PATH."
        ) from exc

    if not pages:
        raise RuntimeError("PDF conversion produced no pages.")

    pages[0].save(converted_path, "PNG")
    return converted_path


def run_pipeline(image_path: Path, output_dir: Path) -> dict[str, Any]:
    from detect_fields import detect_lines, draw_lines, filter_field_lines
    from mapping import draw_mapping, map_labels_to_fields
    from ocr import extract_text
    from pdf_generator import create_pdf_with_fields
    from utils import get_center

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path_str = str(image_path)

    data = extract_text(image_path_str)
    result = []

    for item in data:
        text = item["text"]
        bbox = item["bbox"]
        center = get_center(bbox)
        result.append({"text": text, "bbox": bbox, "center": center})

    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = detect_lines(image_path_str)
    filtered_lines = filter_field_lines(lines, result)
    lines_output_path = output_dir / "lines_detected.png"
    draw_lines(image_path_str, filtered_lines, str(lines_output_path))

    mappings = map_labels_to_fields(result, filtered_lines)
    mappings_path = output_dir / "mappings.json"
    with mappings_path.open("w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    mapping_image_path = output_dir / "mapping.png"
    draw_mapping(image_path_str, mappings, str(mapping_image_path))

    pdf_output_path = output_dir / "output.pdf"
    create_pdf_with_fields(image_path_str, mappings, str(pdf_output_path))

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
        field_line = mapping["field_line"]
        print(f'{mapping["label"]} -> {field_line}')

    return output


if __name__ == "__main__":
    run_default_pipeline()
