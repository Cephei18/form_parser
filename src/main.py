import json
from pathlib import Path

from pdf2image import convert_from_path

from detect_fields import detect_lines, draw_lines, filter_field_lines
from mapping import draw_mapping, map_labels_to_fields
from ocr import extract_text
from pdf_generator import create_pdf_with_fields
from utils import get_center


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


project_root = Path(__file__).resolve().parents[1]
image_path = str(resolve_input_image(project_root))
output_dir = project_root / "output"
output_dir.mkdir(parents=True, exist_ok=True)

data = extract_text(image_path)
result = []

for item in data:
    text = item["text"]
    bbox = item["bbox"]
    center = get_center(bbox)
    result.append({"text": text, "bbox": bbox, "center": center})

    print(f"{text} -> {center}")

result_path = output_dir / "result.json"
with result_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Saved JSON: {result_path}")

lines = detect_lines(image_path)
filtered_lines = filter_field_lines(lines, result)
lines_output_path = output_dir / "lines_detected.png"
draw_lines(image_path, filtered_lines, str(lines_output_path))

mappings = map_labels_to_fields(result, filtered_lines)
mappings_path = output_dir / "mappings.json"
with mappings_path.open("w", encoding="utf-8") as f:
    json.dump(mappings, f, ensure_ascii=False, indent=2)

mapping_image_path = output_dir / "mapping.png"
draw_mapping(image_path, mappings, str(mapping_image_path))

pdf_output_path = output_dir / "output.pdf"
create_pdf_with_fields(image_path, mappings, str(pdf_output_path))

print(f"Detected {len(lines)} lines")
print(f"Filtered to {len(filtered_lines)} candidate field lines")
print(f"Saved lines image: {lines_output_path}")
print(f"Saved mappings: {mappings_path}")
print(f"Saved mapping image: {mapping_image_path}")
print(f"Saved fillable PDF: {pdf_output_path}")

for mapping in mappings:
    field_line = mapping["field_line"]
    print(f'{mapping["label"]} -> {field_line}')
