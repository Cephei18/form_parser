import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _safe_field_name(label: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip())
    if not cleaned:
        cleaned = "field"
    return f"{cleaned}_{index}"


def _field_boxes_from_mapping(mapping):
    boxes = mapping.get("field_bboxes")
    if boxes:
        return boxes

    fallback_boxes = []
    for line in mapping.get("field_lines", []):
        x1, y1 = line["start"]
        x2, y2 = line["end"]
        fallback_boxes.append({
            "x": min(x1, x2),
            "y": max(0, min(y1, y2) - 4),
            "width": abs(x2 - x1),
            "height": 18,
        })
    return fallback_boxes


def create_pdf_with_fields(image_path, mappings, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output), pagesize=letter)
    page_width, page_height = letter

    image_reader = ImageReader(image_path)
    image_width, image_height = image_reader.getSize()

    # Stretch background image to page and scale line coordinates accordingly.
    scale_x = page_width / float(image_width)
    scale_y = page_height / float(image_height)

    c.drawImage(image_reader, 0, 0, width=page_width, height=page_height)

    for index, mapping in enumerate(mappings, start=1):
        field_boxes = _field_boxes_from_mapping(mapping)

        for li, box in enumerate(field_boxes):
            pdf_x = float(box["x"]) * scale_x + 5
            box_y = float(box["y"])
            box_height = float(box.get("height", 18))
            pdf_y = page_height - ((box_y + box_height) * scale_y)

            width = float(box["width"]) * scale_x
            height_field = max(12, box_height * scale_y)

            field_name = _safe_field_name(mapping.get("label", "field") + ("_%d" % (li + 1)), index)

            if mapping.get("field_type") == "checkbox" or box.get("field_type") == "checkbox":
                size = max(10, min(width, height_field))
                c.acroForm.checkbox(
                    name=field_name,
                    tooltip=mapping.get("label", "Checkbox"),
                    x=pdf_x,
                    y=pdf_y,
                    size=size,
                    borderWidth=1,
                    buttonStyle="check",
                    forceBorder=True,
                )
            else:
                c.acroForm.textfield(
                    name=field_name,
                    tooltip=mapping.get("label", "Field"),
                    x=pdf_x,
                    y=pdf_y,
                    width=max(10, width - 10),
                    height=height_field,
                    borderStyle="underlined",
                    forceBorder=True,
                )

    c.save()
