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
        # Some mappings may have multiple lines (multiline fields).
        for li, line in enumerate(mapping.get("field_lines", [])):
            x1, y1 = line["start"]
            x2, y2 = line["end"]

            pdf_x = x1 * scale_x + 5
            pdf_y = page_height - (y1 * scale_y)

            width = (x2 - x1) * scale_x
            height_field = 15

            field_name = _safe_field_name(mapping.get("label", "field") + ("_%d" % (li + 1)), index)

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
