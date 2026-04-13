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
        line = mapping["field_line"]

        x1, y1 = line["start"]
        x2, y2 = line["end"]

        field_x = min(x1, x2) * scale_x
        field_width = max(24.0, abs(x2 - x1) * scale_x)
        field_height = 14.0

        # Convert image-space y (top-origin) to PDF-space y (bottom-origin).
        row_y = ((y1 + y2) / 2.0) * scale_y
        field_y = max(0.0, page_height - row_y - (field_height / 2.0))

        c.acroForm.textfield(
            name=_safe_field_name(mapping.get("label", "field"), index),
            tooltip=mapping.get("label", "Field"),
            x=field_x,
            y=field_y,
            width=field_width,
            height=field_height,
            borderStyle="underlined",
            forceBorder=True,
        )

    c.save()
