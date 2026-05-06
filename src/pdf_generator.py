import re
import logging
import faulthandler
import time
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

logger = logging.getLogger("form_parser.pdf")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


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


def _valid_number(value) -> bool:
    try:
        return float(value) == float(value)
    except (TypeError, ValueError):
        return False


def _validated_field_boxes(mapping, index: int):
    boxes = []
    for box_index, box in enumerate(_field_boxes_from_mapping(mapping), start=1):
        if not isinstance(box, dict):
            logger.warning("[pdf] skip mapping %s box %s: box is not a dict", index, box_index)
            continue

        required = ("x", "y", "width", "height")
        if any(key not in box for key in required):
            logger.warning("[pdf] skip mapping %s box %s: missing coordinate keys", index, box_index)
            continue

        if not all(_valid_number(box[key]) for key in required):
            logger.warning("[pdf] skip mapping %s box %s: non-numeric coordinates", index, box_index)
            continue

        width = float(box["width"])
        height = float(box["height"])
        if width <= 0 or height <= 0:
            logger.warning(
                "[pdf] skip mapping %s box %s: invalid size width=%s height=%s",
                index,
                box_index,
                width,
                height,
            )
            continue

        boxes.append(box)

    return boxes


def _verify_output_writable(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open("ab"):
            pass


def create_pdf_with_fields(image_path, mappings, output_path):
    started = time.perf_counter()
    output = Path(output_path)
    logger.info("[pdf] start create_pdf_with_fields image=%s output=%s mappings=%s", image_path, output, len(mappings or []))

    logger.info("[pdf] verify output path writable start")
    _verify_output_writable(output)
    logger.info("[pdf] verify output path writable end")

    logger.info("[pdf] canvas init start")
    c = canvas.Canvas(str(output), pagesize=letter)
    if c is None:
        raise RuntimeError("ReportLab canvas initialization returned None.")
    logger.info("[pdf] canvas init end")

    page_width, page_height = letter

    logger.info("[pdf] image reader init start")
    image_reader = ImageReader(image_path)
    image_width, image_height = image_reader.getSize()
    if image_width <= 0 or image_height <= 0:
        raise RuntimeError(f"Invalid source image size: {image_width}x{image_height}")
    logger.info("[pdf] image reader init end size=%sx%s", image_width, image_height)

    # Stretch background image to page and scale line coordinates accordingly.
    scale_x = page_width / float(image_width)
    scale_y = page_height / float(image_height)

    logger.info("[pdf] draw background start")
    c.drawImage(image_reader, 0, 0, width=page_width, height=page_height)
    logger.info("[pdf] draw background end")

    for index, mapping in enumerate(mappings, start=1):
        if not isinstance(mapping, dict):
            logger.warning("[pdf] skip mapping %s: mapping is not a dict", index)
            continue

        field_boxes = _validated_field_boxes(mapping, index)
        if not field_boxes:
            logger.warning("[pdf] skip mapping %s: no valid field boxes", index)
            continue

        for li, box in enumerate(field_boxes):
            pdf_x = float(box["x"]) * scale_x + 5
            box_y = float(box["y"])
            box_height = float(box.get("height", 18))
            pdf_y = page_height - ((box_y + box_height) * scale_y)

            width = float(box["width"]) * scale_x
            height_field = max(12, box_height * scale_y)

            field_name = _safe_field_name(mapping.get("label", "field") + ("_%d" % (li + 1)), index)
            logger.info("[pdf] add field start name=%s type=%s", field_name, mapping.get("field_type", "text"))

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
            logger.info("[pdf] add field end name=%s", field_name)

    logger.info("[pdf] save start output=%s", output)
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        c.save()
    except Exception:
        logger.exception("[pdf] save failed output=%s", output)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()

    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(f"PDF save finished but output is missing or empty: {output}")

    logger.info("[pdf] save end output=%s size=%s elapsed=%.2fs", output, output.stat().st_size, time.perf_counter() - started)
