import re
import logging
import faulthandler
import time
from collections import defaultdict
from pathlib import Path

from reportlab.lib.colors import Color
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


TRANSPARENT_FILL = Color(1, 1, 1, alpha=0)
FIELD_BORDER = Color(0.66, 0.69, 0.72)
CHECKBOX_BORDER = Color(0.35, 0.37, 0.39)
FIELD_TEXT = Color(0.08, 0.08, 0.08)


def _safe_field_name(label: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip())
    if not cleaned:
        cleaned = "field"
    return f"{cleaned}_{index}"


def _field_boxes_from_mapping(mapping):
    bbox = mapping.get("bbox")
    if isinstance(bbox, dict) and {"x", "y", "width", "height"}.issubset(bbox.keys()):
        try:
            x = float(bbox["x"])
            y = float(bbox["y"])
            width = float(bbox["width"])
            height = float(bbox["height"])
        except (TypeError, ValueError):
            pass
        else:
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0:
                return [{"bbox_fraction": True, "x": x, "y": y, "width": width, "height": height}]

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


def _safe_page(mapping) -> int:
    try:
        return max(1, int(mapping.get("page") or 1))
    except (TypeError, ValueError):
        return 1


def _clean_label_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalized_label_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_label_text(value).lower()).strip()


def _looks_like_photo_instruction(label: str) -> bool:
    text = _normalized_label_text(label)
    if not text:
        return False
    if text in {"photo", "photograph", "passport size photograph"}:
        return True
    if "photograph" in text:
        return True
    if "passport size" in text and ("photo" in text or "photograph" in text or "affix" in text):
        return True
    if "affix" in text and ("photo" in text or "photograph" in text or "passport" in text):
        return True
    return False


def _answer_anchor_type(mapping, box=None) -> str:
    answer_region = mapping.get("answer_region") if isinstance(mapping, dict) else None
    anchoring = mapping.get("anchoring") if isinstance(mapping, dict) else None
    for value in (
        (box or {}).get("anchor_type") if isinstance(box, dict) else None,
        (answer_region or {}).get("type") if isinstance(answer_region, dict) else None,
        (anchoring or {}).get("anchor_type") if isinstance(anchoring, dict) else None,
    ):
        if value:
            return str(value)
    return ""


def _field_type(mapping, box=None) -> str:
    for value in (
        (box or {}).get("field_type") if isinstance(box, dict) else None,
        mapping.get("field_type") if isinstance(mapping, dict) else None,
    ):
        if value:
            return str(value).lower()
    return "text"


def _is_non_fillable_region(mapping, box=None) -> bool:
    field_type = _field_type(mapping, box)
    anchor_type = _answer_anchor_type(mapping, box)
    label = mapping.get("label", "") if isinstance(mapping, dict) else ""
    return (
        field_type == "photo"
        or anchor_type == "photo_region"
        or _looks_like_photo_instruction(label)
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _multiline_line_count(mapping, height_field: float) -> int:
    value = str(mapping.get("value") or "")
    value_line_count = value.count("\n") + 1 if value.strip() else 0
    continuation_count = 0
    underline_count = 0
    anchoring = mapping.get("anchoring") if isinstance(mapping, dict) else {}
    for candidate in (anchoring or {}).get("top_candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("anchor_type") == "table_cell":
            continuation_count = max(
                continuation_count,
                int(candidate.get("continuation_cell_count") or 0) + 1,
            )
        if candidate.get("anchor_type") == "underline":
            underline_count += 1

    try:
        grouped_count = int(mapping.get("multiline_group_size") or 0)
    except (TypeError, ValueError):
        grouped_count = 0

    if value_line_count > 1:
        return max(2, min(value_line_count, 6))
    if continuation_count:
        return max(2, min(continuation_count, 5))
    if underline_count >= 2:
        return max(2, min(underline_count, 5))
    if grouped_count:
        return max(2, min(grouped_count, 4))
    return max(2, min(int(round(height_field / 14.0)), 4))


def _text_widget_rect(mapping, box, pdf_x: float, pdf_y: float, width: float, height_field: float):
    field_type = _field_type(mapping, box)
    anchor_type = _answer_anchor_type(mapping, box)
    width = max(10.0, width)
    height_field = max(8.0, height_field)
    x_pad = 1.25
    widget_x = pdf_x + x_pad
    widget_width = max(10.0, width - x_pad * 2.0)

    if field_type == "multiline":
        line_count = _multiline_line_count(mapping, height_field)
        line_height = _clamp(height_field / max(line_count, 1), 10.0, 15.0)
        target_height = min(height_field, max(18.0, line_count * line_height + 3.0))
        # Align tall multiline widgets to the top of the detected writable band
        # so they start where handwriting would naturally begin.
        widget_y = pdf_y + max(0.0, height_field - target_height)
        return widget_x, widget_y, widget_width, target_height

    if anchor_type == "underline" or height_field <= 13.5:
        target_height = _clamp(height_field, 10.0, 13.5)
        return widget_x, pdf_y, widget_width, target_height

    target_height = min(height_field, 15.5)
    widget_y = pdf_y + max(0.0, (height_field - target_height) * 0.45)
    return widget_x, widget_y, widget_width, max(10.0, target_height)


def _text_widget_style(mapping, box, height_field: float) -> tuple[str, float]:
    # Textract-only hint: the answer region sits on an already-printed line/box/
    # cell, so drawing our own border would stack a second line over it. Render
    # borderless (border width 0). OCR mappings never set this key, so their
    # rendering is unchanged.
    if isinstance(mapping, dict) and mapping.get("render_border") is False and _field_type(mapping, box) != "checkbox":
        return "underlined", 0.0
    field_type = _field_type(mapping, box)
    anchor_type = _answer_anchor_type(mapping, box)
    if field_type == "multiline" and height_field > 24:
        return "solid", 0.25
    if anchor_type in {"rectangle", "table_cell", "value_block"} and height_field > 18:
        return "solid", 0.25
    return "underlined", 0.35


def _font_size_for_height(field_type: str, height: float) -> float:
    if field_type == "multiline":
        return _clamp(height / 5.0, 7.0, 9.0)
    return _clamp(height - 3.0, 7.5, 9.5)


def _checkbox_widget_rect(pdf_x: float, pdf_y: float, width: float, height_field: float):
    size = _clamp(min(width, height_field), 7.5, 11.0)
    x = pdf_x + max(0.0, (width - size) / 2.0)
    y = pdf_y + max(0.0, (height_field - size) / 2.0)
    return x, y, size


def _fraction_intersection_ratio(box, label_box) -> float:
    try:
        x1 = max(float(box["x"]), float(label_box["x"]))
        y1 = max(float(box["y"]), float(label_box["y"]))
        x2 = min(float(box["x"]) + float(box["width"]), float(label_box["x"]) + float(label_box["width"]))
        y2 = min(float(box["y"]) + float(box["height"]), float(label_box["y"]) + float(label_box["height"]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = max(float(box["width"]) * float(box["height"]), 0.000001)
    except (TypeError, ValueError, KeyError):
        return 0.0
    return intersection / area


def _adjust_fraction_box_away_from_label(mapping, box):
    if not box.get("bbox_fraction"):
        return box
    if mapping.get("field_type") == "photo" or box.get("field_type") == "photo":
        return box

    label_box = mapping.get("label_bbox")
    if not isinstance(label_box, dict):
        return box
    if _fraction_intersection_ratio(box, label_box) <= 0.03:
        return box

    adjusted = dict(box)
    try:
        right = float(adjusted["x"]) + float(adjusted["width"])
        label_right = float(label_box["x"]) + float(label_box["width"])
        label_bottom = float(label_box["y"]) + float(label_box["height"])
        x_gap = max(float(label_box["height"]) * 0.55, 0.006)
        new_x = min(0.99, label_right + x_gap)
    except (TypeError, ValueError, KeyError):
        return box

    if new_x < right and right - new_x >= 0.035:
        adjusted["x"] = new_x
        adjusted["width"] = right - new_x
        return adjusted

    try:
        adjusted["y"] = min(0.98, max(float(adjusted["y"]), label_bottom + 0.004))
    except (TypeError, ValueError, KeyError):
        return box
    return adjusted


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

    mappings_by_page = defaultdict(list)
    for index, mapping in enumerate(mappings or [], start=1):
        if not isinstance(mapping, dict):
            logger.warning("[pdf] skip mapping %s: mapping is not a dict", index)
            continue
        mappings_by_page[_safe_page(mapping)].append((index, mapping))

    page_numbers = sorted(mappings_by_page) or [1]
    for page_position, page_number in enumerate(page_numbers, start=1):
        logger.info("[pdf] draw background start page=%s", page_number)
        c.drawImage(image_reader, 0, 0, width=page_width, height=page_height)
        logger.info("[pdf] draw background end page=%s", page_number)

        for index, mapping in mappings_by_page.get(page_number, []):
            if _is_non_fillable_region(mapping):
                logger.info(
                    "[pdf] skip non-fillable region page=%s label=%r type=%s",
                    page_number,
                    mapping.get("label", ""),
                    mapping.get("field_type", ""),
                )
                continue

            field_boxes = _validated_field_boxes(mapping, index)
            if not field_boxes:
                logger.warning("[pdf] skip mapping %s: no valid field boxes", index)
                continue

            for li, original_box in enumerate(field_boxes):
                if _is_non_fillable_region(mapping, original_box):
                    logger.info("[pdf] skip non-fillable box page=%s label=%r", page_number, mapping.get("label", ""))
                    continue

                box = _adjust_fraction_box_away_from_label(mapping, original_box)
                is_fraction_box = bool(box.get("bbox_fraction"))
                if is_fraction_box:
                    pdf_x = float(box["x"]) * page_width
                    pdf_y = page_height - ((float(box["y"]) + float(box["height"])) * page_height)
                    width = float(box["width"]) * page_width
                    height_field = max(9, float(box["height"]) * page_height)
                else:
                    pdf_x = float(box["x"]) * scale_x
                    box_y = float(box["y"])
                    box_height = float(box.get("height", 18))
                    pdf_y = page_height - ((box_y + box_height) * scale_y)
                    width = float(box["width"]) * scale_x
                    height_field = max(9, box_height * scale_y)

                field_name = _safe_field_name(mapping.get("label", "field") + ("_%d" % (li + 1)), index)
                logger.info("[pdf] add field start page=%s name=%s type=%s", page_number, field_name, mapping.get("field_type", "text"))

                field_type = _field_type(mapping, box)
                if field_type == "checkbox":
                    checkbox_x, checkbox_y, size = _checkbox_widget_rect(pdf_x, pdf_y, width, height_field)
                    c.acroForm.checkbox(
                        name=field_name,
                        tooltip=mapping.get("label", "Checkbox"),
                        x=checkbox_x,
                        y=checkbox_y,
                        size=size,
                        fillColor=TRANSPARENT_FILL,
                        borderColor=CHECKBOX_BORDER,
                        textColor=FIELD_TEXT,
                        borderWidth=0.6,
                        buttonStyle="check",
                        fieldFlags="",
                        forceBorder=False,
                    )
                else:
                    widget_x, widget_y, widget_width, widget_height = _text_widget_rect(
                        mapping,
                        box,
                        pdf_x,
                        pdf_y,
                        width,
                        height_field,
                    )
                    border_style, border_width = _text_widget_style(mapping, box, widget_height)
                    c.acroForm.textfield(
                        name=field_name,
                        tooltip=mapping.get("label", "Field"),
                        x=widget_x,
                        y=widget_y,
                        width=widget_width,
                        height=widget_height,
                        fillColor=TRANSPARENT_FILL,
                        borderColor=FIELD_BORDER,
                        textColor=FIELD_TEXT,
                        borderWidth=border_width,
                        borderStyle=border_style,
                        fieldFlags="multiline" if field_type == "multiline" else "",
                        forceBorder=False,
                        fontSize=_font_size_for_height(field_type, widget_height),
                        maxlen=0 if field_type == "multiline" else 100,
                    )
                logger.info("[pdf] add field end name=%s", field_name)

        if page_position < len(page_numbers):
            c.showPage()

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
