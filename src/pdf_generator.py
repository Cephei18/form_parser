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

from src.widget_model import (
    WidgetRenderContext,
    WidgetRendererRegistry,
    declared_widget_type,
    field_type_for_widget_type,
    widget_registry_enabled,
)

logger = logging.getLogger("form_parser.pdf")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


TRANSPARENT_FILL = Color(1, 1, 1, alpha=0)
CHECKBOX_FILL = Color(1, 1, 1)
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


def _field_tooltip(mapping, default: str) -> str:
    """Prefer the section-qualified label ("Applicant Details › Name") for the
    widget tooltip so repeated labels are distinguishable to the end user. Falls
    back to the raw label, keeping legacy/OCR mappings (no qualified_label)
    unchanged."""
    if isinstance(mapping, dict):
        qualified = mapping.get("qualified_label")
        if qualified:
            return str(qualified)
        label = mapping.get("label")
        if label:
            return str(label)
    return default


def _checkbox_is_checked(mapping) -> bool:
    if not isinstance(mapping, dict):
        return False

    for key in ("checked", "is_checked", "is_selected", "selected"):
        value = mapping.get(key)
        if isinstance(value, bool):
            return value

    value = str(mapping.get("value") or "").strip().lower()
    return value in {"[x]", "x", "yes", "true", "1", "selected", "checked", "on"}


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


def _mapping_with_field_type(mapping, field_type: str):
    updated = dict(mapping)
    updated["field_type"] = field_type
    return updated


def _render_checkbox_field(c, mapping, box, context: WidgetRenderContext) -> None:
    checkbox_x, checkbox_y, size = _checkbox_widget_rect(
        context.pdf_x,
        context.pdf_y,
        context.width,
        context.height,
    )
    c.acroForm.checkbox(
        checked=_checkbox_is_checked(mapping),
        name=context.field_name,
        tooltip=_field_tooltip(mapping, "Checkbox"),
        x=checkbox_x,
        y=checkbox_y,
        size=size,
        fillColor=CHECKBOX_FILL,
        borderColor=CHECKBOX_BORDER,
        textColor=FIELD_TEXT,
        borderWidth=0.6,
        buttonStyle="check",
        fieldFlags="",
        forceBorder=False,
    )


def _safe_radio_value(value: object, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return cleaned or fallback


def _downgrade_singleton_radio_groups(mappings):
    """ReportLab requires every radio group to contain at least two buttons.

    Phase-D review filtering can remove one option from an otherwise valid
    radio group before rendering. Keep the evidence in mappings, but render the
    surviving singleton as a legacy checkbox so PDF generation cannot fail.
    """
    items = list(mappings or [])
    group_counts: dict[str, int] = defaultdict(int)
    group_keys: dict[int, str] = {}
    for index, mapping in enumerate(items):
        if not isinstance(mapping, dict):
            continue
        if declared_widget_type(mapping) != "radio":
            continue
        key = str(mapping.get("radio_group") or f"__radio_singleton_{index}")
        group_keys[index] = key
        group_counts[key] += 1

    if not group_counts:
        return items

    normalized = []
    for index, mapping in enumerate(items):
        if not isinstance(mapping, dict) or index not in group_keys or group_counts[group_keys[index]] >= 2:
            normalized.append(mapping)
            continue
        updated = dict(mapping)
        updated.pop("widget_type", None)
        updated["field_type"] = "checkbox"
        updated["radio_render_fallback"] = "singleton_radio_group"
        normalized.append(updated)
        logger.warning(
            "[pdf] radio group %r has fewer than 2 renderable options; rendering label=%r as checkbox",
            mapping.get("radio_group"),
            mapping.get("label"),
        )
    return normalized


def _radio_is_selected(mapping) -> bool:
    value = mapping.get("radio_selected") if isinstance(mapping, dict) else None
    if isinstance(value, bool):
        return value
    return _checkbox_is_checked(mapping)


def _render_radio_field(c, mapping, box, context: WidgetRenderContext) -> None:
    radio_x, radio_y, size = _checkbox_widget_rect(
        context.pdf_x,
        context.pdf_y,
        context.width,
        context.height,
    )
    group_name = _safe_radio_value(mapping.get("radio_group"), context.field_name)
    export_value = _safe_radio_value(mapping.get("export_value") or mapping.get("option_label"), f"Option_{context.box_index}")
    c.acroForm.radio(
        selected=_radio_is_selected(mapping),
        name=group_name,
        value=export_value,
        tooltip=_field_tooltip(mapping, "Radio button"),
        x=radio_x,
        y=radio_y,
        size=size,
        fillColor=CHECKBOX_FILL,
        borderColor=CHECKBOX_BORDER,
        textColor=FIELD_TEXT,
        borderWidth=0.6,
        buttonStyle="circle",
        shape="circle",
        fieldFlags="radio",
        forceBorder=False,
    )


def _render_text_field(c, mapping, box, context: WidgetRenderContext) -> None:
    field_type = _field_type(mapping, box)
    widget_x, widget_y, widget_width, widget_height = _text_widget_rect(
        mapping,
        box,
        context.pdf_x,
        context.pdf_y,
        context.width,
        context.height,
    )
    border_style, border_width = _text_widget_style(mapping, box, widget_height)
    c.acroForm.textfield(
        name=context.field_name,
        tooltip=_field_tooltip(mapping, "Field"),
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


def _comb_cell_count(mapping) -> int:
    for key in ("comb_cells", "expected_length", "max_length"):
        try:
            value = int(mapping.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return min(value, 64)
    return 0


def _render_comb_field(c, mapping, box, context: WidgetRenderContext) -> None:
    cells = _comb_cell_count(mapping)
    if cells <= 0:
        _render_text_field(c, _mapping_with_field_type(mapping, "text"), box, context)
        return

    inset = 0.5 if mapping.get("comb_boxes") else 1.25
    widget_x = context.pdf_x + inset
    widget_y = context.pdf_y + inset
    widget_width = max(10.0, context.width - inset * 2.0)
    widget_height = max(8.0, context.height - inset * 2.0)
    border_width = 0.0 if mapping.get("render_border") is False or mapping.get("comb_boxes") else 0.35
    c.acroForm.textfield(
        name=context.field_name,
        tooltip=_field_tooltip(mapping, "Comb field"),
        x=widget_x,
        y=widget_y,
        width=widget_width,
        height=widget_height,
        fillColor=TRANSPARENT_FILL,
        borderColor=FIELD_BORDER,
        textColor=FIELD_TEXT,
        borderWidth=border_width,
        borderStyle="solid",
        fieldFlags="comb",
        forceBorder=False,
        fontSize=_clamp(widget_height * 0.62, 7.0, 12.0),
        maxlen=cells,
    )


def _render_legacy_field(c, mapping, box, context: WidgetRenderContext) -> None:
    field_type = _field_type(mapping, box)
    if field_type == "checkbox":
        _render_checkbox_field(c, mapping, box, context)
    else:
        _render_text_field(c, mapping, box, context)


def _render_widget_as(c, mapping, box, context: WidgetRenderContext, widget_type: str) -> None:
    if widget_type == "comb":
        _render_comb_field(c, mapping, box, context)
        return
    if widget_type == "radio":
        _render_radio_field(c, mapping, box, context)
        return
    render_mapping = _mapping_with_field_type(mapping, field_type_for_widget_type(widget_type))
    if widget_type == "checkbox":
        _render_checkbox_field(c, render_mapping, box, context)
    else:
        _render_text_field(c, render_mapping, box, context)


_DEFAULT_WIDGET_REGISTRY: WidgetRendererRegistry | None = None


def get_widget_renderer_registry() -> WidgetRendererRegistry:
    global _DEFAULT_WIDGET_REGISTRY
    if _DEFAULT_WIDGET_REGISTRY is None:
        registry = WidgetRendererRegistry()
        for widget_type in ("text", "multiline", "checkbox", "radio", "comb", "signature"):
            registry.register(
                widget_type,
                lambda c, mapping, box, context, wt=widget_type: _render_widget_as(
                    c,
                    mapping,
                    box,
                    context,
                    wt,
                ),
            )
        _DEFAULT_WIDGET_REGISTRY = registry
    return _DEFAULT_WIDGET_REGISTRY


def _render_declared_widget_if_enabled(c, mapping, box, context: WidgetRenderContext) -> bool:
    if not widget_registry_enabled():
        return False

    widget_type = declared_widget_type(mapping)
    if widget_type is None:
        return False

    rendered = get_widget_renderer_registry().render(c, widget_type, mapping, box, context)
    if not rendered:
        logger.warning("[pdf] no renderer registered for widget_type=%r; using legacy renderer", widget_type)
    return rendered


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


def create_pdf_with_fields(image_path, mappings, output_path, page_images=None, page_sizes=None):
    """Render an empty fillable-form PDF.

    ``page_images`` optionally maps a 1-based page number to that page's
    rasterised background image. When omitted, every page uses ``image_path``
    (the historical single-background behaviour) so the legacy OCR path and the
    single-page Textract path are unchanged. When supplied, each page is drawn
    on its own background, the page count covers every rasterised page (even
    pages with zero detected fields), and each page's widgets are placed using
    that page's coordinate scale.

    ``page_sizes`` optionally maps a 1-based page number to that page's output
    size ``(width_pt, height_pt)`` (Issue 1 / page-geometry preservation). When
    omitted or empty, every page is US Letter, byte-identical to before. When
    supplied, each page is sized to its source geometry (A4 / Legal / landscape /
    mixed-size); pages absent from the map fall back to Letter. Widgets use
    page-local fraction coordinates, so they remain correctly placed at any page
    size.
    """
    started = time.perf_counter()
    output = Path(output_path)
    logger.info(
        "[pdf] start create_pdf_with_fields image=%s output=%s mappings=%s page_images=%s page_sizes=%s",
        image_path,
        output,
        len(mappings or []),
        len(page_images or {}),
        len(page_sizes or {}),
    )

    logger.info("[pdf] verify output path writable start")
    _verify_output_writable(output)
    logger.info("[pdf] verify output path writable end")

    logger.info("[pdf] canvas init start")
    c = canvas.Canvas(str(output), pagesize=letter)
    if c is None:
        raise RuntimeError("ReportLab canvas initialization returned None.")
    logger.info("[pdf] canvas init end")

    default_page_width, default_page_height = letter

    # Per-page output size. An empty map keeps every page at US Letter so the
    # historical behaviour is byte-identical. When ``page_sizes`` is supplied
    # (Issue 1 / page-geometry preservation) each page is sized to its source
    # geometry; pages absent from the map fall back to Letter.
    page_size_map = {
        int(p): (float(w), float(h))
        for p, (w, h) in (page_sizes or {}).items()
        if float(w) > 0 and float(h) > 0
    }

    def _page_dimensions(page_number: int) -> tuple:
        return page_size_map.get(int(page_number), (default_page_width, default_page_height))

    # Per-page background resolver. Each distinct background image is opened once
    # (reader + pixel dims cached). The page-fill scale is derived per page from
    # that page's output size, so mixed-size documents scale correctly. With no
    # page_images map every page falls back to ``image_path``.
    page_image_map = {int(p): img for p, img in (page_images or {}).items()}
    _bg_cache: dict[str, tuple] = {}

    def _background(page_number: int):
        source = page_image_map.get(int(page_number), image_path)
        key = str(source)
        if key not in _bg_cache:
            logger.info("[pdf] image reader init start page=%s src=%s", page_number, key)
            reader = ImageReader(source)
            img_w, img_h = reader.getSize()
            if img_w <= 0 or img_h <= 0:
                raise RuntimeError(f"Invalid source image size: {img_w}x{img_h} ({key})")
            _bg_cache[key] = (reader, img_w, img_h)
            logger.info("[pdf] image reader init end page=%s size=%sx%s", page_number, img_w, img_h)
        return _bg_cache[key]

    render_mappings = _downgrade_singleton_radio_groups(mappings or [])
    mappings_by_page = defaultdict(list)
    for index, mapping in enumerate(render_mappings, start=1):
        if not isinstance(mapping, dict):
            logger.warning("[pdf] skip mapping %s: mapping is not a dict", index)
            continue
        mappings_by_page[_safe_page(mapping)].append((index, mapping))

    # Render EVERY rasterised page (so a page with a background but no detected
    # fields still produces a page) AND every page that carries widgets. This is
    # what guarantees render-page-count == source-page-count.
    page_numbers = sorted(set(mappings_by_page) | set(page_image_map)) or [1]
    for page_position, page_number in enumerate(page_numbers, start=1):
        page_width, page_height = _page_dimensions(page_number)
        # Only resize when custom sizes are in play, so the no-page_sizes path
        # issues no extra canvas calls and stays byte-identical to before. When
        # sizes ARE supplied, set every page (default Letter included) so a prior
        # A4 page never bleeds its size onto a default-sized page.
        if page_size_map:
            c.setPageSize((page_width, page_height))
        image_reader, image_width, image_height = _background(page_number)
        scale_x = page_width / float(image_width)
        scale_y = page_height / float(image_height)
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
                field_type = _field_type(mapping, box)
                logger.info(
                    "[pdf] add field start page=%s name=%s type=%s widget_type=%s",
                    page_number,
                    field_name,
                    field_type,
                    mapping.get("widget_type"),
                )

                context = WidgetRenderContext(
                    field_name=field_name,
                    page_number=page_number,
                    mapping_index=index,
                    box_index=li + 1,
                    pdf_x=pdf_x,
                    pdf_y=pdf_y,
                    width=width,
                    height=height_field,
                    page_width=page_width,
                    page_height=page_height,
                )
                if not _render_declared_widget_if_enabled(c, mapping, box, context):
                    _render_legacy_field(c, mapping, box, context)
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
