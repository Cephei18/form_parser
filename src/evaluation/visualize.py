"""Diff-image rendering for evaluation.

Per page, overlay ground truth and predictions on the source raster (or a blank
canvas when no raster is shipped) and colour-code the outcome so a reviewer can
see *where* the model is wrong at a glance:

    green  = ground-truth widget that was matched (true positive)
    red    = ground-truth widget that was missed  (false negative)
    blue   = prediction that matched a GT widget
    orange = prediction with no GT               (false positive)

Boxes are page-local fractions, so they scale onto whatever canvas is used.
Pillow only — no OpenCV dependency.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.evaluation.flags import EvalConfig
from src.evaluation.geometry import Box
from src.evaluation.matching import MatchResult
from src.evaluation.schema import GroundTruthForm, PredWidget

GREEN = (34, 160, 60)
RED = (220, 40, 40)
BLUE = (40, 90, 220)
ORANGE = (240, 150, 20)
GREY = (120, 120, 120)


def _px(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = int(round(box["x"] * w))
    y0 = int(round(box["y"] * h))
    x1 = int(round((box["x"] + box["width"]) * w))
    y1 = int(round((box["y"] + box["height"]) * h))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _load_canvas(page_image: Path | None, config: EvalConfig) -> Image.Image:
    if page_image and page_image.is_file():
        try:
            return Image.open(page_image).convert("RGB")
        except Exception:  # noqa: BLE001 — corrupt raster must not crash a run
            pass
    return Image.new("RGB", (config.canvas_width, config.canvas_height), (255, 255, 255))


def render_form_diffs(
    form: GroundTruthForm,
    preds: list[PredWidget],
    result: MatchResult,
    out_dir: Path,
    config: EvalConfig,
    page_images: dict[int, Path] | None = None,
) -> list[str]:
    """Write one ``diff_page_<n>.png`` per page; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    page_images = page_images or {}
    matched_gt = {m.gt.widget_id for m in result.matches}
    matched_pred = {id(m.pred) for m in result.matches}

    written: list[str] = []
    for page in range(1, form.page_count + 1):
        canvas = _load_canvas(page_images.get(page), config)
        w, h = canvas.size
        draw = ImageDraw.Draw(canvas)

        # Ground truth first (so predictions render on top).
        for gw in (x for x in form.widgets if x.page == page):
            color = GREEN if gw.widget_id in matched_gt else RED
            box = _px(gw.bbox, w, h)
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0] + 2, box[1] + 2), f"{gw.wtype}", fill=color)

        for pw in (x for x in preds if x.page == page):
            color = BLUE if id(pw) in matched_pred else ORANGE
            box = _px(pw.bbox, w, h)
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, max(0, box[1] - 12)), f"{pw.raw_type}", fill=color)

        _legend(draw)
        path = out_dir / f"diff_page_{page}.png"
        canvas.save(path)
        written.append(str(path))
    return written


def _legend(draw: ImageDraw.ImageDraw) -> None:
    items = [
        ("GT matched", GREEN),
        ("GT missed (FN)", RED),
        ("pred matched", BLUE),
        ("pred extra (FP)", ORANGE),
    ]
    x, y = 8, 8
    for label, color in items:
        draw.rectangle((x, y, x + 14, y + 14), fill=color)
        draw.text((x + 20, y + 1), label, fill=GREY)
        y += 18
