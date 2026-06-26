"""Task 4 — diagnostic visualization artifacts (Pillow only, offline).

Given a page raster + the pipeline ``mappings`` and ``diagnostics``, render three
overlays per page:

* ``page_<n>_overlay.png``   — every widget, coloured by field_type.
* ``page_<n>_structure.png`` — structural cues: table input cells, comb groups,
  dotted leaders.
* ``page_<n>_diff.png``      — problems: suspicious checkboxes, duplicates,
  low/medium-confidence widgets.

This module is **not** imported by the live pipeline; it is called by the offline
replay CLI. It only reads artifacts, so it can never affect production output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

# field_type -> colour
TYPE_COLORS = {
    "checkbox": (40, 90, 220),
    "text": (34, 160, 60),
    "multiline": (0, 170, 170),
    "signature": (150, 60, 200),
    "photo": (130, 130, 130),
    "date": (200, 140, 0),
}
DEFAULT_COLOR = (240, 150, 20)
RED = (220, 40, 40)
ORANGE = (240, 150, 20)
YELLOW = (210, 190, 0)
MAGENTA = (200, 30, 160)
CYAN = (0, 160, 200)
OLIVE = (120, 120, 30)
GREY = (110, 110, 110)


def _px(box: dict[str, Any], w: int, h: int) -> tuple[int, int, int, int] | None:
    try:
        x0 = int(round(float(box["x"]) * w))
        y0 = int(round(float(box["y"]) * h))
        x1 = int(round((float(box["x"]) + float(box["width"])) * w))
        y1 = int(round((float(box["y"]) + float(box["height"])) * h))
    except (KeyError, TypeError, ValueError):
        return None
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _open(page_image: Path | None) -> Image.Image | None:
    if not page_image or not Path(page_image).is_file():
        return None
    try:
        return Image.open(page_image).convert("RGB")
    except Exception:  # noqa: BLE001 — a corrupt raster must not crash a report
        return None


def _legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, tuple[int, int, int]]]) -> None:
    x, y = 8, 8
    for label, color in items:
        draw.rectangle((x, y, x + 14, y + 14), fill=color)
        draw.text((x + 20, y + 1), label, fill=GREY)
        y += 18


def _page_mappings(mappings: list[dict[str, Any]], page: int) -> list[dict[str, Any]]:
    return [m for m in mappings if isinstance(m, dict) and int(m.get("page") or 1) == page]


def render_overlay(img: Image.Image, mappings: list[dict[str, Any]], page: int, out: Path) -> str:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    seen: set[str] = set()
    for m in _page_mappings(mappings, page):
        ftype = str(m.get("field_type") or "other")
        color = TYPE_COLORS.get(ftype, DEFAULT_COLOR)
        seen.add(ftype)
        box = _px(m.get("bbox") or {}, w, h)
        if box:
            draw.rectangle(box, outline=color, width=2)
    _legend(draw, [(t, TYPE_COLORS.get(t, DEFAULT_COLOR)) for t in sorted(seen)])
    canvas.save(out)
    return str(out)


def render_structure(img: Image.Image, diagnostics: dict[str, Any], page: int, out: Path) -> str:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    anchoring = diagnostics.get("anchoring") if isinstance(diagnostics.get("anchoring"), dict) else diagnostics

    # Comb groups (magenta) with cell counts.
    comb = anchoring.get("comb_diagnostics") if isinstance(anchoring.get("comb_diagnostics"), dict) else {}
    for g in comb.get("groups", []) or []:
        if int(g.get("page") or 1) != page:
            continue
        box = _px(g.get("bbox") or {}, w, h)
        if box:
            draw.rectangle(box, outline=MAGENTA, width=2)
            draw.text((box[0], max(0, box[1] - 11)), f"comb x{g.get('cell_count')}", fill=MAGENTA)

    # Dotted leaders (olive).
    dotted = anchoring.get("dotted_underlines") if isinstance(anchoring.get("dotted_underlines"), dict) else {}
    for d in dotted.get("detected", []) or []:
        if int(d.get("page") or 1) != page:
            continue
        box = _px(d.get("bbox") or {}, w, h)
        if box:
            draw.line((box[0], box[1], box[2], box[1]), fill=OLIVE, width=2)

    # Table input cells (cyan), when the classifier surfaced them.
    ti = anchoring.get("table_intelligence") if isinstance(anchoring.get("table_intelligence"), dict) else {}
    for cell in ti.get("input_cells", []) or []:
        if int(cell.get("page") or 1) != page:
            continue
        box = _px(cell.get("bbox") or {}, w, h)
        if box:
            draw.rectangle(box, outline=CYAN, width=1)

    _legend(draw, [("comb group", MAGENTA), ("dotted leader", OLIVE), ("table input cell", CYAN)])
    canvas.save(out)
    return str(out)


def render_diff(img: Image.Image, mappings: list[dict[str, Any]], diagnostics: dict[str, Any], page: int, out: Path) -> str:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    anchoring = diagnostics.get("anchoring") if isinstance(diagnostics.get("anchoring"), dict) else diagnostics

    cbv = anchoring.get("checkbox_validation") if isinstance(anchoring.get("checkbox_validation"), dict) else {}
    suspicious_ids = {
        str(r.get("field_id")) for r in (cbv.get("rejected") or []) if int(r.get("page") or 1) == page
    }

    for m in _page_mappings(mappings, page):
        box = _px(m.get("bbox") or {}, w, h)
        if not box:
            continue
        fid = str(m.get("field_id"))
        reasons = []
        cal = m.get("calibration") if isinstance(m.get("calibration"), dict) else {}
        reasons = cal.get("reasons") or []
        level = str(m.get("confidence_level") or "").upper()
        if fid in suspicious_ids:
            draw.rectangle(box, outline=RED, width=3)
            draw.text((box[0], max(0, box[1] - 11)), "suspicious", fill=RED)
        elif "duplicate_region" in reasons:
            draw.rectangle(box, outline=ORANGE, width=2)
            draw.text((box[0], max(0, box[1] - 11)), "dup", fill=ORANGE)
        elif level == "LOW":
            draw.rectangle(box, outline=RED, width=2)
        elif level == "MEDIUM":
            draw.rectangle(box, outline=YELLOW, width=2)
        else:
            draw.rectangle(box, outline=(60, 170, 60), width=1)

    _legend(draw, [("suspicious checkbox", RED), ("duplicate", ORANGE), ("MEDIUM conf", YELLOW), ("LOW conf", RED)])
    canvas.save(out)
    return str(out)


def render_diagnostics_pages(
    page_images: dict[int, Any],
    mappings: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, list[str]]:
    """Render overlay/structure/diff for every page that has a raster."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {"overlay": [], "structure": [], "diff": []}
    for page in sorted(page_images):
        img = _open(page_images[page])
        if img is None:
            continue
        written["overlay"].append(render_overlay(img, mappings, page, out_dir / f"page_{page}_overlay.png"))
        written["structure"].append(render_structure(img, diagnostics, page, out_dir / f"page_{page}_structure.png"))
        written["diff"].append(render_diff(img, mappings, diagnostics, page, out_dir / f"page_{page}_diff.png"))
    return written
