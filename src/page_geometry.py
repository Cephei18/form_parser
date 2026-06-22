"""
Page geometry preservation for the Textract render path (Issue 1 / Phase 1).

The renderer historically forced every output page to US Letter and stretched
the page background to fit, distorting A4 / Legal / landscape / mixed-size
source documents. This module derives each page's true size (in PDF points)
from its rasterised image so the renderer can size every page to match the
source page's aspect ratio and orientation.

Why deriving from the raster works: ``document_render`` rasterises each PDF page
at a known DPI, and pixels = points / 72 * dpi. Inverting that
(``points = pixels / dpi * 72``) recovers the source page's true point size,
including aspect ratio and orientation, for both portrait and landscape pages.
Standalone image inputs use their embedded DPI when present, else the default.

Design constraints (mirrors the codebase's additive / flag-gated conventions):
  * Pure + dependency-light: only Pillow (already a ReportLab dependency) for a
    header-only image-size read; no cv2 / boto3 / fitz import at module load.
  * Opt-in: gated by ``FORM_PARSER_PRESERVE_PAGE_SIZE`` (default OFF). When the
    flag is off the renderer keeps its byte-identical US-Letter behaviour.
  * Never fatal: any per-page failure returns ``None`` so the caller falls back
    to the legacy Letter size for that page rather than raising.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.document_render import DEFAULT_DPI

logger = logging.getLogger("form_parser.page_geometry")

POINTS_PER_INCH = 72.0

# US Letter in points — the historical default the renderer falls back to.
LETTER_PT: tuple[float, float] = (612.0, 792.0)

# Sanity bounds (points). Guards against a corrupt raster yielding an absurd
# page size. ~1in to ~200in spans A6..A0, Letter, Legal and banner sheets.
_MIN_SIDE_PT = 72.0
_MAX_SIDE_PT = 14400.0


def preserve_page_size_enabled() -> bool:
    """Whether to size output pages to the source geometry.

    Default OFF (rollback-safe): unset/false keeps the legacy US-Letter
    behaviour, so enabling page-size preservation is an explicit per-environment
    opt-in exactly like ``FORM_PARSER_TEXTRACT_ASYNC``.
    """
    return os.getenv("FORM_PARSER_PRESERVE_PAGE_SIZE", "false").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_dpi(dpi: Any) -> tuple[float, float]:
    """Coerce a PIL ``dpi`` value (tuple, scalar, or absent) into a sane (x, y).

    PIL reports ``(0, 0)`` / ``(1, 1)`` / nothing when an image carries no DPI
    metadata (the common case for our freshly-rasterised PNGs), so anything
    non-positive falls back to the rasteriser's ``DEFAULT_DPI``.
    """
    try:
        if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
            dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
        elif isinstance(dpi, (int, float)):
            dpi_x = dpi_y = float(dpi)
        else:
            dpi_x = dpi_y = float(DEFAULT_DPI)
    except (TypeError, ValueError):
        dpi_x = dpi_y = float(DEFAULT_DPI)
    if dpi_x <= 1.0:
        dpi_x = float(DEFAULT_DPI)
    if dpi_y <= 1.0:
        dpi_y = float(DEFAULT_DPI)
    return dpi_x, dpi_y


def _image_pixel_size_and_dpi(image_path: str | Path) -> tuple[int, int, float, float] | None:
    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - Pillow ships with ReportLab
        logger.warning("[geometry] Pillow unavailable; cannot derive page size from %s", image_path)
        return None
    try:
        with Image.open(str(image_path)) as im:
            width_px, height_px = im.size
            raw_dpi = im.info.get("dpi")
    except Exception:
        logger.warning("[geometry] unable to read image size for %s", image_path, exc_info=True)
        return None
    if not width_px or not height_px or width_px <= 0 or height_px <= 0:
        return None
    dpi_x, dpi_y = _normalize_dpi(raw_dpi)
    return int(width_px), int(height_px), dpi_x, dpi_y


def image_point_size(image_path: str | Path, dpi: float | None = None) -> tuple[float, float] | None:
    """Return ``(width_pt, height_pt)`` for a raster image, or ``None`` on failure.

    ``dpi`` overrides the resolution used for the pixels->points conversion;
    when ``None`` the image's embedded DPI is used (falling back to
    ``DEFAULT_DPI`` when absent, which is the case for freshly-rasterised pages).
    """
    info = _image_pixel_size_and_dpi(image_path)
    if not info:
        return None
    width_px, height_px, dpi_x, dpi_y = info
    if dpi is not None:
        try:
            dpi_override = float(dpi)
        except (TypeError, ValueError):
            dpi_override = 0.0
        if dpi_override > 1.0:
            dpi_x = dpi_y = dpi_override

    width_pt = width_px / dpi_x * POINTS_PER_INCH
    height_pt = height_px / dpi_y * POINTS_PER_INCH
    if not (_MIN_SIDE_PT <= width_pt <= _MAX_SIDE_PT and _MIN_SIDE_PT <= height_pt <= _MAX_SIDE_PT):
        logger.warning(
            "[geometry] derived page size out of bounds (%.1fx%.1f pt) for %s; falling back to default",
            width_pt,
            height_pt,
            image_path,
        )
        return None
    return round(width_pt, 2), round(height_pt, 2)


def page_sizes_from_images(
    page_images: dict[int, str | Path] | None,
    dpi: float | None = None,
) -> dict[int, tuple[float, float]]:
    """Map ``{page_no: image_path}`` -> ``{page_no: (width_pt, height_pt)}``.

    Pages whose size cannot be derived are simply omitted, so the renderer falls
    back to its default page size for those pages only. Never raises.
    """
    sizes: dict[int, tuple[float, float]] = {}
    for page_no, image_path in (page_images or {}).items():
        size = image_point_size(image_path, dpi=dpi)
        if size:
            sizes[int(page_no)] = size
    return sizes


def orientation(size: tuple[float, float] | None) -> str:
    """``"portrait"`` / ``"landscape"`` / ``"square"`` / ``"unknown"`` for a size."""
    if not size:
        return "unknown"
    width, height = size
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"
