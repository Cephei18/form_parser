"""
Document rendering helpers for the Textract worker (serverless migration).

Textract's synchronous AnalyzeDocument plus the OpenCV/ReportLab rendering steps
all work most reliably on a raster image. Forms often arrive as PDFs, so the
worker rasterises the first page before processing — mirroring the proven
pdf2image -> PyMuPDF fallback strategy already used by the legacy pipeline, but
kept in its own dependency-light module so the Textract path stays isolated from
the OCR ``main`` module.

Both ``pdf2image`` (needs the Poppler binary) and ``PyMuPDF``/``fitz``
(self-contained) are listed in requirements; the PyMuPDF fallback means PDF
rendering still works in environments without Poppler installed.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

logger = logging.getLogger("form_parser.document_render")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_DPI = 200


def render_pdf_first_page(pdf_path: str | Path, output_path: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    """Rasterise page 1 of a PDF to PNG. Tries pdf2image, falls back to PyMuPDF."""
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pdf2image = importlib.import_module("pdf2image")
        pages = pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=dpi)
        if not pages:
            raise RuntimeError("PDF conversion produced no pages")
        pages[0].save(output_path, "PNG")
        logger.info("[render] pdf2image rendered %s -> %s", pdf_path.name, output_path.name)
        return output_path
    except Exception:
        logger.warning("[render] pdf2image unavailable/failed for %s; trying PyMuPDF fallback", pdf_path.name)
        try:
            fitz = importlib.import_module("fitz")
            doc = fitz.open(str(pdf_path))
            try:
                pix = doc.load_page(0).get_pixmap(dpi=dpi)
                pix.save(str(output_path))
                logger.info("[render] PyMuPDF rendered %s -> %s", pdf_path.name, output_path.name)
                return output_path
            finally:
                doc.close()
        except Exception as exc:
            raise RuntimeError(
                "Failed to convert PDF. Install pdf2image+Poppler or ensure PyMuPDF is available."
            ) from exc


def ensure_image_input(input_path: str | Path, work_dir: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    """Return a raster image path for ``input_path``.

    - Image inputs are returned unchanged.
    - PDF inputs are rendered (page 1) into ``work_dir`` and the PNG is returned.
    - Anything else raises ValueError.

    NOTE: This is the single-page (page-1-only) entry point, preserved verbatim
    for backward compatibility / rollback. Multi-page callers use
    ``ensure_page_images`` instead.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        return input_path
    if suffix == ".pdf":
        output_path = Path(work_dir) / f"{input_path.stem}_page_1.png"
        return render_pdf_first_page(input_path, output_path, dpi=dpi)

    raise ValueError(f"Unsupported input type {suffix!r}; expected one of PDF or {sorted(IMAGE_SUFFIXES)}")


def render_pdf_pages(pdf_path: str | Path, output_dir: str | Path, dpi: int = DEFAULT_DPI) -> list[tuple[int, Path]]:
    """Rasterise **every** page of a PDF to PNG.

    Returns an ordered list of ``(page_number, png_path)`` with 1-based page
    numbers. Mirrors the proven pdf2image -> PyMuPDF fallback used for page 1,
    so multi-page rendering still works in Poppler-less environments.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    def _page_path(page_number: int) -> Path:
        return output_dir / f"{stem}_page_{page_number}.png"

    try:
        pdf2image = importlib.import_module("pdf2image")
        pages = pdf2image.convert_from_path(str(pdf_path), dpi=dpi)
        if not pages:
            raise RuntimeError("PDF conversion produced no pages")
        rendered: list[tuple[int, Path]] = []
        for index, page_image in enumerate(pages, start=1):
            out = _page_path(index)
            page_image.save(out, "PNG")
            rendered.append((index, out))
        logger.info("[render] pdf2image rendered %s page(s) of %s", len(rendered), pdf_path.name)
        return rendered
    except Exception:
        logger.warning("[render] pdf2image unavailable/failed for %s; trying PyMuPDF fallback", pdf_path.name)
        try:
            fitz = importlib.import_module("fitz")
            doc = fitz.open(str(pdf_path))
            try:
                rendered = []
                for index in range(doc.page_count):
                    out = _page_path(index + 1)
                    doc.load_page(index).get_pixmap(dpi=dpi).save(str(out))
                    rendered.append((index + 1, out))
                if not rendered:
                    raise RuntimeError("PDF has no pages")
                logger.info("[render] PyMuPDF rendered %s page(s) of %s", len(rendered), pdf_path.name)
                return rendered
            finally:
                doc.close()
        except Exception as exc:
            raise RuntimeError(
                "Failed to convert PDF. Install pdf2image+Poppler or ensure PyMuPDF is available."
            ) from exc


def ensure_page_images(input_path: str | Path, work_dir: str | Path, dpi: int = DEFAULT_DPI) -> list[tuple[int, Path]]:
    """Return per-page raster images for ``input_path`` as ``[(page_no, path)]``.

    - Image inputs become a single ``[(1, path)]`` entry (already one page).
    - PDF inputs are rendered to one PNG per page (all pages).
    - Anything else raises ValueError.

    This is the multi-page analogue of ``ensure_image_input``; the single-page
    function is left untouched so the legacy/rollback path is unaffected.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        return [(1, input_path)]
    if suffix == ".pdf":
        return render_pdf_pages(input_path, work_dir, dpi=dpi)

    raise ValueError(f"Unsupported input type {suffix!r}; expected one of PDF or {sorted(IMAGE_SUFFIXES)}")
