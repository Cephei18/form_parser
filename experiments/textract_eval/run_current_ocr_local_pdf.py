#!/usr/bin/env python3
"""Run the current EasyOCR-based pipeline on a local PDF and summarize results.

Saves output to experiments/textract_eval/output/<pdf>.ocr.summary.json
"""
import json
import os
import sys
import traceback
from io import BytesIO

# Ensure project root is on sys.path so we can import `src`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import fitz
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None


def render_pdf_pages_to_png_paths(pdf_path, out_dir, zoom=2.0):
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed in this environment")
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    paths = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        png_path = os.path.join(out_dir, f"page_{i+1}.png")
        pix.save(png_path)
        paths.append(png_path)
    return paths


def read_with_easyocr(image_path, reader):
    # reader.readtext returns list of [bbox, text, conf]
    return reader.readtext(image_path, detail=1)


def main(pdf_path):
    out_dir = "experiments/textract_eval/output/current_ocr"
    os.makedirs(out_dir, exist_ok=True)

    try:
        pages = render_pdf_pages_to_png_paths(pdf_path, out_dir)
    except Exception as e:
        print("Failed to render PDF:", e)
        traceback.print_exc()
        sys.exit(1)

    # import the project's ocr module to use the same normalization logic
    try:
        from src import ocr as ocr_module
    except Exception as e:
        print("Failed to import src.ocr:", e)
        traceback.print_exc()
        sys.exit(1)

    reader = ocr_module._get_ocr_model()

    summary = {"pdf": pdf_path, "pages": []}

    for idx, img_path in enumerate(pages, start=1):
        print(f"Running EasyOCR on page {idx}/{len(pages)}...")
        raw_results = read_with_easyocr(img_path, reader)
        raw_items = []
        for res in raw_results:
            normalized = ocr_module._normalize_easyocr_result(res)
            if normalized:
                raw_items.append(normalized)
        cleaned = ocr_module.normalize_ocr_items(raw_items)
        diag = ocr_module.build_ocr_diagnostics(raw_items, cleaned)
        summary["pages"].append({"page": idx, "raw_item_count": len(raw_items), "cleaned_item_count": len(cleaned), "diagnostics": diag})

    out_path = os.path.join(out_dir, os.path.basename(pdf_path) + ".ocr.summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done. OCR summary saved to:", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_current_ocr_local_pdf.py <pdf_path>")
        sys.exit(1)
    main(sys.argv[1])
