#!/usr/bin/env python3
"""Render a local PDF to images using PyMuPDF and call AWS Textract AnalyzeDocument per page.

Saves per-page raw Textract responses and a combined parsed summary.
"""
import json
import os
import sys
import traceback
from io import BytesIO

import boto3
from botocore.exceptions import ClientError

try:
    import fitz
except Exception:
    fitz = None


def render_pdf_pages_to_png_bytes(pdf_path, zoom=2.0):
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed in this environment")
    doc = fitz.open(pdf_path)
    pages = []
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        try:
            png = pix.tobytes("png")
        except Exception:
            png = pix.getPNGData()
        pages.append(png)
    return pages


def analyze_image_bytes(textract_client, image_bytes):
    return textract_client.analyze_document(Document={"Bytes": image_bytes}, FeatureTypes=["TABLES", "FORMS"])


def summarize_blocks(blocks):
    counts = {}
    texts = []
    for b in blocks:
        t = b.get("BlockType")
        counts[t] = counts.get(t, 0) + 1
        if b.get("BlockType") == "LINE":
            texts.append({"Text": b.get("Text"), "Confidence": b.get("Confidence"), "Geometry": b.get("Geometry")})
    return {"counts": counts, "sample_lines": texts[:20]}


def main(pdf_path):
    out_dir = "experiments/textract_eval/output"
    os.makedirs(out_dir, exist_ok=True)

    result = {"pdf": pdf_path, "pages": []}

    try:
        pages = render_pdf_pages_to_png_bytes(pdf_path)
    except Exception as e:
        print("Failed to render PDF:", e)
        traceback.print_exc()
        sys.exit(1)

    textract = boto3.client("textract")

    for idx, img_bytes in enumerate(pages, start=1):
        print(f"Analyzing page {idx}/{len(pages)} with Textract...")
        try:
            resp = analyze_image_bytes(textract, img_bytes)
            raw_path = os.path.join(out_dir, f"{os.path.basename(pdf_path)}.page{idx}.raw.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(resp, f, indent=2, default=str)

            blocks = resp.get("Blocks", [])
            summary = summarize_blocks(blocks)
            result["pages"].append({"page": idx, "blocks_count": len(blocks), "summary": summary, "raw_path": raw_path})
        except ClientError as e:
            err = e.response.get("Error", {})
            print("ClientError:", err)
            result["pages"].append({"page": idx, "error": err})
        except Exception as e:
            print("Error calling Textract:", e)
            result["pages"].append({"page": idx, "error": str(e)})

    summary_path = os.path.join(out_dir, os.path.basename(pdf_path) + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Done. Summary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_textract_local_pdf.py <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    main(pdf_path)
