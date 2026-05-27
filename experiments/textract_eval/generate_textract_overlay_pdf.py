#!/usr/bin/env python3
"""Generate an annotated PDF showing Textract detections (lines, boxes, key-value pairs).

Reads Textract raw JSON files created by `run_textract_local_pdf.py` at
`experiments/textract_eval/output/<pdf>.page{n}.raw.json` and produces
`experiments/textract_eval/output/<pdf>.textract_annotated.pdf`.

Usage:
  python generate_textract_overlay_pdf.py input/form.pdf
"""
import json
import os
import sys
from typing import Dict, Any

try:
    import fitz
except Exception:
    fitz = None


def load_textract_raw(out_dir: str, pdf_basename: str, page_index: int) -> Dict[str, Any] | None:
    path = os.path.join(out_dir, f"{pdf_basename}.page{page_index}.raw.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def blocks_by_id(blocks):
    return {b.get("Id"): b for b in blocks}


def extract_kv_pairs(blocks):
    id_map = blocks_by_id(blocks)
    kvs = []
    for b in blocks:
        if b.get("BlockType") != "KEY_VALUE_SET":
            continue
        entity = b.get("EntityTypes", [])
        if "KEY" in entity:
            key_text = ""
            val_text = ""
            # collect key text from CHILD relationships
            for rel in b.get("Relationships", []) or []:
                if rel.get("Type") == "CHILD":
                    for cid in rel.get("Ids", []):
                        child = id_map.get(cid)
                        if child and child.get("BlockType") == "WORD":
                            key_text += (child.get("Text") or "") + " "
                        elif child and child.get("BlockType") == "LINE":
                            key_text += (child.get("Text") or "") + " "
            # find corresponding VALUE via VALUE relationship
            for rel in b.get("Relationships", []) or []:
                if rel.get("Type") == "VALUE":
                    for vid in rel.get("Ids", []):
                        vb = id_map.get(vid)
                        if not vb:
                            continue
                        for vrel in vb.get("Relationships", []) or []:
                            if vrel.get("Type") == "CHILD":
                                for vcid in vrel.get("Ids", []):
                                    child = id_map.get(vcid)
                                    if child and child.get("BlockType") == "WORD":
                                        val_text += (child.get("Text") or "") + " "
                                    elif child and child.get("BlockType") == "LINE":
                                        val_text += (child.get("Text") or "") + " "
            kvs.append({"key": key_text.strip(), "value": val_text.strip(), "key_block": b})
    return kvs


def draw_overlays_for_page(page, img_w, img_h, blocks):
    idmap = blocks_by_id(blocks)

    # draw lines and words
    for b in blocks:
        t = b.get("BlockType")
        geom = b.get("Geometry") or {}
        bbox = geom.get("BoundingBox")
        if not bbox:
            continue
        left = bbox.get("Left", 0) * img_w
        top = bbox.get("Top", 0) * img_h
        width = bbox.get("Width", 0) * img_w
        height = bbox.get("Height", 0) * img_h
        rect = fitz.Rect(left, top, left + width, top + height)

        if t == "LINE":
            # red translucent rectangle
            page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0, 0, 0.08))
            # draw text label
            try:
                txt = b.get("Text", "")
                if txt:
                    page.insert_text((left + 2, top + 2), txt, fontsize=8, color=(0, 0, 0))
            except Exception:
                pass
        elif t == "TABLE":
            page.draw_rect(rect, color=(0, 0, 1), width=2)
        elif t == "CELL":
            page.draw_rect(rect, color=(0.2, 0.6, 0), width=1)
        elif t == "KEY_VALUE_SET":
            et = b.get("EntityTypes", [])
            if "KEY" in et:
                page.draw_rect(rect, color=(0, 0, 1), fill=(0, 0, 1, 0.06))
            elif "VALUE" in et:
                page.draw_rect(rect, color=(0, 0.6, 0), fill=(0, 0.6, 0, 0.06))


def main(pdf_path: str):
    if fitz is None:
        print("PyMuPDF is required. Install with pip install pymupdf")
        sys.exit(1)

    out_dir = "experiments/textract_eval/output"
    pdf_basename = os.path.basename(pdf_path)

    # render pages to images to know sizes
    doc = fitz.open(pdf_path)
    page_count = len(doc)

    out_pdf_path = os.path.join(out_dir, pdf_basename + ".textract_annotated.pdf")
    out_doc = fitz.open()

    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_bytes = pix.tobytes("png")
        img_w, img_h = pix.width, pix.height

        # create a new page in out_doc sized to image
        new_page = out_doc.new_page(width=img_w, height=img_h)
        # insert image as background
        img_rect = fitz.Rect(0, 0, img_w, img_h)
        new_page.insert_image(img_rect, stream=img_bytes)

        # load textract raw for this page
        raw = load_textract_raw(out_dir, pdf_basename, i + 1)
        if not raw:
            print(f"No Textract raw for page {i+1}; skipping overlays")
            continue
        blocks = raw.get("Blocks", [])

        # draw overlays
        draw_overlays_for_page(new_page, img_w, img_h, blocks)

        # extract key-value pairs and draw small labels
        kvs = extract_kv_pairs(blocks)
        y_offset = 10
        for kv in kvs:
            label = f"{kv.get('key')}: {kv.get('value')}"
            if label.strip():
                try:
                    new_page.insert_text((10, y_offset), label, fontsize=9, color=(0, 0, 0))
                    y_offset += 12
                except Exception:
                    pass

    out_doc.save(out_pdf_path)
    print("Annotated PDF written to:", out_pdf_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generate_textract_overlay_pdf.py <pdf_path>")
        sys.exit(1)
    main(sys.argv[1])
