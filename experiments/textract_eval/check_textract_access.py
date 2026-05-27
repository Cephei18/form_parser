#!/usr/bin/env python3
"""Check AWS Textract access and run a tiny analyze_document request using an in-memory image.

Writes results to experiments/textract_eval/access_result.json and raw_response.json.
"""
import json
import sys
import traceback
from io import BytesIO

import boto3
from botocore.exceptions import ClientError

try:
    from PIL import Image
except Exception:
    Image = None


def make_test_image_bytes():
    if Image is None:
        return None
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    out_dir = "experiments/textract_eval"
    result_path = out_dir + "/access_result.json"
    raw_path = out_dir + "/raw_response.json"

    result = {"sts_identity": None, "textract_ok": False, "error": None}

    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        result["sts_identity"] = identity
    except Exception as e:
        result["error"] = {
            "stage": "sts",
            "message": str(e),
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("STS check failed:", e)
        sys.exit(1)

    try:
        textract = boto3.client("textract")

        img_bytes = make_test_image_bytes()
        if img_bytes is None:
            raise RuntimeError("Pillow not installed in the environment; cannot build test image bytes.")

        # Call analyze_document to test access to Textract (synchronous for small images)
        resp = textract.analyze_document(Document={"Bytes": img_bytes}, FeatureTypes=["TABLES", "FORMS"])

        # save raw response
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(resp, f, indent=2, default=str)

        # simple parsed summary
        blocks = resp.get("Blocks", [])
        counts = {}
        for b in blocks:
            t = b.get("BlockType")
            counts[t] = counts.get(t, 0) + 1

        result.update({"textract_ok": True, "blocks_count": len(blocks), "block_type_counts": counts})
    except ClientError as e:
        err = e.response.get("Error", {})
        result["error"] = {"stage": "textract", "code": err.get("Code"), "message": err.get("Message")}
    except Exception as e:
        result["error"] = {"stage": "textract", "message": str(e), "traceback": traceback.format_exc()}

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
