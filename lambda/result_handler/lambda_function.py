"""result-handler Lambda — returns processed artifacts in the frontend's shape.

Route (API Gateway HTTP API, AWS_PROXY):  GET /result/{job_id}

Responsibilities (slice 5 of the async migration):
  * Verify the job is SUCCEEDED.
  * Presign GET URLs for the processed artifacts the worker published under
    ``textract/<job_id>/{output.pdf,mapping.png,result.json}``.
  * Return them as ``pdf_url`` / ``mapping_preview`` / ``result_url`` plus
    ``stats`` — i.e. the SAME ProcessFormResponse shape the EC2 sync path
    returns — so the existing result page renders with minimal change.

The presigned GET carries the exec role's authority; ~10-min expiry. Keys are
deterministic (the worker writes the same filenames every run), so we derive
them rather than depending on the DDB artifacts map being populated.
"""
from __future__ import annotations

import json
import logging
import os
import re

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DDB_TABLE = os.environ["DDB_TABLE"]
PROCESSED_BUCKET = os.environ["PROCESSED_BUCKET"]
REGION = os.getenv("AWS_REGION", "ap-south-1")
ARTIFACT_PREFIX = os.getenv("FORM_PARSER_ARTIFACT_PREFIX", "textract")
DOWNLOAD_EXPIRY_SECONDS = int(os.getenv("DOWNLOAD_EXPIRY_SECONDS", "600"))

_s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))
_ddb = boto3.client("dynamodb", region_name=REGION)

# A job_id is an opaque capability and is interpolated into the presigned S3 key
# prefix; constrain it to a bounded, safe token set before it is used.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _allowed_origins() -> list[str]:
    return [o.strip() for o in os.getenv("CORS_ALLOW_ORIGIN", "*").split(",") if o.strip()]


def _cors_origin(event: dict) -> str:
    allowed = _allowed_origins()
    if not allowed or "*" in allowed:
        return "*"
    headers = event.get("headers") or {}
    origin = headers.get("origin") or headers.get("Origin") or ""
    return origin if origin in allowed else allowed[0]


def _response(status_code: int, body: dict, origin: str = "*") -> dict:
    headers = {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "Vary": "Origin",
        **_SECURITY_HEADERS,
    }
    return {"statusCode": status_code, "headers": headers, "body": json.dumps(body)}


def _path_job_id(event: dict) -> str:
    params = event.get("pathParameters") or {}
    return str(params.get("job_id") or "").strip()


def _presign(key: str) -> str:
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": PROCESSED_BUCKET, "Key": key},
        ExpiresIn=DOWNLOAD_EXPIRY_SECONDS,
    )


def _num(value: dict, default: int = 0):
    raw = value.get("N")
    if raw is None:
        return default
    return int(raw) if raw.isdigit() else float(raw)


def _stats_from_metrics(metrics_attr: dict) -> dict:
    """Map the worker's metrics map onto the frontend's ProcessingStats shape."""
    m = metrics_attr.get("M") or {}
    fields_detected = _num(m.get("fields_detected", {}))
    checkbox_count = _num(m.get("checkboxes_detected", {}))
    return {
        "ocr_count": 0,
        "line_count": 0,
        "field_candidate_count": fields_detected,
        "mapping_count": fields_detected,
        "checkbox_count": checkbox_count,
        "multi_line_count": _num(m.get("multi_line_count", {})),
    }


def lambda_handler(event, context):
    origin = _cors_origin(event)
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()
    if method == "OPTIONS":
        return _response(200, {"ok": True}, origin)

    job_id = _path_job_id(event)
    if not job_id:
        return _response(400, {"message": "job_id is required."}, origin)
    if not _JOB_ID_RE.match(job_id):
        return _response(400, {"message": "Invalid job_id."}, origin)

    try:
        result = _ddb.get_item(TableName=DDB_TABLE, Key={"job_id": {"S": job_id}})
    except ClientError:
        logger.exception("[result-handler] get_item failed job_id=%s", job_id)
        return _response(500, {"message": "Could not read job."}, origin)

    item = result.get("Item")
    if not item:
        return _response(404, {"message": "Job not found.", "job_id": job_id}, origin)

    status = item.get("status", {}).get("S", "UNKNOWN")
    if status != "SUCCEEDED":
        return _response(409, {"message": "Result not ready.", "job_id": job_id, "status": status}, origin)

    base = f"{ARTIFACT_PREFIX}/{job_id}"
    try:
        pdf_url = _presign(f"{base}/output.pdf")
        mapping_preview = _presign(f"{base}/mapping.png")
        result_url = _presign(f"{base}/result.json")
    except ClientError:
        logger.exception("[result-handler] presign failed job_id=%s", job_id)
        return _response(500, {"message": "Could not create download URLs."}, origin)

    stats = _stats_from_metrics(item.get("metrics", {}))

    logger.info("[result-handler] returned result job_id=%s", job_id)
    return _response(
        200,
        {
            "job_id": job_id,
            "status": "success",
            "mode": item.get("mode", {}).get("S", "textract"),
            "pdf_url": pdf_url,
            "mapping_preview": mapping_preview,
            "result_url": result_url,
            "stats": stats,
        },
        origin,
    )
