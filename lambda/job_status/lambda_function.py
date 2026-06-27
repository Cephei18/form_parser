"""job-status Lambda — the frontend's polling target.

Route (API Gateway HTTP API, AWS_PROXY):  GET /jobs/{job_id}

Responsibilities (slice 4 of the async migration):
  * GetItem by job_id and return the job's current state + a ``result_ready``
    boolean the frontend uses to decide when to call result-handler.

Read-only and cache-safe; DDB is on-demand so polling is cheap.
"""
from __future__ import annotations

import json
import logging
import os
import re

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DDB_TABLE = os.environ["DDB_TABLE"]
REGION = os.getenv("AWS_REGION", "ap-south-1")

_ddb = boto3.client("dynamodb", region_name=REGION)

# Defence-in-depth: a job_id is an opaque capability used directly as the DDB
# partition key. Reject anything that is not a bounded, safe token before it
# touches AWS (real job_ids are uuid4 hex; the charset also covers test ids).
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

TERMINAL = {"SUCCEEDED", "FAILED", "DEAD_LETTER"}


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


def _unwrap_metrics(metrics_attr: dict) -> dict:
    """Convert a DynamoDB Map of metrics into a plain JSON-able dict."""
    out: dict = {}
    for key, value in (metrics_attr.get("M") or {}).items():
        if "N" in value:
            num = value["N"]
            out[key] = int(num) if num.isdigit() else float(num)
        elif "S" in value:
            out[key] = value["S"]
    return out


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
        logger.exception("[job-status] get_item failed job_id=%s", job_id)
        return _response(500, {"message": "Could not read job status."}, origin)

    item = result.get("Item")
    if not item:
        return _response(404, {"message": "Job not found.", "job_id": job_id}, origin)

    status = item.get("status", {}).get("S", "UNKNOWN")
    body = {
        "job_id": job_id,
        "status": status,
        "mode": item.get("mode", {}).get("S"),
        "created_at": item.get("created_at", {}).get("S"),
        "updated_at": item.get("updated_at", {}).get("S"),
        "result_ready": status == "SUCCEEDED",
        "terminal": status in TERMINAL,
    }
    if "metrics" in item:
        body["metrics"] = _unwrap_metrics(item["metrics"])
    if "error" in item:
        # Surface only the coarse error type to the client; the detailed message
        # (which can contain internal paths/identifiers) stays in CloudWatch.
        error_map = item["error"].get("M") or {}
        body["error"] = {"type": (error_map.get("type") or {}).get("S", "ProcessingError")}

    return _response(200, body, origin)
