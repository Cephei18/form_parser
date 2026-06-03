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

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DDB_TABLE = os.environ["DDB_TABLE"]
REGION = os.getenv("AWS_REGION", "ap-south-1")

_ddb = boto3.client("dynamodb", region_name=REGION)

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": os.getenv("CORS_ALLOW_ORIGIN", "*"),
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,OPTIONS",
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
}

TERMINAL = {"SUCCEEDED", "FAILED", "DEAD_LETTER"}


def _response(status_code: int, body: dict) -> dict:
    return {"statusCode": status_code, "headers": _CORS_HEADERS, "body": json.dumps(body)}


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
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()
    if method == "OPTIONS":
        return _response(200, {"ok": True})

    job_id = _path_job_id(event)
    if not job_id:
        return _response(400, {"message": "job_id is required."})

    try:
        result = _ddb.get_item(TableName=DDB_TABLE, Key={"job_id": {"S": job_id}})
    except ClientError:
        logger.exception("[job-status] get_item failed job_id=%s", job_id)
        return _response(500, {"message": "Could not read job status."})

    item = result.get("Item")
    if not item:
        return _response(404, {"message": "Job not found.", "job_id": job_id})

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
        error_map = item["error"].get("M") or {}
        body["error"] = {k: v.get("S") for k, v in error_map.items()}

    return _response(200, body)
