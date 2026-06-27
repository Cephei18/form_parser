"""presigned-url Lambda — API entry point that starts an async Textract job.

Route (API Gateway HTTP API, AWS_PROXY):  POST /uploads

Responsibilities (slice 1 of the async migration):
  1. Validate the requested filename / content-type / size against the same
     limits the EC2 sync path enforces (PNG, JPG, JPEG, PDF; <= 20 MB).
  2. Mint a job_id (uuid4) and derive a deterministic raw key
     ``uploads/<job_id>/<sanitized_filename>``.
  3. Write the canonical DynamoDB job item in state ``QUEUED`` (conditional
     put, so a job_id collision can never clobber an existing job).
  4. Return a presigned **POST** for the browser to upload directly to the raw
     bucket, plus the job_id the frontend will poll.

Design notes:
  * boto3 ships in the Lambda py3.12 runtime — no extra dependencies.
  * The handler never raises out to API Gateway; every path returns a proper
    JSON HTTP response so the frontend always gets a structured error.
  * generate_presigned_post enforces content-length-range + Content-Type *at
    S3*, so an oversized/forged upload is rejected by S3 itself, not just here.
  * mode + user_id are stamped into the S3 object metadata at presign time so
    the upload-handler can recover them from the ObjectCreated event without a
    DynamoDB round-trip.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RAW_BUCKET = os.environ["RAW_BUCKET"]
DDB_TABLE = os.environ["DDB_TABLE"]
REGION = os.getenv("AWS_REGION", "ap-south-1")

# Mirror the EC2 path's limits exactly (src/api.py).
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
ALLOWED_MODES = {"rule", "ml", "textract"}
PRESIGN_EXPIRY_SECONDS = int(os.getenv("PRESIGN_EXPIRY_SECONDS", "300"))
# TTL for un-completed jobs so abandoned QUEUED rows self-prune.
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(7 * 24 * 3600)))
# Bound attacker-controllable fields that get persisted into S3 user-metadata
# (which has a hard 2 KB total limit) and the DDB job item.
MAX_USER_ID_LEN = int(os.getenv("MAX_USER_ID_LEN", "128"))
MAX_FILENAME_LEN = int(os.getenv("MAX_FILENAME_LEN", "200"))

# Presigned URLs must be SigV4 so POST policy conditions are honoured.
_s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))
_ddb = boto3.client("dynamodb", region_name=REGION)

# Hardening: response security headers + an origin allowlist. CORS_ALLOW_ORIGIN
# is a comma-separated allowlist; when unset it defaults to "*" (preserves the
# current behaviour), and when configured the request Origin is reflected back
# only if it is on the list. Set it to the real frontend origin to lock CORS.
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
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Content-Type": "application/json",
        "Vary": "Origin",
        **_SECURITY_HEADERS,
    }
    return {"statusCode": status_code, "headers": headers, "body": json.dumps(body)}


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-._")
    return cleaned or "upload"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_body(event: dict) -> dict:
    raw = event.get("body")
    if raw is None:
        return {}
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def lambda_handler(event, context):
    origin = _cors_origin(event)
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "POST"
    ).upper()
    if method == "OPTIONS":
        return _response(200, {"ok": True}, origin)

    body = _parse_body(event)
    filename = str(body.get("filename") or "").strip()[:MAX_FILENAME_LEN]
    content_type = str(body.get("content_type") or "").split(";", 1)[0].strip().lower()
    mode = str(body.get("mode") or "textract").strip().lower()
    # Sanitize + bound user_id: it is persisted into S3 user-metadata (2 KB cap)
    # and the DDB item, so an unbounded/forged value must not break either.
    user_id = re.sub(r"[^A-Za-z0-9._@-]+", "-", str(body.get("user_id") or "anonymous").strip())[:MAX_USER_ID_LEN]
    user_id = user_id.strip("-._") or "anonymous"

    if not filename:
        return _response(400, {"message": "filename is required."}, origin)
    if content_type not in ALLOWED_CONTENT_TYPES:
        return _response(400, {"message": "Only PNG, JPG, JPEG, and PDF are supported."}, origin)
    if mode not in ALLOWED_MODES:
        return _response(400, {"message": "Mode must be one of rule, ml, textract."}, origin)

    safe_name = _sanitize_filename(filename)
    extension = os.path.splitext(safe_name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        # Force the extension to match the declared content-type.
        extension = ALLOWED_CONTENT_TYPES[content_type]
        safe_name = f"{safe_name}{extension}"

    job_id = uuid.uuid4().hex
    raw_key = f"uploads/{job_id}/{safe_name}"
    created_at = _now_iso()

    # 1) Record the job as QUEUED before handing out the upload URL.
    try:
        _ddb.put_item(
            TableName=DDB_TABLE,
            Item={
                "job_id": {"S": job_id},
                "status": {"S": "QUEUED"},
                "created_at": {"S": created_at},
                "updated_at": {"S": created_at},
                "user_id": {"S": user_id},
                "engine": {"S": "textract"},
                "mode": {"S": mode},
                "input": {
                    "M": {
                        "raw_bucket": {"S": RAW_BUCKET},
                        "raw_key": {"S": raw_key},
                        "content_type": {"S": content_type},
                    }
                },
                "expires_at": {"N": str(int(__import__("time").time()) + JOB_TTL_SECONDS)},
            },
            ConditionExpression="attribute_not_exists(job_id)",
        )
    except ClientError as exc:
        logger.exception("[presigned-url] DDB put_item failed job_id=%s", job_id)
        return _response(500, {"message": "Could not create job.", "error": exc.response["Error"]["Code"]}, origin)

    # 2) Mint a presigned POST that enforces size + content-type at S3.
    try:
        presigned = _s3.generate_presigned_post(
            Bucket=RAW_BUCKET,
            Key=raw_key,
            Fields={
                "Content-Type": content_type,
                "x-amz-meta-job_id": job_id,
                "x-amz-meta-mode": mode,
                "x-amz-meta-user_id": user_id,
            },
            Conditions=[
                {"Content-Type": content_type},
                {"x-amz-meta-job_id": job_id},
                {"x-amz-meta-mode": mode},
                {"x-amz-meta-user_id": user_id},
                ["content-length-range", 1, MAX_UPLOAD_BYTES],
            ],
            ExpiresIn=PRESIGN_EXPIRY_SECONDS,
        )
    except ClientError:
        logger.exception("[presigned-url] presign failed job_id=%s", job_id)
        return _response(500, {"message": "Could not create upload URL."}, origin)

    logger.info("[presigned-url] job created job_id=%s key=%s mode=%s", job_id, raw_key, mode)
    return _response(
        200,
        {
            "job_id": job_id,
            "raw_key": raw_key,
            "expires_in": PRESIGN_EXPIRY_SECONDS,
            "upload": {
                "method": "POST",
                "url": presigned["url"],
                "fields": presigned["fields"],
            },
        },
        origin,
    )
