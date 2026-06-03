"""upload-handler Lambda — bridges a raw S3 upload to the processing queue.

Trigger:  S3 ObjectCreated on the raw bucket, prefix ``uploads/`` (NOT an API
route). Wired by an admin via PutBucketNotification + a Lambda resource policy
allowing s3.amazonaws.com to invoke this function.

Responsibilities (slice 2 of the async migration):
  1. For each ObjectCreated record, recover the job_id from the key
     (``uploads/<job_id>/<file>``) and the mode/user_id from the object's
     user-metadata (set by presigned-url at presign time).
  2. Conditionally stamp the job as enqueued in DynamoDB (idempotent — S3 can
     deliver the same event more than once).
  3. SendMessage to SQS so the worker can pick it up:
        { job_id, bucket, key, mode, user_id }

Design notes:
  * boto3 ships in the runtime — no extra dependencies.
  * If SQS send fails we RAISE, so S3 async invocation retries (and, failing
    that, lands in the function's async dead-letter destination if configured).
  * The conditional DDB update means duplicate S3 deliveries enqueue exactly
    once; even if two slip through, the worker's conditional claim absorbs it.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DDB_TABLE = os.environ["DDB_TABLE"]
SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
REGION = os.getenv("AWS_REGION", "ap-south-1")

_s3 = boto3.client("s3", region_name=REGION)
_sqs = boto3.client("sqs", region_name=REGION)
_ddb = boto3.client("dynamodb", region_name=REGION)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _job_id_from_key(key: str) -> str | None:
    # Expected layout: uploads/<job_id>/<filename>
    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "uploads" and parts[1]:
        return parts[1]
    return None


def _object_metadata(bucket: str, key: str) -> dict:
    try:
        head = _s3.head_object(Bucket=bucket, Key=key)
    except ClientError:
        logger.warning("[upload-handler] head_object failed for s3://%s/%s", bucket, key)
        return {}
    return head.get("Metadata", {}) or {}


def _mark_enqueued(job_id: str) -> bool:
    """Conditionally flag the job as enqueued. Returns False if already done."""
    try:
        _ddb.update_item(
            TableName=DDB_TABLE,
            Key={"job_id": {"S": job_id}},
            UpdateExpression="SET enqueued_at = :ts, updated_at = :ts",
            ConditionExpression="attribute_not_exists(enqueued_at)",
            ExpressionAttributeValues={":ts": {"S": _now_iso()}},
        )
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            logger.info("[upload-handler] job already enqueued job_id=%s; skipping", job_id)
            return False
        # A missing job row (presigned-url failed) or DDB error — log and still
        # enqueue so processing is not silently lost.
        logger.warning("[upload-handler] enqueue flag update failed job_id=%s: %s", job_id, exc)
        return True


def _enqueue(job_id: str, bucket: str, key: str, mode: str, user_id: str) -> None:
    _sqs.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps(
            {
                "job_id": job_id,
                "bucket": bucket,
                "key": key,
                "mode": mode,
                "user_id": user_id,
            }
        ),
    )
    logger.info("[upload-handler] enqueued job_id=%s key=%s", job_id, key)


def lambda_handler(event, context):
    records = event.get("Records") or []
    processed = 0
    for record in records:
        s3 = record.get("s3") or {}
        bucket = (s3.get("bucket") or {}).get("name")
        raw_key = (s3.get("object") or {}).get("key")
        if not bucket or not raw_key:
            continue
        key = unquote_plus(str(raw_key))

        job_id = _job_id_from_key(key)
        if not job_id:
            logger.warning("[upload-handler] could not derive job_id from key=%s; skipping", key)
            continue

        metadata = _object_metadata(bucket, key)
        mode = metadata.get("mode", "textract")
        user_id = metadata.get("user_id", "anonymous")

        if not _mark_enqueued(job_id):
            continue  # duplicate delivery — already enqueued

        _enqueue(job_id, bucket, key, mode, user_id)
        processed += 1

    return {"processed": processed, "received": len(records)}
