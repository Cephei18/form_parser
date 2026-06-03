"""DynamoDB job-state transitions for the async Textract worker.

This is the thin, lazy-boto3 layer that records the canonical job lifecycle
(QUEUED → PROCESSING → SUCCEEDED | FAILED) with **conditional writes**, so that
at-least-once SQS delivery is idempotent:

  * claim   : QUEUED|PROCESSING → PROCESSING   (terminal state → claim refused)
  * succeed : PROCESSING        → SUCCEEDED     (+ artifacts, metrics)
  * fail    : (not SUCCEEDED)   → FAILED        (+ error)

Design goals (consistent with artifact_store.py):
  * Additive + isolated: importing this module never requires boto3/AWS. The
    local FastAPI + OCR paths never touch it.
  * Safe no-op: if no table is configured (``DDB_TABLE`` unset) every call is a
    silent no-op, so the worker's existing S3/direct-invoke test paths are
    byte-identical to before.
  * Never the cause of a lost job: a ConditionalCheckFailed on claim is surfaced
    to the caller (duplicate/late delivery → ack), but unexpected DDB errors are
    raised so the caller can decide on SQS retry.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("form_parser.job_state")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ConditionRefused(Exception):
    """Raised when a conditional transition is refused (duplicate/late delivery)."""


class JobStateStore:
    """DynamoDB-backed job state. A no-op when ``table_name`` is falsy."""

    def __init__(self, table_name: str | None = None, region: str | None = None, client: Any | None = None) -> None:
        self.table_name = table_name or os.getenv("DDB_TABLE") or os.getenv("FORM_PARSER_JOBS_TABLE")
        self.region = region or os.getenv("AWS_REGION") or os.getenv("FORM_PARSER_AWS_REGION")
        self._client = client

    @property
    def enabled(self) -> bool:
        return bool(self.table_name)

    def _ddb(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("dynamodb", region_name=self.region) if self.region else boto3.client("dynamodb")
        return self._client

    # --- transitions -------------------------------------------------------

    def claim(self, job_id: str) -> bool:
        """Move QUEUED|PROCESSING → PROCESSING. Returns False if already terminal."""
        if not self.enabled:
            return True
        from botocore.exceptions import ClientError

        try:
            self._ddb().update_item(
                TableName=self.table_name,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET #s = :processing, updated_at = :ts ADD #at :one",
                ConditionExpression="attribute_not_exists(#s) OR #s IN (:queued, :processing)",
                ExpressionAttributeNames={"#s": "status", "#at": "attempts"},
                ExpressionAttributeValues={
                    ":processing": {"S": "PROCESSING"},
                    ":queued": {"S": "QUEUED"},
                    ":ts": {"S": _now_iso()},
                    ":one": {"N": "1"},
                },
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.info("[job_state] claim refused (already terminal) job_id=%s", job_id)
                return False
            raise

    def succeed(self, job_id: str, *, artifacts: dict[str, Any], metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        from botocore.exceptions import ClientError

        try:
            self._ddb().update_item(
                TableName=self.table_name,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET #s = :succeeded, updated_at = :ts, #ar = :art, #m = :met",
                ConditionExpression="#s = :processing",
                ExpressionAttributeNames={"#s": "status", "#ar": "artifacts", "#m": "metrics"},
                ExpressionAttributeValues={
                    ":succeeded": {"S": "SUCCEEDED"},
                    ":processing": {"S": "PROCESSING"},
                    ":ts": {"S": _now_iso()},
                    ":art": {"M": _to_attr_map(artifacts)},
                    ":met": {"M": _to_attr_map(metrics)},
                },
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.info("[job_state] succeed skipped (not PROCESSING) job_id=%s", job_id)
                return
            raise

    def fail(self, job_id: str, *, error: dict[str, Any]) -> None:
        if not self.enabled:
            return
        from botocore.exceptions import ClientError

        try:
            self._ddb().update_item(
                TableName=self.table_name,
                Key={"job_id": {"S": job_id}},
                UpdateExpression="SET #s = :failed, updated_at = :ts, #e = :err",
                ConditionExpression="#s <> :succeeded",
                ExpressionAttributeNames={"#s": "status", "#e": "error"},
                ExpressionAttributeValues={
                    ":failed": {"S": "FAILED"},
                    ":succeeded": {"S": "SUCCEEDED"},
                    ":ts": {"S": _now_iso()},
                    ":err": {"M": _to_attr_map(error)},
                },
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.info("[job_state] fail skipped (already SUCCEEDED) job_id=%s", job_id)
                return
            raise


def _to_attr_map(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat-ish dict into DynamoDB AttributeValue map form.

    Only the value types the job item uses (str, int/float, None) are handled;
    nested dicts are recursed. Unsupported values are stringified defensively so
    a metric shape change can never break a state write.
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            out[key] = {"NULL": True}
        elif isinstance(value, bool):
            out[key] = {"BOOL": value}
        elif isinstance(value, (int, float)):
            out[key] = {"N": str(value)}
        elif isinstance(value, dict):
            out[key] = {"M": _to_attr_map(value)}
        else:
            out[key] = {"S": str(value)}
    return out
