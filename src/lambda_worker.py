"""
Lambda-compatible Textract worker (serverless migration, step 2).

This module provides an event-driven execution path for the Textract pipeline
that is suitable for AWS Lambda (container image), while leaving the existing
local FastAPI workflow and the OCR pipeline completely untouched.

Responsibilities (intentionally minimal):
  1. Resolve input from an S3 event notification OR a direct invocation payload.
  2. Stage the raw document into a per-job /tmp working directory.
  3. Run the existing ``run_textract_pipeline`` against that directory (which
     already publishes artifacts to S3 via the storage abstraction from step 1).
  4. Return a structured, JSON-serialisable JobResult.
  5. Never raise out of a job: failures are captured into a ``failed`` result so
     the surrounding orchestration (added in a later slice) can decide on retries
     / DLQ behaviour.

Out of scope for this slice: API Gateway, SQS orchestration, DynamoDB writes.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote_plus

from src.artifact_store import download_s3_object
from src.document_render import ensure_image_input
from src.pipelines.textract_pipeline import run_textract_pipeline

logger = logging.getLogger("form_parser.lambda_worker")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Only /tmp is writable in Lambda; overridable for local testing.
WORK_ROOT = Path(os.getenv("FORM_PARSER_WORK_ROOT", "/tmp/form_parser_jobs"))

# Whether to delete the per-job working directory after publishing. Defaults to
# True so warm Lambda containers do not accumulate files in /tmp.
CLEANUP_WORKDIR = os.getenv("FORM_PARSER_WORKER_CLEANUP", "true").strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_job_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "textract-job"


def _job_id_from_key(key: str) -> str:
    # Use the object's path stem so artifacts are traceable back to the source.
    name = Path(key).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return _sanitize_job_id(stem or key)


def _resolve_inputs(event: Any) -> list[dict[str, str]]:
    """Normalise supported event shapes into a list of {bucket, key, job_id}.

    Supported:
      * S3 event notification: {"Records": [{"s3": {"bucket": {"name": ...},
        "object": {"key": ...}}}]}
      * Direct invocation payload: {"bucket": ..., "key": ..., "job_id"?: ...}
    """
    if not isinstance(event, dict):
        return []

    inputs: list[dict[str, str]] = []

    records = event.get("Records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            s3 = record.get("s3")
            if not isinstance(s3, dict):
                continue
            bucket = (s3.get("bucket") or {}).get("name")
            raw_key = (s3.get("object") or {}).get("key")
            if not bucket or not raw_key:
                continue
            key = unquote_plus(str(raw_key))
            inputs.append({"bucket": str(bucket), "key": key, "job_id": _job_id_from_key(key)})

    # Direct invocation payload (no S3 Records).
    if not inputs and event.get("bucket") and event.get("key"):
        key = str(event["key"])
        job_id = event.get("job_id")
        inputs.append(
            {
                "bucket": str(event["bucket"]),
                "key": key,
                "job_id": _sanitize_job_id(str(job_id)) if job_id else _job_id_from_key(key),
            }
        )

    return inputs


def process_job(
    bucket: str,
    key: str,
    job_id: str,
    *,
    work_root: Path | None = None,
    s3_client: Any | None = None,
    region: str | None = None,
    pipeline_fn: Callable[..., dict[str, Any]] = run_textract_pipeline,
) -> dict[str, Any]:
    """Download one document, run the Textract pipeline, return a JobResult.

    ``pipeline_fn`` and ``s3_client`` are injectable so the worker orchestration
    can be tested without AWS or a real Textract call.
    """
    work_root = Path(work_root) if work_root is not None else WORK_ROOT
    job_dir = work_root / job_id
    started_at = time.time()
    result: dict[str, Any] = {
        "job_id": job_id,
        "status": "failed",
        "pipeline_mode": "textract",
        "input": {"bucket": bucket, "key": key},
        "artifacts": None,
        "metrics": {},
        "error": None,
        "started_at": started_at,
        "finished_at": None,
        "duration_ms": None,
    }

    try:
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        job_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(key).suffix or ".bin"
        input_path = job_dir / f"input{suffix}"
        download_s3_object(bucket, key, input_path, region=region, client=s3_client)

        # Rasterise PDFs (and pass images through) so Textract + the OpenCV/
        # ReportLab render steps always receive an image.
        image_path = ensure_image_input(input_path, job_dir)

        logger.info("[worker] running Textract pipeline job_id=%s input=%s", job_id, image_path.name)
        pipeline_output = pipeline_fn(str(image_path), str(job_dir), reference_image_path=str(image_path))

        result["status"] = "succeeded"
        result["artifacts"] = pipeline_output.get("artifact_store")
        result["metrics"] = {
            "fields_detected": _safe_len(pipeline_output.get("fields")),
            "checkboxes_detected": len(pipeline_output.get("checkboxes") or []),
            "tables_detected": pipeline_output.get("tables_detected"),
            "processing_time_ms": pipeline_output.get("processing_time_ms"),
        }
        logger.info("[worker] job succeeded job_id=%s metrics=%s", job_id, result["metrics"])
    except Exception as exc:  # production-safe: never raise out of a job
        logger.exception("[worker] job failed job_id=%s", job_id)
        result["status"] = "failed"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }
    finally:
        if CLEANUP_WORKDIR:
            shutil.rmtree(job_dir, ignore_errors=True)
        finished_at = time.time()
        result["finished_at"] = finished_at
        result["duration_ms"] = round((finished_at - started_at) * 1000.0, 2)

    return result


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def handler(event: Any, context: Any | None = None) -> dict[str, Any]:
    """AWS Lambda entry point. Returns an aggregate, JSON-serialisable summary."""
    inputs = _resolve_inputs(event)
    if not inputs:
        logger.warning("[worker] no resolvable input in event; nothing to process")
        return {"job_count": 0, "succeeded": 0, "failed": 0, "results": [], "warning": "no_input_resolved"}

    results = [process_job(item["bucket"], item["key"], item["job_id"]) for item in inputs]
    succeeded = sum(1 for r in results if r.get("status") == "succeeded")
    summary = {
        "job_count": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    }
    logger.info("[worker] batch complete jobs=%s succeeded=%s failed=%s", summary["job_count"], succeeded, summary["failed"])
    return summary
