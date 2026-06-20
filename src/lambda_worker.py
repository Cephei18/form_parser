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

import json
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
from src.document_render import ensure_image_input, ensure_page_images
from src.job_state import JobStateStore
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

# Multi-page document handling. Defaults ON: every page of a PDF is rasterised,
# analysed, mapped and rendered. KILL-SWITCH: set FORM_PARSER_MULTIPAGE=false to
# instantly revert to the legacy page-1-only behaviour (no redeploy needed),
# which is the rollback path if multi-page ever misbehaves in production.
MULTIPAGE_ENABLED = os.getenv("FORM_PARSER_MULTIPAGE", "true").strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_job_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "textract-job"


def _job_id_from_key(key: str) -> str:
    # Use the object's path stem so artifacts are traceable back to the source.
    name = Path(key).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return _sanitize_job_id(stem or key)


def _resolve_inputs(event: Any) -> list[dict[str, str]]:
    """Normalise supported event shapes into a list of input descriptors.

    Each descriptor is ``{bucket, key, job_id, source, mode?, user_id?}`` where
    ``source`` is one of ``"sqs"`` / ``"s3"`` / ``"direct"``.

    Supported:
      * SQS event (async production path): {"Records": [{"eventSource":
        "aws:sqs", "body": "{\"job_id\":..,\"bucket\":..,\"key\":..}"}]}.
        The job_id from the message is authoritative (drives DDB + the processed
        artifact prefix), so the frontend's polling lines up.
      * S3 event notification (legacy/direct trigger): {"Records": [{"s3":
        {"bucket": {"name": ...}, "object": {"key": ...}}}]}.
      * Direct invocation payload: {"bucket": ..., "key": ..., "job_id"?: ...}.
    """
    if not isinstance(event, dict):
        return []

    inputs: list[dict[str, str]] = []

    records = event.get("Records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue

            # SQS message (the async worker path).
            if record.get("eventSource") == "aws:sqs" or "body" in record:
                payload = _parse_sqs_body(record.get("body"))
                bucket = payload.get("bucket")
                raw_key = payload.get("key")
                if not bucket or not raw_key:
                    logger.warning("[worker] SQS message missing bucket/key; skipping")
                    continue
                key = unquote_plus(str(raw_key))
                job_id = payload.get("job_id")
                inputs.append(
                    {
                        "bucket": str(bucket),
                        "key": key,
                        "job_id": _sanitize_job_id(str(job_id)) if job_id else _job_id_from_key(key),
                        "source": "sqs",
                        "mode": str(payload.get("mode", "textract")),
                        "user_id": str(payload.get("user_id", "anonymous")),
                    }
                )
                continue

            # S3 event notification.
            s3 = record.get("s3")
            if not isinstance(s3, dict):
                continue
            bucket = (s3.get("bucket") or {}).get("name")
            raw_key = (s3.get("object") or {}).get("key")
            if not bucket or not raw_key:
                continue
            key = unquote_plus(str(raw_key))
            inputs.append({"bucket": str(bucket), "key": key, "job_id": _job_id_from_key(key), "source": "s3"})

    # Direct invocation payload (no Records).
    if not inputs and event.get("bucket") and event.get("key"):
        key = str(event["key"])
        job_id = event.get("job_id")
        inputs.append(
            {
                "bucket": str(event["bucket"]),
                "key": key,
                "job_id": _sanitize_job_id(str(job_id)) if job_id else _job_id_from_key(key),
                "source": "direct",
            }
        )

    return inputs


def _parse_sqs_body(body: Any) -> dict[str, Any]:
    if not isinstance(body, str):
        return body if isinstance(body, dict) else {}
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        logger.warning("[worker] could not parse SQS message body as JSON")
        return {}
    return parsed if isinstance(parsed, dict) else {}


# Error types that warrant an SQS retry (transient infra/throttling) rather
# than an immediate FAILED + ack. Everything else is treated as a permanent
# failure so retries are not burned on un-processable input.
_TRANSIENT_ERROR_TYPES = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "ServiceUnavailable",
    "ServiceUnavailableException",
    "InternalServerError",
    "RequestTimeout",
    "RequestTimeoutException",
    "ConnectionError",
    "EndpointConnectionError",
    "ReadTimeoutError",
    "ConnectTimeoutError",
}
_TRANSIENT_MESSAGE_HINTS = ("throttl", "timed out", "timeout", "503", "500", "temporarily")


def _is_transient_error(error: dict[str, Any] | None) -> bool:
    if not error:
        return False
    if str(error.get("type")) in _TRANSIENT_ERROR_TYPES:
        return True
    message = str(error.get("message", "")).lower()
    return any(hint in message for hint in _TRANSIENT_MESSAGE_HINTS)


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

        if MULTIPAGE_ENABLED:
            # Multi-page path: rasterise EVERY page. Page 1 remains the reference
            # image for CV fallback / previews; the full set drives per-page
            # extraction, mapping and rendering. document_location enables the
            # opt-in native async Textract path (FORM_PARSER_TEXTRACT_ASYNC).
            page_images = ensure_page_images(input_path, job_dir)
            image_path = page_images[0][1]
            logger.info(
                "[worker] running Textract pipeline (multi-page) job_id=%s pages=%s input=%s",
                job_id,
                len(page_images),
                image_path.name,
            )
            pipeline_output = pipeline_fn(
                str(image_path),
                str(job_dir),
                reference_image_path=str(image_path),
                page_images=[(page_no, str(img)) for page_no, img in page_images],
                document_location={"bucket": bucket, "key": key},
            )
        else:
            # Legacy single-page path (rollback target). Unchanged: rasterise only
            # page 1 (or pass images through) and analyse that single image.
            image_path = ensure_image_input(input_path, job_dir)
            logger.info("[worker] running Textract pipeline job_id=%s input=%s", job_id, image_path.name)
            pipeline_output = pipeline_fn(str(image_path), str(job_dir), reference_image_path=str(image_path))

        page_observability = pipeline_output.get("page_observability") or {}
        result["status"] = "succeeded"
        result["artifacts"] = pipeline_output.get("artifact_store")
        result["metrics"] = {
            "fields_detected": _safe_len(pipeline_output.get("fields")),
            "checkboxes_detected": len(pipeline_output.get("checkboxes") or []),
            "tables_detected": pipeline_output.get("tables_detected"),
            "processing_time_ms": pipeline_output.get("processing_time_ms"),
            "pages_detected": page_observability.get("pages_detected"),
            "pages_analyzed": len(page_observability.get("pages_analyzed") or []),
            "pages_rendered": len(page_observability.get("pages_rendered") or []),
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


def _artifacts_for_ddb(result: dict[str, Any]) -> dict[str, Any]:
    """Distil the pipeline artifact manifest into the canonical DDB shape."""
    store = result.get("artifacts") or {}
    artifacts = store.get("artifacts") or {}
    out: dict[str, Any] = {"base_uri": store.get("base_uri")}
    for name, attr in (
        ("output.pdf", "output_pdf"),
        ("mapping.png", "mapping_png"),
        ("mappings.json", "mappings_json"),
        ("result.json", "result_json"),
    ):
        if name in artifacts:
            out[attr] = artifacts[name]
    return {k: v for k, v in out.items() if v is not None}


def _process_with_state(item: dict[str, Any], store: JobStateStore) -> dict[str, Any]:
    """Process one SQS-sourced job under the DDB state machine.

    Returns the JobResult. Raises only on transient failures, to trigger an SQS
    retry; permanent failures are recorded as FAILED and acked (return normally).
    """
    job_id = item["job_id"]

    # Idempotent claim. A refused claim means a duplicate/late delivery for a
    # job that already reached a terminal state → ack without reprocessing.
    if not store.claim(job_id):
        logger.info("[worker] skipping already-terminal job_id=%s", job_id)
        return {"job_id": job_id, "status": "skipped", "reason": "claim_refused"}

    result = process_job(item["bucket"], item["key"], job_id)

    if result.get("status") == "succeeded":
        store.succeed(
            job_id,
            artifacts=_artifacts_for_ddb(result),
            metrics=result.get("metrics") or {},
        )
        return result

    # Failed: classify for retry vs. permanent.
    error = result.get("error") or {"type": "UnknownError", "message": "job failed"}
    if _is_transient_error(error):
        logger.warning("[worker] transient failure job_id=%s type=%s; raising for SQS retry", job_id, error.get("type"))
        raise RuntimeError(f"transient worker failure for job {job_id}: {error.get('type')}")

    logger.error("[worker] permanent failure job_id=%s type=%s; marking FAILED", job_id, error.get("type"))
    store.fail(job_id, error={"type": str(error.get("type")), "message": str(error.get("message"))[:1000]})
    return result


def handler(event: Any, context: Any | None = None) -> dict[str, Any]:
    """AWS Lambda entry point. Returns an aggregate, JSON-serialisable summary.

    SQS-sourced jobs are driven through the DynamoDB state machine; S3/direct
    invocations (used for manual validation) keep the original DDB-free path so
    those test flows are unchanged.
    """
    inputs = _resolve_inputs(event)
    if not inputs:
        logger.warning("[worker] no resolvable input in event; nothing to process")
        return {"job_count": 0, "succeeded": 0, "failed": 0, "results": [], "warning": "no_input_resolved"}

    store = JobStateStore()
    results: list[dict[str, Any]] = []
    for item in inputs:
        if item.get("source") == "sqs" and store.enabled:
            results.append(_process_with_state(item, store))
        else:
            results.append(process_job(item["bucket"], item["key"], item["job_id"]))

    succeeded = sum(1 for r in results if r.get("status") == "succeeded")
    summary = {
        "job_count": len(results),
        "succeeded": succeeded,
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }
    logger.info(
        "[worker] batch complete jobs=%s succeeded=%s failed=%s",
        summary["job_count"],
        succeeded,
        summary["failed"],
    )
    return summary
