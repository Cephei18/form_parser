"""
Asynchronous (multi-page) AWS Textract analysis.

The synchronous ``AnalyzeDocument`` API (used by the current pipeline via
``Document={"Bytes": ...}``) only accepts a single rasterised image, so the
live path silently drops every page after page 1 of a multi-page PDF. This
module adds the async ``StartDocumentAnalysis`` + ``GetDocumentAnalysis`` flow,
which Textract runs natively over a multi-page PDF stored in S3.

Design goals:
  * Produce a ``raw_response`` dict that is **shape-compatible** with the
    synchronous response (``{"Blocks": [...], "DocumentMetadata": {...},
    "JobStatus": ..., "AnalyzeDocumentModelVersion": ...}``). Every downstream
    consumer (``textract_parser`` / ``field_anchor_engine``) then works
    unchanged — the only difference is that ``Blocks`` now span all pages and
    each block carries its native ``Page`` field.
  * Stay fully testable offline: the Textract client and the sleep function are
    injectable, so the polling/pagination contract is asserted with a fake
    client and no AWS, no real wall-clock waits.
  * Never hang: bounded by ``max_wait``; a timeout raises with a "timed out"
    message so the worker's existing transient-error classifier triggers an SQS
    retry rather than a permanent FAILED.

Requires the worker role to grant ``textract:StartDocumentAnalysis`` and
``textract:GetDocumentAnalysis`` (see ``infra/worker_role_textract_async_policy.json``);
the document must live in an S3 bucket the caller can ``s3:GetObject``.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

logger = logging.getLogger("form_parser.textract_analysis")

DEFAULT_FEATURE_TYPES: tuple[str, ...] = ("TABLES", "FORMS")

# Terminal Textract job statuses we treat as usable output. PARTIAL_SUCCESS
# still returns blocks for the pages that did parse, so we keep it (and surface
# the warnings) rather than discarding a mostly-good multi-page result.
_USABLE_STATUSES = {"SUCCEEDED", "PARTIAL_SUCCESS"}


class TextractAnalysisError(RuntimeError):
    """Raised when the async Textract job cannot produce a usable response."""


def _get_client(client: Any | None, region: str | None) -> Any:
    if client is not None:
        return client
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without boto3
        raise TextractAnalysisError("boto3 is required for async Textract analysis") from exc
    return boto3.client("textract", region_name=region) if region else boto3.client("textract")


def start_analysis(
    bucket: str,
    key: str,
    *,
    client: Any,
    feature_types: Sequence[str] = DEFAULT_FEATURE_TYPES,
) -> str:
    """Kick off StartDocumentAnalysis against an S3 object; return the JobId."""
    response = client.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=list(feature_types),
    )
    job_id = response.get("JobId") if isinstance(response, dict) else None
    if not job_id:
        raise TextractAnalysisError("StartDocumentAnalysis returned no JobId")
    return str(job_id)


def _accumulate_blocks(
    client: Any,
    job_id: str,
    first_page: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], Any, list[Any]]:
    """Walk GetDocumentAnalysis pagination, collecting all blocks across pages."""
    blocks: list[dict[str, Any]] = list(first_page.get("Blocks") or [])
    document_metadata: dict[str, Any] = first_page.get("DocumentMetadata") or {}
    model_version: Any = first_page.get("AnalyzeDocumentModelVersion")
    warnings: list[Any] = list(first_page.get("Warnings") or [])

    next_token = first_page.get("NextToken")
    while next_token:
        page = client.get_document_analysis(JobId=job_id, NextToken=next_token)
        if not isinstance(page, dict):
            break
        blocks.extend(page.get("Blocks") or [])
        if not document_metadata:
            document_metadata = page.get("DocumentMetadata") or {}
        model_version = model_version or page.get("AnalyzeDocumentModelVersion")
        warnings.extend(page.get("Warnings") or [])
        next_token = page.get("NextToken")

    return blocks, document_metadata, model_version, warnings


def analyze_document_async(
    bucket: str,
    key: str,
    *,
    client: Any | None = None,
    region: str | None = None,
    feature_types: Sequence[str] = DEFAULT_FEATURE_TYPES,
    poll_interval: float = 2.0,
    max_wait: float = 270.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run a multi-page Textract analysis and return a sync-compatible response.

    ``client`` and ``sleep_fn`` are injectable so the polling/pagination
    contract is unit-tested without AWS or real sleeps.
    """
    client = _get_client(client, region)
    job_id = start_analysis(bucket, key, client=client, feature_types=feature_types)
    logger.info("[async] StartDocumentAnalysis job_id=%s s3://%s/%s", job_id, bucket, key)

    waited = 0.0
    first_page: dict[str, Any] = {}
    status = "IN_PROGRESS"
    while True:
        page = client.get_document_analysis(JobId=job_id)
        page = page if isinstance(page, dict) else {}
        status = str(page.get("JobStatus") or "IN_PROGRESS")
        if status != "IN_PROGRESS":
            first_page = page
            break
        if waited >= max_wait:
            # "timed out" => worker classifies as transient => SQS retry.
            raise TextractAnalysisError(f"Textract job {job_id} timed out after {waited:.0f}s (still IN_PROGRESS)")
        sleep_fn(poll_interval)
        waited += poll_interval

    if status == "FAILED":
        message = first_page.get("StatusMessage") or "Textract job failed"
        raise TextractAnalysisError(f"Textract job {job_id} FAILED: {message}")
    if status not in _USABLE_STATUSES:
        raise TextractAnalysisError(f"Textract job {job_id} ended with unexpected status {status!r}")

    blocks, document_metadata, model_version, warnings = _accumulate_blocks(client, job_id, first_page)

    response: dict[str, Any] = {
        "DocumentMetadata": document_metadata,
        "Blocks": blocks,
        "JobStatus": status,
    }
    if model_version is not None:
        response["AnalyzeDocumentModelVersion"] = model_version
    if warnings:
        response["Warnings"] = warnings

    logger.info(
        "[async] job_id=%s status=%s blocks=%s pages=%s warnings=%s",
        job_id,
        status,
        len(blocks),
        document_metadata.get("Pages"),
        len(warnings),
    )
    return response
