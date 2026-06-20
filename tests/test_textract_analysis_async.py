"""Unit tests for async (multi-page) Textract analysis.

No AWS, no real sleeps: a scripted fake client drives the polling +
pagination contract, and sleep is captured so we can assert the loop without
wall-clock waits.
"""
from __future__ import annotations

import pytest

from src.textract_analysis import (
    DEFAULT_FEATURE_TYPES,
    TextractAnalysisError,
    analyze_document_async,
    start_analysis,
)


class FakeTextract:
    """Scripts StartDocumentAnalysis + a queue of GetDocumentAnalysis replies."""

    def __init__(self, job_id="job-1", get_responses=None):
        self.job_id = job_id
        self._get_responses = list(get_responses or [])
        self.start_calls: list[dict] = []
        self.get_calls: list[dict] = []

    def start_document_analysis(self, **kwargs):
        self.start_calls.append(kwargs)
        return {"JobId": self.job_id}

    def get_document_analysis(self, **kwargs):
        self.get_calls.append(kwargs)
        if not self._get_responses:
            raise AssertionError("unexpected extra get_document_analysis call")
        return self._get_responses.pop(0)


def _block(block_id, page, block_type="WORD"):
    return {"Id": block_id, "BlockType": block_type, "Page": page}


def test_start_analysis_passes_s3_location_and_features():
    client = FakeTextract()
    job_id = start_analysis("raw-bucket", "uploads/x/form.pdf", client=client)
    assert job_id == "job-1"
    call = client.start_calls[0]
    assert call["DocumentLocation"] == {"S3Object": {"Bucket": "raw-bucket", "Name": "uploads/x/form.pdf"}}
    assert call["FeatureTypes"] == list(DEFAULT_FEATURE_TYPES)


def test_start_analysis_no_jobid_raises():
    class NoJob(FakeTextract):
        def start_document_analysis(self, **kwargs):
            return {}

    with pytest.raises(TextractAnalysisError):
        start_analysis("b", "k", client=NoJob())


def test_immediate_success_single_page():
    client = FakeTextract(
        get_responses=[
            {
                "JobStatus": "SUCCEEDED",
                "DocumentMetadata": {"Pages": 1},
                "Blocks": [_block("a", 1)],
                "AnalyzeDocumentModelVersion": "1.0",
            }
        ]
    )
    resp = analyze_document_async("b", "k", client=client, sleep_fn=lambda _s: None)
    assert resp["JobStatus"] == "SUCCEEDED"
    assert resp["DocumentMetadata"]["Pages"] == 1
    assert [b["Id"] for b in resp["Blocks"]] == ["a"]
    assert resp["AnalyzeDocumentModelVersion"] == "1.0"


def test_polls_while_in_progress_then_succeeds():
    slept: list[float] = []
    client = FakeTextract(
        get_responses=[
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "SUCCEEDED", "DocumentMetadata": {"Pages": 1}, "Blocks": [_block("a", 1)]},
        ]
    )
    resp = analyze_document_async("b", "k", client=client, poll_interval=2.0, sleep_fn=slept.append)
    assert resp["JobStatus"] == "SUCCEEDED"
    assert slept == [2.0, 2.0]  # slept once per IN_PROGRESS poll, not after success


def test_pagination_accumulates_blocks_across_pages():
    client = FakeTextract(
        get_responses=[
            {
                "JobStatus": "SUCCEEDED",
                "DocumentMetadata": {"Pages": 3},
                "Blocks": [_block("p1", 1)],
                "NextToken": "t1",
            },
            {"Blocks": [_block("p2", 2)], "NextToken": "t2"},
            {"Blocks": [_block("p3", 3)]},
        ]
    )
    resp = analyze_document_async("b", "k", client=client, sleep_fn=lambda _s: None)
    assert [b["Id"] for b in resp["Blocks"]] == ["p1", "p2", "p3"]
    assert {b["Page"] for b in resp["Blocks"]} == {1, 2, 3}
    assert resp["DocumentMetadata"]["Pages"] == 3
    # 1 status/first-page call + 2 pagination calls
    assert len(client.get_calls) == 3
    assert client.get_calls[1]["NextToken"] == "t1"
    assert client.get_calls[2]["NextToken"] == "t2"


def test_partial_success_is_kept_with_warnings():
    client = FakeTextract(
        get_responses=[
            {
                "JobStatus": "PARTIAL_SUCCESS",
                "DocumentMetadata": {"Pages": 2},
                "Blocks": [_block("a", 1)],
                "Warnings": [{"ErrorCode": "X", "Pages": [2]}],
            }
        ]
    )
    resp = analyze_document_async("b", "k", client=client, sleep_fn=lambda _s: None)
    assert resp["JobStatus"] == "PARTIAL_SUCCESS"
    assert resp["Warnings"]


def test_failed_status_raises():
    client = FakeTextract(get_responses=[{"JobStatus": "FAILED", "StatusMessage": "bad doc"}])
    with pytest.raises(TextractAnalysisError, match="FAILED"):
        analyze_document_async("b", "k", client=client, sleep_fn=lambda _s: None)


def test_timeout_raises_transient_message():
    # Always IN_PROGRESS; max_wait reached quickly. Message must contain
    # "timed out" so the worker's transient classifier triggers an SQS retry.
    client = FakeTextract(get_responses=[{"JobStatus": "IN_PROGRESS"}] * 10)
    with pytest.raises(TextractAnalysisError, match="timed out"):
        analyze_document_async(
            "b", "k", client=client, poll_interval=1.0, max_wait=2.0, sleep_fn=lambda _s: None
        )
