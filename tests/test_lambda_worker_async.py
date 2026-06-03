"""Unit tests for the async worker delta (SQS shape + DDB state machine).

These run with no AWS: process_job is monkeypatched and a fake JobStateStore
records the transitions, so we assert the contract (claim → succeed/fail, retry
classification, idempotent claim) deterministically.
"""
from __future__ import annotations

import json

import pytest

from src import lambda_worker
from src.lambda_worker import _resolve_inputs, _is_transient_error


def _sqs_event(job_id="job123", bucket="raw", key="uploads/job123/form.pdf", mode="textract"):
    return {
        "Records": [
            {
                "eventSource": "aws:sqs",
                "body": json.dumps({"job_id": job_id, "bucket": bucket, "key": key, "mode": mode}),
            }
        ]
    }


class FakeStore:
    def __init__(self, claim_ok=True):
        self.enabled = True
        self._claim_ok = claim_ok
        self.calls: list[tuple] = []

    def claim(self, job_id):
        self.calls.append(("claim", job_id))
        return self._claim_ok

    def succeed(self, job_id, *, artifacts, metrics):
        self.calls.append(("succeed", job_id, artifacts, metrics))

    def fail(self, job_id, *, error):
        self.calls.append(("fail", job_id, error))


def test_resolve_inputs_sqs_uses_message_job_id():
    inputs = _resolve_inputs(_sqs_event(job_id="abc"))
    assert len(inputs) == 1
    assert inputs[0]["job_id"] == "abc"
    assert inputs[0]["source"] == "sqs"
    assert inputs[0]["mode"] == "textract"


def test_resolve_inputs_s3_still_works():
    event = {"Records": [{"s3": {"bucket": {"name": "raw"}, "object": {"key": "uploads/x/form.pdf"}}}]}
    inputs = _resolve_inputs(event)
    assert inputs[0]["source"] == "s3"


def test_transient_classification():
    assert _is_transient_error({"type": "ThrottlingException", "message": "slow down"})
    assert _is_transient_error({"type": "X", "message": "request timed out"})
    assert not _is_transient_error({"type": "ValueError", "message": "bad file"})
    assert not _is_transient_error(None)


def test_handler_succeeds_and_records_state(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(lambda_worker, "JobStateStore", lambda *a, **k: store)

    def fake_process_job(bucket, key, job_id, **kwargs):
        return {
            "job_id": job_id,
            "status": "succeeded",
            "metrics": {"fields_detected": 5, "checkboxes_detected": 2},
            "artifacts": {
                "base_uri": "s3://proc/textract/job123",
                "artifacts": {
                    "output.pdf": "s3://proc/textract/job123/output.pdf",
                    "mapping.png": "s3://proc/textract/job123/mapping.png",
                    "result.json": "s3://proc/textract/job123/result.json",
                },
            },
        }

    monkeypatch.setattr(lambda_worker, "process_job", fake_process_job)
    summary = lambda_worker.handler(_sqs_event())

    assert summary["succeeded"] == 1
    kinds = [c[0] for c in store.calls]
    assert kinds == ["claim", "succeed"]
    succeed_call = store.calls[1]
    assert succeed_call[2]["output_pdf"].endswith("output.pdf")
    assert succeed_call[3]["fields_detected"] == 5


def test_handler_permanent_failure_marks_failed(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(lambda_worker, "JobStateStore", lambda *a, **k: store)
    monkeypatch.setattr(
        lambda_worker,
        "process_job",
        lambda *a, **k: {"job_id": "job123", "status": "failed", "error": {"type": "ValueError", "message": "bad file"}},
    )

    summary = lambda_worker.handler(_sqs_event())
    assert summary["failed"] == 1
    assert [c[0] for c in store.calls] == ["claim", "fail"]


def test_handler_transient_failure_raises_for_retry(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(lambda_worker, "JobStateStore", lambda *a, **k: store)
    monkeypatch.setattr(
        lambda_worker,
        "process_job",
        lambda *a, **k: {"job_id": "job123", "status": "failed", "error": {"type": "ThrottlingException", "message": "slow"}},
    )

    with pytest.raises(RuntimeError):
        lambda_worker.handler(_sqs_event())
    # claim happened; no terminal write (let SQS retry)
    assert [c[0] for c in store.calls] == ["claim"]


def test_handler_claim_refused_skips(monkeypatch):
    store = FakeStore(claim_ok=False)
    monkeypatch.setattr(lambda_worker, "JobStateStore", lambda *a, **k: store)
    called = {"n": 0}

    def fake_process_job(*a, **k):
        called["n"] += 1
        return {"status": "succeeded"}

    monkeypatch.setattr(lambda_worker, "process_job", fake_process_job)
    summary = lambda_worker.handler(_sqs_event())
    assert called["n"] == 0  # never processed
    assert summary["results"][0]["status"] == "skipped"
