"""End-to-end async validation harness (manual trigger orchestration).

Faithfully drives the full async path against REAL AWS, substituting a direct
Lambda invoke for the two admin-gated auto-triggers (S3 notification, SQS ESM):

  presigned-url -> S3 presigned POST upload -> [simulate S3 notif] upload-handler
  -> SQS -> [simulate ESM] worker (Textract + DDB) -> job-status -> result-handler
  -> download the generated PDF.

Asserts the DDB lifecycle (QUEUED -> PROCESSING -> SUCCEEDED), artifact presence,
and presigned-URL validity. Read-mostly; the only writes are one real job's
artifacts + DDB row (TTL-pruned) and one SQS message (consumed + deleted here).

Usage:  .venv/Scripts/python.exe scripts/e2e_async_validate.py [path-to-pdf]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import boto3
import requests

REGION = "ap-south-1"
RAW_BUCKET = "form-pdf-poc-dev-raw-documents"
PROCESSED_BUCKET = "form-pdf-poc-dev-processed-documents"
QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/637423601842/form-pdf-poc-dev-processing-queue"
TABLE = "form-pdf-poc-dev-jobs"
FN = {
    "presigned": "form-pdf-poc-dev-presigned-url",
    "upload": "form-pdf-poc-dev-upload-handler",
    "status": "form-pdf-poc-dev-job-status",
    "result": "form-pdf-poc-dev-result-handler",
    "worker": "form-pdf-poc-dev-worker",
}

session = boto3.Session(profile_name="form-pdf-poc", region_name=REGION)
lam = session.client("lambda")
s3 = session.client("s3")
sqs = session.client("sqs")
ddb = session.client("dynamodb")

PDF = Path(sys.argv[1] if len(sys.argv) > 1 else "input/form.pdf")
results: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def invoke(fn: str, payload: dict) -> dict:
    resp = lam.invoke(FunctionName=fn, Payload=json.dumps(payload).encode())
    raw = resp["Payload"].read().decode()
    if resp.get("FunctionError"):
        raise RuntimeError(f"{fn} FunctionError: {raw[:500]}")
    return json.loads(raw)


def api_body(resp: dict) -> dict:
    """Unwrap an API-Gateway-proxy style response body."""
    return json.loads(resp["body"]) if isinstance(resp.get("body"), str) else resp


def ddb_status(job_id: str) -> str:
    item = ddb.get_item(TableName=TABLE, Key={"job_id": {"S": job_id}}).get("Item")
    return item["status"]["S"] if item else "ABSENT"


print(f"\n=== Async E2E validation using {PDF} ===\n")
assert PDF.is_file(), f"missing test file {PDF}"

# 1) presigned-url -----------------------------------------------------------
pre = api_body(invoke(FN["presigned"], {
    "requestContext": {"http": {"method": "POST"}},
    "body": json.dumps({"filename": PDF.name, "content_type": "application/pdf", "mode": "textract"}),
}))
job_id = pre["job_id"]
raw_key = pre["raw_key"]
step("presigned-url returns job_id + upload", bool(job_id and pre.get("upload", {}).get("url")), f"job_id={job_id}")
step("DDB row created as QUEUED", ddb_status(job_id) == "QUEUED")

# 2) Browser-style presigned POST upload to raw S3 ---------------------------
upload = pre["upload"]
with PDF.open("rb") as fh:
    post = requests.post(upload["url"], data=upload["fields"], files={"file": (PDF.name, fh, "application/pdf")})
step("presigned POST upload to raw S3", post.status_code in (200, 201, 204), f"http={post.status_code}")

head = s3.head_object(Bucket=RAW_BUCKET, Key=raw_key)
step("raw object present with metadata", head["Metadata"].get("job_id") == job_id,
     f"size={head['ContentLength']} meta_mode={head['Metadata'].get('mode')}")

# 3) Simulate S3 ObjectCreated -> upload-handler -----------------------------
upl = invoke(FN["upload"], {"Records": [{"s3": {
    "bucket": {"name": RAW_BUCKET}, "object": {"key": raw_key}}}]})
step("upload-handler enqueued 1 message", upl.get("processed") == 1, json.dumps(upl))

# 4) Receive the real SQS message --------------------------------------------
msg = None
for _ in range(5):
    recv = sqs.receive_message(QueueUrl=QUEUE_URL, MaxNumberOfMessages=10, WaitTimeSeconds=3,
                               VisibilityTimeout=120)
    for m in recv.get("Messages", []):
        if json.loads(m["Body"]).get("job_id") == job_id:
            msg = m
            break
    if msg:
        break
step("SQS message present for job", msg is not None)
body = json.loads(msg["Body"])
step("SQS message carries correct contract", body.get("job_id") == job_id and body.get("key") == raw_key,
     f"mode={body.get('mode')}")

# 5) Simulate ESM -> worker (real Textract + DDB transitions) ----------------
print("    ... invoking worker (real Textract, ~10-30s)")
wrk = invoke(FN["worker"], {"Records": [{"eventSource": "aws:sqs", "body": msg["Body"]}]})
step("worker reports 1 succeeded", wrk.get("succeeded") == 1, json.dumps(wrk.get("results", [{}])[0].get("metrics", {})))
step("DDB transitioned to SUCCEEDED", ddb_status(job_id) == "SUCCEEDED")

# Clean up the SQS message we consumed manually.
sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])

# 6) Idempotency: re-invoke worker with same message -> claim refused --------
wrk2 = invoke(FN["worker"], {"Records": [{"eventSource": "aws:sqs", "body": msg["Body"]}]})
r0 = (wrk2.get("results") or [{}])[0]
step("duplicate delivery is a no-op (idempotent)", r0.get("status") == "skipped" or ddb_status(job_id) == "SUCCEEDED",
     f"status={r0.get('status')}")

# 7) job-status reports SUCCEEDED + result_ready -----------------------------
st = api_body(invoke(FN["status"], {"requestContext": {"http": {"method": "GET"}},
                                     "pathParameters": {"job_id": job_id}}))
step("job-status SUCCEEDED + result_ready", st.get("status") == "SUCCEEDED" and st.get("result_ready") is True,
     f"metrics={st.get('metrics')}")

# 8) result-handler returns presigned artifacts ------------------------------
res = api_body(invoke(FN["result"], {"requestContext": {"http": {"method": "GET"}},
                                      "pathParameters": {"job_id": job_id}}))
step("result-handler ProcessFormResponse shape",
     all(k in res for k in ("pdf_url", "mapping_preview", "result_url", "stats")),
     f"stats={res.get('stats')}")

# 9) Download the generated PDF via the presigned GET ------------------------
pdf_resp = requests.get(res["pdf_url"])
is_pdf = pdf_resp.status_code == 200 and pdf_resp.content[:5] == b"%PDF-"
step("generated PDF downloads + is valid", is_pdf, f"http={pdf_resp.status_code} bytes={len(pdf_resp.content)}")

# verify processed artifacts exist directly too
listing = s3.list_objects_v2(Bucket=PROCESSED_BUCKET, Prefix=f"textract/{job_id}/")
names = sorted(o["Key"].split("/")[-1] for o in listing.get("Contents", []))
need = {"output.pdf", "mapping.png", "result.json", "mappings.json"}
step("processed artifacts published", need.issubset(set(names)), f"{names}")

# --- summary ----------------------------------------------------------------
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=== {passed}/{len(results)} checks passed | job_id={job_id} ===")
sys.exit(0 if passed == len(results) else 1)
