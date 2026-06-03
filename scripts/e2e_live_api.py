"""Live HTTP E2E against the deployed API Gateway /dev stage.

Unlike scripts/e2e_async_validate.py (which invoked Lambdas directly), this drives
the EXACT HTTP surface the browser frontend uses — proving routes, integrations,
the S3-notification -> SQS -> worker AUTO-trigger, and the polling contract:

  POST {API}/uploads -> presigned POST -> upload to S3 -> [auto] S3 notif -> SQS
  -> worker -> Textract -> DDB -> poll GET {API}/jobs/{id} -> GET {API}/result/{id}
  -> download PDF.

Polling mirrors the frontend exactly (1.5->4s backoff, 3-min cap). boto3 is used
ONLY for read-only observability (DDB/SQS), not to drive the flow.

Usage:  .venv/Scripts/python.exe scripts/e2e_live_api.py [path-to-pdf]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import boto3
import requests

API = "https://58is64i9kb.execute-api.ap-south-1.amazonaws.com/dev"
REGION = "ap-south-1"
TABLE = "form-pdf-poc-dev-jobs"
PROCESSED_BUCKET = "form-pdf-poc-dev-processed-documents"
QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/637423601842/form-pdf-poc-dev-processing-queue"
DLQ_URL = "https://sqs.ap-south-1.amazonaws.com/637423601842/form-pdf-poc-dev-processing-dlq"

PDF = Path(sys.argv[1] if len(sys.argv) > 1 else "input/form.pdf")

sess = boto3.Session(profile_name="form-pdf-poc", region_name=REGION)
ddb = sess.client("dynamodb")
sqs = sess.client("sqs")
s3 = sess.client("s3")

results: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def ddb_status(job_id: str) -> str:
    item = ddb.get_item(TableName=TABLE, Key={"job_id": {"S": job_id}}).get("Item")
    return item["status"]["S"] if item else "ABSENT"


def queue_depth(url: str) -> tuple[str, str]:
    a = sqs.get_queue_attributes(
        QueueUrl=url,
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
    )["Attributes"]
    return a.get("ApproximateNumberOfMessages", "?"), a.get("ApproximateNumberOfMessagesNotVisible", "?")


# Mirror the frontend's normalizeAsyncFileUrl guard.
def is_safe_async_url(url: str) -> bool:
    from urllib.parse import urlparse

    u = urlparse(url or "")
    return u.scheme == "https" and (u.hostname or "").endswith("amazonaws.com")


print(f"\n=== Live HTTP E2E via {API} using {PDF} ===\n")
assert PDF.is_file(), f"missing {PDF}"
dlq_before = queue_depth(DLQ_URL)[0]

# 1) POST /uploads ------------------------------------------------------------
r = requests.post(
    f"{API}/uploads",
    json={"filename": PDF.name, "content_type": "application/pdf", "mode": "textract"},
    timeout=30,
)
step("POST /dev/uploads -> 200", r.status_code == 200, f"http={r.status_code}")
pre = r.json()
job_id = pre.get("job_id", "")
step("response has job_id + presigned upload", bool(job_id and pre.get("upload", {}).get("url")), f"job_id={job_id}")
step("DDB row QUEUED right after presign", ddb_status(job_id) == "QUEUED")

# 2) Browser-style presigned POST upload to S3 -------------------------------
up = pre["upload"]
step("presigned upload host is *.amazonaws.com", is_safe_async_url(up["url"]), up["url"])
with PDF.open("rb") as fh:
    post = requests.post(up["url"], data=up.get("fields", {}), files={"file": (PDF.name, fh, "application/pdf")}, timeout=60)
step("upload to S3 (presigned POST)", post.status_code in (200, 201, 204), f"http={post.status_code}")

# 3) Poll GET /jobs/{id} like the frontend (auto S3->SQS->worker must drive it)
intervals = [1.5, 2, 3, 3, 4]
deadline = time.time() + 180
attempt = 0
last = None
terminal = None
print("    ... polling /dev/jobs (auto-trigger: S3 notif -> SQS -> worker -> DDB)")
while time.time() < deadline:
    time.sleep(intervals[min(attempt, len(intervals) - 1)])
    attempt += 1
    jr = requests.get(f"{API}/jobs/{job_id}", timeout=30)
    if jr.status_code != 200:
        continue
    js = jr.json()
    status = js.get("status")
    if status != last:
        print(f"        t+{int(time.time()-(deadline-180))}s  status={status}")
        last = status
    if js.get("terminal"):
        terminal = js
        break

step("polling reached a terminal state (no deadlock)", terminal is not None, f"status={(terminal or {}).get('status')}")
step("auto-pipeline drove job to SUCCEEDED", bool(terminal) and terminal.get("status") == "SUCCEEDED",
     f"metrics={(terminal or {}).get('metrics')}")

# 4) GET /result/{id} ---------------------------------------------------------
rr = requests.get(f"{API}/result/{job_id}", timeout=30)
step("GET /dev/result -> 200", rr.status_code == 200, f"http={rr.status_code}")
res = rr.json()
step("result has ProcessFormResponse shape",
     all(k in res for k in ("pdf_url", "mapping_preview", "result_url", "stats")), f"stats={res.get('stats')}")
step("all artifact URLs pass the frontend safety guard",
     all(is_safe_async_url(res.get(k, "")) for k in ("pdf_url", "mapping_preview", "result_url")))

# 5) Download the generated PDF via the presigned GET ------------------------
pdf = requests.get(res["pdf_url"], timeout=60)
step("generated PDF downloads + valid", pdf.status_code == 200 and pdf.content[:5] == b"%PDF-",
     f"http={pdf.status_code} bytes={len(pdf.content)}")

# 6) result.json fetch (the result-page overlay does this cross-origin) ------
rj = requests.get(res["result_url"], timeout=30)
step("result.json fetch ok (overlay source)", rj.status_code == 200 and isinstance(rj.json(), list),
     f"http={rj.status_code} mappings={len(rj.json()) if rj.status_code==200 else '?'}")

# 7) Observability ------------------------------------------------------------
listing = s3.list_objects_v2(Bucket=PROCESSED_BUCKET, Prefix=f"textract/{job_id}/")
names = sorted(o["Key"].split("/")[-1] for o in listing.get("Contents", []))
need = {"output.pdf", "mapping.png", "result.json", "mappings.json"}
step("processed artifacts present", need.issubset(set(names)), f"{names}")
mq, mqn = queue_depth(QUEUE_URL)
dlq_after = queue_depth(DLQ_URL)[0]
step("no DLQ growth (no failed/retried jobs)", dlq_after == dlq_before, f"dlq {dlq_before}->{dlq_after}")
print(f"    main queue: visible={mq} in-flight={mqn} | DLQ={dlq_after}")

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=== {passed}/{len(results)} checks passed | job_id={job_id} ===")
sys.exit(0 if passed == len(results) else 1)
