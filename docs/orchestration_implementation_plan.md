# Implementation plan — placeholder scaffolding → real Textract orchestration

Concrete, slice-by-slice plan to turn the inert async scaffolding into a working
serverless Textract flow, **reusing** the provisioned resources and the existing
Lambda placeholders. Design only — no code here.

Grounded in verified facts:
- DynamoDB `…-jobs`: PK `job_id` (S); GSIs `status-created_at-index`,
  `user_id-created_at-index`; on-demand; attrs `job_id,status,created_at,user_id`.
- SQS `…-processing-queue`: standard; visibility 900s; DLQ at maxReceiveCount=3; 4-day retention.
- Stub Lambdas: py3.12 Zip, entrypoint `lambda_function.lambda_handler`, role
  `FormPdfPocLambdaExecutionRole`, env `{DDB_TABLE, SQS_QUEUE_URL, RAW_BUCKET, PROCESSED_BUCKET, PROJECT, ENV}`.
- Worker: container, built/pushed, **not yet created** (IAM-gated); emits
  `textract/<job_id>/{output.pdf,mappings.json,result.json,mapping.png,...}`.
- Frontend contract (`frontend/lib/api.ts`): `processForm` → `POST {base}/process-form`
  (multipart `file`+`mode`) → `{status, mode, pdf_url, mapping_preview?, result_url?, stats}`;
  artifacts fetched same-origin under `/files/...`; `NEXT_PUBLIC_API_BASE_URL`.
- boto3 is in the Lambda py3.12 runtime → the Zip handlers need **no extra deps** (code-only updates).

Guiding rules: reuse the 4 placeholders in place (keep `lambda_function.lambda_handler`);
keep EC2/FastAPI + OCR live and isolated throughout; additive + flag-gated; every slice
independently reversible.

---

## 0. Canonical job item (DynamoDB) — the contract all slices share

```
job_id      S  (PK)        uuid4
status      S  (GSI hash)  QUEUED → PROCESSING → SUCCEEDED | FAILED | DEAD_LETTER
created_at  S  (GSI range) ISO-8601 (presign time)
updated_at  S              ISO-8601 (last transition)
user_id     S  (GSI hash)  caller identity or "anonymous"
engine      S              "textract"
mode        S              processing mode echoed from the frontend
input       M              { raw_bucket, raw_key, content_type, size }
artifacts   M              { base_uri, output_pdf, mapping_png, mappings_json, result_json }  (SUCCEEDED)
metrics     M              { fields_detected, checkboxes_detected, tables_detected, processing_time_ms }
attempts    N              SQS receive count (observability)
error       M              { type, message }  (FAILED | DEAD_LETTER)
expires_at  N              TTL epoch (auto-prune)
```
State machine is enforced with **conditional writes** (see §6).

---

## 1. presigned-url  (API entry: start a job)

- **Responsibilities:** mint `job_id` (uuid4), derive raw key
  `uploads/<job_id>/<sanitized_filename>`, write DDB `QUEUED` item, return a presigned
  upload + the `job_id`.
- **Route:** `POST /uploads`  (Lambda proxy).
- **Request:** `{ "filename": str, "content_type": str, "mode": str, "user_id"?: str }`
- **Response 200:** `{ "job_id", "upload": { "url", "method", "fields"?|"headers" }, "raw_key", "expires_in" }`
- **Presign mechanism:** prefer **`generate_presigned_post`** with conditions
  `content-length-range (0, 20MB)` + `Content-Type` (enforces the 20MB/type limits at S3);
  simpler alternative `generate_presigned_url('put_object')` (no size enforcement).
- **Permissions (role):** `s3:PutObject` on `raw/uploads/*` (signs the URL),
  `dynamodb:PutItem` on jobs.
- **Failure:** invalid/oversized type → 400 (no DDB write); DDB failure → 500 (client retries → new job).
- **Idempotency:** each call = a new job. Un-uploaded jobs stay `QUEUED` and are pruned by TTL.
- **Frontend impact:** new first call (JSON metadata) before the file leaves the browser.
- **Deploy order:** slice 3 (with job-status/result + routes).
- **Rollback:** revert to placeholder zip; remove route.

## 2. upload-handler  (S3 → enqueue)

- **Responsibilities:** on raw `ObjectCreated`, derive `job_id` from the key, enrich, and
  `SendMessage` to SQS `{ job_id, bucket, key, version_id, mode, user_id }`; optional DDB
  `updated_at`/`uploaded` flag.
- **Trigger:** S3 event notification on raw bucket, prefix `uploads/` (NOT an API route).
- **Permissions:** `sqs:SendMessage` on the queue; `dynamodb:UpdateItem` (optional);
  `s3:GetObject`/HeadObject on raw (metadata). Plus an S3→Lambda **resource policy**
  (`lambda:InvokeFunction` for `s3.amazonaws.com`, source = raw bucket) — currently absent.
- **Request (event):** S3 notification `Records[*].s3`.
- **Failure:** SQS send fails → raise → S3-async retry; configure a Lambda async failure
  destination/DLQ for safety.
- **Idempotency:** S3 may double-deliver; carry `job_id`; optionally conditional
  `UpdateItem … attribute_not_exists(enqueued_at)` to enqueue once. Duplicate SQS messages
  are absorbed by the worker's conditional claim (§6).
- **Frontend impact:** none.
- **Alternative (simpler):** **S3 → SQS notification directly** (drop this Lambda); then
  `mode`/`user_id` must travel via S3 object metadata set at presign. Recommended to keep
  `upload-handler` (reuses a provisioned stub; richer message; one place to validate).
- **Deploy order:** slice 4.
- **Rollback:** remove the S3 notification (stops enqueues instantly); revert stub.

## 3. worker  (SQS → process)  — already built + planned delta

- **Responsibilities:** per message: claim `QUEUED→PROCESSING` (conditional), download raw
  → render/Textract/anchor/PDF (existing pipeline) → publish `textract/<job_id>/…` →
  `SUCCEEDED` (+artifacts/metrics) or `FAILED`.
- **Trigger:** SQS **event-source mapping** (batch=1 initially; `maximum_concurrency` set).
- **Permissions:** `sqs:ReceiveMessage/DeleteMessage/GetQueueAttributes`; `s3:GetObject`
  (raw), `s3:PutObject` (processed); `textract:AnalyzeDocument`; `dynamodb:UpdateItem` (jobs); logs.
- **Failure:** transient (Textract/S3 5xx, throttle, timeout) → **raise** → SQS retry → DLQ
  at 3; permanent (bad/unsupported file, missing object) → DDB `FAILED` + **ack** (don't
  burn retries). `ReportBatchItemFailures` once batch>1.
- **Idempotency:** conditional claim (only `QUEUED`/`PROCESSING`→proceed; terminal→skip+ack);
  deterministic artifact keys (overwrite-safe); at-least-once safe.
- **Code delta (separate PR, already specced):** accept SQS event shape (`Records[*].body`);
  wrap `process_job` with DDB transitions + retry classification; thin lazy-boto3 `dynamodb`
  module. **No pipeline/OCR change.**
- **Frontend impact:** none.
- **Deploy order:** slice 2 (deploy function dark) then slice 5 (create ESM).
- **Rollback:** delete/disable the ESM (stops consumption); `delete-function` removes worker; ECR image retained.

## 4. job-status  (API: poll)

- **Responsibilities:** `GetItem` by `job_id`; return status + readiness.
- **Route:** `GET /jobs/{job_id}`.
- **Response 200:** `{ "job_id", "status", "mode", "created_at", "updated_at", "metrics"?,
  "error"?, "result_ready": status=="SUCCEEDED" }`; **404** if absent.
- **Permissions:** `dynamodb:GetItem` on jobs.
- **Failure:** not found → 404; DDB error → 500 (client keeps polling).
- **Idempotency:** read-only.
- **Frontend impact:** the poll target.
- **Deploy order:** slice 3.
- **Rollback:** revert stub; remove route.

## 5. result-handler  (API: fetch outputs)

- **Responsibilities:** verify `SUCCEEDED`; presign **GET** URLs for the processed
  artifacts; return them in the **frontend's existing response shape** to minimize change.
- **Route:** `GET /result/{job_id}`.
- **Response 200 (mirrors `ProcessFormResponse`):**
  `{ "job_id","status","mode","pdf_url","mapping_preview","result_url","stats" }`
  where `pdf_url`/`mapping_preview`/`result_url` are **presigned S3 GET** URLs for
  `output.pdf`/`mapping.png`/`result.json`; `stats` = `metrics`.
- **Permissions:** `dynamodb:GetItem`; `s3:GetObject` on `processed/textract/*` (signs GETs).
- **Failure:** not `SUCCEEDED` → 409 + current status; not found → 404.
- **Idempotency:** read-only (fresh presigned URLs each call; ~10-min expiry).
- **Frontend impact:** consumes the same fields — but note the **origin/`/files/` guard**
  in `lib/api.ts` must be relaxed for the async path (presigned URLs are cross-origin and
  not `/files/`). See §7.
- **Deploy order:** slice 3.
- **Rollback:** revert stub; remove route.

## 6. DynamoDB state transitions (conditional, idempotent)

```
presign      : PutItem  status=QUEUED   (ConditionExpression: attribute_not_exists(job_id))
worker claim : UpdateItem status=PROCESSING
                 Condition: status IN (QUEUED, PROCESSING)        # dup after terminal → skip+ack
worker done  : UpdateItem status=SUCCEEDED, SET artifacts, metrics, updated_at
                 Condition: status = PROCESSING
worker fail  : UpdateItem status=FAILED, SET error, updated_at
                 Condition: status <> SUCCEEDED
dlq consumer : UpdateItem status=DEAD_LETTER, SET error           # later slice
```
A `ConditionalCheckFailedException` on claim/done means a duplicate/late delivery → treat
as a no-op success (ack the message). This is the core of at-least-once safety.

## 7. Frontend async integration (additive, no rewrite)

Single integration point: `frontend/lib/api.ts`. **Add** `processFormAsync`, keep
`processForm` (sync EC2) for OCR + fallback. Route by mode/flag.

```
processFormAsync(file, mode):
  1. POST {ASYNC_BASE}/uploads { filename, content_type, mode }   → { job_id, upload }
  2. upload file to S3 via the presigned POST/PUT
  3. poll GET {ASYNC_BASE}/jobs/{job_id}  (backoff)  until terminal
  4. if SUCCEEDED → GET {ASYNC_BASE}/result/{job_id} → ProcessFormResponse
     if FAILED    → throw error.message
  5. return the SAME ProcessFormResponse  → result page unchanged
```
- **Routing:** `mode === "textract" && NEXT_PUBLIC_TEXTRACT_ASYNC === "true"` → async; else
  the existing sync `POST /process-form` (EC2). One env flag = instant cutover/rollback.
- **URL guard:** add an async-aware consumer that accepts presigned S3 GET URLs (the
  current `normalizeBackendFileUrl` enforces same-origin + `/files/` — keep it for the sync
  path, branch for async). This is the main code change.
- **Config:** `NEXT_PUBLIC_API_BASE_URL` (EC2 sync, unchanged) + `NEXT_PUBLIC_ASYNC_API_BASE_URL`
  (API GW). Additive env; existing builds keep working.
- **Result page / upload page:** result page consumes the same shape (no change); upload
  page gains QUEUED/PROCESSING polling states.

### How polling should work
Backoff: 1.5s → 2s → 3s (cap ~3-5s), overall timeout ~3 min; stop on terminal status;
show progress; on timeout show "still processing, check history". Read-only `job-status`
is cache-safe and cheap (DDB GetItem, on-demand).

### How presigned URLs are generated
- Upload: `presigned-url` λ → `generate_presigned_post` (size+type conditions) on raw,
  ~5-min expiry. Requires **raw-bucket CORS** allowing the frontend origin + `PUT/POST`.
- Download: `result-handler` λ → `generate_presigned_url('get_object')` on processed,
  ~10-min expiry. Browser GET is cross-origin → fine for `<a download>`/`<img>`; if fetched
  via JS, processed-bucket CORS may be needed for `GET`.

### How processed artifacts are surfaced
`result-handler` maps `textract/<job_id>/output.pdf|mapping.png|result.json` → presigned
GETs and returns them as `pdf_url|mapping_preview|result_url`, exactly the fields the
result page already renders.

### How EC2 fallback coexists during rollout
EC2/FastAPI stays fully live. OCR mode → **always** EC2 sync (isolated, untouched).
Textract mode → flag-routed: async (new) or EC2 sync (current). Both emit the same output
contract, so the result page is agnostic. Flip `NEXT_PUBLIC_TEXTRACT_ASYNC` off → 100% EC2
instantly; async infra can stay deployed and idle.

## 8. API Gateway route structure (HTTP API 58is64i9kb — reuse)

| Method | Route | Integration | Auth |
|---|---|---|---|
| POST | `/uploads` | presigned-url λ (AWS_PROXY) | (open/JWT later) |
| GET | `/jobs/{job_id}` | job-status λ | same |
| GET | `/result/{job_id}` | result-handler λ | same |
| (n/a) | S3 ObjectCreated | upload-handler λ (S3 trigger) | — |

- **CORS:** configure allowed origin (frontend), methods `GET,POST,OPTIONS`, headers
  `content-type`. HTTP API handles `OPTIONS` preflight.
- **Permissions:** each route adds `lambda:InvokeFunction` for `apigateway.amazonaws.com`
  (source = the API/route ARN). Currently absent on all functions.

---

## 9. Deployment sequencing (ordered, each reversible)

| # | Step | Reversible by |
|---|---|---|
| 0 | **(gate)** Admin IAM: `CreateFunction`+`PassRole`; role gains Textract/DDB/SQS/S3 + per-λ invoke | n/a (perm grant) |
| 1 | Update `FormPdfPocLambdaExecutionRole` policy (Textract, dynamodb, sqs, s3) | detach policy |
| 2 | Deploy **worker** from ECR (no ESM) — **dark**; validate via manual invoke (`scripts/lambda_test_event.json`) | `delete-function` |
| 3 | Implement+deploy **presigned-url / job-status / result-handler**; add API GW routes + CORS + invoke perms; curl-test API surface (jobs stay QUEUED) | revert stubs; delete routes |
| 4 | Implement+deploy **upload-handler**; add raw-bucket S3 notification + S3→λ perm | remove notification |
| 5 | Create **SQS→worker ESM** (batch=1, max-concurrency) → end-to-end works; test a real upload | delete ESM |
| 6 | Raw-bucket **CORS**, **S3 lifecycle**, **CloudWatch alarms** (DLQ depth, queue age, worker errors) | remove rules/alarms |
| 7 | Frontend: add async client + flag (default OFF); deploy build pointing at API GW; validate | flag stays OFF |
| 8 | Flip `NEXT_PUBLIC_TEXTRACT_ASYNC=true` for Textract; monitor; EC2 retained | flip flag OFF |

Steps 2–6 are invisible to users (frontend still on EC2). Real user exposure happens only
at step 8 and is a single env flag.

## 10. Rollback strategy

- **Master switch:** `NEXT_PUBLIC_TEXTRACT_ASYNC=false` → 100% EC2 sync (current prod),
  instantly, no infra teardown.
- **Stop processing:** delete the SQS→worker ESM and the raw S3 notification (uploads stop
  enqueuing/consuming; in-flight messages age out to DLQ).
- **Revert handlers:** redeploy the saved placeholder zips (keep copies) — functions return
  to inert stubs.
- **Remove worker:** `scripts/deploy_lambda_worker.ps1 -Delete` (ECR image kept).
- **EC2 + OCR untouched** at every step → the synchronous path is always a live fallback.

## 11. Permissions roll-up (single shared role)

`FormPdfPocLambdaExecutionRole` (used by all): `dynamodb:PutItem/GetItem/UpdateItem/Query`
on jobs (+indexes); `sqs:SendMessage` + `sqs:ReceiveMessage/DeleteMessage/GetQueueAttributes`;
`s3:PutObject`+`s3:GetObject` on raw+processed; `textract:AnalyzeDocument`; CloudWatch Logs.
*(Hardening option: split per-function least-privilege roles later; one role is simpler now
and matches the current setup.)*

## 12. Open confirmations (admin)
Whether API GW already has any routes/integrations (`apigateway:GET` denied), and the exact
role policy contents (IAM reads denied). Evidence says none/insufficient, but confirm before
slice 1/3 to avoid duplicate routes.
