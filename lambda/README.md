# Async Textract Lambda handlers

The four API/event handlers that turn the deployed worker into an end-to-end
async flow. Each is a single dependency-free `lambda_function.py` (boto3 ships in
the py3.12 runtime) deployed into the **existing** stub functions in place —
handler stays `lambda_function.lambda_handler`, role stays
`FormPdfPocLambdaExecutionRole`.

```
frontend ──POST /uploads──▶ presigned-url ──┐ DDB QUEUED
                                            └─▶ presigned POST
browser ──PUT/POST file──▶ raw S3 (uploads/<job_id>/<file>)
raw S3 ──ObjectCreated──▶ upload-handler ──SendMessage──▶ SQS
SQS ──ESM──▶ worker (claim→Textract→publish→SUCCEEDED/FAILED)  processed S3 (textract/<job_id>/…)
frontend ──GET /jobs/{id}──▶ job-status (poll DDB)
frontend ──GET /result/{id}──▶ result-handler (presigned GETs, ProcessFormResponse shape)
```

| Function | Trigger | Route | Reads/Writes |
|---|---|---|---|
| `presigned_url` | API GW | `POST /uploads` | DDB PutItem (QUEUED), S3 presign POST (raw) |
| `upload_handler` | S3 ObjectCreated | — | S3 HeadObject (raw), DDB UpdateItem, SQS SendMessage |
| `job_status` | API GW | `GET /jobs/{job_id}` | DDB GetItem |
| `result_handler` | API GW | `GET /result/{job_id}` | DDB GetItem, S3 presign GET (processed) |

## The shared contract

- **job_id** is minted by `presigned-url` (uuid4) and is authoritative end to
  end. The raw key is always `uploads/<job_id>/<sanitized_filename>`.
- `upload-handler` recovers `job_id` from the key and `mode`/`user_id` from the
  object's user-metadata (stamped at presign time).
- The worker uses the SQS message's `job_id` for both the DynamoDB row and the
  processed prefix `textract/<job_id>/…`, so polling + result fetch line up.
- `result-handler` returns the **same `ProcessFormResponse` shape** as the EC2
  sync path (`pdf_url`, `mapping_preview`, `result_url`, `stats`) so the result
  page renders with only a URL-validation branch (`src=async`).

## DynamoDB job item

```
job_id (PK) | status (QUEUED→PROCESSING→SUCCEEDED|FAILED) | created_at | updated_at
user_id | engine="textract" | mode | input{raw_bucket,raw_key,content_type}
artifacts{base_uri,output_pdf,mapping_png,mappings_json,result_json} | metrics{…}
attempts | error{type,message} | enqueued_at | expires_at (TTL)
```

State transitions use **conditional writes** (`src/job_state.py`) so at-least-once
SQS delivery is idempotent.

## Deploy

Code (deploy identity, the only piece intern.form-pdf-poc can likely do):

```powershell
pwsh scripts/deploy_async_handlers.ps1 -Backup   # snapshot stubs for rollback
pwsh scripts/deploy_async_handlers.ps1           # push all 4 handlers
```

Wiring (ELEVATED identity — IAM, API routes, S3 trigger, SQS ESM, CORS):

```powershell
pwsh scripts/wire_async_infra.ps1 -All -FrontendOrigin https://your-app
```

Worker code delta (SQS shape + DDB state machine) ships in the container — rebuild:

```powershell
pwsh scripts/deploy_lambda_worker.ps1            # now sets DDB_TABLE too
```

## Rollback

- **User-facing master switch:** `NEXT_PUBLIC_TEXTRACT_ASYNC=false` → 100% EC2
  sync, no infra teardown.
- Stop processing: delete the SQS→worker ESM and the raw S3 notification.
- Revert handlers: `pwsh scripts/deploy_async_handlers.ps1 -Restore`.
- EC2/FastAPI + OCR are untouched throughout and remain a live fallback.

## Frontend env

```
NEXT_PUBLIC_API_BASE_URL=https://<ec2-or-existing>        # sync path (unchanged)
NEXT_PUBLIC_ASYNC_API_BASE_URL=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com
NEXT_PUBLIC_TEXTRACT_ASYNC=false                          # flip to true at cutover
```
