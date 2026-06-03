# Admin activation runbook — async Textract go-live

Hand-off for an **elevated/admin identity**. The async backend logic is fully
implemented, deployed, and validated end-to-end against real AWS (14/14 checks,
real Textract, full DDB lifecycle). What remains is the IAM-gated trigger wiring
that the `intern.form-pdf-poc` identity cannot perform.

Account `637423601842` · region `ap-south-1` · API `58is64i9kb`.

---

## Already done (by the dev identity — do NOT repeat)

- ✅ 4 API handlers deployed into the existing functions (`presigned-url`,
  `upload-handler`, `job-status`, `result-handler`).
- ✅ Worker rebuilt from `Dockerfile.lambda` with the SQS-event + DDB
  state-machine delta; `DDB_TABLE=form-pdf-poc-dev-jobs` set; `FORM_PARSER_WORK_ROOT=/tmp/form_parser_jobs`.
- ✅ Runtime IAM on `FormPdfPocLambdaExecutionRole` is **already sufficient**
  (verified live: DDB Put/Get/Update, S3 get/put + presign, Textract all work).
  → The `-Role` slice below is **optional** (least-privilege documentation only).
- ✅ E2E validated by direct invoke (substituting for the not-yet-wired triggers):
  presigned → S3 upload → upload-handler → SQS → worker → Textract → DDB
  SUCCEEDED → job-status → result-handler → valid PDF. Idempotency + DLQ clean.
- ✅ Bug fixed during validation: DDB reserved keywords (`metrics`/`artifacts`/
  `attempts`) now aliased in `src/job_state.py`.

## Remaining (admin) — the actual go-live switches

Run from the repo root with an elevated profile. Each slice is idempotent and
independently reversible.

```powershell
# 1. API Gateway routes + invoke perms + CORS  (frontend cannot reach the async
#    API without this — processFormAsync calls /uploads,/jobs,/result)
.\scripts\wire_async_infra.ps1 -ApiRoutes -Profile <admin> -FrontendOrigin https://<frontend-origin>

# 2. SQS -> worker event-source mapping (worker is validated; safe to enable)
.\scripts\wire_async_infra.ps1 -Esm -Profile <admin>

# 3. S3 raw ObjectCreated(uploads/) -> upload-handler + invoke perm
.\scripts\wire_async_infra.ps1 -S3Notify -Profile <admin>

# 4. Bucket CORS (browser POST to raw, GET from processed)
.\scripts\wire_async_infra.ps1 -Cors -Profile <admin> -FrontendOrigin https://<frontend-origin>

# (optional) least-privilege runtime policy — already effectively present
.\scripts\wire_async_infra.ps1 -Role -Profile <admin>
```

**Ordering matters:** worker first (already done) → ESM → S3 notification. Never
enable the S3 notification before the ESM exists and the worker is the validated
image, or enqueued messages would have no consumer. (Both are now satisfied.)

## Validate after wiring (curl the real API)

```bash
A=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com
# presign
curl -s -XPOST $A/uploads -H 'content-type: application/json' \
  -d '{"filename":"form.pdf","content_type":"application/pdf","mode":"textract"}'
# → upload the file to the returned presigned POST, then:
curl -s $A/jobs/<job_id>     # QUEUED → PROCESSING → SUCCEEDED
curl -s $A/result/<job_id>   # pdf_url / mapping_preview / result_url / stats
```

Watch: `aws logs tail /aws/lambda/form-pdf-poc-dev-worker --since 10m` and the
DLQ depth (`form-pdf-poc-dev-processing-dlq` should stay 0).

## Frontend cutover (last)

1. Build with `NEXT_PUBLIC_ASYNC_API_BASE_URL=$A` and keep
   `NEXT_PUBLIC_TEXTRACT_ASYNC=false`; deploy to the frontend bucket.
2. Smoke-test by temporarily flipping the flag locally.
3. Flip `NEXT_PUBLIC_TEXTRACT_ASYNC=true` and redeploy → Textract uploads go async;
   `rule`/`ml` stay on EC2 sync.

## Rollback ladder (cheapest first)

1. `NEXT_PUBLIC_TEXTRACT_ASYNC=false` + redeploy → 100% EC2 sync, instant, no infra teardown.
2. `delete-event-source-mapping` + empty S3 notification → auto-processing stops; in-flight ages to DLQ.
3. `.\scripts\deploy_async_handlers.ps1 -Restore` → handlers revert to stubs.
4. Redeploy prior worker image tag (ECR retains it).
5. EC2 + OCR untouched throughout = always-live fallback.

## Known residue (benign)

- One test job (`863c1612…`) is stuck `PROCESSING` from a pre-fix run; its SQS
  message is in-flight. When the ESM is enabled it will be reprocessed to
  SUCCEEDED (conditional `claim` permits re-entry from PROCESSING). No action needed.
