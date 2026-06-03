# Go-live checklist — async Textract rollout (Task 2)

Operator runbook for activating the serverless Textract flow. Specific to this
system (account `637423601842`, region `ap-south-1`, API `58is64i9kb`). Each step
has a concrete verify and a rollback. Backend logic is already validated E2E (14/14);
this is the activation + cutover sequence.

Resources: raw `form-pdf-poc-dev-raw-documents` · processed `form-pdf-poc-dev-processed-documents`
· DDB `form-pdf-poc-dev-jobs` · queue `…-processing-queue` · DLQ `…-processing-dlq`
· worker `…-dev-worker` · handlers `…-dev-{presigned-url,upload-handler,job-status,result-handler}`.

---

## 0. Pre-flight (already done — verify, don't redo)

- [ ] Handlers deployed (not stubs): `aws lambda get-function-configuration --function-name form-pdf-poc-dev-presigned-url --query CodeSize` → > 363.
- [ ] Worker on the async image with `DDB_TABLE` + `/tmp` work root:
      `aws lambda get-function-configuration --function-name form-pdf-poc-dev-worker --query "Environment.Variables.{DDB:DDB_TABLE,WR:FORM_PARSER_WORK_ROOT}"`
      → `{DDB: form-pdf-poc-dev-jobs, WR: /tmp/form_parser_jobs}`.
- [ ] DLQ empty: `aws sqs get-queue-attributes --queue-url <DLQ> --attribute-names ApproximateNumberOfMessages`.

## 1. Infra activation (ELEVATED identity — `wire_async_infra.ps1`)

Run in this **order** (worker must be able to consume before any producer exists):

- [ ] **API Gateway routes + CORS** — `wire_async_infra.ps1 -ApiRoutes -FrontendOrigin https://<app>`
      Verify: `curl -s -XPOST $A/uploads -H 'content-type: application/json' -d '{"filename":"f.pdf","content_type":"application/pdf","mode":"textract"}'` → JSON with `job_id` + `upload`. (404/{"message":"Not Found"} ⇒ route missing; 500 ⇒ invoke-permission missing.)
- [ ] **SQS → worker ESM** — `wire_async_infra.ps1 -Esm`
      Verify: `aws lambda list-event-source-mappings --function-name form-pdf-poc-dev-worker --query "EventSourceMappings[].State"` → `Enabled`.
- [ ] **S3 → upload-handler notification** — `wire_async_infra.ps1 -S3Notify`
      Verify: upload a test object under `uploads/` and confirm a message lands on the queue (or a worker log line appears).
- [ ] **Bucket CORS** — `wire_async_infra.ps1 -Cors -FrontendOrigin https://<app>`
      Verify (browser): result-page overlay `fetch(result.json)` succeeds (no CORS error in console).

`A=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com`

## 2. Backend E2E validation (before frontend)

- [ ] `.venv/Scripts/python.exe scripts/e2e_async_validate.py` → **14/14** (or curl the API per §1).
- [ ] DDB transitions observed: `QUEUED → PROCESSING → SUCCEEDED`
      `aws dynamodb get-item --table-name form-pdf-poc-dev-jobs --key '{"job_id":{"S":"<id>"}}' --query "Item.status.S"`.
- [ ] Processed artifacts present: `aws s3 ls s3://form-pdf-poc-dev-processed-documents/textract/<id>/` → output.pdf, mapping.png, result.json, mappings.json.
- [ ] DLQ still 0; worker log shows `succeeded=1 failed=0`.

## 3. Frontend deployment (see docs/frontend_deployment.md)

- [ ] Build with shell env vars, **flag false**: `aws s3 sync frontend/out/ s3://<FRONTEND_BUCKET>/ --delete` + CF invalidate.
- [ ] Site loads; sync path unaffected.
- [ ] Rebuild **flag true**; redeploy; CF invalidate.

## 4. Production validation (real user path)

- [ ] Upload one real form via the UI → progress shows QUEUED → PROCESSING.
- [ ] Result page renders; **Download PDF** works (presigned GET 200, `%PDF-`).
- [ ] Advanced-details overlay renders (confirms processed-bucket CORS).
- [ ] Repeat once more; both jobs reach SUCCEEDED; DLQ stays 0.

## 5. Observability watch (first 30 min) — see docs/observability_runbook.md

- [ ] Worker errors: `aws logs tail /aws/lambda/form-pdf-poc-dev-worker --since 30m --filter-pattern "ERROR"` (use `MSYS_NO_PATHCONV=1` in Git Bash).
- [ ] Queue not backing up: `ApproximateNumberOfMessages` trends to 0.
- [ ] DLQ depth = 0. Any DLQ message ⇒ investigate before continuing rollout.

---

## Rollback ladder (cheapest first)

1. **Flag off** → rebuild `NEXT_PUBLIC_TEXTRACT_ASYNC=false` + redeploy frontend → 100% sync path. Instant, no infra teardown.
2. **Stop consumption:** `aws lambda delete-event-source-mapping --uuid <UUID>` → worker stops pulling SQS (in-flight ages to DLQ).
3. **Stop ingestion:** set raw-bucket notification config to `{}` → uploads no longer enqueue.
4. **Revert handlers:** `pwsh scripts/deploy_async_handlers.ps1 -Restore` → stubs.
5. **Revert worker:** redeploy the previous ECR image tag (image retained).
6. EC2 + OCR code untouched throughout = always-available fallback (when EC2 is restored).

## Critical failure indicators → fast recovery

| Symptom | Likely cause | Fast action |
|---|---|---|
| API `/uploads` 500 | invoke-permission missing on handler | re-run `-ApiRoutes` (AddPermission) |
| API `/uploads` 404 | route not created | re-run `-ApiRoutes` |
| Upload 403 in browser | raw-bucket CORS or presign expiry | re-run `-Cors`; re-request presign |
| Jobs stuck `QUEUED` | ESM disabled / S3 notification missing | check ESM `Enabled`; verify notification |
| Jobs stuck `PROCESSING` | worker timeout / crash mid-job | check worker logs; message retries → DLQ at 3 |
| Messages piling in DLQ | permanent worker failure (bad input / bug) | inspect DLQ body + worker logs; fix; redrive |
| Overlay fails, PDF OK | processed-bucket GET CORS missing | re-run `-Cors` (non-blocking; PDF still works) |
| Polling never resolves | job_id mismatch / result-handler error | compare job_id across DDB/SQS/processed prefix |

## Final sequences (quick reference)

**Deploy:** `-ApiRoutes` → `-Esm` → `-S3Notify` → `-Cors` → frontend (flag false) → frontend (flag true).
**Validate:** `e2e_async_validate.py` → DDB status → S3 artifacts → real UI upload → DLQ=0.
**Rollback:** flag off → delete ESM → empty notification → restore handlers → revert worker image.
