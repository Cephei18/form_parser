# Observability & operations runbook — async Textract (Task 5)

Practical debugging for the live async flow. Account `637423601842`, region
`ap-south-1`, profile `form-pdf-poc`. All commands assume those are set.

> **Git Bash gotcha:** prefix log/DDB commands that contain `/aws/...` paths with
> `MSYS_NO_PATHCONV=1`, or MSYS rewrites `/aws/lambda/...` into a Windows path and
> you get a confusing AccessDenied on a mangled log-group name. PowerShell is unaffected.

## Log locations (CloudWatch)

| Component | Log group |
|---|---|
| Worker | `/aws/lambda/form-pdf-poc-dev-worker` |
| presigned-url | `/aws/lambda/form-pdf-poc-dev-presigned-url` |
| upload-handler | `/aws/lambda/form-pdf-poc-dev-upload-handler` |
| job-status | `/aws/lambda/form-pdf-poc-dev-job-status` |
| result-handler | `/aws/lambda/form-pdf-poc-dev-result-handler` |

```bash
# Live tail (worker is the one you'll watch most)
MSYS_NO_PATHCONV=1 aws logs tail /aws/lambda/form-pdf-poc-dev-worker --since 15m --follow
# Errors only
MSYS_NO_PATHCONV=1 aws logs tail /aws/lambda/form-pdf-poc-dev-worker --since 1h --filter-pattern "ERROR"
# Trace one job across handlers (job_id appears in every log line)
MSYS_NO_PATHCONV=1 aws logs tail /aws/lambda/form-pdf-poc-dev-worker --since 1h --format short | grep <job_id>
```

Useful worker log markers: `[worker] running Textract pipeline job_id=…`,
`[worker] job succeeded job_id=… metrics=…`, `[worker] batch complete jobs=N succeeded=N failed=N`,
`[job_state] claim refused …` (idempotent dup), `transient failure … raising for SQS retry`.

## Queue / DLQ inspection

```bash
Q=https://sqs.ap-south-1.amazonaws.com/637423601842/form-pdf-poc-dev-processing-queue
DLQ=https://sqs.ap-south-1.amazonaws.com/637423601842/form-pdf-poc-dev-processing-dlq
# Backlog + in-flight
aws sqs get-queue-attributes --queue-url $Q   --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible --query Attributes
# DLQ depth (should be 0)
aws sqs get-queue-attributes --queue-url $DLQ --attribute-names ApproximateNumberOfMessages --query Attributes
# Peek a DLQ message WITHOUT deleting (visibility 0 so it returns)
aws sqs receive-message --queue-url $DLQ --max-number-of-messages 1 --visibility-timeout 0 --query "Messages[].Body"
```
`ApproximateNumberOfMessagesNotVisible > 0` = jobs currently being processed (in the
worker's visibility window of 900s). Steady non-zero `…NumberOfMessages` = backlog.

## DynamoDB job inspection

```bash
T=form-pdf-poc-dev-jobs
# One job
aws dynamodb get-item --table-name $T --key '{"job_id":{"S":"<id>"}}' \
  --query "Item.{status:status.S,attempts:attempts.N,updated:updated_at.S,err:error.M}"
# All PROCESSING jobs (uses the status GSI) — find stuck work
aws dynamodb query --table-name $T --index-name status-created_at-index \
  --key-condition-expression "#s = :s" --expression-attribute-names '{"#s":"status"}' \
  --expression-attribute-values '{":s":{"S":"PROCESSING"}}' \
  --query "Items[].{job:job_id.S,updated:updated_at.S,attempts:attempts.N}"
```
Swap `:s` for `FAILED` / `DEAD_LETTER` to audit failures.

## API Gateway debugging

- 500 on a route → Lambda not authorized to be invoked (resource policy). Re-run the
  `-ApiRoutes` slice (it adds `lambda:add-permission` with the path-param wildcard).
- 404 → route/integration missing. Re-run `-ApiRoutes`.
- Reads of routes need an elevated identity (`apigateway:GET` is denied to the intern).
  Use the curl smoke test in the go-live checklist instead.

## Troubleshooting scenarios

**Stuck PROCESSING.** Worker crashed/timed out after `claim`. The SQS message
redelivers (visibility 900s); the conditional `claim` allows re-entry from PROCESSING,
so it self-heals on the next delivery. If it persists past 3 receives it lands in the
DLQ. Diagnose via worker logs for that `job_id`; check Textract throttling / timeout.

**Duplicate delivery.** Expected (at-least-once). Second delivery → `claim refused`
log + `status:"skipped"`; DDB already SUCCEEDED → no reprocessing, no double artifacts
(keys are deterministic/overwrite-safe). Nothing to do.

**Missing artifacts.** result-handler 409/404 or empty `s3 ls textract/<id>/`. Check:
(1) job actually SUCCEEDED in DDB; (2) worker log shows `artifacts published`; (3)
`FORM_PARSER_ARTIFACT_PREFIX=textract` + `FORM_PARSER_PROCESSED_BUCKET` on the worker.

**Polling failures / never resolves.** Usually a `job_id` mismatch — confirm the
same id in DDB, the SQS body, and the `textract/<id>/` prefix. Or `job-status` 404 =
presigned-url never wrote the row (check its logs).

**Expired signed URL (403 on download).** Presigned GET TTL is ~10 min. Re-call
`GET /result/{job_id}` to mint fresh URLs (job stays SUCCEEDED; read-only).

**Retry behavior.** Worker classifies failures: transient (throttle/timeout/5xx/conn)
→ raises → SQS retry → DLQ after 3 receives; permanent (bad/unsupported input) →
DDB FAILED + ack (no retry burn). See `_is_transient_error` in `src/lambda_worker.py`.

**DLQ redrive (after a fix).** Once the root cause is fixed, move messages back:
```bash
aws sqs start-message-move-task --source-arn arn:aws:sqs:ap-south-1:637423601842:form-pdf-poc-dev-processing-dlq \
  --destination-arn arn:aws:sqs:ap-south-1:637423601842:form-pdf-poc-dev-processing-queue
```

## Quick-debug checklist

1. DLQ depth 0? Backlog draining?
2. Worker logs clean (`succeeded`, no `ERROR`)?
3. Job row status sane (not stuck PROCESSING > a few min)?
4. Artifacts in `s3://…-processed-documents/textract/<id>/`?
5. API smoke (`POST /uploads`) returns a presigned POST?
6. Browser console free of CORS errors on the result page?

## Rollback references

Full ladder in `docs/go_live_checklist.md` §Rollback. Fastest: frontend flag off
(100% sync) → delete SQS ESM (stop consumption) → empty raw S3 notification (stop ingest).
