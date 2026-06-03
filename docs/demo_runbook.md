# Demo & review runbook — async Textract (Task 6)

Optimized for a short, reliable demo that showcases engineering maturity (async
serverless orchestration, rollback-safe migration) over flash. Assume ~10 minutes.

## The artifacts to use

- **Sample form:** `input/form.pdf` — a real multi-field form; validated to yield
  **12 fields + 1 table** via Textract in ~3–6s. Known-good, deterministic.
- **Clean generated PDF:** the `output.pdf` from a validated run (e.g. the E2E job).
  For the crispest preview screenshot, rebuild the worker with
  `FORM_PARSER_PREVIEW_LABELS=false` (boxes-only `mapping.png`; see `docs/pdf_polish_notes.md`).
- **Stable workflow:** the async path end-to-end (presigned → poll → result).

## Architecture summary (1 minute)

> A scanned form is uploaded directly to S3 via a presigned URL. An S3 event hands
> it to a queue; a containerized Lambda worker runs AWS Textract, anchors fields,
> and renders a fillable PDF to a processed bucket. DynamoDB tracks each job's
> lifecycle; the frontend polls status and fetches the result via presigned URLs.
> Fully serverless, decoupled, and rollback-safe behind a feature flag.

```
Browser → API GW → presigned-url λ → S3 (raw) → upload-handler λ → SQS
        → worker λ (Textract + anchoring + PDF) → S3 (processed) + DynamoDB
Browser ← job-status λ (poll) ← DynamoDB ;  Browser ← result-handler λ (presigned GETs)
```

## Request lifecycle (the story to tell)

1. **Presign** — frontend asks for a presigned POST; a `QUEUED` job row is written (uuid4 `job_id`).
2. **Direct upload** — the browser uploads straight to S3 (no bytes through our API; 20 MB enforced at S3).
3. **Enqueue** — S3 `ObjectCreated` → upload-handler → SQS (carries `job_id`, `mode`).
4. **Process** — SQS triggers the worker: conditional `claim` (QUEUED→PROCESSING) → Textract → publish `textract/<job_id>/…` → `SUCCEEDED`.
5. **Poll & retrieve** — frontend polls `job-status` until SUCCEEDED, then `result-handler` returns presigned PDF/preview/result URLs.

## Talking points (engineering maturity)

- **Lambda architecture:** container-image worker (OpenCV/Poppler/ReportLab) + 4 lightweight zip handlers — no servers to manage.
- **Textract integration:** `AnalyzeDocument` (FORMS+TABLES); field anchoring maps labels → writable regions → AcroForm widgets.
- **Async orchestration:** SQS decouples ingestion from processing; visibility 900s; DLQ at 3 receives.
- **DDB lifecycle + idempotency:** conditional writes make at-least-once delivery safe — duplicates are skipped, never double-processed (verified).
- **Rollback-safe migration:** one feature flag flips 100% back to the prior path; OCR/EC2 code untouched and isolated.
- **Frontend async UX:** live QUEUED/PROCESSING states, bounded polling, presigned-URL handling, graceful errors — no UI rewrite.
- **Validated:** 14/14 automated E2E against real AWS; a real reserved-keyword bug was caught and fixed during stabilization.

## Demo execution checklist

1. [ ] Confirm green before the room: `.venv/Scripts/python.exe scripts/e2e_async_validate.py` → 14/14.
2. [ ] DLQ depth 0; worker logs clean.
3. [ ] Have a **pre-generated** `output.pdf` open as a guaranteed fallback.
4. [ ] Live: upload `input/form.pdf` → narrate QUEUED → PROCESSING → SUCCEEDED.
5. [ ] Open the result page; show the fillable PDF (click a field, type) + stats (12 fields, 1 table).
6. [ ] (Optional) show DynamoDB row transitions and a worker log line for credibility.

## Failure fallback strategy

- **Live upload stalls/fails:** switch to the pre-generated `output.pdf` + the
  `scripts/e2e_async_validate.py` transcript (14/14) as proof of the working path.
- **API not reachable** (routes not yet wired): demo via the E2E harness (direct
  Lambda invokes) — same chain, no API GW dependency — and show the architecture diagram.
- **Textract slow:** it's normally 3–6s; if Textract throttles, the job retries
  automatically — narrate that as the resilience story rather than an error.

## Quick troubleshooting during the demo

| Symptom | 10-second action |
|---|---|
| Upload 403 | presigned link expired → re-trigger upload (re-presigns) |
| Stuck PROCESSING | mention auto-retry; switch to pre-generated PDF |
| Result page overlay blank | processed-bucket CORS; PDF still downloads — proceed |
| Anything red | fall back to pre-generated PDF + E2E transcript; keep narrating architecture |

Reference docs: `docs/admin_activation_runbook.md`, `docs/go_live_checklist.md`,
`docs/observability_runbook.md`, `docs/frontend_deployment.md`.
