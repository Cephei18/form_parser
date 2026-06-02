# Orchestration plan — async Textract slice (design only)

**Status:** design/planning. No infra changes, no SQS/S3 triggers, no code in this slice.
**Scope:** wire the already-validated Textract worker into the *existing* async
scaffolding: `S3 upload → enqueue → SQS → worker Lambda → DynamoDB job tracking →
frontend polling`. Preserve the Textract worker architecture and keep the OCR
pipeline (EC2/FastAPI) fully isolated and untouched.

This plan is grounded in the **provisioned** resources (read-only verified):

| Resource | Fact |
|---|---|
| DynamoDB `form-pdf-poc-dev-jobs` | PK `job_id` (S); on-demand; GSIs `status-created_at-index`, `user_id-created_at-index`; attrs `job_id,status,created_at,user_id` |
| SQS `form-pdf-poc-dev-processing-queue` | standard; VisibilityTimeout **900s**; Redrive → `…-processing-dlq` at **maxReceiveCount=3**; retention 4 days |
| SQS `form-pdf-poc-dev-processing-dlq` | dead-letter target |
| Lambdas (Zip, py3.12) | `presigned-url`, `upload-handler`, `result-handler`, `job-status` already exist |
| Lambda `form-pdf-poc-dev-worker` | container worker (this project) — the missing SQS consumer |
| Buckets | `…-raw-documents` (input), `…-processed-documents` (artifacts) |

The infra shape **dictates the design**: SQS with a DLQ + a jobs table with status/user
GSIs is an asynchronous, decoupled, retry-with-DLQ, pollable-status architecture.

---

## 1. Recommended event flow

```
                          (1) GET presigned PUT url
  ┌──────────┐  ───────────────────────────────────►  ┌─────────────────────┐
  │ Frontend │                                          │ presigned-url λ     │
  │  (S3 UI) │  ◄───────────────────────────────────   │ (API GW)            │
  └────┬─────┘        presigned URL                     └─────────────────────┘
       │ (2) PUT file directly to S3 (raw bucket)
       ▼
  ┌─────────────────────────┐  (3) ObjectCreated event   ┌──────────────────────┐
  │ S3 raw-documents bucket │ ─────────────────────────► │ upload-handler λ     │
  └─────────────────────────┘                            │  (S3 trigger)        │
                                                          │  • create DDB job    │
                                                          │    status=QUEUED     │
                                                          │  • send SQS message  │
                                                          └──────────┬───────────┘
                                                                     │ (4) {job_id,bucket,key,user_id}
                                                                     ▼
                                                          ┌──────────────────────┐
                                              maxReceive=3 │ SQS processing-queue │ ──► DLQ (after 3)
                                                          └──────────┬───────────┘
                                                                     │ (5) Event source mapping (batch=1)
                                                                     ▼
                                            ┌────────────────────────────────────────────┐
                                            │ form-pdf-poc-dev-worker (container λ)        │
                                            │  • DDB: QUEUED → PROCESSING (conditional)    │
                                            │  • download raw → /tmp                       │
                                            │  • render (PDF→png) + Textract + anchor + pdf│
                                            │  • publish artifacts → processed bucket      │
                                            │  • DDB: → SUCCEEDED (+artifact ptrs) / FAILED│
                                            └───────────────┬───────────────┬─────────────┘
                                                            │               │
                       (6) artifacts                        ▼               ▼  (6) status write
                          ┌──────────────────────────┐   ┌─────────────────────────────┐
                          │ S3 processed-documents    │   │ DynamoDB jobs table         │
                          │ textract/<job_id>/...     │   │ job_id → status/artifacts   │
                          └──────────────────────────┘   └─────────────────────────────┘
                                                            ▲
       ┌──────────┐  (7) poll GET /jobs/{job_id}            │
       │ Frontend │ ───────────────────────────► job-status λ (reads DDB)
       │          │ ◄─── status (+ presigned download when SUCCEEDED via result-handler)
       └──────────┘
```

Steps: **(1)** presigned PUT, **(2)** direct browser→S3 upload, **(3)** S3 event →
`upload-handler` creates the `QUEUED` job + enqueues, **(4)** SQS buffers,
**(5)** event-source-mapping invokes the worker, **(6)** worker publishes artifacts +
writes terminal status, **(7)** frontend polls `job-status`, then downloads via a
presigned GET (issued by `result-handler`).

---

## 2. S3-trigger vs SQS-trigger tradeoffs

| Concern | Direct S3 → worker Lambda | **SQS → worker Lambda (recommended)** |
|---|---|---|
| Buffering / spikes | S3 invokes immediately; bursts hit concurrency limits | SQS absorbs bursts; smooth, throttle-controlled drain |
| Retry semantics | S3→Lambda async = 2 retries, opaque | SQS visibility-timeout retries, **maxReceiveCount=3 → DLQ** (already configured) |
| Poison messages | Lost to Lambda async DLQ (if any) | Land in `…-processing-dlq` for inspection/redrive |
| Concurrency control | None (S3 fans out) | `maximum_concurrency` on the event source mapping caps Textract TPS/cost |
| Decoupling / replay | Tightly coupled to the put event | Messages replayable; producers/consumers independent |
| Ordering/batching | One event per object | Batched receives, partial-batch failure reporting |
| Extra moving part | Fewer pieces | One queue (already provisioned) |

**Recommendation: SQS-trigger.** The DLQ + redrive policy already exist, which is the
decisive signal. Use S3 only to *notify* (via `upload-handler`), and SQS as the
worker's event source. This gives back-pressure, a real DLQ, and Textract-rate
control — all production essentials.

**Producer choice (S3 → SQS):** prefer **S3 → `upload-handler` → SQS** over a raw
S3→SQS notification, because `upload-handler` should create the `QUEUED` DDB item
*first* (so the frontend sees a job immediately and a stuck/failed worker is still
visible), assign a stable `job_id`, and attach `user_id`. (Raw S3→SQS is the simpler
fallback if `upload-handler` is not yet wired, but then the worker must create the
job item on first receive.)

---

## 3. DynamoDB job schema (matches the existing table)

Single item per job, keyed by `job_id`. Indexed for ops + per-user history via the
two existing GSIs. Suggested attributes:

```
job_id        (S, PK)        e.g. "j-3f9a..."  (uuid v4 assigned by upload-handler)
status        (S, GSI HASH)  QUEUED | PROCESSING | SUCCEEDED | FAILED | DEAD_LETTER
created_at    (S, GSI RANGE) ISO-8601 (upload time)
updated_at    (S)            ISO-8601 (last transition)
user_id       (S, GSI HASH)  owner (from auth/presign); enables history listing
engine        (S)            "textract"  (keeps OCR jobs distinguishable if ever tracked)
input         (M)            { bucket, key, version_id, content_type, bytes }
artifacts     (M)            { base_uri, output_pdf, mappings, diagnostics, count }  (on SUCCEEDED)
metrics       (M)            { fields_detected, checkboxes_detected, tables_detected, processing_time_ms }
attempts      (N)            receive count (for observability)
error         (M)            { type, message } (on FAILED/DEAD_LETTER)
expires_at    (N)            TTL epoch (optional; auto-expire old job rows)
```

- **GSI `status-created_at-index`** → ops queries: "all FAILED today", "QUEUED backlog",
  reconciliation sweeps.
- **GSI `user_id-created_at-index`** → frontend "my recent jobs", newest-first.
- **TTL** on `expires_at` keeps the table small (e.g. 30–90 days) without scans.
- Status is a strict state machine: `QUEUED → PROCESSING → {SUCCEEDED|FAILED}`, plus
  `→ DEAD_LETTER` written by a DLQ consumer.

---

## 4. Retry / DLQ behavior (already provisioned)

- Visibility timeout **900s** ≥ worker timeout (120s) with wide margin → no premature
  re-delivery mid-run. (Rule: visibility ≥ function timeout; keep it.)
- **maxReceiveCount = 3** → a message that fails 3 receive cycles moves to
  `…-processing-dlq`. So: up to 3 worker attempts, then DLQ.
- DLQ retention = inspect/triage; add a **DLQ alarm on depth > 0**.
- A future small **DLQ consumer** marks the job `DEAD_LETTER` in DDB (so the frontend
  shows a definitive failure) and optionally supports **redrive** back to the main queue
  after a fix. (Out of this slice; note as the immediate follow-up.)

---

## 5. Failure handling — retryable vs permanent

The worker must classify failures, because blind retry of a permanent error just burns
3 attempts + DLQ noise:

| Class | Examples | Action |
|---|---|---|
| **Transient** (retryable) | Textract throttling/5xx, S3 5xx, timeout | **Raise** → SQS re-delivers (visibility backoff) up to 3× → DLQ |
| **Permanent** (non-retryable) | Unsupported/corrupt file, object missing, malformed input | Write `FAILED` to DDB, **report success to SQS** (delete message) — do NOT waste retries |
| **Partial batch** | batch size > 1 | Return `batchItemFailures` so only failed records re-deliver |

Recommended: keep event-source `batch_size = 1` initially (simplest, one job per
invocation, clean isolation), and use `ReportBatchItemFailures` once batching is
introduced. The worker's existing "never raise out of a job" behavior must become
**raise-on-transient / swallow-on-permanent** for SQS semantics — a small, contained
change to `process_job`/`handler` (planned, not implemented here).

---

## 6. Idempotency strategy

SQS standard queue = **at-least-once**; the worker may run twice for one message. Make
reprocessing safe and side-effect-free:

1. **Stable, unique `job_id`** assigned once by `upload-handler` (uuid), carried in the
   SQS message — *not* derived from filename (avoids collisions for same-named files).
2. **Deterministic artifact keys** `textract/<job_id>/<file>` → a re-run overwrites the
   same objects (idempotent publish, no duplicates).
3. **DDB conditional transitions:**
   - Claim: `UpdateItem … SET status=PROCESSING` with
     `ConditionExpression: status IN (QUEUED, PROCESSING)` (or `attribute_not_exists`),
     so a duplicate that arrives after a terminal state is a **no-op skip**.
   - Finalize: only write `SUCCEEDED/FAILED` if not already terminal.
4. **Object version pinning:** include S3 `version_id` in the message so a re-upload of
   the same key is a distinct, traceable job.

Net effect: a duplicate delivery either no-ops (already terminal) or re-produces the
identical artifacts under the same keys — never double-charges the user or corrupts state.

---

## 7. Artifact lifecycle strategy

Per job the pipeline emits ~12 artifacts (~1.3 MB) under `textract/<job_id>/`. Strategy:

- **Tier by value:** the *user-facing* `output.pdf` (+ `result.json`) are the products;
  `textract_raw_response.json`, `*_debug.png`, `mapping.png` are diagnostics.
- **Prod minimization:** run prod with debug artifacts off (the pipeline already has
  debug flags) to cut storage/PII surface; keep raw Textract JSON only if needed for
  re-anchoring/audit.
- **S3 lifecycle rules (processed bucket):**
  - diagnostics prefix → expire ~7–14 days;
  - `output.pdf`/`result.json` → Standard-IA at 30 days, expire at 90–180 days (policy).
- **Raw bucket:** expire uploads ~1–7 days post-processing (they're re-derivable only
  from the user; keep short for cost/PII).
- **DDB TTL** (`expires_at`) aligned with artifact expiry so status rows don't outlive
  their artifacts.
- **Encryption:** SSE-S3/KMS on both buckets; block public access (already expected).

---

## 8. Frontend polling / status strategy

Async, poll-based (no websockets initially):

1. `presigned-url` → browser uploads directly to raw bucket (no payload through API).
2. Response returns the `job_id` (assigned at presign/upload time).
3. Frontend **polls `GET /jobs/{job_id}`** (`job-status` λ → DDB `GetItem`) every ~2–3s
   with capped exponential backoff, until `status ∈ {SUCCEEDED, FAILED, DEAD_LETTER}`.
4. On `SUCCEEDED` → request a **presigned GET** for `output.pdf` (via `result-handler`)
   and render/download.
5. History view → `user_id-created_at-index` (newest-first).
6. Show states `QUEUED/PROCESSING/SUCCEEDED/FAILED` + `error.message` on failure.

Polling is simple, cache-friendly, and stateless. (Later upgrade path: API Gateway
WebSocket or SNS→push for instant completion — not needed initially.)

---

## 9. Production monitoring / logging

- **Structured JSON logs** with `job_id` correlation on every worker line (already
  logging per stage). Ship to CloudWatch Logs; set retention (e.g. 30 days).
- **CloudWatch alarms (minimum set):**
  - `…-processing-dlq` `ApproximateNumberOfMessagesVisible > 0` (poison messages).
  - main queue `ApproximateAgeOfOldestMessage` high (consumer falling behind).
  - worker `Errors` rate, `Throttles > 0`, `Duration` p99 near timeout.
  - DynamoDB throttled requests (`ReadThrottleEvents`/`WriteThrottleEvents`).
  - Textract throttling surfaced via worker error metric / log filter.
- **Dashboard:** queue depth, age, worker concurrency/duration, success vs FAILED vs
  DLQ counts (from DDB GSI by status).
- **Metrics:** emit EMF/custom metrics (jobs succeeded/failed, fields/checkboxes per
  doc, Textract latency) for cost + quality tracking.
- **Tracing:** optional X-Ray on the worker for S3/Textract/DDB span timing.
- **Cost guard:** event-source `maximum_concurrency` caps parallel Textract calls.

---

## 10. Minimal-change migration path (from current frontend/API flow)

Today: **frontend → API Gateway/FastAPI (EC2) → synchronous pipeline**. Keep that intact
for OCR; introduce async Textract alongside it, flag-gated.

1. **Dark deploy:** deploy the worker + SQS event-source mapping with **no producer
   wired** (queue stays empty). Zero user impact. Validate via manual test events.
2. **Shadow:** wire `upload-handler` enqueue behind an internal flag; submit known docs;
   confirm DDB transitions + artifacts + DLQ behavior end-to-end.
3. **Opt-in routing:** add an async submission path in the frontend used only when
   `engine=textract` (or a beta flag). OCR requests stay on the existing sync EC2 path —
   **OCR fallback untouched and isolated.**
4. **Ramp:** route a growing % of Textract traffic to async; watch alarms/DLQ/latency.
5. **Cutover:** make async the default for Textract; retain EC2/FastAPI as OCR runtime +
   emergency fallback. Decommission the synchronous *Textract* path only after stability.

Backwards compatibility: the worker already emits the **same artifact set + output
contract** (`output.pdf`, `mappings.json`, `result.json`) the frontend consumes today;
only the *delivery* changes (S3/poll vs inline response). No output-format change.

---

## 11. AWS resource mapping

| Component | Resource | New / existing | Change in this slice (future) |
|---|---|---|---|
| Upload URL | `presigned-url` λ | existing | none (verify it returns `job_id`) |
| Job create + enqueue | `upload-handler` λ | existing | add: create `QUEUED` DDB item + SQS send on S3 event |
| Buffer/retry | `…-processing-queue` (+DLQ) | existing | add event-source mapping → worker; DLQ alarm |
| Processing | `…-worker` container λ | built, not deployed | add SQS-event handling + DDB writes + retry classing |
| Job state | `…-jobs` DynamoDB | existing | worker writes status/artifacts/metrics |
| Status read | `job-status` λ | existing | none (reads DDB) |
| Download URL | `result-handler` λ | existing | presigned GET for `output.pdf` |
| Inputs | `…-raw-documents` | existing | S3→`upload-handler` notification; lifecycle expiry |
| Outputs | `…-processed-documents` | existing | lifecycle tiering/expiry |
| Exec role | `FormPdfPocLambdaExecutionRole` | existing | ensure `dynamodb:UpdateItem/GetItem/PutItem` on jobs + `textract:AnalyzeDocument` |

---

## 12. Deployment sequencing (future slices, ordered)

1. **(gate)** IAM confirmations from the deploy slice (CreateFunction/PassRole; Textract on role).
2. Deploy the worker container (`scripts/deploy_lambda_worker.ps1`).
3. Add/confirm role perms: `dynamodb:UpdateItem/GetItem/PutItem` on `…-jobs`,
   `sqs:ReceiveMessage/DeleteMessage/GetQueueAttributes` on the queue.
4. Worker code delta (separate PR): SQS event shape + DDB state machine + retry classing
   + idempotent conditional writes. Validate with the local harness, then manual SQS msgs.
5. Create the **SQS → worker event-source mapping** (batch=1, `maximum_concurrency` set).
   Queue empty ⇒ safe/no-op until a producer is wired.
6. Wire `upload-handler` (create job + enqueue) behind a flag; shadow test.
7. CloudWatch alarms (DLQ depth, queue age, worker errors) + dashboard.
8. S3 lifecycle policies + DDB TTL.
9. Frontend async submit+poll path (flagged) → ramp → cutover.

Each step is independently reversible (delete the event-source mapping / disable the
flag / delete the function) and leaves OCR and the existing sync path untouched.

---

## 13. Worker code delta required (planned, NOT in this slice)

For reference, the future worker change is small and contained:
- `handler`: accept the **SQS event shape** (`Records[*].body` = JSON message) in
  addition to the existing S3-event/direct-payload shapes.
- `process_job`: wrap with **DDB state transitions** (conditional claim → terminal) and
  **retry classification** (raise transient / swallow permanent).
- Add a thin `dynamodb` client module (lazy boto3, like `artifact_store`), keeping the
  worker isolated from OCR and the pipeline unchanged.
- Add `batchItemFailures` support if batch size > 1.

No change to the Textract pipeline, anchoring, dedup, rendering, or OCR.

---

## 14. Open items to confirm with an admin / infra owner

- Does `upload-handler` already create the DDB job + enqueue, or only one of those?
- Does `result-handler` already issue presigned GETs (its exact role)?
- Does `presigned-url` assign/return the `job_id`?
- Execution-role policy contents (Textract + DynamoDB write) — currently unreadable by
  the scoped user.
These determine how much of the producer side is "wire-up" vs "already done".
