# AWS infrastructure reconciliation (discovery, read-only)

Discovery pass before orchestration/frontend work. **No infrastructure was mutated.**
Findings are evidence-based (AWS `describe`/`list`, downloaded Lambda code, frontend
source + deployed bundle). Some reads were denied to the scoped `intern.form-pdf-poc`
user (noted inline) and inferred from adjacent evidence.

## TL;DR — the single most important finding
There is **no conflicting or duplicate flow risk**, because the entire serverless
async stack is **provisioned but inert**:

- The current, *working* production flow is **100% EC2/FastAPI, synchronous**.
- The 4 backend Lambdas are **placeholder stubs** (echo only), wired to nothing
  (no resource policies, no triggers), and **not referenced by the frontend**.
- The worker Lambda **does not exist** yet (image built/pushed; function blocked on IAM).

So this slice is *implement the inert scaffolding*, not *deconflict competing systems*.

---

## 1. Current-state architecture map

### A. Live production path (EC2, synchronous) — what actually serves users today
```
 Browser (Next.js "FormFlow AI", static on S3 frontend bucket)
   │  POST {NEXT_PUBLIC_API_BASE_URL}/process-form   (multipart: file, mode)
   ▼
 EC2 FastAPI  (http://15.207.134.4, PLAIN HTTP)   src/api.py
   • POST /process-form  → runs pipeline (OCR or Textract by `mode`) SYNCHRONOUSLY
   • returns { status, mode, pdf_url, mapping_preview, result_url, stats }
   • GET /files/{path}   → serves artifacts (output.pdf, mapping.png, mappings.json)
   ▼
 Browser /result page  fetches pdf_url + mapping_preview from EC2 /files/...
```
Evidence: `frontend/lib/api.ts` (`processForm` → `POST /process-form`, expects
`pdf_url`, same-origin `/files/...`), `src/api.py:386 @app.post("/process-form")` +
`:261 @app.get("/files/{file_path:path}")`, deployed bundle hardcodes
`http://15.207.134.4`.

### B. Serverless async stack — PROVISIONED BUT INERT (not used by anyone)
```
 API Gateway HTTP API  58is64i9kb        ── exists; routes unverified (apigateway:GET denied);
                                              frontend does NOT point at it
 Lambda (Zip, py3.12, role FormPdfPocLambdaExecutionRole, env→DDB/SQS/buckets):
   • presigned-url     ┐
   • upload-handler    │  ALL FOUR ARE PLACEHOLDER STUBS — return
   • job-status        │  {"message":"form-pdf-poc placeholder lambda"}
   • result-handler    ┘  no DDB/SQS/presign logic; NO resource policy (untriggered)
 Lambda worker (container)  ── DOES NOT EXIST (image built; CreateFunction IAM-blocked)
 SQS processing-queue (+DLQ, maxReceive=3, visibility 900s, retention 4d) ── no producer/consumer
 DynamoDB jobs (PK job_id; GSIs status-created_at, user_id-created_at; on-demand) ── unused
 ECR form-pdf-poc-dev-worker ── repo exists; worker image built/pushed
```
Evidence: `get-function` code = identical echo stub ×4; `get-policy` →
ResourceNotFound ×4 (no triggers); `get-function-configuration` (worker) → not found.

### C. S3 buckets
| Bucket | Observed | Notes |
|---|---|---|
| `…-frontend` | `index.html`, `404.html`, `_next/`, `result/` | Next.js static export ("FormFlow AI", deployed 2026-05-11). Website/policy read denied. |
| `…-raw-documents` | `uploads/` (`form_page_1.png`, `mapping.png`) | only our smoke-test objects; notification/lifecycle read denied |
| `…-processed-documents` | `textract/<job_id>/…` | only our smoke-test runs; lifecycle read denied |

---

## 2. Identified legacy / OCR paths
- **EC2 FastAPI `POST /process-form` + `GET /files/...`** is the legacy *and current*
  runtime for BOTH OCR and Textract (selected by the `mode` form field). It writes
  artifacts to local disk and serves them over plain HTTP.
- **OCR runs only here** and must stay here (isolated). The serverless worker is
  **Textract-only**; OCR is never moved into Lambda in this migration.
- No other OCR code paths touch the serverless stack — isolation already holds.

## 3. Reusable existing infrastructure (prefer reuse over new)
Everything needed already exists — the work is *implementation*, not *provisioning*:
- **API Gateway HTTP API `58is64i9kb`** → add routes (don't create a new API).
- **4 Lambda function shells** → replace stub code in place (keep name/role/env).
- **SQS queue + DLQ** → wire worker as consumer (redrive already configured).
- **DynamoDB `jobs` + 2 GSIs** → job tracking (schema already fits — see orchestration_plan.md §3).
- **`FormPdfPocLambdaExecutionRole`** → reuse (add Textract + DynamoDB perms).
- **ECR repo + built worker image** → deploy as the worker function.
- **Both S3 buckets** → raw input / processed artifacts (already used by the worker).
- **Frontend `NEXT_PUBLIC_API_BASE_URL` + `mode` param + `/files`-style artifact fetch**
  → re-point to API GW; artifact fetch maps cleanly to presigned S3 GETs.

## 4. Missing orchestration pieces (to implement)
1. **Worker Lambda function** (create from ECR image) + **SQS event-source mapping**.
2. **Real handler logic** in the 4 stubs:
   - `presigned-url`: create DDB `QUEUED` job (uuid `job_id`, `user_id`) + return presigned **PUT** url + `job_id`.
   - enqueue step: S3 `ObjectCreated` → `upload-handler` → SQS (or presigned-url enqueues on confirm).
   - `job-status`: `GetItem` by `job_id` → status JSON.
   - `result-handler`: presigned **GET** for `output.pdf` / `mapping.png` / `result.json`.
3. **Worker DDB writes** (QUEUED→PROCESSING→SUCCEEDED/FAILED) — the planned worker delta.
4. **API Gateway routes → handlers** (+ `lambda:InvokeFunction` permissions; currently absent).
5. **Producer wiring** (S3 notification or upload-handler → SQS).
6. **Raw-bucket CORS** (browser presigned PUT needs it), **S3 lifecycle**, **CloudWatch alarms**.
7. **Frontend async client** (presign → PUT → poll → result) + rebuild + redeploy + repoint API base.

---

## 5. How frontend & backend should connect (final architecture)

Reuse the provisioned async stack; move the frontend off the EC2 sync call:

```
 Browser ──HTTPS──► API Gateway 58is64i9kb
   1) POST /presigned-url {filename, mode}        → presigned-url λ  → {job_id, upload_url}
   2) PUT file ──directly──► S3 raw (presigned, CORS-enabled)
   3) S3 ObjectCreated → upload-handler λ → SQS → worker λ (container)
         worker: DDB PROCESSING → Textract+render → publish processed/ → DDB SUCCEEDED/FAILED
   4) poll GET /jobs/{job_id}                      → job-status λ (DDB GetItem)
   5) on SUCCEEDED: GET /result/{job_id}           → result-handler λ → presigned PDF/mapping URLs
   6) Browser renders output.pdf + mapping overlay from presigned S3 GET
```

Why this shape: it reuses **every** provisioned resource, gives back-pressure + DLQ +
HTTPS, removes the EC2 single point of failure for Textract, and keeps OCR on EC2.

### Frontend change is contained (not a rewrite)
`frontend/lib/api.ts` is the single integration point. Replace the one `processForm`
(POST `/process-form`) with `presign → PUT → poll job-status → fetch result`; the upload
page gains a polling state, the result page consumes presigned URLs instead of `/files`.
`NEXT_PUBLIC_API_BASE_URL`, the `mode` selector, accepted types, and the result UI all
stay. This is ~1 module + 2 page tweaks, not a rebuild of the app.

### Transitional option (smallest change, not the end-state)
Add a single API GW `POST /process-form` route → the worker invoked **synchronously**,
returning the existing `{pdf_url,...}` shape (with presigned URLs). Keeps the frontend
*as-is*, but inherits API GW's 29s timeout + ~10MB payload limit and doesn't use the
queue. Use only as a bridge if an immediate cutover is needed; prefer the async flow.

---

## 6. What must be updated in S3 / frontend
- **Frontend (`frontend/`):** implement async client in `lib/api.ts`; update upload +
  result pages; set `NEXT_PUBLIC_API_BASE_URL` to the API GW base; rebuild (`next build`/
  export) and redeploy to `…-frontend` (invalidate caches).
- **Config drift to fix:** deployed bundle baked `http://15.207.134.4`; repo `.env.local`
  defaults to `http://localhost:8000`; `.env.example` `https://api.example.com`. None
  point at API GW yet — standardize on the API GW HTTPS endpoint for prod.
- **Security:** current EC2 call is **plain HTTP**; the async API GW path is HTTPS
  (removes mixed-content risk if the site is served over HTTPS/CloudFront).
- **S3:** add **CORS** to the raw bucket (browser presigned PUT), **lifecycle** rules to
  raw + processed (expiry/tiering), and confirm the frontend bucket's website/CDN setup
  (reads were denied).

## 7. Minimal remaining implementation slices (ordered, before prod rollout)
1. **(gate)** Admin IAM: `lambda:CreateFunction`+`iam:PassRole`; add `textract:AnalyzeDocument`
   + `dynamodb:*Item` to `FormPdfPocLambdaExecutionRole` (see lambda_deployment.md).
2. Deploy **worker** Lambda (ECR image) + **SQS event-source mapping** + worker **DDB delta**.
3. Implement **presigned-url** + **upload-handler** (create job + enqueue).
4. Implement **job-status** + **result-handler**.
5. **API GW routes** (`/presigned-url`, `/jobs/{id}`, `/result/{id}`) + invoke permissions.
6. **Raw CORS** + S3 lifecycle + CloudWatch alarms (DLQ depth, queue age, worker errors).
7. **Frontend** async client → rebuild → redeploy → repoint `NEXT_PUBLIC_API_BASE_URL`.
8. Cutover with **EC2/FastAPI retained** as OCR runtime + Textract fallback.

Each slice is independently reversible and leaves the live EC2 path (and OCR) untouched
until the final cutover.

## 8. Reads denied to the scoped user (confirm with an admin)
`apigateway:GET` (routes/integrations), `s3:GetBucketNotification`,
`s3:GetLifecycleConfiguration`, `s3:GetBucketWebsite`, `lambda:ListEventSourceMappings`,
all IAM reads. These would confirm: whether API GW already has routes/integrations,
whether any S3 notification already exists, and the exact role policy. Current evidence
(no Lambda resource policies, stub code, frontend→EC2) strongly indicates **none are
wired**, but an admin read would make it definitive.
