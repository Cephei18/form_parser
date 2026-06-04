# Production activation validation summary — async Textract

Date: 2026-06-03 · account `637423601842` · region `ap-south-1` · API `58is64i9kb` (stage `/dev`).
Phase: frontend activation + real-user async validation against the LIVE backend.

## Verdict: PRODUCTION-READY (local-origin validated; one prod-origin CORS step remains)

The full async flow is validated end-to-end through the real HTTP edge. The only
gate before a public production cutover is re-wiring CORS for the production frontend
origin (currently wired for `http://localhost:3000`).

## What was validated (live, this phase)

| Area | Result |
|---|---|
| API stage discovery | Routes live under `/dev`; `/dev/uploads` → 200, bare path → 404 |
| Live HTTP E2E (`scripts/e2e_live_api.py`) | **14/14** through API Gateway |
| Auto-trigger chain | S3 upload → S3 notif → SQS → worker fired **automatically** (no manual invoke) |
| DDB lifecycle | QUEUED → PROCESSING → SUCCEEDED in ~9s, observed via polling |
| Real Textract | 12 fields, 1 table, ~2.6s processing |
| Result contract | `pdf_url`/`mapping_preview`/`result_url`/`stats` returned; all pass frontend URL guard |
| PDF retrieval | Presigned GET → valid `%PDF-`, 138 KB |
| Overlay source | `result.json` fetch → 200, 12 mappings (cross-origin GET works) |
| CORS (localhost:3000) | API Gateway + raw (POST/PUT) + processed (GET) — all return correct ACAO |
| Worker logs | Exactly one run per job, no errors, no `claim refused`/duplicates |
| SQS / DLQ | Main queue drained to 0; **DLQ 0** (no retries/failures) |
| Stuck jobs | 0 PROCESSING (prior stuck job self-healed when ESM was enabled) |
| Frontend build | Typecheck PASS; static export; `/dev` URL + flag inlined into bundle |

## Frontend activation state

- `frontend/.env.production`: `NEXT_PUBLIC_ASYNC_API_BASE_URL=…/dev`, `NEXT_PUBLIC_TEXTRACT_ASYNC=true`.
- `frontend/.env.local`: left as the local sync/dev config (localhost) — unchanged.
- Build must inject the async vars via **shell env** (highest precedence; `.env.local` overrides `.env.production`).
- One stabilization fix applied: with async live, the mode selector shows **Textract only**
  (rule/ml hidden — EC2 is down; they’d 404). Reversible via the flag; OCR code untouched.

## Highest-risk points (all addressed)

1. **`/dev` stage in the async URL** — fixed in env + docs; verified inlined into the build.
2. **Browser CORS** — verified for `localhost:3000` on all three layers; **must be re-wired
   for the prod origin** before public cutover (`wire_async_infra.ps1 -ApiRoutes -Cors -FrontendOrigin https://<prod>`).
3. **`.env.local` precedence** — documented; prod build uses shell env.
4. **Presigned-URL host guard** — all live artifact URLs are `*.amazonaws.com` (pass `normalizeAsyncFileUrl`).

## Low-risk observations (no change required)

- Result page passes presigned URLs via the client-side route query (`router.push`). This
  lives in browser history state (not an HTTP request) for SPA nav, so URL length is a
  non-issue for the validated flow. A hard-refresh of `/result?…` sends the long query to
  S3/CloudFront (generous limits); acceptable. Re-architecting to pass `job_id` only would
  be a design change, not a stabilization fix — deferred.
- Presigned GET TTL ~10 min: a user lingering past that then clicking Download gets a 403;
  re-calling `/result/{id}` re-mints. Documented in the stabilization review.

## Remaining steps for public cutover

1. Re-wire CORS for the production frontend origin (one command, admin).
2. Build with shell env (`/dev`, flag true) → `aws s3 sync frontend/out/ s3://<FRONTEND_BUCKET>/ --delete` → CF invalidate.
3. Smoke test on the deployed origin; watch worker logs + DLQ (0).

## Rollback

`NEXT_PUBLIC_TEXTRACT_ASYNC=false` → rebuild + redeploy → sync path; modes reappear; no infra
teardown. Full ladder in `docs/go_live_checklist.md`.

## Reproduce this validation

```bash
.venv/Scripts/python.exe scripts/e2e_live_api.py        # live HTTP E2E, 14/14
```

---

## Production rollout status (frontend deployed)

- **Frontend DEPLOYED** to `s3://form-pdf-poc-dev-frontend/` (the live hosting bucket).
  Backup of the prior build saved to `./frontend-prod-backup/` (23 objects). Deploy was
  34 uploads / 17 deletes; verified served:
  - `http://form-pdf-poc-dev-frontend.s3-website.ap-south-1.amazonaws.com/` → 200
  - `/result/` → 200; app chunk serves the `/dev` async URL; old chunks 404 (clean `--delete`).
- **Public origin:** `http://form-pdf-poc-dev-frontend.s3-website.ap-south-1.amazonaws.com` (S3 website endpoint, HTTP). HTTP page → HTTPS API/S3 calls is allowed (no mixed-content block).

### Production CORS — DONE ✅ (2026-06-04)

IAM perms were granted (`apigateway:GET/update-api`, `s3:Get/PutBucketCors` now work).
CORS was applied **additively** (kept `http://localhost:3000`, **added** the prod origin),
preserving every existing field — non-destructive, no route/integration changes:
- API Gateway `AllowOrigins`: `[localhost:3000, <prod>]`
- raw bucket (POST/PUT) + processed bucket (GET): both origins.

Validated on all three layers, both origins:

| Layer | Prod origin | localhost:3000 |
|---|---|---|
| API GW `/dev/uploads` preflight | 204 + ACAO=prod | 204 + ACAO=localhost |
| API GW actual POST response | 200 + ACAO=prod | — |
| raw bucket preflight (POST) | 200 + ACAO=prod | 200 + ACAO=localhost |
| processed bucket preflight (GET) | 200 + ACAO=prod | 200 + ACAO=localhost |

Fresh full pipeline E2E after cutover: **14/14** (QUEUED→PROCESSING→SUCCEEDED ~8s, valid PDF,
single worker execution, 0 stuck, DLQ 0).

**The deployed site is now fully live in the browser.** Open
`http://form-pdf-poc-dev-frontend.s3-website.ap-south-1.amazonaws.com`, upload a form,
watch the async flow, download the PDF.

### Known hardening item (not a blocker)
The frontend is served over the **S3 website endpoint (HTTP)**. It works (HTTP page →
HTTPS API/S3 calls are allowed; no mixed-content block), but browsers mark it "Not Secure".
For a hardened public deployment, front the bucket with CloudFront + HTTPS (+ a custom
domain) and re-run the CORS cutover for that HTTPS origin.

### Re-validate CORS anytime (read-only)

```bash
SITE=http://form-pdf-poc-dev-frontend.s3-website.ap-south-1.amazonaws.com
A=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com/dev
curl -i -X OPTIONS "$A/uploads" -H "Origin: $SITE" -H "Access-Control-Request-Method: POST" -H "Access-Control-Request-Headers: content-type"
curl -i -X OPTIONS "https://form-pdf-poc-dev-raw-documents.s3.amazonaws.com/"       -H "Origin: $SITE" -H "Access-Control-Request-Method: POST"
curl -i -X OPTIONS "https://form-pdf-poc-dev-processed-documents.s3.amazonaws.com/" -H "Origin: $SITE" -H "Access-Control-Request-Method: GET"
```

### Frontend rollback (if needed)

```bash
aws s3 sync ./frontend-prod-backup/ s3://form-pdf-poc-dev-frontend/ --delete --profile form-pdf-poc
```
Or flip `NEXT_PUBLIC_TEXTRACT_ASYNC=false`, rebuild, redeploy → 100% sync (no infra teardown).
