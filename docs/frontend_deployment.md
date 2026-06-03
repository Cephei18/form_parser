# Frontend deployment — production cutover (Task 1)

Static Next.js export (`output: "export"`, `trailingSlash: true`) → plain HTML/JS
in `frontend/out/`, served from S3 (optionally fronted by CloudFront). The async
Textract integration is additive and flag-gated; this doc makes the cutover instant.

## Inspection results (verified)

- **Build:** `next build` (Next 15.5) emits a static export to `frontend/out/`.
  There is no server runtime — no SSR, no API routes.
- **Env handling:** `NEXT_PUBLIC_*` are **inlined at build time**. Three vars matter:
  `NEXT_PUBLIC_ASYNC_API_BASE_URL`, `NEXT_PUBLIC_TEXTRACT_ASYNC`, `NEXT_PUBLIC_API_BASE_URL`.
- **No stale prod URLs.** The only `localhost` is the *fallback default* in
  `lib/constants.ts` (`?? "http://localhost:8000"`); `/process-form` is the
  intentional sync EC2 endpoint kept for OCR fallback. No hardcoded prod hosts.
- **Env precedence footgun (important):** all `.env*` except `.env.example` are
  gitignored, so a clean CI checkout has neither `.env.local` nor `.env.production`.
  In Next.js the lookup order is: shell `process.env` → `.env.production.local` →
  `.env.local` → `.env.production` → `.env`. So **a local `.env.local`
  (`localhost:8000`) overrides `.env.production`** for local builds. Canonical fix:
  **set the vars in the shell** (highest precedence) for any production build.

## Final production env values

The API Gateway routes live under the **`/dev` stage** (verified live: `/dev/uploads`
→ 200, bare path → 404). The async base URL MUST include `/dev`.

| Var | Value |
|---|---|
| `NEXT_PUBLIC_ASYNC_API_BASE_URL` | `https://58is64i9kb.execute-api.ap-south-1.amazonaws.com/dev` |
| `NEXT_PUBLIC_TEXTRACT_ASYNC` | `true` (backend is live + validated 14/14) |
| `NEXT_PUBLIC_API_BASE_URL` | sync EC2 origin if restored; otherwise the API GW `/dev` origin as a safe non-localhost placeholder (EC2 is stopped → rule/ml modes are non-functional until restored) |

> ⚠️ **CORS is per-origin.** It is currently wired for `http://localhost:3000` only
> (API Gateway + raw + processed buckets — all verified). Before deploying to the real
> production frontend origin, re-run the wiring for that origin:
> `wire_async_infra.ps1 -ApiRoutes -Cors -FrontendOrigin https://<prod-origin> -Profile <admin>`
> — otherwise the browser upload/poll/result-fetch will fail CORS even though the API works.

## Exact build command (CI / clean checkout — canonical)

```bash
cd frontend
npm ci
NEXT_PUBLIC_ASYNC_API_BASE_URL=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com/dev \
NEXT_PUBLIC_TEXTRACT_ASYNC=true \
NEXT_PUBLIC_API_BASE_URL=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com/dev \
  npm run build
# → static site in frontend/out/
```

For a **local** prod-style build, the committed-locally `.env.production` supplies
these — but first `mv .env.local .env.local.bak` (or it overrides the async vars’ siblings).

## Exact deploy command

Hosting bucket (discovered live): **`form-pdf-poc-dev-frontend`** (ap-south-1). It
currently holds an old pre-async export (2026-05-11). A dry-run deploy of the new
build shows **34 uploads / 17 deletes**. Whether a CloudFront distribution fronts it
is unconfirmed (`cloudfront:ListDistributions` + `s3:GetBucketWebsite` are denied to
the intern identity) — **confirm the public origin + any CF distribution with the admin.**

```bash
# 0) BACK UP the current live build first (enables a true frontend rollback;
#    the deploy below uses --delete which removes the old objects).
aws s3 sync s3://form-pdf-poc-dev-frontend/ ./frontend-prod-backup/ --profile <deploy>

# 1) Preview the change (read-only — safe):
aws s3 sync frontend/out/ s3://form-pdf-poc-dev-frontend/ --delete --dryrun --profile <deploy>

# 2) Deploy:
aws s3 sync frontend/out/ s3://form-pdf-poc-dev-frontend/ --delete --profile <deploy> \
  --cache-control "public,max-age=300"

# 3) If CloudFront fronts the bucket, invalidate so the new build is served:
aws cloudfront create-invalidation --distribution-id <CF_DIST_ID> --paths "/*" --profile <deploy>
```

`--delete` removes stale files (prevents old chunks lingering). `trailingSlash:true`
means routes are `index.html` under each path — S3 static hosting needs the index
document set to `index.html` (one-time bucket config).

## Rollback commands

1. **Flag rollback (instant, no redeploy):** rebuild with `NEXT_PUBLIC_TEXTRACT_ASYNC=false`
   and redeploy → 100% synchronous path. (If EC2 is down, this is only meaningful
   once EC2 is restored; otherwise rollback = redeploy the previous build.)
2. **Build rollback:** restore the pre-deploy snapshot from step 0 above:
   ```bash
   aws s3 sync ./frontend-prod-backup/ s3://form-pdf-poc-dev-frontend/ --delete --profile <deploy>
   aws cloudfront create-invalidation --distribution-id <CF_DIST_ID> --paths "/*" --profile <deploy>
   ```
3. Frontend is static + additive → no schema/state to roll back.

## Async flow validation (LIVE — verified)

- **Live HTTP E2E: 14/14** via `scripts/e2e_live_api.py` — exercises the exact browser
  calls through the `/dev` stage: POST `/uploads` → presigned POST upload → auto
  S3-notif→SQS→worker→Textract → poll `/jobs/{id}` (QUEUED→PROCESSING→SUCCEEDED in
  ~9s) → GET `/result/{id}` → PDF download + `result.json` fetch. DLQ stayed 0.
- **CORS verified for `http://localhost:3000`** (API Gateway preflight + raw POST/PUT +
  processed GET). So local browser validation works: `npm run dev` then open
  `http://localhost:3000`. Re-wire CORS for the real prod origin before prod cutover.
- Polling: 1.5→4s backoff, 3-min ceiling; stops on `SUCCEEDED`/`FAILED` (no deadlock).
- Signed URLs: result page accepts `https://*.amazonaws.com` only (async branch),
  same-origin `/files/` only (sync branch) — selected by `?src=async`.

### Local browser validation (against the live backend)
```bash
cd frontend
NEXT_PUBLIC_ASYNC_API_BASE_URL=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com/dev \
NEXT_PUBLIC_TEXTRACT_ASYNC=true \
  npm run dev   # open http://localhost:3000 (CORS already allows this origin)
```

## Rollback env config

| Mode | `NEXT_PUBLIC_TEXTRACT_ASYNC` | Effect |
|---|---|---|
| **Active (now)** | `true` | Textract uploads go async via `/dev` |
| **Rollback** | `false` | `shouldUseAsync()` false → 100% sync path; rebuild + redeploy |

Rollback needs no infra teardown — flag off is sufficient (the async API stays up, idle).

## Frontend cutover checklist

1. [x] Backend wired + LIVE: API GW `/dev` routes, S3 notification, SQS ESM (verified 14/14).
2. [x] `curl POST <API>/dev/uploads` returns a presigned POST (verified).
3. [ ] **Re-wire CORS for the prod origin** (currently only `http://localhost:3000`):
       `wire_async_infra.ps1 -ApiRoutes -Cors -FrontendOrigin https://<prod-origin> -Profile <admin>`.
4. [ ] Build with shell env vars (`/dev` async URL, flag **true**); `aws s3 sync frontend/out/` to `<FRONTEND_BUCKET>`; CF invalidate.
5. [ ] Open the deployed site → upload one real form → QUEUED→PROCESSING→SUCCEEDED → PDF downloads.
6. [ ] Watch worker logs + DLQ depth (0) during first real uploads.
7. [ ] Rollback ready: previous `out/` retained; flag-off rebuild one command away.
