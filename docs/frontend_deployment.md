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

| Var | Value |
|---|---|
| `NEXT_PUBLIC_ASYNC_API_BASE_URL` | `https://58is64i9kb.execute-api.ap-south-1.amazonaws.com` |
| `NEXT_PUBLIC_TEXTRACT_ASYNC` | `false` → flip to `true` at cutover |
| `NEXT_PUBLIC_API_BASE_URL` | sync EC2 origin if restored; otherwise the API GW origin as a safe non-localhost placeholder (EC2 is stopped → rule/ml modes are non-functional until restored) |

## Exact build command (CI / clean checkout — canonical)

```bash
cd frontend
npm ci
NEXT_PUBLIC_ASYNC_API_BASE_URL=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com \
NEXT_PUBLIC_TEXTRACT_ASYNC=false \
NEXT_PUBLIC_API_BASE_URL=https://58is64i9kb.execute-api.ap-south-1.amazonaws.com \
  npm run build
# → static site in frontend/out/
```

For a **local** prod-style build, the committed-locally `.env.production` supplies
these — but first `mv .env.local .env.local.bak` (or it overrides the async vars’ siblings).

## Exact deploy command (operator supplies bucket / distribution)

```bash
# Replace <FRONTEND_BUCKET> and (optional) <CF_DIST_ID>. No frontend hosting
# bucket is recorded in the repo — confirm it with the account admin.
aws s3 sync frontend/out/ s3://<FRONTEND_BUCKET>/ --delete --profile <deploy> \
  --cache-control "public,max-age=300"
# If CloudFront fronts the bucket, invalidate so the new build is served:
aws cloudfront create-invalidation --distribution-id <CF_DIST_ID> --paths "/*" --profile <deploy>
```

`--delete` removes stale files (prevents old chunks lingering). `trailingSlash:true`
means routes are `index.html` under each path — S3 static hosting needs the index
document set to `index.html` (one-time bucket config).

## Rollback commands

1. **Flag rollback (instant, no redeploy):** rebuild with `NEXT_PUBLIC_TEXTRACT_ASYNC=false`
   and redeploy → 100% synchronous path. (If EC2 is down, this is only meaningful
   once EC2 is restored; otherwise rollback = redeploy the previous build.)
2. **Build rollback:** keep the previous `out/` (or git tag) and re-`s3 sync` it:
   ```bash
   aws s3 sync ./out-previous/ s3://<FRONTEND_BUCKET>/ --delete --profile <deploy>
   aws cloudfront create-invalidation --distribution-id <CF_DIST_ID> --paths "/*" --profile <deploy>
   ```
3. Frontend is static + additive → no schema/state to roll back.

## Async flow validation (pre-cutover, against a wired API)

- `processFormAsync`: POST `/uploads` → presigned POST upload → poll `/jobs/{id}` →
  GET `/result/{id}`. Verified end-to-end at the backend (14/14); from the browser
  it requires the admin to have wired the 3 API GW routes + raw-bucket CORS.
- Polling: 1.5→4s backoff, 3-min ceiling; stops on `SUCCEEDED`/`FAILED`.
- Signed URLs: result page accepts `https://*.amazonaws.com` only (async branch),
  same-origin `/files/` only (sync branch) — selected by `?src=async`.
- Rollback: flag off → `shouldUseAsync()` returns false → sync path.

## Frontend cutover checklist

1. [ ] Admin has wired API GW routes + S3 notification + SQS ESM + raw/processed CORS.
2. [ ] `curl POST <API>/uploads` returns a presigned POST (smoke test).
3. [ ] Build with shell env vars, flag **false**; `aws s3 sync` to `<FRONTEND_BUCKET>`; CF invalidate.
4. [ ] Manually verify the site loads and the sync path is unaffected.
5. [ ] Rebuild with flag **true**; redeploy; CF invalidate.
6. [ ] Upload one real form → QUEUED→PROCESSING→SUCCEEDED → PDF downloads.
7. [ ] Watch worker logs + DLQ depth (0) during first real uploads.
8. [ ] Rollback ready: previous `out/` retained; flag-off build one command away.
