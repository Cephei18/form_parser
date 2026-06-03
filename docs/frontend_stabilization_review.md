# Frontend stabilization review — async Textract (Task 3)

Production-readiness review of the async integration. Scope: `lib/api.ts`,
`app/page.tsx`, `app/result/page.tsx`, `lib/constants.ts`, `lib/types.ts`.
**Findings + recommendations only** — no architectural changes. Verdict: the
async path is sound and rollback-safe; a few low-risk hardening items below.

## What is solid (verified)

- **No polling deadlock.** `processFormAsync` loops with 1.5→4s backoff under a
  hard 3-min ceiling (`POLL_TIMEOUT_MS`); transient `job-status` read errors are
  caught and retried; terminal `SUCCEEDED`/`FAILED`/`DEAD_LETTER` exit immediately;
  timeout throws a clear message. Bounded in all paths.
- **State machine coverage.** QUEUED/PROCESSING surface via `onStatus`; SUCCEEDED →
  `fetchResult`; FAILED/DEAD_LETTER → thrown `error.message`. All four handled.
- **Signed-URL safety.** `normalizeAsyncFileUrl` accepts only `https://*.amazonaws.com`;
  the sync `normalizeBackendFileUrl` still enforces same-origin `/files/`. The two
  are selected by `?src=async`, so neither path can be fed a URL it shouldn't trust.
- **Rollback.** `shouldUseAsync()` gates on `TEXTRACT_ASYNC_ENABLED && mode==='textract'
  && async base set`; flag off → pure sync path. Build-time inlined, instant.
- **Artifact-name contract** matches the worker (`output.pdf`/`mapping.png`/`result.json`).

## Findings & recommendations (low-risk, optional)

### F1 — Result-overlay fetch needs processed-bucket CORS (dependency, not a bug)
`result/page.tsx` does `fetch(result_url)` (presigned `result.json`) to render the
Advanced-details overlay. That JS fetch is cross-origin → **requires processed-bucket
GET CORS** (covered by `wire_async_infra.ps1 -Cors`). Degrades gracefully: if CORS
is missing, only the overlay errors — the **PDF download/preview and stats still work**
(they use `<a>`/`<img>`, which don't need CORS). *Action: ensure -Cors is run; no code change.*

### F2 — Dead OCR modes are still selectable while EC2 is stopped
With the flag on, `MODE_OPTIONS = ["textract","rule","ml"]`. EC2 is stopped, so
picking rule/ml posts to a dead endpoint and errors. Not a crash, but a confusing path.
*Recommended one-line change (reversible — flag off restores all modes):*
```ts
// app/page.tsx
const MODE_OPTIONS: ProcessingMode[] = TEXTRACT_ASYNC_ENABLED ? ["textract"] : ["rule", "ml"];
```
Keeps the OCR code path fully intact (only the selector hides). *Left unapplied —
product decision: apply if rule/ml should be hidden until EC2 returns.*

### F3 — Presigned GET expiry vs. a slow user (10-min window)
Result URLs are minted at `fetchResult` time with ~10-min expiry and carried in the
result-page query string. A user who lingers >10 min then clicks Download gets a 403.
*Recommended (optional, small): on a failed PDF load, show "link expired — reprocess".*
Low priority; the common path is well within 10 min.

### F4 — `fetchResult` is not retried
On SUCCEEDED, a single transient `result-handler` failure throws the whole flow even
though the job is done. *Optional: wrap `fetchResult` in a 2-try retry.* Low priority
(result-handler is a DDB read + presign; very reliable).

### F5 — Sync fallback base URL must not bake `localhost`
`lib/constants.ts` falls back to `http://localhost:8000`. For prod, set
`NEXT_PUBLIC_API_BASE_URL` via shell env at build (see `docs/frontend_deployment.md`).
*No code change; deployment discipline.*

## Explicitly checked — no issue

- No race between the cosmetic progress timer and `onStatus` (timer cleared in
  success branch + `finally`).
- No stale `process-form`/EC2 assumption on the async path (separate base URL + branch).
- `encodeURIComponent(jobId)` on both `/jobs` and `/result` — safe path building.
- `cache: "no-store"` on status/result fetches — no stale polling reads.
- Static export (`output: export`) has no SSR/runtime to misconfigure.

## Recommendation summary

| ID | Item | Risk | Action |
|---|---|---|---|
| F1 | Processed-bucket CORS for overlay fetch | none | run `-Cors`; verify |
| F2 | Hide dead rule/ml modes | very low | apply 1-liner if desired |
| F3 | Presigned expiry UX | low | optional polish |
| F4 | Retry `fetchResult` | low | optional |
| F5 | Prod env via shell | none | deployment discipline |

No blocking issues. F1 (CORS) is the only item that can visibly affect the demo, and
it degrades gracefully.
