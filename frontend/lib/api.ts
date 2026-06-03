import { API_BASE_URL, ASYNC_API_BASE_URL, TEXTRACT_ASYNC_ENABLED } from "@/lib/constants";
import type {
  JobStatus,
  JobStatusResponse,
  PresignResponse,
  ProcessingMode,
  ProcessFormResponse
} from "@/lib/types";

export function getApiBaseUrl(): string {
  const baseUrl = API_BASE_URL.trim();
  if (baseUrl) {
    return baseUrl.replace(/\/$/, "");
  }

  // Safe runtime fallback for local testing when .env is not populated.
  if (typeof window !== "undefined" && window.location) {
    // eslint-disable-next-line no-console
    console.warn(
      "NEXT_PUBLIC_API_BASE_URL is not set; falling back to window.location.origin"
    );
    return window.location.origin.replace(/\/$/, "");
  }

  throw new Error("NEXT_PUBLIC_API_BASE_URL is not configured.");
}

export function normalizeBackendFileUrl(url: string | null | undefined): string {
  const rawUrl = (url ?? "").trim();
  if (!rawUrl) {
    return "";
  }

  const apiBaseUrl = getApiBaseUrl();

  let apiOrigin: string;
  let parsedUrl: URL;
  try {
    apiOrigin = new URL(apiBaseUrl).origin;
    parsedUrl = new URL(rawUrl, apiBaseUrl);
  } catch {
    return "";
  }

  if (!["http:", "https:"].includes(parsedUrl.protocol)) {
    return "";
  }

  if (parsedUrl.origin !== apiOrigin) {
    return "";
  }

  if (!parsedUrl.pathname.startsWith("/files/")) {
    return "";
  }

  parsedUrl.hash = "";
  return parsedUrl.toString();
}

export function mappingJsonUrlFromPreview(mappingPreview: string | null | undefined): string {
  const safePreviewUrl = normalizeBackendFileUrl(mappingPreview);
  if (!safePreviewUrl) {
    return "";
  }

  const parsedUrl = new URL(safePreviewUrl);
  if (!/\/mapping\.png$/i.test(parsedUrl.pathname)) {
    return "";
  }

  parsedUrl.pathname = parsedUrl.pathname.replace(/\/mapping\.png$/i, "/mappings.json");
  parsedUrl.search = "";
  return parsedUrl.toString();
}

function toAbsoluteUrl(url: string): string {
  const safeUrl = normalizeBackendFileUrl(url);
  if (!safeUrl) {
    throw new Error("Response contained an unsafe file URL.");
  }

  return safeUrl;
}

export async function processForm(file: File, mode: ProcessingMode): Promise<ProcessFormResponse> {
  const endpoint = `${getApiBaseUrl()}/process-form`;
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mode", mode);

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const fallbackMessage = `Request failed with status ${response.status}`;
    let message = fallbackMessage;

    try {
      const data = (await response.json()) as { detail?: string; message?: string };
      message = data.detail ?? data.message ?? fallbackMessage;
    } catch {
      // Ignore JSON parsing issues and keep fallback.
    }

    throw new Error(message);
  }

  const data = (await response.json()) as ProcessFormResponse;

  if (!data.pdf_url) {
    throw new Error("Response missing required field: pdf_url");
  }

  return {
    status: data.status,
    mode: data.mode,
    pdf_url: toAbsoluteUrl(data.pdf_url),
    mapping_preview: data.mapping_preview ? toAbsoluteUrl(data.mapping_preview) : undefined,
    result_url: data.result_url ? toAbsoluteUrl(data.result_url) : undefined,
    stats: data.stats
  };
}

// ===========================================================================
// Async (serverless Textract) flow — additive. The synchronous EC2 path above
// is unchanged; OCR (rule/ml) always uses it. Textract uploads route here only
// when the feature flag + async base URL are both configured.
// ===========================================================================

export function getAsyncApiBaseUrl(): string {
  const baseUrl = ASYNC_API_BASE_URL.trim();
  if (!baseUrl) {
    throw new Error("NEXT_PUBLIC_ASYNC_API_BASE_URL is not configured.");
  }
  return baseUrl.replace(/\/$/, "");
}

/** Decide whether a given mode should use the async serverless path. */
export function shouldUseAsync(mode: ProcessingMode): boolean {
  return TEXTRACT_ASYNC_ENABLED && mode === "textract" && ASYNC_API_BASE_URL.trim().length > 0;
}

/**
 * Validate a presigned S3 URL returned by result-handler. Async artifact URLs
 * are cross-origin (S3), so they cannot pass the same-origin `/files/` guard
 * used by the sync path; instead we require https + an amazonaws.com host.
 */
export function normalizeAsyncFileUrl(url: string | null | undefined): string {
  const rawUrl = (url ?? "").trim();
  if (!rawUrl) {
    return "";
  }
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return "";
  }
  if (parsed.protocol !== "https:") {
    return "";
  }
  if (!/(^|\.)amazonaws\.com$/i.test(parsed.hostname)) {
    return "";
  }
  return parsed.toString();
}

function contentTypeForFile(file: File): string {
  if (file.type) {
    return file.type.split(";", 1)[0].trim().toLowerCase();
  }
  const name = file.name.toLowerCase();
  if (name.endsWith(".pdf")) return "application/pdf";
  if (name.endsWith(".png")) return "image/png";
  if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
  return "application/octet-stream";
}

async function requestPresignedUpload(file: File, mode: ProcessingMode): Promise<PresignResponse> {
  const response = await fetch(`${getAsyncApiBaseUrl()}/uploads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      content_type: contentTypeForFile(file),
      mode
    })
  });

  if (!response.ok) {
    let message = `Could not start upload (status ${response.status}).`;
    try {
      const data = (await response.json()) as { message?: string };
      message = data.message ?? message;
    } catch {
      // keep fallback
    }
    throw new Error(message);
  }

  const data = (await response.json()) as PresignResponse;
  if (!data.job_id || !data.upload?.url) {
    throw new Error("Upload could not be initialized.");
  }
  return data;
}

async function uploadToS3(presign: PresignResponse, file: File): Promise<void> {
  const formData = new FormData();
  Object.entries(presign.upload.fields ?? {}).forEach(([key, value]) => {
    formData.append(key, value);
  });
  // The file part MUST be appended last for an S3 POST policy.
  formData.append("file", file);

  const response = await fetch(presign.upload.url, { method: "POST", body: formData });
  if (!response.ok && response.status !== 201 && response.status !== 204) {
    throw new Error(`Upload to storage failed (status ${response.status}).`);
  }
}

async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await fetch(`${getAsyncApiBaseUrl()}/jobs/${encodeURIComponent(jobId)}`, {
    method: "GET",
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error(`Could not read job status (status ${response.status}).`);
  }
  return (await response.json()) as JobStatusResponse;
}

async function fetchResult(jobId: string): Promise<ProcessFormResponse> {
  const response = await fetch(`${getAsyncApiBaseUrl()}/result/${encodeURIComponent(jobId)}`, {
    method: "GET",
    cache: "no-store"
  });
  if (!response.ok) {
    let message = `Could not fetch result (status ${response.status}).`;
    try {
      const data = (await response.json()) as { message?: string };
      message = data.message ?? message;
    } catch {
      // keep fallback
    }
    throw new Error(message);
  }

  const data = (await response.json()) as ProcessFormResponse;
  const pdfUrl = normalizeAsyncFileUrl(data.pdf_url);
  if (!pdfUrl) {
    throw new Error("Result contained an invalid PDF URL.");
  }
  return {
    status: data.status,
    mode: data.mode,
    pdf_url: pdfUrl,
    mapping_preview: data.mapping_preview ? normalizeAsyncFileUrl(data.mapping_preview) || undefined : undefined,
    result_url: data.result_url ? normalizeAsyncFileUrl(data.result_url) || undefined : undefined,
    stats: data.stats
  };
}

const POLL_TIMEOUT_MS = 3 * 60 * 1000;
const POLL_INTERVALS_MS = [1500, 2000, 3000, 3000, 4000];

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Full async flow: presign → upload → poll → fetch result. Returns the SAME
 * ProcessFormResponse shape as the sync path so the result page is unchanged.
 * `onStatus` lets the upload page surface QUEUED/PROCESSING transitions.
 */
export async function processFormAsync(
  file: File,
  mode: ProcessingMode,
  onStatus?: (status: JobStatus) => void
): Promise<ProcessFormResponse> {
  const presign = await requestPresignedUpload(file, mode);
  await uploadToS3(presign, file);

  const startedAt = Date.now();
  let attempt = 0;
  let lastStatus: JobStatus = "QUEUED";
  onStatus?.(lastStatus);

  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    await delay(POLL_INTERVALS_MS[Math.min(attempt, POLL_INTERVALS_MS.length - 1)]);
    attempt += 1;

    let job: JobStatusResponse;
    try {
      job = await fetchJobStatus(presign.job_id);
    } catch {
      // Transient read error (e.g. row not yet visible) — keep polling.
      continue;
    }

    if (job.status !== lastStatus) {
      lastStatus = job.status;
      onStatus?.(lastStatus);
    }

    if (job.status === "SUCCEEDED") {
      return fetchResult(presign.job_id);
    }
    if (job.status === "FAILED" || job.status === "DEAD_LETTER") {
      throw new Error(job.error?.message || "Processing failed. Please try another document.");
    }
  }

  throw new Error("Still processing — this is taking longer than expected. Please try again shortly.");
}
