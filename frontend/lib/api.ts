import { API_BASE_URL } from "@/lib/constants";
import type { ProcessingMode, ProcessFormResponse } from "@/lib/types";

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
