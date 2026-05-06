import { API_BASE_URL } from "@/lib/constants";
import type { ProcessingMode, ProcessFormResponse } from "@/lib/types";

function getApiBaseUrl(): string {
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

function toAbsoluteUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) {
    return url;
  }

  const baseUrl = API_BASE_URL.trim();

  if (url.startsWith("/")) {
    return baseUrl ? `${baseUrl.replace(/\/$/, "")}${url}` : url;
  }

  return baseUrl ? `${baseUrl.replace(/\/$/, "")}/${url}` : url;
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
