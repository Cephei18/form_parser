import { API_BASE_URL } from "@/lib/constants";
import type { ProcessFormResponse } from "@/lib/types";

const ENDPOINT = `${API_BASE_URL.replace(/\/$/, "")}/process-form`;

function toAbsoluteUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) {
    return url;
  }

  if (url.startsWith("/")) {
    return `${API_BASE_URL.replace(/\/$/, "")}${url}`;
  }

  return `${API_BASE_URL.replace(/\/$/, "")}/${url}`;
}

export async function processForm(file: File): Promise<ProcessFormResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(ENDPOINT, {
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
    pdf_url: toAbsoluteUrl(data.pdf_url),
    mapping_preview: data.mapping_preview ? toAbsoluteUrl(data.mapping_preview) : undefined
  };
}
