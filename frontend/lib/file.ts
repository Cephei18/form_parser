import {
  ACCEPTED_EXTENSIONS,
  MAX_FILE_SIZE_MB
} from "@/lib/constants";
import type { SupportedFileType } from "@/lib/types";

const bytesToMb = (bytes: number) => bytes / (1024 * 1024);

const MIME_TYPES_BY_EXTENSION: Record<string, SupportedFileType[]> = {
  ".pdf": ["application/pdf"],
  ".png": ["image/png"],
  ".jpg": ["image/jpeg"],
  ".jpeg": ["image/jpeg"]
};

export function validateFile(file: File): string | null {
  const extension = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
  const hasValidExtension = ACCEPTED_EXTENSIONS.includes(extension);
  const acceptedMimeTypes = MIME_TYPES_BY_EXTENSION[extension] ?? [];
  const normalizedMimeType = file.type.toLowerCase();
  const hasValidMime = acceptedMimeTypes.includes(normalizedMimeType as SupportedFileType);

  if (!hasValidExtension) {
    return "Only PNG, JPG, and PDF files are allowed.";
  }

  if (!normalizedMimeType) {
    return "This file type could not be verified. Please upload a PNG, JPG, or PDF file.";
  }

  if (!hasValidMime) {
    return "File extension and type do not match. Please upload a valid PNG, JPG, or PDF file.";
  }

  if (bytesToMb(file.size) > MAX_FILE_SIZE_MB) {
    return `File must be smaller than ${MAX_FILE_SIZE_MB}MB.`;
  }

  return null;
}

export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to read file for preview."));
    reader.readAsDataURL(file);
  });
}
