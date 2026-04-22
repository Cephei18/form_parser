import {
  ACCEPTED_EXTENSIONS,
  ACCEPTED_MIME_TYPES,
  MAX_FILE_SIZE_MB
} from "@/lib/constants";

const bytesToMb = (bytes: number) => bytes / (1024 * 1024);

export function validateFile(file: File): string | null {
  const extension = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
  const hasValidMime = ACCEPTED_MIME_TYPES.includes(file.type as never);
  const hasValidExtension = ACCEPTED_EXTENSIONS.includes(extension);

  if (!hasValidMime && !hasValidExtension) {
    return "Only PNG, JPG, and PDF files are allowed.";
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
