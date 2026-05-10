import type { SupportedFileType } from "@/lib/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://15.207.134.4:8000";

export const ACCEPTED_MIME_TYPES: SupportedFileType[] = [
  "image/png",
  "image/jpeg",
  "application/pdf"
];

export const ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf"];
export const MAX_FILE_SIZE_MB = 20;
