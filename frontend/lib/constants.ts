import type { SupportedFileType } from "@/lib/types";

// Default to the textract-enabled backend on localhost:8001 so the app uses
// the Textract pipeline by default during local development. Override with
// `NEXT_PUBLIC_API_BASE_URL` when needed.
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

// Async (serverless Textract) API surface — API Gateway. Additive: when unset
// or the feature flag is off, the app uses the synchronous EC2 path unchanged.
export const ASYNC_API_BASE_URL =
  process.env.NEXT_PUBLIC_ASYNC_API_BASE_URL ?? "";

// Master switch for the async Textract flow. "true" routes Textract uploads
// through API Gateway → presigned upload → polling; anything else keeps the
// current synchronous POST /process-form behaviour (instant rollback).
export const TEXTRACT_ASYNC_ENABLED =
  (process.env.NEXT_PUBLIC_TEXTRACT_ASYNC ?? "false").trim().toLowerCase() === "true";

export const ACCEPTED_MIME_TYPES: SupportedFileType[] = [
  "image/png",
  "image/jpeg",
  "application/pdf"
];

export const ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf"];
export const MAX_FILE_SIZE_MB = 20;
