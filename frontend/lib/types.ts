export type SupportedFileType = "image/png" | "image/jpeg" | "application/pdf";

export interface ProcessFormResponse {
  pdf_url: string;
  mapping_preview?: string;
}

export interface UploadSessionData {
  originalPreview: string;
  originalMimeType: string;
  originalFileName: string;
}
