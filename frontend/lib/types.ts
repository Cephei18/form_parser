export type SupportedFileType = "image/png" | "image/jpeg" | "application/pdf";
// "rule" / "ml" → synchronous EC2 OCR pipeline; "textract" → async serverless flow.
export type ProcessingMode = "rule" | "ml" | "textract";

export interface ProcessingStats {
  ocr_count: number;
  line_count: number;
  field_candidate_count: number;
  mapping_count: number;
  checkbox_count: number;
  multi_line_count: number;
}

export interface FieldBox {
  x: number;
  y: number;
  width: number;
  height: number;
  field_type?: string;
}

export interface MappingItem {
  label: string;
  label_pos: [number, number];
  field_bboxes: FieldBox[];
  field_type: string;
}

export interface ResultPayload {
  status: string;
  mode: ProcessingMode;
  stats: ProcessingStats;
  mappings: MappingItem[];
}

export interface ProcessFormResponse {
  status?: string;
  mode?: ProcessingMode;
  pdf_url: string;
  mapping_preview?: string;
  result_url?: string;
  stats?: ProcessingStats;
}

export interface UploadSessionData {
  originalPreview: string;
  originalMimeType: string;
  originalFileName: string;
  mode: ProcessingMode;
  stages: string[];
}

// --- Async (serverless Textract) flow ---------------------------------------

export type JobStatus =
  | "QUEUED"
  | "PROCESSING"
  | "SUCCEEDED"
  | "FAILED"
  | "DEAD_LETTER"
  | "UNKNOWN";

export interface PresignResponse {
  job_id: string;
  raw_key: string;
  expires_in: number;
  upload: {
    method: string;
    url: string;
    fields: Record<string, string>;
  };
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  mode?: string;
  created_at?: string;
  updated_at?: string;
  result_ready: boolean;
  terminal: boolean;
  metrics?: Record<string, number | string>;
  error?: { type?: string; message?: string };
}
