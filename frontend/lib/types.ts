export type SupportedFileType = "image/png" | "image/jpeg" | "application/pdf";
export type ProcessingMode = "rule" | "ml";

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
