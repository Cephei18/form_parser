"use client";

import { useRouter } from "next/navigation";
import { useCallback, useMemo, useState } from "react";
import { FileDropzone } from "@/components/file-dropzone";
import { FilePreview } from "@/components/file-preview";
import { LoadingButton } from "@/components/loading-button";
import { PageCard } from "@/components/page-card";
import { StatusAlert } from "@/components/status-alert";
import { processForm } from "@/lib/api";
import { fileToDataUrl, validateFile } from "@/lib/file";
import type { UploadSessionData } from "@/lib/types";

const SESSION_KEY = "form-parser:last-upload";

export default function UploadPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewSource, setPreviewSource] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);

  const previewMimeType = useMemo(() => selectedFile?.type ?? "", [selectedFile]);

  const handleFileSelect = useCallback(async (file: File) => {
    setError("");
    setSuccess("");

    const validationError = validateFile(file);
    if (validationError) {
      setSelectedFile(null);
      setPreviewSource("");
      setError(validationError);
      return;
    }

    try {
      const dataUrl = await fileToDataUrl(file);
      setSelectedFile(file);
      setPreviewSource(dataUrl);
      setSuccess("File is ready to be processed.");
    } catch {
      setSelectedFile(null);
      setPreviewSource("");
      setError("Unable to generate file preview.");
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!selectedFile) {
      setError("Please choose a file first.");
      return;
    }

    try {
      setIsProcessing(true);
      setError("");
      setSuccess("");

      const result = await processForm(selectedFile);

      const uploadSession: UploadSessionData = {
        originalPreview: previewSource,
        originalMimeType: selectedFile.type,
        originalFileName: selectedFile.name
      };

      window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(uploadSession));

      const params = new URLSearchParams({
        pdf_url: result.pdf_url
      });

      if (result.mapping_preview) {
        params.set("mapping_preview", result.mapping_preview);
      }

      router.push(`/result?${params.toString()}`);
    } catch (unknownError) {
      const message =
        unknownError instanceof Error
          ? unknownError.message
          : "Something went wrong while processing the file.";
      setError(message);
    } finally {
      setIsProcessing(false);
    }
  }, [previewSource, router, selectedFile]);

  return (
    <PageCard
      title="Form to Fillable PDF"
      subtitle="Upload a scanned form or PDF and convert it into a fillable PDF."
    >
      <div className="space-y-5">
        <FileDropzone onFileSelect={handleFileSelect} disabled={isProcessing} />

        {previewSource && selectedFile ? (
          <FilePreview
            src={previewSource}
            mimeType={previewMimeType}
            title={`Selected file: ${selectedFile.name}`}
          />
        ) : null}

        {success ? <StatusAlert kind="success" message={success} /> : null}
        {error ? <StatusAlert kind="error" message={error} /> : null}

        <div className="flex flex-wrap gap-3">
          <LoadingButton
            isLoading={isProcessing}
            loadingLabel="Processing form..."
            onClick={handleSubmit}
            disabled={!selectedFile}
            aria-label="Process uploaded form"
          >
            Process Form
          </LoadingButton>
        </div>
      </div>
    </PageCard>
  );
}
