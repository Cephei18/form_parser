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
import type { ProcessingMode, UploadSessionData } from "@/lib/types";

const SESSION_KEY = "form-parser:last-upload";
const STAGES = ["Reading document", "Understanding layout", "Detecting fields", "Generating PDF"];

export default function UploadPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewSource, setPreviewSource] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [mode, setMode] = useState<ProcessingMode>("rule");
  const [activeStage, setActiveStage] = useState(0);

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
      setSuccess("Your document is ready.");
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

    let stageTimer: number | undefined;
    try {
      setIsProcessing(true);
      setActiveStage(0);
      setError("");
      setSuccess("");

      stageTimer = window.setInterval(() => {
        setActiveStage((stage) => Math.min(stage + 1, STAGES.length - 1));
      }, 900);

      const result = await processForm(selectedFile, mode);
      setActiveStage(STAGES.length - 1);

      const uploadSession: UploadSessionData = {
        originalPreview: previewSource,
        originalMimeType: selectedFile.type,
        originalFileName: selectedFile.name,
        mode,
        stages: STAGES
      };

      window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(uploadSession));

      const params = new URLSearchParams({
        pdf_url: result.pdf_url
      });

      if (result.mapping_preview) {
        params.set("mapping_preview", result.mapping_preview);
      }
      if (result.result_url) {
        params.set("result_url", result.result_url);
      }
      if (result.stats) {
        params.set("mapping_count", String(result.stats.mapping_count));
      }

      router.push(`/result?${params.toString()}`);
    } catch (unknownError) {
      const message =
        unknownError instanceof Error
          ? unknownError.message
          : "Something went wrong while processing the file.";
      setError(message);
    } finally {
      if (stageTimer !== undefined) {
        window.clearInterval(stageTimer);
      }
      setIsProcessing(false);
    }
  }, [mode, previewSource, router, selectedFile]);

  return (
    <div className="space-y-8">
      <section className="grid gap-8 py-8 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:py-12">
        <div className="space-y-6">
          <div className="inline-flex rounded-full border border-white/70 bg-white/75 px-4 py-2 text-sm font-medium text-slate-700 shadow-sm backdrop-blur">
            AI-powered document conversion
          </div>
          <div className="space-y-4">
            <h1 className="max-w-3xl text-4xl font-semibold leading-tight text-ink sm:text-5xl lg:text-6xl">
              Convert scanned forms into fillable PDFs
            </h1>
            <p className="max-w-2xl text-base leading-7 text-slate-600 sm:text-lg">
              Upload a form and let FormFlow AI read the document, understand the layout, detect fillable fields, and prepare a polished PDF.
            </p>
          </div>
          <div className="grid max-w-2xl gap-3 sm:grid-cols-3">
            {["Private processing", "Layout-aware fields", "Ready to download"].map((item) => (
              <div key={item} className="rounded-2xl border border-white/70 bg-white/75 px-4 py-3 text-sm font-semibold text-slate-700 shadow-sm backdrop-blur">
                {item}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-[2rem] border border-white/70 bg-white/70 p-3 shadow-panel backdrop-blur">
          <FileDropzone onFileSelect={handleFileSelect} disabled={isProcessing} />
        </div>
      </section>

      <PageCard
        title="Prepare your document"
        subtitle="Preview the upload, then start the conversion."
      >
        <div className="space-y-5">
          {previewSource && selectedFile ? (
            <FilePreview
              src={previewSource}
              mimeType={previewMimeType}
              title={`Uploaded document: ${selectedFile.name}`}
            />
          ) : null}

          {success ? <StatusAlert kind="success" message={success} /> : null}
          {error ? <StatusAlert kind="error" message={error} /> : null}

          <details className="rounded-2xl border border-edge bg-white px-4 py-3">
            <summary className="cursor-pointer text-sm font-semibold text-ink">
              Advanced options
            </summary>
            <div className="mt-3 inline-flex rounded-2xl border border-edge bg-slate-50 p-1">
              {(["rule", "ml"] as ProcessingMode[]).map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setMode(item)}
                  className={`rounded-xl px-3 py-1.5 text-sm font-medium ${
                    mode === item ? "bg-ink text-white" : "text-slate-700 hover:bg-white"
                  }`}
                  disabled={isProcessing}
                >
                  {item === "rule" ? "Standard AI" : "ML beta"}
                </button>
              ))}
            </div>
          </details>

          {isProcessing ? (
            <div className="rounded-3xl border border-brand/20 bg-brand/5 p-4">
              <div className="mb-4 h-2 overflow-hidden rounded-full bg-white">
                <div
                  className="h-full rounded-full bg-brand transition-all duration-500"
                  style={{ width: `${((activeStage + 1) / STAGES.length) * 100}%` }}
                />
              </div>
              <div className="grid gap-2 sm:grid-cols-4">
                {STAGES.map((stage, index) => (
                  <div
                    key={stage}
                    className={`rounded-2xl border px-3 py-3 text-sm font-semibold ${
                      index <= activeStage
                        ? "border-brand/30 bg-white text-brand shadow-sm"
                        : "border-edge bg-white text-slate-500"
                    }`}
                  >
                    {stage}
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          <div className="flex flex-wrap gap-3">
            <LoadingButton
              isLoading={isProcessing}
              loadingLabel="Creating your PDF..."
              onClick={handleSubmit}
              disabled={!selectedFile}
              aria-label="Convert uploaded form"
              className="w-full sm:w-auto"
            >
              Convert to fillable PDF
            </LoadingButton>
          </div>
        </div>
      </PageCard>
    </div>
  );
}
