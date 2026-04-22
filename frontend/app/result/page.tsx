"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { FilePreview } from "@/components/file-preview";
import { PageCard } from "@/components/page-card";
import { StatusAlert } from "@/components/status-alert";
import type { UploadSessionData } from "@/lib/types";

const SESSION_KEY = "form-parser:last-upload";

export default function ResultPage() {
  const searchParams = useSearchParams();
  const [uploadData, setUploadData] = useState<UploadSessionData | null>(null);

  const pdfUrl = useMemo(() => searchParams.get("pdf_url") ?? "", [searchParams]);
  const mappingPreview = useMemo(
    () => searchParams.get("mapping_preview") ?? "",
    [searchParams]
  );

  useEffect(() => {
    const raw = window.sessionStorage.getItem(SESSION_KEY);
    if (!raw) {
      return;
    }

    try {
      const parsed = JSON.parse(raw) as UploadSessionData;
      setUploadData(parsed);
    } catch {
      // Ignore invalid session state.
    }
  }, []);

  if (!pdfUrl) {
    return (
      <PageCard title="Result unavailable" subtitle="No processing result found.">
        <div className="space-y-4">
          <StatusAlert kind="error" message="Missing required result data (pdf_url)." />
          <Link
            href="/"
            className="inline-flex rounded-lg border border-edge px-4 py-2 text-sm font-medium text-ink hover:bg-slate-50"
          >
            Back to upload
          </Link>
        </div>
      </PageCard>
    );
  }

  return (
    <PageCard title="Processing complete" subtitle="Your fillable PDF is ready.">
      <div className="space-y-5">
        {uploadData?.originalPreview ? (
          <FilePreview
            src={uploadData.originalPreview}
            mimeType={uploadData.originalMimeType}
            title={`Original upload: ${uploadData.originalFileName}`}
          />
        ) : (
          <StatusAlert
            kind="error"
            message="Original preview is not available in this session."
          />
        )}

        {mappingPreview ? (
          <FilePreview src={mappingPreview} mimeType="image/png" title="Detected mapping preview" />
        ) : null}

        <div className="flex flex-wrap gap-3">
          <a
            href={pdfUrl}
            target="_blank"
            rel="noreferrer"
            download
            className="inline-flex rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white hover:bg-brand/90"
            aria-label="Download generated fillable PDF"
          >
            Download Generated PDF
          </a>

          <Link
            href="/"
            className="inline-flex rounded-lg border border-edge px-4 py-2 text-sm font-medium text-ink hover:bg-slate-50"
          >
            Process another file
          </Link>
        </div>
      </div>
    </PageCard>
  );
}
