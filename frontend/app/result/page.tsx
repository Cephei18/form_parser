"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { FilePreview } from "@/components/file-preview";
import { MappingOverlay } from "@/components/mapping-overlay";
import { PageCard } from "@/components/page-card";
import { StatusAlert } from "@/components/status-alert";
import type { ResultPayload, UploadSessionData } from "@/lib/types";

const SESSION_KEY = "form-parser:last-upload";

type RawMapping = {
  label?: string;
  label_pos?: [number, number];
  field_type?: string;
  field_bboxes?: Array<{ x: number; y: number; width: number; height: number; field_type?: string }>;
  field_lines?: Array<{ start: [number, number]; end: [number, number]; field_type?: string }>;
};

function mappingJsonUrl(mappingPreview: string): string {
  return mappingPreview.replace(/mapping\.png(?:\?.*)?$/i, "mappings.json");
}

function normalizeMappings(raw: RawMapping[]) {
  return raw.map((mapping) => {
    const fieldBboxes =
      mapping.field_bboxes ??
      (mapping.field_lines ?? []).map((line) => ({
        x: Math.min(line.start[0], line.end[0]),
        y: Math.max(0, Math.min(line.start[1], line.end[1]) - 4),
        width: Math.abs(line.end[0] - line.start[0]),
        height: 18,
        field_type: line.field_type ?? "line"
      }));

    return {
      label: mapping.label ?? "Field",
      label_pos: mapping.label_pos ?? [0, 0],
      field_bboxes: fieldBboxes,
      field_type: mapping.field_type ?? (fieldBboxes.length > 1 ? "multi_line" : "line")
    };
  });
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-3xl border border-white/70 bg-white/80 px-5 py-4 shadow-sm backdrop-blur">
      <div className="text-sm text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-ink">{value}</div>
    </div>
  );
}

function ResultContent() {
  const searchParams = useSearchParams();
  const [uploadData, setUploadData] = useState<UploadSessionData | null>(null);
  const [resultData, setResultData] = useState<ResultPayload | null>(null);
  const [resultError, setResultError] = useState("");

  const pdfUrl = useMemo(() => searchParams.get("pdf_url") ?? "", [searchParams]);
  const resultUrl = useMemo(() => searchParams.get("result_url") ?? "", [searchParams]);
  const mappingCount = useMemo(() => searchParams.get("mapping_count") ?? "", [searchParams]);
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

  useEffect(() => {
    const fallbackMappingsUrl = mappingPreview ? mappingJsonUrl(mappingPreview) : "";
    const sourceUrl = resultUrl || fallbackMappingsUrl;
    if (!sourceUrl) {
      return;
    }

    let isMounted = true;
    fetch(sourceUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load result JSON (${response.status}).`);
        }
        return response.json() as Promise<ResultPayload | RawMapping[]>;
      })
      .then((data) => {
        if (isMounted) {
          if (Array.isArray(data)) {
            const mappings = normalizeMappings(data);
            setResultData({
              status: "success",
              mode: uploadData?.mode ?? "rule",
              mappings,
              stats: {
                ocr_count: 0,
                line_count: 0,
                field_candidate_count: mappings.length,
                mapping_count: mappings.length,
                checkbox_count: mappings.filter((item) => item.field_type === "checkbox").length,
                multi_line_count: mappings.filter((item) => item.field_type === "multi_line").length
              }
            });
          } else {
            setResultData(data);
          }
        }
      })
      .catch((error: unknown) => {
        if (isMounted) {
          setResultError(error instanceof Error ? error.message : "Unable to load result data.");
        }
      });

    return () => {
      isMounted = false;
    };
  }, [mappingPreview, resultUrl, uploadData?.mode]);

  if (!pdfUrl) {
    return (
      <PageCard title="Result unavailable" subtitle="No processing result found.">
        <div className="space-y-4">
          <StatusAlert kind="error" message="Missing required result data." />
          <Link
            href="/"
            className="inline-flex rounded-2xl border border-edge bg-white px-4 py-2 text-sm font-semibold text-ink hover:border-brand hover:text-brand"
          >
            Start again
          </Link>
        </div>
      </PageCard>
    );
  }

  const stats = resultData?.stats;

  return (
    <div className="space-y-8">
      <section className="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-panel backdrop-blur sm:p-8">
        <div className="grid gap-6 lg:grid-cols-[0.85fr_1.15fr] lg:items-center">
          <div className="space-y-5">
            <div className="inline-flex rounded-full bg-ok/10 px-4 py-2 text-sm font-semibold text-ok">
              Conversion complete
            </div>
            <div>
              <h1 className="text-3xl font-semibold text-ink sm:text-4xl">
                Your fillable PDF is ready
              </h1>
              <p className="mt-3 text-base leading-7 text-slate-600">
                Review the generated document, download it, or run another form through the assistant.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <a
                href={pdfUrl}
                target="_blank"
                rel="noreferrer"
                download
                className="inline-flex rounded-2xl bg-ink px-5 py-3 text-sm font-semibold text-white shadow-soft transition hover:-translate-y-0.5 hover:bg-slate-800"
                aria-label="Download generated fillable PDF"
              >
                Download PDF
              </a>

              <Link
                href="/"
                className="inline-flex rounded-2xl border border-edge bg-white px-5 py-3 text-sm font-semibold text-ink transition hover:border-brand hover:text-brand"
              >
                Process another form
              </Link>
            </div>
          </div>

          <FilePreview src={pdfUrl} mimeType="application/pdf" title="Generated fillable PDF" />
        </div>
      </section>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Fields detected" value={stats?.mapping_count ?? mappingCount ?? "-"} />
        <StatCard label="Multiline fields" value={stats?.multi_line_count ?? "-"} />
        <StatCard label="Checkboxes" value={stats?.checkbox_count ?? "-"} />
      </div>

      {uploadData?.originalPreview ? (
        <FilePreview
          src={uploadData.originalPreview}
          mimeType={uploadData.originalMimeType}
          title={`Original form: ${uploadData.originalFileName}`}
        />
      ) : null}

      {resultError ? <StatusAlert kind="error" message={resultError} /> : null}

      <details className="rounded-3xl border border-edge bg-white/80 p-5 shadow-sm backdrop-blur">
        <summary className="cursor-pointer text-sm font-semibold text-ink">
          Advanced details
        </summary>
        <div className="mt-5 space-y-5">
          {mappingPreview ? (
            <FilePreview src={mappingPreview} mimeType="image/png" title="Field detection preview" />
          ) : null}

          {resultData && mappingPreview ? (
            <MappingOverlay
              imageSrc={mappingPreview}
              mappings={resultData.mappings}
              title="Field overlay"
            />
          ) : null}
        </div>
      </details>
    </div>
  );
}

export default function ResultPage() {
  return (
    <Suspense
      fallback={
        <PageCard title="Loading result" subtitle="Preparing your processed form.">
          <div className="h-24 rounded-3xl border border-edge bg-slate-50" />
        </PageCard>
      }
    >
      <ResultContent />
    </Suspense>
  );
}
