import type { SupportedFileType } from "@/lib/types";

interface FilePreviewProps {
  src: string;
  mimeType: string;
  title: string;
}

export function FilePreview({ src, mimeType, title }: FilePreviewProps) {
  const isPdf = mimeType === "application/pdf";
  const isImage = ["image/png", "image/jpeg"].includes(mimeType as SupportedFileType);

  return (
    <section aria-label={title} className="overflow-hidden rounded-xl border border-edge bg-white">
      <header className="border-b border-edge px-4 py-3 text-sm font-medium">{title}</header>
      <div className="h-[420px] w-full bg-slate-50">
        {isPdf ? (
          <iframe src={src} title={title} className="h-full w-full" />
        ) : isImage ? (
          // Native img supports data URLs and arbitrary backend URLs without extra image config.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={src} alt={title} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full items-center justify-center px-4 text-sm text-slate-600">
            Preview unavailable for this file type.
          </div>
        )}
      </div>
    </section>
  );
}
