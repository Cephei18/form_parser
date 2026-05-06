"use client";

import type { ChangeEvent, DragEvent } from "react";
import { useCallback, useRef } from "react";

interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export function FileDropzone({ onFileSelect, disabled }: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      if (disabled) {
        return;
      }

      const file = event.dataTransfer.files?.[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [disabled, onFileSelect]
  );

  const onDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  }, []);

  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      className="group rounded-3xl border border-dashed border-brand/40 bg-white/80 p-8 text-center shadow-panel backdrop-blur transition hover:-translate-y-0.5 hover:border-brand hover:bg-white"
      aria-disabled={disabled}
    >
      <input
        ref={inputRef}
        type="file"
        className="sr-only"
        id="form-upload"
        accept=".png,.jpg,.jpeg,.pdf"
        onChange={onInputChange}
        disabled={disabled}
      />
      <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-brand/10 text-2xl text-brand transition group-hover:scale-105">
        +
      </div>
      <label htmlFor="form-upload" className="block text-lg font-semibold text-ink">
        Drop a scanned form here
      </label>
      <p className="mt-2 text-sm text-slate-600">PDF, PNG, or JPG. Your fillable PDF is created in one flow.</p>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-5 rounded-2xl border border-edge bg-white px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-brand hover:text-brand disabled:cursor-not-allowed disabled:opacity-60"
        disabled={disabled}
      >
        Choose file
      </button>
    </div>
  );
}
