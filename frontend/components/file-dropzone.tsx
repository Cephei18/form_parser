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
      className="rounded-xl border-2 border-dashed border-edge bg-white p-6 text-center shadow-panel"
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
      <label htmlFor="form-upload" className="block text-sm font-semibold text-ink">
        Drag and drop a form file here
      </label>
      <p className="mt-1 text-sm text-slate-600">PNG, JPG, or PDF up to 20MB</p>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-4 rounded-lg border border-edge px-4 py-2 text-sm font-medium transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
        disabled={disabled}
      >
        Browse files
      </button>
    </div>
  );
}
