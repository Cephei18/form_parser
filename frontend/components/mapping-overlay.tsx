"use client";

import { useState } from "react";
import type { MappingItem } from "@/lib/types";

interface MappingOverlayProps {
  imageSrc: string;
  mappings: MappingItem[];
  title: string;
}

export function MappingOverlay({ imageSrc, mappings, title }: MappingOverlayProps) {
  const [size, setSize] = useState({ width: 0, height: 0 });

  return (
    <section aria-label={title} className="overflow-hidden rounded-xl border border-edge bg-white">
      <header className="flex items-center justify-between border-b border-edge px-4 py-3">
        <span className="text-sm font-medium">{title}</span>
        <span className="text-xs text-slate-500">{mappings.length} mapped fields</span>
      </header>
      <div className="bg-slate-50 p-3">
        <div className="relative mx-auto max-h-[620px] w-fit overflow-hidden rounded-lg border border-edge bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageSrc}
            alt={title}
            className="max-h-[620px] w-auto max-w-full object-contain"
            onLoad={(event) => {
              setSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight
              });
            }}
          />
          {mappings.map((mapping, mappingIndex) =>
            mapping.field_bboxes.map((box, boxIndex) => (
              <div
                key={`${mappingIndex}-${boxIndex}`}
                className={`absolute border-2 ${
                  mapping.field_type === "checkbox"
                    ? "border-amber-500 bg-amber-300/20"
                    : "border-emerald-500 bg-emerald-300/10"
                }`}
                style={{
                  left: size.width ? `${(box.x / size.width) * 100}%` : 0,
                  top: size.height ? `${(box.y / size.height) * 100}%` : 0,
                  width: size.width ? `${(box.width / size.width) * 100}%` : 0,
                  height: size.height ? `${(box.height / size.height) * 100}%` : 0
                }}
                title={`${mapping.label} (${mapping.field_type})`}
              />
            ))
          )}
        </div>
      </div>
    </section>
  );
}
