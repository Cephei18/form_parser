import type { PropsWithChildren } from "react";

interface PageCardProps extends PropsWithChildren {
  title: string;
  subtitle?: string;
}

export function PageCard({ title, subtitle, children }: PageCardProps) {
  return (
    <section className="rounded-2xl border border-edge bg-white p-6 shadow-panel sm:p-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {subtitle ? <p className="mt-1 text-sm text-slate-600">{subtitle}</p> : null}
      </header>
      {children}
    </section>
  );
}
