export default function TermsPage() {
  return (
    <section className="mx-auto max-w-3xl rounded-lg border border-edge bg-white/85 p-6 shadow-panel sm:p-8">
      <p className="text-sm font-semibold uppercase tracking-[0.08em] text-accent">Terms</p>
      <h1 className="mt-3 text-3xl font-semibold text-ink">Terms of use</h1>
      <div className="mt-6 space-y-5 text-sm leading-6 text-slate-700">
        <p>
          This service converts uploaded forms into generated fillable PDF outputs. Users are responsible for ensuring they have the right to upload and process each document.
        </p>
        <p>
          Generated outputs should be reviewed before official use. OCR, layout reconstruction, and field detection can make mistakes, especially on low-quality scans or complex forms.
        </p>
        <p>
          Operators should configure production limits, retention, logging, monitoring, and support channels that match their business and compliance obligations.
        </p>
      </div>
    </section>
  );
}
