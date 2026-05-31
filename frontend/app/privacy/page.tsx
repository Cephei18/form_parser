export default function PrivacyPage() {
  return (
    <section className="mx-auto max-w-3xl rounded-lg border border-edge bg-white/85 p-6 shadow-panel sm:p-8">
      <p className="text-sm font-semibold uppercase tracking-[0.08em] text-accent">Privacy</p>
      <h1 className="mt-3 text-3xl font-semibold text-ink">Privacy notice</h1>
      <div className="mt-6 space-y-5 text-sm leading-6 text-slate-700">
        <p>
          FormFlow AI accepts PDF and image uploads to generate fillable PDF outputs and mapping files. Uploaded files and generated artifacts are stored on the backend only for processing, retrieval, troubleshooting, and configured retention cleanup.
        </p>
        <p>
          Do not upload documents unless you are authorized to process them. Documents may contain personal, financial, or official information, and access to generated URLs should be limited to the intended user or operator.
        </p>
        <p>
          Production deployments should configure explicit storage retention, backups, access controls, monitoring, and an incident response contact before handling sensitive or regulated documents.
        </p>
      </div>
    </section>
  );
}
