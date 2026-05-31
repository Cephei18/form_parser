export default function DisclaimerPage() {
  return (
    <section className="mx-auto max-w-3xl rounded-lg border border-edge bg-white/85 p-6 shadow-panel sm:p-8">
      <p className="text-sm font-semibold uppercase tracking-[0.08em] text-accent">Disclaimer</p>
      <h1 className="mt-3 text-3xl font-semibold text-ink">Document output disclaimer</h1>
      <div className="mt-6 space-y-5 text-sm leading-6 text-slate-700">
        <p>
          FormFlow AI provides automated document reconstruction and fillable field generation. It does not guarantee legal, regulatory, accessibility, or submission acceptance for any generated document.
        </p>
        <p>
          Review every generated PDF before signing, filing, submitting, or distributing it. Important fields, labels, tables, and photo areas should be checked against the source document.
        </p>
        <p>
          This product is not a substitute for professional review where official, legal, financial, medical, or government document accuracy is required.
        </p>
      </div>
    </section>
  );
}
