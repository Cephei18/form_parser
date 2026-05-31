import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FormFlow AI",
  description: "Convert scanned forms into fillable PDFs"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <main className="mx-auto min-h-screen w-full max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
          {children}
        </main>
        <footer className="mx-auto flex w-full max-w-6xl flex-col gap-3 px-4 pb-8 text-sm text-slate-600 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <p>FormFlow AI processes uploads only to generate form outputs.</p>
          <nav aria-label="Legal" className="flex flex-wrap gap-x-4 gap-y-2">
            <a href="/privacy">Privacy</a>
            <a href="/terms">Terms</a>
            <a href="/disclaimer">Disclaimer</a>
          </nav>
        </footer>
      </body>
    </html>
  );
}
