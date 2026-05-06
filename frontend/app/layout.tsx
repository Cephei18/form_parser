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
      </body>
    </html>
  );
}
