import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Form Parser",
  description: "Upload forms and generate fillable PDFs"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <main className="mx-auto min-h-screen w-full max-w-5xl px-4 py-8 sm:px-6 sm:py-10">
          {children}
        </main>
      </body>
    </html>
  );
}
