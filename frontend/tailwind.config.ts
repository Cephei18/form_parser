import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#101828",
        mist: "#f6f8fb",
        brand: "#2563eb",
        accent: "#0f766e",
        blush: "#fff1f2",
        ok: "#15803d",
        warn: "#b42318",
        edge: "#d9e2ec"
      },
      boxShadow: {
        panel: "0 18px 55px rgba(16, 24, 40, 0.10)",
        soft: "0 12px 34px rgba(37, 99, 235, 0.14)"
      }
    }
  },
  plugins: []
};

export default config;
