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
        ink: "#101820",
        mist: "#f4f6f8",
        brand: "#1c7ed6",
        ok: "#2f9e44",
        warn: "#d9480f",
        edge: "#d0d7de"
      },
      boxShadow: {
        panel: "0 10px 30px rgba(16, 24, 32, 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
