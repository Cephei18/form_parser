/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "export",
  // Use a dedicated dev build directory to avoid permission-locked .next/trace
  // This can be overridden via NEXT_DIST_DIR env var when needed.
  distDir: process.env.NEXT_DIST_DIR ?? ".next_dev",
  trailingSlash: true,
  images: {
    unoptimized: true
  }
};

export default nextConfig;
