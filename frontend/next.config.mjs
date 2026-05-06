/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "export",
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
  trailingSlash: true,
  images: {
    unoptimized: true
  }
};

export default nextConfig;
