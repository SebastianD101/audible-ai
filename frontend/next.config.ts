import type { NextConfig } from "next";

// Static export: the app is one client-side page, so `next build` writes plain files to
// `out/`, which the API's serve image hosts on the same origin (see docs/deploy.md).
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
