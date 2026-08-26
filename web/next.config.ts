import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The Modal URL is read at request time in `lib/server.ts`, never inlined
  // into the client bundle, so one build works against any deployment.
  reactStrictMode: true,
};

export default nextConfig;
