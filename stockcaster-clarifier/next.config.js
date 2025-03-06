/** @type {import('next').NextConfig} */
const nextConfig = {
  /* config options here */
  reactStrictMode: false, // Disable strict mode for better performance
  images: {
    unoptimized: true, // Use unoptimized images for better performance
  },
  experimental: {
    // Optimize package imports for better performance
    optimizePackageImports: ['recharts'],
  },
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Warning: This allows production builds to successfully complete even if
    // your project has TypeScript errors.
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig; 