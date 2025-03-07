/** @type {import('next').NextConfig} */

// Check if using Turbopack
const isTurbopack = process.env.npm_lifecycle_script?.includes('--turbo') || 
                   process.env.NEXT_TURBOPACK === '1';

// Base configuration shared between bundlers
const baseConfig = {
  reactStrictMode: true,
  // swcMinify is deprecated in Next.js 15, removed
  
  transpilePackages: ['lightweight-charts'],
  experimental: {
    optimizePackageImports: ['recharts'],
  },
  
  images: {
    unoptimized: true, // Use unoptimized images for better performance
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

// Webpack-specific configuration
const webpackConfig = {
  ...baseConfig,
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        module: false,
      };
    }
    return config;
  },
};

// Turbopack-specific configuration (no webpack config)
const turbopackConfig = {
  ...baseConfig,
  // No webpack configuration for Turbopack
};

// Export the appropriate configuration based on the bundler
const nextConfig = isTurbopack ? turbopackConfig : webpackConfig;

module.exports = nextConfig; 