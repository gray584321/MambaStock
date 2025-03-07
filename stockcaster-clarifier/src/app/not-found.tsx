import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
      <div className="max-w-xl w-full bg-white dark:bg-gray-800 shadow-lg rounded-lg p-8 text-center">
        <h2 className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-4">
          404
        </h2>
        <h3 className="text-2xl font-semibold mb-6">Page Not Found</h3>
        <p className="text-gray-600 dark:text-gray-300 mb-8">
          The page you are looking for doesn't exist or has been moved.
        </p>
        <Link 
          href="/" 
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors inline-block"
        >
          Return Home
        </Link>
      </div>
    </div>
  );
} 