'use client';

import React from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    // Log the error to the console
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
      <div className="max-w-xl w-full bg-white dark:bg-gray-800 shadow-lg rounded-lg p-8">
        <h2 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-4">
          Something went wrong!
        </h2>
        <div className="bg-red-50 dark:bg-red-900/30 p-4 rounded-md mb-6">
          <p className="text-red-800 dark:text-red-200 font-mono text-sm whitespace-pre-wrap break-words">
            {error.message || 'An unexpected error occurred'}
          </p>
        </div>
        <div className="flex flex-col gap-4">
          <button
            onClick={reset}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
          >
            Try again
          </button>
          <a
            href="/"
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-center rounded-md transition-colors"
          >
            Return to home page
          </a>
        </div>
      </div>
    </div>
  );
} 