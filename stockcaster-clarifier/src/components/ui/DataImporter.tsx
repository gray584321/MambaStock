import React, { useRef, useState } from 'react';
import { useStockData } from '@/lib/StockDataContext';

const DataImporter: React.FC = () => {
  const { loadData, isLoading } = useStockData();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [importStatus, setImportStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      
      // Automatically start import
      setImportStatus('loading');
      
      try {
        const reader = new FileReader();
        
        // Create a Promise-based wrapper for the FileReader
        const readFileAsync = new Promise<string | ArrayBuffer | null>((resolve, reject) => {
          reader.onload = () => resolve(reader.result);
          reader.onerror = () => reject(reader.error);
        });
        
        reader.readAsArrayBuffer(file);
        
        // Wait for file to load
        const result = await readFileAsync;
        
        if (result) {
          // Create a blob from the file content
          const blob = new Blob([result as ArrayBuffer], { type: 'text/csv' });
          const url = URL.createObjectURL(blob);
          
          // Load data from the blob URL
          await loadData(url);
          
          // Clean up
          URL.revokeObjectURL(url);
          setImportStatus('success');
          
          // Reset file input after successful import
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        }
      } catch (error) {
        console.error('Error importing file:', error);
        setImportStatus('error');
      }
    }
  };

  const handleUseDefaultData = async () => {
    setImportStatus('loading');
    try {
      await loadData('/data/SPY_featured.csv');
      setImportStatus('success');
    } catch (error) {
      console.error('Error loading sample data:', error);
      setImportStatus('error');
    }
  };
  
  // Reset status after a delay when success or error
  React.useEffect(() => {
    if (importStatus === 'success' || importStatus === 'error') {
      const timer = setTimeout(() => {
        setImportStatus('idle');
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [importStatus]);

  return (
    <div className="bg-white rounded-lg shadow-md p-4 dark:bg-gray-800 dark:border dark:border-gray-700">
      <h3 className="font-bold text-lg mb-3 dark:text-white">Import Data</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1 dark:text-gray-300">
            Select CSV File
          </label>
          <div className="relative">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv"
              disabled={isLoading || importStatus === 'loading'}
              className="block w-full text-sm text-gray-500 dark:text-gray-400
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                dark:file:bg-blue-900 dark:file:text-blue-200
                hover:file:bg-blue-100 dark:hover:file:bg-blue-800
                disabled:opacity-50 disabled:cursor-not-allowed"
            />
            
            {/* Status indicator */}
            {importStatus === 'loading' && (
              <div className="mt-2 flex items-center text-blue-600 dark:text-blue-400">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Importing data...</span>
              </div>
            )}
            
            {importStatus === 'success' && (
              <div className="mt-2 text-green-600 dark:text-green-400">
                ✓ Data imported successfully
              </div>
            )}
            
            {importStatus === 'error' && (
              <div className="mt-2 text-red-600 dark:text-red-400">
                ✗ Error importing data
              </div>
            )}
          </div>
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            CSV files will be imported automatically when selected
          </p>
        </div>
        
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Or use sample data</p>
          <button
            onClick={handleUseDefaultData}
            disabled={isLoading || importStatus === 'loading'}
            className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 dark:focus:ring-offset-gray-800 disabled:opacity-50"
          >
            {importStatus === 'loading' ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Loading...
              </>
            ) : 'Load Sample Data (SPY)'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DataImporter; 