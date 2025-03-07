'use client';

import React, { useEffect, useCallback, useState } from 'react';
import AppLayout from '@/components/layout/AppLayout';
import { StockDataProvider } from '@/lib/StockDataContext';
import { useStockData } from '@/lib/StockDataContext';
import TimeFrameSelector from '@/components/ui/TimeFrameSelector';
import IndicatorSelector from '@/components/ui/IndicatorSelector';
import DataImporter from '@/components/ui/DataImporter';
import DataStats from '@/components/ui/DataStats';
import dynamic from 'next/dynamic';
import ClientOnly from '@/components/charts/ClientOnly';

// Dynamically import the chart component with SSR disabled
const EnhancedStockChart = dynamic(
  () => import('@/components/charts/EnhancedStockChart'),
  { ssr: false }
);

// This is the main page component for Next.js App Router
export default function Page() {
  return (
    <StockDataProvider>
      <AppLayout>
        <MainContent />
      </AppLayout>
    </StockDataProvider>
  );
}

const MainContent = () => {
  const { loadData, filteredData, isLoading, error } = useStockData();
  const [dataLoadAttempted, setDataLoadAttempted] = useState(false);
  const [dataSource, setDataSource] = useState<string | null>(null);

  // Load sample data on first render
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        // First try to load the smaller sample file
        console.log('Attempting to load sample data...');
        try {
          await loadData('/data/sample.csv');
          console.log('Sample data loaded successfully');
          setDataSource('sample.csv');
          setDataLoadAttempted(true);
        } catch (sampleError) {
          console.error('Failed to load sample data, trying main file:', sampleError);
          // If sample fails, try the main file
          await loadData('/data/SPY_featured.csv');
          console.log('Main data loaded successfully');
          setDataSource('SPY_featured.csv');
          setDataLoadAttempted(true);
        }
      } catch (error) {
        console.error('Failed to load any data files:', error);
        setDataLoadAttempted(true);
      }
    };
    
    loadInitialData();
  }, []); // Empty dependency array since we only want to load once

  return (
    <div className="space-y-6">
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
        <h2 className="text-xl font-bold mb-4 dark:text-gray-100">StockCaster Clarifier</h2>
        <p className="text-gray-600 dark:text-gray-300">
          Advanced Stock Data Visualization Platform
        </p>
        {dataSource && (
          <p className="text-sm text-gray-500 mt-1">
            Data source: {dataSource}
          </p>
        )}
        {error && (
          <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            <p className="font-medium">Error loading data:</p>
            <p>{error}</p>
          </div>
        )}
        {isLoading && (
          <div className="mt-4 p-3 bg-blue-100 border border-blue-400 text-blue-700 rounded">
            <p>Loading data...</p>
          </div>
        )}
        {dataLoadAttempted && !isLoading && filteredData.length === 0 && !error && (
          <div className="mt-4 p-3 bg-yellow-100 border border-yellow-400 text-yellow-700 rounded">
            <p>No data found. Please try importing a CSV file using the Data Importer below.</p>
          </div>
        )}
      </section>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-4 space-y-6">
          <ClientOnly
            fallback={
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700 h-[500px] flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                  <p className="mt-4 text-gray-600 dark:text-gray-300">Loading chart...</p>
                </div>
              </div>
            }
          >
            <EnhancedStockChart />
          </ClientOnly>
          <DataStats />
        </div>
        
        <div className="lg:col-span-4 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <DataImporter />
            <IndicatorSelector />
          </div>
        </div>
      </div>

      {filteredData.length > 0 && (
        <section className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
          <h2 className="text-xl font-bold mb-4 dark:text-gray-100">Data Preview</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  {Object.keys(filteredData[0]).slice(0, 8).map((key) => (
                    <th
                      key={key}
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                    >
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {filteredData.slice(0, 10).map((row, i) => (
                  <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    {Object.keys(row).slice(0, 8).map((key) => (
                      <td
                        key={`${i}-${key}`}
                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300"
                      >
                        {typeof row[key] === 'number'
                          ? row[key].toFixed(2)
                          : row[key]?.toString()}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            Showing 10 of {filteredData.length} records
          </p>
        </section>
      )}
    </div>
  );
}; 