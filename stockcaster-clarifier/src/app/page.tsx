'use client';

import React, { useEffect } from 'react';
import { StockDataProvider } from '@/lib/StockDataContext';
import AppLayout from '@/components/layout/AppLayout';
import StockChart from '@/components/charts/StockChart';
import IndicatorSelector from '@/components/ui/IndicatorSelector';
import TimeFrameSelector from '@/components/ui/TimeFrameSelector';
import DataImporter from '@/components/ui/DataImporter';
import DataStats from '@/components/ui/DataStats';
import { useStockData } from '@/lib/StockDataContext';

const App = () => {
  return (
    <StockDataProvider>
      <AppLayout>
        <MainContent />
      </AppLayout>
    </StockDataProvider>
  );
};

const MainContent = () => {
  const { loadData, filteredData } = useStockData();

  // Load sample data on first render
  useEffect(() => {
    loadData('/data/SPY_featured.csv');
  }, []);

  return (
    <div className="space-y-6">
      <section className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-xl font-bold mb-4">Chart Controls</h2>
        <div className="space-y-4">
          <TimeFrameSelector />
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3 space-y-6">
          <StockChart />
          <DataStats />
        </div>
        
        <div className="space-y-6">
          <DataImporter />
          <IndicatorSelector />
        </div>
      </div>

      {filteredData.length > 0 && (
        <section className="bg-white rounded-lg shadow-md p-4">
          <h2 className="text-xl font-bold mb-4">Data Preview</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {Object.keys(filteredData[0]).slice(0, 8).map((key) => (
                    <th
                      key={key}
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredData.slice(0, 10).map((row, i) => (
                  <tr key={i}>
                    {Object.keys(row).slice(0, 8).map((key) => (
                      <td
                        key={`${i}-${key}`}
                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
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
          <p className="mt-2 text-sm text-gray-500">
            Showing 10 of {filteredData.length} records
          </p>
        </section>
      )}
    </div>
  );
};

export default App; 