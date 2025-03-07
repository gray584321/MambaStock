import React from 'react';
import { useStockData } from '@/lib/StockDataContext';
import { formatValue, formatPercentage } from '@/lib/dataUtils';

const DataStats: React.FC = () => {
  const { filteredData } = useStockData();
  
  if (filteredData.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
        <h3 className="font-bold text-lg mb-3 dark:text-white">Statistics</h3>
        <p className="text-gray-500 dark:text-gray-400 text-center py-4">No data available</p>
      </div>
    );
  }
  
  // Calculate basic statistics
  const firstPoint = filteredData[0];
  const lastPoint = filteredData[filteredData.length - 1];
  
  const startPrice = firstPoint.close;
  const endPrice = lastPoint.close;
  const priceChange = endPrice - startPrice;
  const percentChange = (endPrice / startPrice) - 1;
  
  // Use reduce instead of Math.max/min with spread to avoid stack overflow
  const { high, low } = filteredData.reduce((acc, d) => ({
    high: Math.max(acc.high, d.high),
    low: Math.min(acc.low, d.low)
  }), { high: -Infinity, low: Infinity });
  
  const startDate = new Date(firstPoint.datetime);
  const endDate = new Date(lastPoint.datetime);
  
  // Calculate additional statistics for improved UI
  const volume = filteredData.reduce((sum, d) => sum + (d.volume || 0), 0);
  const avgVolume = volume / filteredData.length;
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700 transition-all hover:shadow-lg">
      <h3 className="font-bold text-lg mb-4 dark:text-white">Market Statistics</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
        <div className="transition-all hover:bg-gray-50 dark:hover:bg-gray-750 p-2 rounded-md">
          <p className="text-sm text-gray-500 dark:text-gray-400">Period</p>
          <p className="font-semibold dark:text-gray-200">
            {startDate.toLocaleDateString()} - {endDate.toLocaleDateString()}
          </p>
        </div>
        
        <div className="transition-all hover:bg-gray-50 dark:hover:bg-gray-750 p-2 rounded-md">
          <p className="text-sm text-gray-500 dark:text-gray-400">Data Points</p>
          <p className="font-semibold dark:text-gray-200">{filteredData.length.toLocaleString()}</p>
        </div>
        
        <div className="transition-all hover:bg-gray-50 dark:hover:bg-gray-750 p-2 rounded-md">
          <p className="text-sm text-gray-500 dark:text-gray-400">Price Range</p>
          <p className="font-semibold dark:text-gray-200">
            {formatValue(low)} - {formatValue(high)}
          </p>
        </div>
        
        <div className="transition-all hover:bg-gray-50 dark:hover:bg-gray-750 p-2 rounded-md">
          <p className="text-sm text-gray-500 dark:text-gray-400">Open</p>
          <p className="font-semibold dark:text-gray-200">{formatValue(startPrice)}</p>
        </div>
        
        <div className="transition-all hover:bg-gray-50 dark:hover:bg-gray-750 p-2 rounded-md">
          <p className="text-sm text-gray-500 dark:text-gray-400">Close</p>
          <p className="font-semibold dark:text-gray-200">{formatValue(endPrice)}</p>
        </div>
        
        <div className="transition-all hover:bg-gray-50 dark:hover:bg-gray-750 p-2 rounded-md">
          <p className="text-sm text-gray-500 dark:text-gray-400">Change</p>
          <p className={`font-semibold ${priceChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
            {priceChange >= 0 ? '+' : ''}{formatValue(priceChange)} ({percentChange >= 0 ? '+' : ''}{(percentChange * 100).toFixed(2)}%)
          </p>
        </div>
        
        {volume > 0 && (
          <div className="transition-all hover:bg-gray-50 dark:hover:bg-gray-750 p-2 rounded-md">
            <p className="text-sm text-gray-500 dark:text-gray-400">Avg. Volume</p>
            <p className="font-semibold dark:text-gray-200">{Math.round(avgVolume).toLocaleString()}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataStats; 