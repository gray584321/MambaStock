import React from 'react';
import { useStockData } from '@/lib/StockDataContext';
import { formatValue, formatPercentage } from '@/lib/dataUtils';

const DataStats: React.FC = () => {
  const { filteredData } = useStockData();
  
  if (filteredData.length === 0) {
    return null;
  }
  
  // Calculate basic statistics
  const firstPoint = filteredData[0];
  const lastPoint = filteredData[filteredData.length - 1];
  
  const startPrice = firstPoint.close;
  const endPrice = lastPoint.close;
  const priceChange = endPrice - startPrice;
  const percentChange = (endPrice / startPrice) - 1;
  
  const high = Math.max(...filteredData.map(d => d.high));
  const low = Math.min(...filteredData.map(d => d.low));
  
  const startDate = new Date(firstPoint.datetime);
  const endDate = new Date(lastPoint.datetime);
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
      <h3 className="font-bold text-lg mb-3 dark:text-white">Statistics</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Period</p>
          <p className="font-semibold dark:text-gray-200">
            {startDate.toLocaleDateString()} - {endDate.toLocaleDateString()}
          </p>
        </div>
        
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Data Points</p>
          <p className="font-semibold dark:text-gray-200">{filteredData.length}</p>
        </div>
        
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Price Range</p>
          <p className="font-semibold dark:text-gray-200">
            {formatValue(low)} - {formatValue(high)}
          </p>
        </div>
        
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Open</p>
          <p className="font-semibold dark:text-gray-200">{formatValue(startPrice)}</p>
        </div>
        
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Close</p>
          <p className="font-semibold dark:text-gray-200">{formatValue(endPrice)}</p>
        </div>
        
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">Change</p>
          <p className={`font-semibold ${priceChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
            {priceChange >= 0 ? '+' : ''}{formatValue(priceChange)} ({percentChange >= 0 ? '+' : ''}{(percentChange * 100).toFixed(2)}%)
          </p>
        </div>
      </div>
    </div>
  );
};

export default DataStats; 