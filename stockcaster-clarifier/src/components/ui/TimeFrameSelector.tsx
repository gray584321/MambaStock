import React from 'react';
import { useStockData } from '@/lib/StockDataContext';
import { timeframes } from '@/lib/StockDataContext';

const TimeFrameSelector: React.FC = () => {
  const { selectedTimeframe, setSelectedTimeframe } = useStockData();

  return (
    <div className="flex flex-wrap gap-2">
      {timeframes.map((timeframe) => (
        <button
          key={timeframe.id}
          onClick={() => setSelectedTimeframe(timeframe)}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
            selectedTimeframe.id === timeframe.id
              ? 'bg-blue-600 text-white dark:bg-blue-700'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
          }`}
        >
          {timeframe.label}
        </button>
      ))}
    </div>
  );
};

export default TimeFrameSelector; 