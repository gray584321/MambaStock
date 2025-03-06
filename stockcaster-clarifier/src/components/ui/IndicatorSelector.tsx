import React from 'react';
import { useStockData } from '@/lib/StockDataContext';

const IndicatorSelector: React.FC = () => {
  const { chartConfigs, toggleIndicator } = useStockData();

  // Group indicators by type
  const groupedIndicators = chartConfigs.reduce((acc, indicator) => {
    const type = indicator.id.split('_')[0] || 'other';
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(indicator);
    return acc;
  }, {} as Record<string, typeof chartConfigs>);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
      <h3 className="font-bold text-lg mb-3 dark:text-white">Indicators</h3>
      <div className="space-y-4">
        {Object.entries(groupedIndicators).map(([group, indicators]) => (
          <div key={group} className="space-y-2">
            <h4 className="font-semibold text-gray-700 dark:text-gray-300 capitalize">{group}</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {indicators.map((indicator) => (
                <div key={indicator.id} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id={indicator.id}
                    checked={indicator.visible}
                    onChange={() => toggleIndicator(indicator.id)}
                    className="h-4 w-4 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500 dark:focus:ring-blue-400 dark:bg-gray-700"
                  />
                  <label
                    htmlFor={indicator.id}
                    className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center"
                  >
                    <span
                      className="inline-block w-3 h-3 mr-1 rounded-full"
                      style={{ backgroundColor: indicator.color }}
                    ></span>
                    {indicator.label}
                  </label>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default IndicatorSelector; 