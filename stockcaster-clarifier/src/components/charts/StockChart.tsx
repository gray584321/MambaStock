import React, { useRef, useState, useEffect, useMemo, useCallback, memo } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  Bar,
  ReferenceLine,
} from 'recharts';
import { formatDate, formatValue } from '@/lib/dataUtils';
import { useStockData } from '@/lib/StockDataContext';

// Simple counter for generating unique keys
let uniqueKeyCounter = 1;
const getUniqueKey = (prefix: string): string => `${prefix}-${uniqueKeyCounter++}`;

// Memoized Tooltip component to prevent unnecessary re-renders
const CustomTooltip = memo(({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null;
  
  // Group indicators by type
  const priceItems: any[] = [];
  const indicatorItems: any[] = [];
  const volumeItems: any[] = [];
  
  payload.forEach(entry => {
    if (!entry || entry.value === undefined || entry.value === null) return;
    
    if (entry.dataKey === 'volume') {
      volumeItems.push(entry);
    } else if (entry.dataKey === 'close' || entry.dataKey === 'open' || 
              entry.dataKey === 'high' || entry.dataKey === 'low') {
      priceItems.push(entry);
    } else {
      indicatorItems.push(entry);
    }
  });
  
  return (
    <div className="bg-white p-4 border border-gray-200 shadow-lg rounded-md max-w-xs">
      <div className="font-medium text-gray-700 border-b pb-1 mb-2">
        {formatDate(label)}
      </div>
      
      {priceItems.length > 0 && (
        <div className="mb-2">
          <div className="text-xs font-semibold text-gray-500 mb-1">Price</div>
          {priceItems.map((entry, index) => {
            const value = typeof entry.value === 'number' ? 
              formatValue(entry.value) : 'N/A';
            
            return (
              <div 
                key={`price-${index}`}
                className="flex items-center justify-between gap-4"
              >
                <span className="text-xs font-medium" style={{ color: entry.color }}>
                  {entry.name}:
                </span>
                <span className="text-xs font-mono">{value}</span>
              </div>
            );
          })}
        </div>
      )}
      
      {indicatorItems.length > 0 && (
        <div className="mb-2">
          <div className="text-xs font-semibold text-gray-500 mb-1">Indicators</div>
          {indicatorItems.map((entry, index) => {
            const value = typeof entry.value === 'number' ? 
              formatValue(entry.value) : 'N/A';
            
            return (
              <div 
                key={`indicator-${index}`}
                className="flex items-center justify-between gap-4"
              >
                <span className="text-xs font-medium" style={{ color: entry.color }}>
                  {entry.name}:
                </span>
                <span className="text-xs font-mono">{value}</span>
              </div>
            );
          })}
        </div>
      )}
      
      {volumeItems.length > 0 && (
        <div className="mb-1">
          <div className="text-xs font-semibold text-gray-500 mb-1">Volume</div>
          {volumeItems.map((entry, index) => {
            const value = typeof entry.value === 'number' ? 
              Intl.NumberFormat('en-US', { 
                notation: 'compact',
                compactDisplay: 'short'
              }).format(entry.value) : 'N/A';
            
            return (
              <div 
                key={`volume-${index}`}
                className="flex items-center justify-between gap-4"
              >
                <span className="text-xs font-medium" style={{ color: entry.color }}>
                  {entry.name}:
                </span>
                <span className="text-xs font-mono">{value}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
});

CustomTooltip.displayName = 'CustomTooltip';

// Memoized time range tabs component
const TimeRangeTabs = memo(({ selectedRange, onRangeChange }: { 
  selectedRange: string, 
  onRangeChange: (range: string) => void 
}) => (
  <div className="flex space-x-1 border-b border-gray-200 mb-4">
    {[
      { id: '1d', label: '1D' },
      { id: '1w', label: '1W' },
      { id: '1m', label: '1M' },
      { id: '3m', label: '3M' },
      { id: '6m', label: '6M' },
      { id: '1y', label: '1Y' },
      { id: 'all', label: 'ALL' }
    ].map(range => (
      <button
        key={range.id}
        className={`
          px-4 py-2 text-sm font-medium 
          ${selectedRange === range.id 
            ? 'text-blue-600 border-b-2 border-blue-600' 
            : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'}
        `}
        onClick={() => onRangeChange(range.id)}
      >
        {range.label}
      </button>
    ))}
  </div>
));

TimeRangeTabs.displayName = 'TimeRangeTabs';

// Main component
const StockChart: React.FC = () => {
  const { 
    filteredData, 
    chartConfigs, 
    visibleIndicators, 
    selectedTimeframe, 
    toggleIndicator,
    setSelectedTimeframe,
    chartRange,
    setChartRange,
    isLoading
  } = useStockData();
  
  const [processedData, setProcessedData] = useState<any[]>([]);
  const [chartHeight, setChartHeight] = useState(500);
  const [chartType, setChartType] = useState<'line' | 'candle'>('line');
  const [showVolume, setShowVolume] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [uniqueId] = useState(() => Date.now().toString());
  const [availableIndicators, setAvailableIndicators] = useState<string[]>([]);
  const [selectedRange, setSelectedRange] = useState<'all' | '1y' | '6m' | '3m' | '1m' | '1w' | '1d'>('3m');
  const [dataSubset, setDataSubset] = useState<any[]>([]);
  const [subsamplingRate, setSubsamplingRate] = useState(1);
  const chartRef = useRef<HTMLDivElement>(null);
  const brushRef = useRef<any>(null);

  // Performance optimization: Calculate optimal sampling rate based on screen width
  useEffect(() => {
    const calculateSamplingRate = () => {
      const chartWidth = chartRef.current?.clientWidth || 1000;
      // Aim for 1-2 data points per pixel for optimal performance
      const optimalDataPoints = Math.floor(chartWidth * 1.5);
      const rate = Math.max(1, Math.ceil(filteredData.length / optimalDataPoints));
      setSubsamplingRate(rate);
    };

    calculateSamplingRate();
    
    // Add resize listener to recalculate when window size changes
    const handleResize = () => calculateSamplingRate();
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [filteredData.length]);

  // Handle time range selection
  const handleRangeChange = useCallback((range: string) => {
    setSelectedRange(range as any);
    
    // Find matching timeframe in the context
    const timeframes = [
      { id: '1d', label: '1 Day', days: 1 },
      { id: '1w', label: '1 Week', days: 7 },
      { id: '1m', label: '1 Month', days: 30 },
      { id: '3m', label: '3 Months', days: 90 },
      { id: '6m', label: '6 Months', days: 180 },
      { id: '1y', label: '1 Year', days: 365 },
      { id: 'all', label: 'All', days: 9999 },
    ];
    
    const timeframe = timeframes.find(t => t.id === range);
    if (timeframe) {
      setSelectedTimeframe(timeframe);
    }
  }, [setSelectedTimeframe]);
  
  // Process and validate data for chart rendering
  useEffect(() => {
    if (filteredData.length === 0) {
      setProcessedData([]);
      return;
    }

    try {
      // Performance optimization: Process data in a web worker if available
      const processData = () => {
        // Process the data for visualization
        const processed = filteredData.map((item, index) => {
          if (!item) return null;

          const dateTime = item.datetime ? new Date(item.datetime) : null;
          if (!dateTime) return null;
          
          // Create base object with valid values
          const baseItem: any = {
            id: `item-${index}`,
            datetime: dateTime,
            timestamp: dateTime.getTime(),
            formattedDate: dateTime ? formatDate(item.datetime) : '',
            open: typeof item.open === 'number' && !isNaN(item.open) ? Number(item.open) : null,
            high: typeof item.high === 'number' && !isNaN(item.high) ? Number(item.high) : null,
            low: typeof item.low === 'number' && !isNaN(item.low) ? Number(item.low) : null,
            close: typeof item.close === 'number' && !isNaN(item.close) ? Number(item.close) : null,
            volume: typeof item.volume === 'number' && !isNaN(item.volume) ? Number(item.volume) : 0,
          };

          // Collect all available indicators in the dataset
          const indicators: string[] = [];
          
          // Copy all indicator fields
          Object.entries(item).forEach(([key, value]) => {
            if (
              !['id', 'datetime', 'timestamp', 'formattedDate', 'open', 'high', 'low', 'close', 'volume'].includes(key) &&
              typeof value === 'number' &&
              !isNaN(value)
            ) {
              baseItem[key] = Number(value);
              indicators.push(key);
            }
          });
          
          // Record available indicators
          if (index === 0) {
            setAvailableIndicators([...new Set(indicators)]);
          }

          return baseItem;
        }).filter(Boolean); // Remove any null items
        
        // Sort data by datetime for consistent display
        processed.sort((a, b) => a.timestamp - b.timestamp);
        
        setProcessedData(processed);
      };

      // Use requestIdleCallback if available for better performance
      if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
        // @ts-ignore
        window.requestIdleCallback(() => processData());
      } else {
        processData();
      }
    } catch (error) {
      console.error('Error processing chart data:', error);
      setProcessedData([]);
    }
  }, [filteredData]);

  // Performance optimization: Subsample data for chart rendering
  useEffect(() => {
    if (processedData.length === 0) {
      setDataSubset([]);
      return;
    }

    // Subsample data for better performance
    const subset = processedData.filter((_, i) => i % subsamplingRate === 0);
    setDataSubset(subset);
  }, [processedData, subsamplingRate]);
  
  // Memoize price chart elements to prevent unnecessary rerenders
  const priceChartElements = useMemo(() => {
    if (dataSubset.length === 0) return null;
    
    // Base price line - always show this
    return (
      <Line
        key={`price-line-${uniqueId}`}
        type="monotone"
        dataKey="close"
        stroke="#2563eb"
        dot={false}
        strokeWidth={2}
        name="Price"
        yAxisId="left"
        isAnimationActive={false}
        connectNulls={true}
      />
    );
  }, [dataSubset, uniqueId]);
  
  // Memoize indicator elements - filter by available indicators
  const indicatorElements = useMemo(() => {
    if (dataSubset.length === 0) return [];
    
    // Performance optimization: Limit the number of active indicators
    const visibleConfigs = chartConfigs.filter(config => 
      config.visible && 
      config.id !== 'candlestick' && 
      config.id !== 'volume' &&
      availableIndicators.includes(config.id)
    ).slice(0, 5); // Limit to 5 active indicators for performance
    
    return visibleConfigs.map((config, index) => {
      // For line indicators like moving averages
      if (config.type === 'area') {
        return (
          <Area 
            key={`indicator-${config.id}-${index}-${uniqueId}`}
            type="monotone"
            dataKey={config.id}
            name={config.label || config.id}
            stroke={config.color || '#000'}
            fill={config.color || '#000'}
            fillOpacity={0.1}
            strokeWidth={1.5}
            dot={false}
            yAxisId="left"
            isAnimationActive={false}
            connectNulls={true}
          />
        );
      }
      
      // Default to line indicators
      return (
        <Line 
          key={`indicator-${config.id}-${index}-${uniqueId}`}
          type="monotone"
          dataKey={config.id}
          name={config.label || config.id}
          stroke={config.color || '#000'}
          strokeWidth={1.5}
          dot={false}
          yAxisId="left"
          isAnimationActive={false}
          connectNulls={true}
        />
      );
    });
  }, [chartConfigs, dataSubset, availableIndicators, uniqueId]);

  // Memoize volume element - with a stable unique key
  const volumeElement = useMemo(() => {
    if (dataSubset.length === 0 || !showVolume) return null;
    
    return (
      <Bar
        key={`volume-bar-${uniqueId}`}
        dataKey="volume"
        name="Volume"
        yAxisId="right"
        fill="#64748b"
        opacity={0.5}
        isAnimationActive={false}
      />
    );
  }, [dataSubset, showVolume, uniqueId]);

  // Generate reference lines for important price levels
  const referenceLines = useMemo(() => {
    if (dataSubset.length === 0) return null;
    
    // Calculate some common reference values
    const priceData = dataSubset.map(d => d.close).filter(Boolean);
    if (priceData.length === 0) return null;
    
    const latestPrice = priceData[priceData.length - 1];
    
    return (
      <ReferenceLine
        y={latestPrice}
        label={{ 
          value: formatValue(latestPrice), 
          position: 'right',
          fill: '#6b7280',
          fontSize: 12
        }}
        stroke="#9CA3AF"
        strokeDasharray="3 3"
        yAxisId="left"
      />
    );
  }, [dataSubset]);

  // Memoized indicator toggle handler
  const handleToggleIndicator = useCallback((id: string) => {
    toggleIndicator(id);
  }, [toggleIndicator]);

  // Memoized chart toolbar for better performance
  const ChartToolbar = useCallback(() => {
    // Filter indicators to only show ones that are available in the data
    const availableChartConfigs = chartConfigs.filter(
      config => availableIndicators.includes(config.id) || 
                config.id === 'candlestick' || 
                config.id === 'volume'
    );
    
    // Group indicators by type
    const movingAverages = availableChartConfigs.filter(c => 
      c.id.startsWith('sma_') || c.id.startsWith('ema_')
    );
    
    const bands = availableChartConfigs.filter(c => 
      c.id.includes('bollinger') || c.id.includes('keltner') || c.id.includes('band')
    );
    
    const oscillators = availableChartConfigs.filter(c => 
      c.id.includes('rsi') || c.id.includes('macd') || 
      c.id.includes('stoch') || c.id.includes('cci')
    );
    
    const others = availableChartConfigs.filter(c => 
      !movingAverages.includes(c) && 
      !bands.includes(c) && 
      !oscillators.includes(c) && 
      c.id !== 'candlestick' && 
      c.id !== 'volume'
    );
    
    return (
      <div className="mb-4">
        <TimeRangeTabs 
          selectedRange={selectedRange} 
          onRangeChange={handleRangeChange}
        />
        <div className="flex flex-wrap justify-between">
          <div className="flex items-center gap-3 mb-2">
            <button 
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${chartType === 'line' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
              onClick={() => setChartType('line')}
            >
              Line
            </button>
            <button 
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${chartType === 'candle' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
              onClick={() => setChartType('candle')}
            >
              Candle
            </button>
            
            <div className="flex items-center gap-2 ml-2">
              <label className="inline-flex items-center cursor-pointer">
                <input 
                  type="checkbox" 
                  checked={showVolume} 
                  onChange={e => setShowVolume(e.target.checked)} 
                  className="sr-only peer"
                />
                <div className="relative w-9 h-5 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                <span className="ms-2 text-sm font-medium text-gray-700">Volume</span>
              </label>
              
              <label className="inline-flex items-center cursor-pointer ml-4">
                <input 
                  type="checkbox" 
                  checked={showGrid} 
                  onChange={e => setShowGrid(e.target.checked)} 
                  className="sr-only peer"
                />
                <div className="relative w-9 h-5 bg-gray-200 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
                <span className="ms-2 text-sm font-medium text-gray-700">Grid</span>
              </label>
            </div>
          </div>
          
          <div className="space-y-2 w-full mt-2">
            {movingAverages.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-1">
                <span className="text-xs font-medium text-gray-500 mr-2 py-1">MAs:</span>
                {movingAverages.map(config => (
                  <button
                    key={config.id}
                    className={`
                      px-2 py-1 rounded text-xs font-medium border transition-colors
                      ${config.visible 
                        ? 'bg-blue-50 border-blue-300 text-blue-700' 
                        : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'}
                    `}
                    onClick={() => handleToggleIndicator(config.id)}
                    style={{ borderLeftColor: config.color, borderLeftWidth: '3px' }}
                  >
                    {config.label}
                  </button>
                ))}
              </div>
            )}
            
            {bands.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-1">
                <span className="text-xs font-medium text-gray-500 mr-2 py-1">Bands:</span>
                {bands.map(config => (
                  <button
                    key={config.id}
                    className={`
                      px-2 py-1 rounded text-xs font-medium border transition-colors
                      ${config.visible 
                        ? 'bg-blue-50 border-blue-300 text-blue-700' 
                        : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'}
                    `}
                    onClick={() => handleToggleIndicator(config.id)}
                    style={{ borderLeftColor: config.color, borderLeftWidth: '3px' }}
                  >
                    {config.label}
                  </button>
                ))}
              </div>
            )}
            
            {oscillators.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-1">
                <span className="text-xs font-medium text-gray-500 mr-2 py-1">Oscillators:</span>
                {oscillators.map(config => (
                  <button
                    key={config.id}
                    className={`
                      px-2 py-1 rounded text-xs font-medium border transition-colors
                      ${config.visible 
                        ? 'bg-blue-50 border-blue-300 text-blue-700' 
                        : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'}
                    `}
                    onClick={() => handleToggleIndicator(config.id)}
                    style={{ borderLeftColor: config.color, borderLeftWidth: '3px' }}
                  >
                    {config.label}
                  </button>
                ))}
              </div>
            )}
            
            {others.length > 0 && (
              <div className="flex flex-wrap gap-1">
                <span className="text-xs font-medium text-gray-500 mr-2 py-1">Others:</span>
                {others.map(config => (
                  <button
                    key={config.id}
                    className={`
                      px-2 py-1 rounded text-xs font-medium border transition-colors
                      ${config.visible 
                        ? 'bg-blue-50 border-blue-300 text-blue-700' 
                        : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'}
                    `}
                    onClick={() => handleToggleIndicator(config.id)}
                    style={{ borderLeftColor: config.color, borderLeftWidth: '3px' }}
                  >
                    {config.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        
        {/* Performance stats */}
        {processedData.length > 0 && (
          <div className="text-xs text-gray-400 mt-1">
            Data: {processedData.length} points | Displayed: {dataSubset.length} points | Sample rate: {subsamplingRate}x
          </div>
        )}
      </div>
    );
  }, [chartConfigs, availableIndicators, selectedRange, chartType, showVolume, showGrid, handleRangeChange, handleToggleIndicator, processedData.length, dataSubset.length, subsamplingRate]);

  // Loading state
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-2/3 mb-4"></div>
          <div className="h-[500px] bg-gray-100 rounded-lg flex items-center justify-center">
            <div className="flex items-center justify-center">
              <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="ml-2 text-gray-600">Loading chart data...</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // No data state
  if (processedData.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
        <ChartToolbar />
        <div className="flex items-center justify-center h-96 bg-gray-100 dark:bg-gray-700 rounded-lg">
          <div className="text-center">
            <p className="text-lg text-gray-600 dark:text-gray-300 mb-2">No data available</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">Please load stock data to visualize</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
      <ChartToolbar />
      <div ref={chartRef} className="w-full h-[500px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={dataSubset}
            margin={{ top: 10, right: 55, left: 10, bottom: 30 }}
          >
            {showGrid && <CartesianGrid strokeDasharray="3 3" opacity={0.3} />}
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => {
                if (!value) return '';
                try {
                  const date = new Date(value);
                  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                } catch (e) {
                  return '';
                }
              }}
              type="number"
              domain={['auto', 'auto']}
              padding={{ left: 5, right: 5 }}
              scale="time"
              tick={{ fontSize: 12, fill: '#6b7280' }}
              interval="preserveStartEnd"
            />
            <YAxis
              yAxisId="left"
              orientation="right"
              domain={['auto', 'auto']}
              tickFormatter={(value) => formatValue(value)}
              padding={{ top: 10, bottom: 10 }}
              tick={{ fontSize: 12, fill: '#6b7280' }}
              width={55}
              allowDecimals={false}
            />
            <YAxis
              yAxisId="right"
              orientation="left"
              domain={[0, 'auto']}
              tickCount={5}
              hide={!showVolume}
              tick={{ fontSize: 11, fill: '#6b7280' }}
            />
            <Tooltip 
              content={<CustomTooltip />} 
              isAnimationActive={false}
            />
            <Legend 
              verticalAlign="top" 
              formatter={(value) => <span className="text-xs font-medium">{value}</span>}
              iconSize={10}
              wrapperStyle={{ paddingBottom: 10 }}
            />
            <Brush 
              ref={brushRef}
              dataKey="timestamp" 
              height={25} 
              stroke="#2563eb"
              fill="#f1f5f9"
              tickFormatter={(value) => {
                if (!value) return '';
                try {
                  const date = new Date(value);
                  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                } catch (e) {
                  return '';
                }
              }}
              travellerWidth={7}
              startIndex={Math.floor(dataSubset.length * 0.7)}
              endIndex={dataSubset.length - 1}
            />
            
            {/* Price line */}
            {priceChartElements}
            
            {/* Technical indicators */}
            {indicatorElements}
            
            {/* Volume */}
            {volumeElement}
            
            {/* Reference lines */}
            {referenceLines}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 flex justify-between text-xs text-gray-500">
        <p>
          Period: {selectedTimeframe?.label || 'All'} | 
          Data Points: {processedData.length}
        </p>
        <p>
          Last: {processedData.length > 0 ? formatValue(processedData[processedData.length - 1].close) : 'N/A'} | 
          Updated: {processedData.length > 0 ? processedData[processedData.length - 1].formattedDate : 'N/A'}
        </p>
      </div>
    </div>
  );
};

export default StockChart; 