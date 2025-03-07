"use client";

import React, { useState, useEffect, useRef, useCallback, memo, useLayoutEffect, useMemo } from 'react';
import { createChart, IChartApi, CandlestickSeries, LineSeries, AreaSeries, HistogramSeries } from 'lightweight-charts';
import { useStockData } from '@/lib/StockDataContext';
import { formatDate, formatValue } from '@/lib/dataUtils';

// Time Range Tabs Component with improved responsive design
const TimeRangeTabs = memo(({ selectedRange, onRangeChange }: { 
  selectedRange: string, 
  onRangeChange: (range: string) => void 
}) => {
  const timeRanges = [
    { id: '1d', label: '1D', description: 'Last 24 hours' },
    { id: '1w', label: '1W', description: 'Last 7 days' },
    { id: '1m', label: '1M', description: 'Last 30 days' },
    { id: '3m', label: '3M', description: 'Last 90 days' },
    { id: '6m', label: '6M', description: 'Last 180 days' },
    { id: '1y', label: '1Y', description: 'Last 365 days' },
    { id: 'all', label: 'ALL', description: 'Full range' }
  ];

  return (
    <div className="w-full">
      <div className="flex space-x-1 overflow-x-auto scrollbar-hide py-1 w-full">
        {timeRanges.map(range => (
          <button
            key={range.id}
            className={`
              px-3 py-1.5 text-sm font-medium rounded transition-colors min-w-[3rem] relative
              ${selectedRange === range.id 
                ? 'bg-blue-600 text-white font-bold shadow-sm' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'}
            `}
            onClick={() => onRangeChange(range.id)}
            aria-pressed={selectedRange === range.id}
            aria-label={`View ${range.description}`}
            title={range.description}
          >
            {range.label}
            {selectedRange === range.id && (
              <span className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-white dark:bg-blue-300"></span>
            )}
          </button>
        ))}
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 pl-1 hidden sm:block">
        {timeRanges.find(r => r.id === selectedRange)?.description}
      </div>
    </div>
  );
});

TimeRangeTabs.displayName = 'TimeRangeTabs';

// Custom Legend Component
const ChartLegend = memo(({ data, indicatorSettings, onToggleIndicator }: {
  data: any,
  indicatorSettings: any[],
  onToggleIndicator: (id: string) => void
}) => {
  if (!data) return null;
  
  return (
    <div className="flex flex-wrap gap-3 p-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md">
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 bg-blue-500 rounded-sm"></div>
        <span className="text-sm font-semibold dark:text-white">Price: {formatValue(data.value || data.close)}</span>
      </div>
      
      {indicatorSettings
        .filter(indicator => indicator.visible && data[indicator.id] !== undefined)
        .map(indicator => (
          <div 
            key={indicator.id} 
            className="flex items-center gap-2 cursor-pointer transition-all hover:bg-gray-100 dark:hover:bg-gray-700 px-2 py-1 rounded-sm" 
            onClick={() => onToggleIndicator(indicator.id)}
          >
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: indicator.color }}></div>
            <span className="text-sm font-semibold dark:text-gray-200">{indicator.label}: {formatValue(data[indicator.id])}</span>
          </div>
        ))}
    </div>
  );
});

ChartLegend.displayName = 'ChartLegend';

// Chart Type Selector Component
const ChartTypeSelector = memo(({ chartType, onChartTypeChange }: {
  chartType: 'line' | 'area' | 'candle',
  onChartTypeChange: (type: 'line' | 'area' | 'candle') => void
}) => (
  <div className="bg-gray-100 dark:bg-gray-800 p-1 rounded-md flex">
    <button 
      className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${chartType === 'line' ? 'bg-blue-600 text-white' : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'}`}
      onClick={() => onChartTypeChange('line')}
      aria-label="Display as line chart"
    >
      <span className="flex items-center gap-1">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        Line
      </span>
    </button>
    <button 
      className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${chartType === 'area' ? 'bg-blue-600 text-white' : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'}`}
      onClick={() => onChartTypeChange('area')}
      aria-label="Display as area chart"
    >
      <span className="flex items-center gap-1">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 12C22 12 19 18 12 18C5 18 2 12 2 12C2 12 5 6 12 6C19 6 22 12 22 12Z"></path>
        </svg>
        Area
      </span>
    </button>
    <button 
      className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${chartType === 'candle' ? 'bg-blue-600 text-white' : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'}`}
      onClick={() => onChartTypeChange('candle')}
      aria-label="Display as candlestick chart"
    >
      <span className="flex items-center gap-1">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 3v18M10 6h4v12h-4z"></path>
          <path d="M17 3v18M14 6h6M14 18h6"></path>
        </svg>
        Candle
      </span>
    </button>
  </div>
));

ChartTypeSelector.displayName = 'ChartTypeSelector';

// Indicators Panel Component
const IndicatorsPanel = memo(({ chartConfigs, toggleIndicator, availableIndicators }: {
  chartConfigs: any[],
  toggleIndicator: (id: string) => void,
  availableIndicators: string[]
}) => {
  const [activeTab, setActiveTab] = useState('moving_averages');
  
  const tabs = [
    { id: 'moving_averages', label: 'Moving Averages' },
    { id: 'oscillators', label: 'Oscillators' },
    { id: 'bands', label: 'Bands' },
    { id: 'volumes', label: 'Volume' },
    { id: 'others', label: 'Others' },
  ];
  
  // Filter indicators by type
  const getIndicatorsByType = (type: string) => {
    let filteredConfigs;
    
    switch (type) {
      case 'moving_averages':
        filteredConfigs = chartConfigs.filter(c => 
          (c.id.startsWith('sma_') || c.id.startsWith('ema_')) &&
          availableIndicators.includes(c.id)
        );
        break;
      case 'oscillators':
        filteredConfigs = chartConfigs.filter(c => 
          (c.id.includes('rsi') || c.id.includes('macd') || 
           c.id.includes('stoch') || c.id.includes('cci')) &&
          availableIndicators.includes(c.id)
        );
        break;
      case 'bands':
        filteredConfigs = chartConfigs.filter(c => 
          (c.id.includes('bollinger') || c.id.includes('keltner') || 
           c.id.includes('band')) &&
          availableIndicators.includes(c.id)
        );
        break;
      case 'volumes':
        filteredConfigs = chartConfigs.filter(c => 
          (c.id === 'volume' || c.id.includes('volume') || 
           c.id.includes('obv')) &&
          availableIndicators.includes(c.id)
        );
        break;
      default:
        filteredConfigs = chartConfigs.filter(c => 
          !c.id.startsWith('sma_') && 
          !c.id.startsWith('ema_') && 
          !c.id.includes('rsi') && 
          !c.id.includes('macd') && 
          !c.id.includes('stoch') && 
          !c.id.includes('cci') && 
          !c.id.includes('bollinger') && 
          !c.id.includes('keltner') && 
          !c.id.includes('band') && 
          c.id !== 'volume' && 
          !c.id.includes('volume') && 
          !c.id.includes('obv') &&
          availableIndicators.includes(c.id)
        );
    }
    
    return filteredConfigs;
  };
  
  return (
    <div id="indicator-panel" className="mb-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium dark:text-white">Technical Indicators</h3>
        <button
          className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          onClick={() => document.getElementById('indicator-panel')?.classList.add('hidden')}
          aria-label="Close indicator panel"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
      
      <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4 overflow-x-auto pb-1">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`px-4 py-2 text-sm font-medium whitespace-nowrap ${
              activeTab === tab.id 
                ? 'text-blue-600 border-b-2 border-blue-600 dark:text-blue-400' 
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {getIndicatorsByType(activeTab).map(config => (
          <button
            key={config.id}
            className={`
              flex items-center gap-2 px-3 py-2 rounded border transition-colors
              ${config.visible 
                ? 'bg-blue-50 border-blue-300 text-blue-700 dark:bg-blue-900/50 dark:border-blue-700 dark:text-blue-300' 
                : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-600'}
            `}
            onClick={() => toggleIndicator(config.id)}
            style={{ borderLeftColor: config.color, borderLeftWidth: '3px' }}
          >
            <span className="text-sm truncate">{config.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
});

IndicatorsPanel.displayName = 'IndicatorsPanel';

/**
 * Ultimate performance sampling that strictly limits points for immediate rendering
 */
const ultraSampleData = (data: any[], maxPoints: number = 100) => {
  if (!data || data.length === 0) return [];
  if (data.length <= maxPoints) return data;
  
  const result = [];
  // Always include first and last points
  result.push(data[0]);
  
  // For very large datasets, use more aggressive constant-time sampling
  if (data.length > 10000) {
    // Create evenly distributed sample points
    const step = Math.max(1, Math.floor(data.length / (maxPoints - 2)));
    for (let i = step; i < data.length - step; i += step) {
      result.push(data[i]);
    }
  } else {
    // For smaller datasets, use a uniform sampling approach
    const step = Math.max(1, Math.floor(data.length / (maxPoints - 2)));
    for (let i = step; i < data.length - step; i += step) {
      result.push(data[i]);
    }
  }
  
  // Add the last point
  result.push(data[data.length - 1]);
  
  return result;
};

/**
 * Simple Web Worker inline implementation for data processing
 */
const createDataProcessingWorker = () => {
  const workerCode = `
    self.onmessage = function(e) {
      const { data, operation, maxPoints } = e.data;
      
      if (operation === 'sample') {
        const result = sampleData(data, maxPoints);
        self.postMessage({ result });
      } else if (operation === 'format') {
        const result = formatTimeData(data);
        self.postMessage({ result });
      }
    };
    
    function sampleData(data, maxPoints) {
      if (!data || data.length === 0) return data;
      if (data.length <= maxPoints) return data;
      
      const result = [];
      // Always include first point
      result.push(data[0]);
      
      // Sample middle points
      const step = Math.max(1, Math.floor(data.length / (maxPoints - 2)));
      for (let i = step; i < data.length - step; i += step) {
        result.push(data[i]);
      }
      
      // Add the last point
      result.push(data[data.length - 1]);
      
      return result;
    }
    
    function formatTimeData(data) {
      return data.map(point => {
        const newPoint = {...point};
        
        if (typeof newPoint.time === 'number') {
          if (newPoint.time > 2000000000) {
            newPoint.time = Math.floor(newPoint.time / 1000);
          }
        } else if (typeof newPoint.time === 'string') {
          newPoint.time = Math.floor(new Date(newPoint.time).getTime() / 1000);
        }
        
        return newPoint;
      }).filter(point => point.time && !isNaN(point.time));
    }
  `;
  
  const blob = new Blob([workerCode], { type: 'application/javascript' });
  return new Worker(URL.createObjectURL(blob));
};

// Main Enhanced Stock Chart Component
const EnhancedStockChart: React.FC = () => {
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
  const [candleData, setCandleData] = useState<any[]>([]);
  const [volumeData, setVolumeData] = useState<any[]>([]);
  const [selectedRange, setSelectedRange] = useState<string>('3m'); // Default to 3 months
  const [lastDataPoint, setLastDataPoint] = useState<any>(null);
  const [chartType, setChartType] = useState<'line' | 'area' | 'candle'>('candle');
  const [showVolume, setShowVolume] = useState(true);
  const [availableIndicators, setAvailableIndicators] = useState<string[]>([]);
  const [sampledDataCount, setSampledDataCount] = useState<number>(0);
  const [showIndicators, setShowIndicators] = useState<boolean>(false);
  
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const activeSeries = useRef<any[]>([]);
  const [chartCreated, setChartCreated] = useState(false);
  const isInitialized = useRef(false);
  
  // Enhanced state for staged rendering
  const [renderStage, setRenderStage] = useState<'preview' | 'basic' | 'complete'>('preview');
  const workerRef = useRef<Worker | null>(null);
  const [isProcessingData, setIsProcessingData] = useState<boolean>(false);
  
  // Refs to store data at different processing stages
  const rawDataRef = useRef<any[]>([]);
  const previewDataRef = useRef<any[]>([]);
  
  // Initialize selected range based on selectedTimeframe
  useEffect(() => {
    if (selectedTimeframe?.id) {
      setSelectedRange(selectedTimeframe.id);
    }
  }, [selectedTimeframe?.id]);
  
  // Define setTimeRange function to connect range selection to timeframe context
  const setTimeRange = useCallback((range: string) => {
    // Map the range to a timeframe object
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
  
  // Handle time range selection
  const handleRangeChange = useCallback((range: string) => {
    console.log("Changing time range to:", range);
    
    // Update the state
    setSelectedRange(range);
    
    // This will trigger context updates to filter data
    setTimeRange(range);
    
    // For immediate UI feedback, reset rendering stage to basic
    setRenderStage('basic');
    
    // Force chart to fit content after changing time range
    setTimeout(() => {
      if (chartInstanceRef.current) {
        console.log("Fitting content to new time range");
        chartInstanceRef.current.timeScale().fitContent();
        
        // Optionally update legends and other UI components
        // to reflect the new time range
      }
    }, 300);
  }, [setTimeRange]);
  
  // Calculate max points based on screen width
  const calculateMaxPoints = useCallback(() => {
    if (!containerRef.current) return 1000;
    
    // Approximately 1 data point per 2 pixels works well for most charts
    const containerWidth = containerRef.current.clientWidth;
    return Math.max(500, Math.min(2000, Math.floor(containerWidth / 2)));
  }, []);
  
  // Initialize worker once
  useEffect(() => {
    // Only create worker in browser environment
    if (typeof window !== 'undefined') {
      try {
        // Create worker reference
        workerRef.current = createDataProcessingWorker();
        
        // Set up message handler
        workerRef.current.onmessage = (e) => {
          const { result } = e.data;
          if (result) {
            // Process the results from the worker
            setProcessedData(result);
            setIsProcessingData(false);
            
            // Move to next stage if in preview
            if (renderStage === 'preview') {
              setRenderStage('basic');
            }
            
            console.log(`Received ${result.length} processed data points from worker`);
          }
        };
        
        return () => {
          // Clean up worker when component unmounts
          if (workerRef.current) {
            workerRef.current.terminate();
          }
        };
      } catch (error) {
        console.error("Error initializing web worker:", error);
        // Fallback to main thread processing if worker fails
      }
    }
  }, []);
  
  // IMMEDIATE render with ultra-sampled data (no waiting)
  useEffect(() => {
    if (filteredData.length === 0 || typeof window === 'undefined') return;
    
    console.log("Starting immediate preview rendering with", filteredData.length, "data points");
    
    // Store the raw data for later processing
    rawDataRef.current = filteredData;
    
    // Generate preview data on main thread - keep it very light (max 50-100 points)
    const generatePreviewData = () => {
      try {
        // Extract just close prices with timestamps for a very basic preview
        const previewData = filteredData
          .filter(item => item && item.datetime && typeof item.close === 'number')
          .map(item => ({
            time: Math.floor(new Date(item.datetime).getTime() / 1000),
            value: item.close
          }));
        
        // Ultra-sample to ensure immediate rendering (50 points max for preview)
        const sampledPreview = ultraSampleData(previewData, 50);
        
        // Store and use preview data
        previewDataRef.current = sampledPreview;
        setProcessedData(sampledPreview);
        
        // Log success
        console.log("Preview data ready:", sampledPreview.length, "points");
        
        // Once preview is ready, schedule more detailed processing
        setTimeout(() => {
          setRenderStage('basic');
        }, 100);
        
      } catch (error) {
        console.error("Error generating preview data:", error);
      }
    };
    
    // Run immediate preview generation
    generatePreviewData();
  }, [filteredData]);
  
  // Handle next stages of data processing
  useEffect(() => {
    if (renderStage === 'basic' && !isProcessingData && rawDataRef.current.length > 0) {
      setIsProcessingData(true);
      
      const processBasicData = () => {
        try {
          if (rawDataRef.current.length === 0) return;
          
          // Generate light-weight basic data including OHLC
          const lineAreaData: any[] = [];
          const ohlcData: any[] = [];
          const indicators = new Set<string>();
          
          rawDataRef.current.forEach((item) => {
            if (!item || !item.datetime) return;
            
            const dateTime = new Date(item.datetime);
            if (isNaN(dateTime.getTime())) return;
            
            // Extract timestamp for proper x-axis
            const timestamp = Math.floor(dateTime.getTime() / 1000);
            
            // Basic line data with indicators
            if (typeof item.close === 'number' && !isNaN(item.close)) {
              const dataPoint = {
                time: timestamp,
                value: item.close
              };
              
              // Copy indicators
              Object.entries(item).forEach(([key, value]) => {
                if (
                  !['datetime', 'open', 'high', 'low', 'close', 'volume'].includes(key) &&
                  typeof value === 'number' &&
                  !isNaN(value)
                ) {
                  dataPoint[key] = value;
                  indicators.add(key);
                }
              });
              
              lineAreaData.push(dataPoint);
            }
            
            // Basic OHLC data for candles
            if (
              typeof item.open === 'number' && !isNaN(item.open) &&
              typeof item.high === 'number' && !isNaN(item.high) &&
              typeof item.low === 'number' && !isNaN(item.low) &&
              typeof item.close === 'number' && !isNaN(item.close)
            ) {
              ohlcData.push({
                time: timestamp,
                open: item.open,
                high: item.high,
                low: item.low,
                close: item.close
              });
            }
          });
          
          // Sample data for faster rendering
          const maxPoints = 200;
          
          if (workerRef.current) {
            // Use the worker to sample the line data
            workerRef.current.postMessage({
              data: lineAreaData,
              operation: 'sample',
              maxPoints: 200
            });
            
            // Also update available indicators
            setAvailableIndicators(Array.from(indicators));
            
            // Set candle data directly (it's smaller and less processing-intensive)
            setCandleData(ohlcData);
            console.log(`Basic processing: Set ${ohlcData.length} candle data points`);
          }
        } catch (error) {
          console.error("Error processing basic data:", error);
          setIsProcessingData(false);
        }
      };
      
      processBasicData();
    }
    else if (renderStage === 'complete' && !isProcessingData && rawDataRef.current.length > 0) {
      // Process complete data including candlestick data if needed
      setIsProcessingData(true);
      
      const processCompleteData = () => {
        try {
          // Create main chart data with all necessary indicators
          // This is the full processing similar to our previous implementation
          // but we're now doing it as the final stage after showing something to the user
          
          // Process the data for visualization - using our existing logic
          const lineAreaData: any[] = [];
          const ohlcData: any[] = [];
          const volumeData: any[] = [];
          const indicators = new Set<string>();
          
          rawDataRef.current.forEach((item) => {
            if (!item || !item.datetime) return;
    
            const dateTime = new Date(item.datetime);
            if (isNaN(dateTime.getTime())) return;
            
            // Extract timestamp for proper x-axis
            const timestamp = Math.floor(dateTime.getTime() / 1000);
            
            // Process line/area data
            if (typeof item.close === 'number' && !isNaN(item.close)) {
              const dataPoint = {
                time: timestamp,
                value: item.close
              };
              
              // Copy indicators
              Object.entries(item).forEach(([key, value]) => {
                if (
                  !['datetime', 'open', 'high', 'low', 'close', 'volume'].includes(key) &&
                  typeof value === 'number' &&
                  !isNaN(value)
                ) {
                  dataPoint[key] = value;
                  indicators.add(key);
                }
              });
              
              lineAreaData.push(dataPoint);
            }
            
            // Process candlestick data
            if (
              typeof item.open === 'number' && !isNaN(item.open) &&
              typeof item.high === 'number' && !isNaN(item.high) &&
              typeof item.low === 'number' && !isNaN(item.low) &&
              typeof item.close === 'number' && !isNaN(item.close)
            ) {
              ohlcData.push({
                time: timestamp,
                open: item.open,
                high: item.high,
                low: item.low,
                close: item.close
              });
            }
            
            // Process volume data
            if (typeof item.volume === 'number' && !isNaN(item.volume)) {
              volumeData.push({
                time: timestamp,
                value: item.volume,
                color: (item.close > item.open) ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
              });
            }
          });
          
          // Sample data appropriately for different data types
          const maxPoints = calculateMaxPoints();
          
          // For line data, sample based on screen width
          const sampledLineData = sampleData(lineAreaData, maxPoints);
          
          // For candle data, we need to sample differently to preserve OHLC integrity
          // Sample candlestick data while preserving its structure
          const sampledCandleData = ohlcData.length > maxPoints 
            ? sampleData(ohlcData, maxPoints)
            : ohlcData;
          
          // Set the processed data
          setProcessedData(sampledLineData);
          setCandleData(sampledCandleData);
          setVolumeData(volumeData);
          
          // Debug - check if candleData has values
          console.log(`Candle data processed: ${sampledCandleData.length} points`);
          if (sampledCandleData.length > 0) {
            console.log('Sample candle data point:', sampledCandleData[0]);
          }
          
          // Set the available indicators
          setAvailableIndicators(Array.from(indicators));
          
          // Record the sample count
          setSampledDataCount(sampledLineData.length);
          
          // Save last data point for latest values
          setLastDataPoint(rawDataRef.current[rawDataRef.current.length - 1]);
          
          // Update status
          setIsProcessingData(false);
          
          console.log(`Complete data processing finished: ${sampledLineData.length} points, ${ohlcData.length} candles, ${volumeData.length} volume bars`);
        } catch (error) {
          console.error("Error processing complete data:", error);
          setIsProcessingData(false);
        }
      };
      
      // Run the complete data processing
      processCompleteData();
    }
  }, [renderStage, isProcessingData, calculateMaxPoints]);
  
  // Create optimized chart options only once
  const chartOptions = useMemo(() => ({
    layout: {
      background: { type: 'solid', color: 'transparent' },
      textColor: '#4B5563',
      fontSize: 13,
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
    },
    grid: {
      vertLines: { color: 'rgba(229, 231, 235, 0.3)' },
      horzLines: { color: 'rgba(229, 231, 235, 0.3)' },
    },
    timeScale: {
      timeVisible: true,
      secondsVisible: false,
      borderColor: 'rgba(229, 231, 235, 0.8)',
    },
    rightPriceScale: {
      borderColor: 'rgba(229, 231, 235, 0.8)',
      scaleMargins: {
        top: 0.1,
        bottom: 0.1,
      },
    },
    crosshair: {
      mode: 1,
      vertLine: {
        color: 'rgba(37, 99, 235, 0.6)',
        width: 1,
        style: 1,
        visible: true,
        labelVisible: true,
        labelBackgroundColor: 'rgba(37, 99, 235, 0.9)',
      },
      horzLine: {
        color: 'rgba(37, 99, 235, 0.6)',
        width: 1,
        style: 1,
        visible: true,
        labelVisible: true,
        labelBackgroundColor: 'rgba(37, 99, 235, 0.9)',
      },
    },
    handleScale: {
      mouseWheel: true,
      pinch: true,
      axisPressedMouseMove: true,
    },
    handleScroll: {
      pressedMouseMove: true,
      mouseWheel: true,
    },
  }), []);

  // Instant chart initialization
  useEffect(() => {
    if (!chartContainerRef.current || isInitialized.current) {
      return;
    }
    
    try {
      console.log("Creating chart immediately...");
      
      // Set initialized flag
      isInitialized.current = true;
      
      // Get container dimensions
      const containerWidth = chartContainerRef.current.clientWidth || 800;
      const containerHeight = 500;
      
      // Immediately create chart - no waiting
      const chart = createChart(chartContainerRef.current, {
        width: containerWidth,
        height: containerHeight,
        ...chartOptions,
        layout: {
          ...chartOptions.layout,
          background: { 
            type: 'solid', 
            color: window.matchMedia('(prefers-color-scheme: dark)').matches ? '#1f2937' : '#ffffff' 
          },
        },
      });
      
      // Store the chart instance
      chartInstanceRef.current = chart;
      
      // Log chart type and data availability for debugging
      console.log(`Chart type: ${chartType}, Candle data length: ${candleData.length}`);
      
      // Create a bare-bones line series immediately
      let mainSeries;
      if (chartType === 'candle' && candleData.length > 0) {
        console.log("Creating candlestick chart with data:", candleData.length, "points");
        mainSeries = chart.addSeries(CandlestickSeries, {
          priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
          upColor: '#26A69A',
          downColor: '#EF5350',
          borderUpColor: '#26A69A',
          borderDownColor: '#EF5350',
          wickUpColor: '#26A69A',
          wickDownColor: '#EF5350',
        });
        mainSeries.setData(candleData);
      } else {
        console.log("Creating line chart as fallback");
        mainSeries = chart.addSeries(LineSeries, {
          priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
          lineWidth: 2,
          color: '#2563EB',
        });
        mainSeries.setData(processedData);
      }
      
      // Add to active series
      activeSeries.current = [mainSeries];
      
      // Set as created
      setChartCreated(true);
      console.log("Chart created instantly!");

      // Fit content to view
      setTimeout(() => {
        if (chartInstanceRef.current) {
          chartInstanceRef.current.timeScale().fitContent();
          
          // If filtering is done based on time range, also ensure proper scale
          if (selectedTimeframe && selectedTimeframe.days) {
            console.log(`Setting visible time range for ${selectedTimeframe.label}`);
            // Use visible range method if needed for precise time control
            // chartInstanceRef.current.timeScale().setVisibleRange({ from: <timestamp>, to: <timestamp> });
          }
        }
      }, 50);
      
      // Set up resize handler
      const resizeObserver = new ResizeObserver(() => {
        if (chartInstanceRef.current && chartContainerRef.current) {
          chartInstanceRef.current.resize(
            chartContainerRef.current.clientWidth,
            chartContainerRef.current.clientHeight
          );
        }
      });
      
      resizeObserver.observe(chartContainerRef.current);
      
      // Clean up
      return () => {
        resizeObserver.disconnect();
        
        if (chartInstanceRef.current) {
          chartInstanceRef.current.remove();
          chartInstanceRef.current = null;
        }
        
        isInitialized.current = false;
        setChartCreated(false);
      };
    } catch (error) {
      console.error("Error creating chart:", error);
      isInitialized.current = false;
    }
  }, [chartOptions, processedData, candleData, chartType, selectedTimeframe]);
  
  // Update chart when processed data changes
  useEffect(() => {
    if (!chartInstanceRef.current || !chartCreated || activeSeries.current.length === 0) {
      return;
    }
    
    // Skip if no data is available
    if (processedData.length === 0 && candleData.length === 0) {
      console.warn("No data available to update chart");
      return;
    }
    
    try {
      // Get the main series
      const mainSeries = activeSeries.current[0];
      
      // Determine which data to use based on chart type
      const dataToUse = chartType === 'candle' ? candleData : processedData;
      
      // Only update if we have data
      if (dataToUse.length > 0) {
        console.log(`Updating chart with ${dataToUse.length} data points at stage: ${renderStage}`);
        
        // Simple and fast update
        mainSeries.setData(dataToUse);
        
        // Fit content to view
        setTimeout(() => {
          if (chartInstanceRef.current) {
            chartInstanceRef.current.timeScale().fitContent();
          }
        }, 10);
      } else {
        console.warn(`No data available for chart type: ${chartType}`);
      }
    } catch (error) {
      console.error("Error updating chart:", error);
    }
  }, [processedData, candleData, chartCreated, renderStage, chartType]);
  
  // New effect to manage indicator series
  useEffect(() => {
    if (!chartInstanceRef.current || !chartCreated || processedData.length === 0) {
      return;
    }
    
    try {
      // Debug visible indicators
      console.log("Current visible indicators:", visibleIndicators);
      
      // Debug first data point to check fields
      if (processedData.length > 0) {
        console.log("Sample data point for indicators:", processedData[0]);
      }
      
      // Keep track of active indicator series (separate from the main price series)
      const indicatorSeries = activeSeries.current.slice(1);
      
      // Remove all existing indicator series
      indicatorSeries.forEach(series => {
        try {
          chartInstanceRef.current?.removeSeries(series);
        } catch (error) {
          console.error("Error removing indicator series:", error);
        }
      });
      
      // Reset the active series array to only contain the main price series
      activeSeries.current = activeSeries.current.slice(0, 1);
      
      // Filter for visible indicator configs
      const visibleIndicatorConfigs = chartConfigs.filter(
        config => config.visible && config.id !== 'candlestick'
      );
      
      console.log(`Adding ${visibleIndicatorConfigs.length} indicator series to chart:`, 
        visibleIndicatorConfigs.map(c => c.id));
      
      // Create new series for each visible indicator
      visibleIndicatorConfigs.forEach(config => {
        // Handle volume indicator specially
        if (config.id === 'volume' && volumeData.length > 0 && showVolume) {
          const volumeSeries = chartInstanceRef.current!.addSeries(HistogramSeries, {
            color: config.color || '#26a69a',
            priceFormat: {
              type: 'volume',
            },
            priceScaleId: 'volume',
            scaleMargins: {
              top: 0.8,
              bottom: 0,
            },
          });
          
          volumeSeries.setData(volumeData);
          activeSeries.current.push(volumeSeries);
          console.log(`Added volume series with ${volumeData.length} data points`);
          return;
        }
        
        // Fix for Bollinger Bands
        if (config.id.includes('bollinger')) {
          // We'll handle all bollinger bands when processing the middle band
          if (config.id === 'bollinger_middle_20') {
            console.log("Processing Bollinger Bands");
            
            // Check if the data contains Bollinger Band fields
            const hasBollingerData = processedData.some(point => 
              point.bollinger_middle_20 !== undefined &&
              point.bollinger_upper_20 !== undefined &&
              point.bollinger_lower_20 !== undefined
            );
            
            if (!hasBollingerData) {
              console.log("No Bollinger Band data available");
              return;
            }
            
            // Create data arrays for each band with proper time format
            const upperBandData = processedData
              .filter(point => point.bollinger_upper_20 !== undefined)
              .map(point => ({
                time: point.time,
                value: point.bollinger_upper_20
              }));
              
            const middleBandData = processedData
              .filter(point => point.bollinger_middle_20 !== undefined)
              .map(point => ({
                time: point.time,
                value: point.bollinger_middle_20
              }));
              
            const lowerBandData = processedData
              .filter(point => point.bollinger_lower_20 !== undefined)
              .map(point => ({
                time: point.time,
                value: point.bollinger_lower_20
              }));
            
            console.log(`Bollinger data points: Upper (${upperBandData.length}), Middle (${middleBandData.length}), Lower (${lowerBandData.length})`);
            
            // Only add bands if we have data
            if (middleBandData.length > 0) {
              // Add upper band
              const upperSeries = chartInstanceRef.current!.addSeries(LineSeries, {
                color: 'rgba(76, 175, 80, 0.5)',
                lineWidth: 1,
                priceLineVisible: false,
              });
              upperSeries.setData(upperBandData);
              activeSeries.current.push(upperSeries);
              
              // Add middle band
              const middleSeries = chartInstanceRef.current!.addSeries(LineSeries, {
                color: 'rgba(76, 175, 80, 1)',
                lineWidth: 1,
                priceLineVisible: false,
              });
              middleSeries.setData(middleBandData);
              activeSeries.current.push(middleSeries);
              
              // Add lower band
              const lowerSeries = chartInstanceRef.current!.addSeries(LineSeries, {
                color: 'rgba(76, 175, 80, 0.5)',
                lineWidth: 1,
                priceLineVisible: false,
              });
              lowerSeries.setData(lowerBandData);
              activeSeries.current.push(lowerSeries);
              
              console.log("Successfully added Bollinger Bands");
            }
          }
          return; // Skip individual Bollinger Band components
        }
        
        // For standard indicators, check if the field exists in the data
        if (!processedData[0] || processedData[0][config.id] === undefined) {
          console.log(`Indicator field '${config.id}' not found in data`);
          
          // Check if data might have a different field name (common naming variations)
          let found = false;
          const possibleNames = [
            config.id,
            config.id.toLowerCase(),
            config.id.toUpperCase(),
            config.id.replace(/_/g, ''),
            config.id.replace(/^([a-z]+)_(\d+)$/, '$1$2') // Convert ema_20 to ema20
          ];
          
          for (const name of possibleNames) {
            if (name !== config.id && processedData[0][name] !== undefined) {
              console.log(`Found indicator field '${name}' instead of '${config.id}'`);
              
              // Create indicator data with the correct field name
              const indicatorData = processedData
                .filter(point => point[name] !== undefined)
                .map(point => ({
                  time: point.time,
                  value: point[name]
                }));
              
              if (indicatorData.length > 0) {
                const indicatorSeries = chartInstanceRef.current!.addSeries(LineSeries, {
                  color: config.color || '#2196F3',
                  lineWidth: 1,
                  priceLineVisible: false,
                });
                
                indicatorSeries.setData(indicatorData);
                activeSeries.current.push(indicatorSeries);
                console.log(`Added indicator series: ${config.id} with ${indicatorData.length} points using field ${name}`);
                found = true;
                break;
              }
            }
          }
          
          if (!found) {
            return; // Skip if we still can't find the indicator
          }
        } else {
          // Standard indicator as a line series
          const indicatorData = processedData
            .filter(point => point[config.id] !== undefined && !isNaN(point[config.id]))
            .map(point => ({
              time: point.time,
              value: point[config.id]
            }));
          
          if (indicatorData.length === 0) {
            console.log(`No valid data for indicator: ${config.id}`);
            return;
          }
          
          // Debug first and last data point
          console.log(`Indicator ${config.id} first point:`, indicatorData[0]);
          console.log(`Indicator ${config.id} last point:`, indicatorData[indicatorData.length - 1]);
          
          const indicatorSeries = chartInstanceRef.current!.addSeries(LineSeries, {
            color: config.color || '#2196F3',
            lineWidth: 1,
            priceLineVisible: false,
          });
          
          indicatorSeries.setData(indicatorData);
          activeSeries.current.push(indicatorSeries);
          console.log(`Added indicator series: ${config.id} with ${indicatorData.length} points`);
        }
      });
      
      // Fit content again after adding indicators
      setTimeout(() => {
        if (chartInstanceRef.current) {
          chartInstanceRef.current.timeScale().fitContent();
        }
      }, 10);
      
    } catch (error) {
      console.error("Error managing indicator series:", error);
    }
  }, [visibleIndicators, chartCreated, processedData, candleData, chartConfigs, showVolume, showIndicators]);
  
  // Fix the handling of showIndicators state and indicator panel visibility
  useEffect(() => {
    const panel = document.getElementById('indicator-panel');
    if (!panel) return;
    
    if (showIndicators) {
      panel.classList.remove('hidden');
      
      // Also force a redraw of the indicator series to ensure they're visible
      if (chartInstanceRef.current && chartCreated) {
        // Small delay to ensure panel is visible first
        setTimeout(() => {
          const visibleIndicatorConfigs = chartConfigs.filter(
            config => config.visible && config.id !== 'candlestick'
          );
          console.log(`There are ${visibleIndicatorConfigs.length} visible indicators`);
        }, 50);
      }
    } else {
      panel.classList.add('hidden');
    }
  }, [showIndicators, chartCreated, chartConfigs, chartInstanceRef, visibleIndicators]);
  
  // Updated toggleIndicator function to improve performance
  const handleToggleIndicators = useCallback(() => {
    console.log(`Toggling indicators panel visibility: ${!showIndicators}`);
    setShowIndicators(prev => !prev);
  }, [showIndicators]);
  
  // Add a wrapper for toggleIndicator to fix any issues
  const handleToggleIndicator = useCallback((id: string) => {
    console.log(`Toggling indicator: ${id}`);
    
    // Call the context's toggleIndicator function
    toggleIndicator(id);
    
    // Force re-render of indicator series after a short delay
    setTimeout(() => {
      if (chartInstanceRef.current) {
        // This will trigger the useEffect for indicators
        setRenderStage(prev => {
          console.log("Forcing indicator update");
          return prev; // Keep the same value, but force re-render
        });
      }
    }, 50);
  }, [toggleIndicator]);
  
  // Add optimized handleChartTypeChange function that directly updates the chart
  const handleChartTypeChange = useCallback((type: 'line' | 'area' | 'candle') => {
    setChartType(type);
    
    // Only try to update the chart directly if it exists already
    if (chartInstanceRef.current && activeSeries.current.length > 0) {
      // Store indicator series
      const indicatorSeries = activeSeries.current.slice(1);
      
      // Remove existing main series (keep indicator series for now)
      if (activeSeries.current[0]) {
        try {
          chartInstanceRef.current.removeSeries(activeSeries.current[0]);
        } catch (e) {
          console.error("Error removing main series during type change:", e);
        }
      }
      
      // Update the main series to reflect new chart type
      let newMainSeries;
      
      if (type === 'line') {
        newMainSeries = chartInstanceRef.current.addSeries(LineSeries, {
          priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
          lineWidth: 2,
          color: '#2563EB',
        });
        newMainSeries.setData(processedData);
      } else if (type === 'area') {
        newMainSeries = chartInstanceRef.current.addSeries(AreaSeries, {
          priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
          lineWidth: 2,
          lineColor: '#2563EB',
          topColor: 'rgba(37, 99, 235, 0.4)',
          bottomColor: 'rgba(37, 99, 235, 0.05)',
        });
        newMainSeries.setData(processedData);
      } else {
        newMainSeries = chartInstanceRef.current.addSeries(CandlestickSeries, {
          priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
          upColor: '#26A69A',
          downColor: '#EF5350',
          borderUpColor: '#26A69A',
          borderDownColor: '#EF5350',
          wickUpColor: '#26A69A',
          wickDownColor: '#EF5350',
        });
        newMainSeries.setData(candleData);
      }
      
      // Update the active series array with the new main series and preserve indicator series
      activeSeries.current = [newMainSeries, ...indicatorSeries];
      
      // Force a redraw of indicators
      setTimeout(() => {
        // This will trigger the useEffect for visibleIndicators
        setRenderStage(prev => {
          console.log("Forcing indicator redraw after chart type change");
          return prev;
        });
      }, 50);
    }
  }, [processedData, candleData]);
  
  // Add a new useEffect to monitor timeframe changes and ensure chart updates correctly
  useEffect(() => {
    if (!chartInstanceRef.current || !chartCreated || !selectedTimeframe) {
      return;
    }
    
    console.log(`Selected timeframe changed to: ${selectedTimeframe.label} (${selectedTimeframe.days} days)`);
    
    // Give time for filtered data to be updated
    setTimeout(() => {
      if (chartInstanceRef.current) {
        console.log(`Ensuring chart displays ${selectedTimeframe.label} properly`);
        
        // Fit content to display all data
        chartInstanceRef.current.timeScale().fitContent();
        
        // Update visible range if needed
        // const now = new Date();
        // const startTs = Math.floor(new Date(now.getTime() - (selectedTimeframe.days * 24 * 60 * 60 * 1000)).getTime() / 1000);
        // const endTs = Math.floor(now.getTime() / 1000);
        // chartInstanceRef.current.timeScale().setVisibleRange({ from: startTs, to: endTs });
      }
    }, 300);
  }, [selectedTimeframe, chartCreated]);
  
  // Loading state
  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-2/3 mb-4"></div>
          <div className="h-[500px] bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center">
            <div className="flex items-center justify-center">
              <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="ml-2 text-gray-600 dark:text-gray-300">Loading chart data...</span>
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
        <TimeRangeTabs 
          selectedRange={selectedRange} 
          onRangeChange={handleRangeChange} 
        />
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
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 dark:border dark:border-gray-700" ref={containerRef}>
      {/* Unified control panel */}
      <div className="flex flex-col gap-4 mb-4">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          {/* Chart type control */}
          <div className="flex-shrink-0">
            <ChartTypeSelector 
              chartType={chartType} 
              onChartTypeChange={handleChartTypeChange} 
            />
          </div>
          
          {/* Only one instance of TimeRangeTabs */}
          <div className="flex-grow">
            <TimeRangeTabs 
              selectedRange={selectedRange} 
              onRangeChange={handleRangeChange} 
            />
          </div>
        </div>
        
        <div className="flex flex-wrap items-center gap-2 mt-1">
          {/* Feature toggles */}
          <button
            onClick={handleToggleIndicators}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center gap-1 ${
              showIndicators 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300'
            }`}
            aria-label="Toggle indicators"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4" y1="21" x2="4" y2="14"></line>
              <line x1="4" y1="10" x2="4" y2="3"></line>
              <line x1="12" y1="21" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12" y2="3"></line>
              <line x1="20" y1="21" x2="20" y2="16"></line>
              <line x1="20" y1="12" x2="20" y2="3"></line>
              <line x1="1" y1="14" x2="7" y2="14"></line>
              <line x1="9" y1="8" x2="15" y2="8"></line>
              <line x1="17" y1="16" x2="23" y2="16"></line>
            </svg>
            <span className="hidden xs:inline">Indicators</span> <span className="inline xs:hidden">Ind</span> {showIndicators ? 'On' : 'Off'}
          </button>
          
          <button
            onClick={() => setShowVolume(prev => !prev)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center gap-1 ${
              showVolume
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300'
            }`}
            aria-label="Toggle volume display"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 3v18h18"></path>
              <rect x="7" y="10" width="2" height="8"></rect>
              <rect x="12" y="7" width="2" height="11"></rect>
              <rect x="17" y="4" width="2" height="14"></rect>
            </svg>
            <span className="hidden xs:inline">Volume</span> <span className="inline xs:hidden">Vol</span> {showVolume ? 'On' : 'Off'}
          </button>
          
          {/* Display current time range information - Moved to TimeRangeTabs */}
          
          {lastDataPoint && (
            <div className="ml-auto flex flex-wrap gap-3">
              <span className="text-sm font-medium dark:text-gray-200">
                <span className="text-gray-500 dark:text-gray-400">Last:</span> ${typeof lastDataPoint.close === 'number' ? lastDataPoint.close.toFixed(2) : 'N/A'}
              </span>
              <span className="text-sm font-medium dark:text-gray-200">
                <span className="text-gray-500 dark:text-gray-400">Date:</span> {new Date(lastDataPoint.datetime).toLocaleDateString()}
              </span>
            </div>
          )}
        </div>
      </div>
      
      {/* Always render the panel but rely on CSS for visibility */}
      <IndicatorsPanel 
        chartConfigs={chartConfigs}
        toggleIndicator={handleToggleIndicator}
        availableIndicators={availableIndicators}
      />
      
      {/* Chart legend */}
      {lastDataPoint && visibleIndicators.length > 0 && (
        <ChartLegend 
          data={lastDataPoint} 
          indicatorSettings={chartConfigs.filter(
            config => config.visible && availableIndicators.includes(config.id)
          )}
          onToggleIndicator={handleToggleIndicator}
        />
      )}
      
      {/* Chart container */}
      <div 
        ref={chartContainerRef} 
        className="w-full h-[500px] rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 mt-4"
        style={{ height: '500px' }}
      >
        {(isProcessingData || !chartCreated) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-700 bg-opacity-70 dark:bg-opacity-70 z-10">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-2"></div>
            <p className="text-sm text-gray-600 dark:text-gray-300 font-medium">
              {renderStage === 'preview' ? 'Creating preview...' : 
               renderStage === 'basic' ? 'Enhancing chart...' : 
               'Finalizing details...'}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {renderStage !== 'complete' && 'Showing simplified view while processing'}
            </p>
          </div>
        )}
      </div>
      
      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
        {renderStage !== 'complete' && (
          <span>Loading chart... ({renderStage === 'preview' ? 'Preview' : 'Basic'} view)</span>
        )}
        {renderStage === 'complete' && processedData.length > 0 && (
          <span>{processedData.length} data points</span>
        )}
      </div>
    </div>
  );
};

export default EnhancedStockChart; 