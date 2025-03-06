export interface StockData {
  datetime: string;
  cum_return: number;
  low: number;
  high: number;
  open: number;
  close: number;
  close_future_1: number;
  volume_sma_20?: number;
  volume_sma_50?: number;
  sma_5?: number;
  sma_10?: number;
  sma_20?: number;
  sma_50?: number;
  sma_200?: number;
  ema_5?: number;
  ema_10?: number;
  ema_20?: number;
  ema_50?: number;
  ema_200?: number;
  bollinger_upper_20?: number;
  bollinger_middle_20?: number;
  bollinger_lower_20?: number;
  vwap?: number;
  [key: string]: any; // For any additional fields
}

export interface ChartConfig {
  id: string;
  label: string;
  color: string;
  visible: boolean;
  type: 'line' | 'area' | 'bar' | 'candlestick' | 'volume';
  yAxis?: 'left' | 'right';
  valueFormatter?: (value: number) => string;
}

export interface ChartRange {
  startDate: Date;
  endDate: Date;
}

export interface TimeFrame {
  id: string;
  label: string;
  days: number;
} 