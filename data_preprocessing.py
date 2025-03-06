import pandas as pd
import numpy as np
import os
from scipy import stats
from datetime import datetime, time
import pytz
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from scipy.fft import fft, ifft, fftfreq, rfft, rfftfreq, irfft
import pywt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import (
    mutual_info_regression, SelectKBest, SelectFromModel,
    f_regression, RFE, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks
import warnings
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
import traceback

# Try to import numba for optimization if available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
# Function to be optimized with numba if available
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _calculate_zscore(data):
        """Calculate z-score using numba for performance.
        Handles division by zero cases safely.
        """
        # Convert data to float to avoid type inference issues
        data_float = data.astype(np.float64)
        mean = np.mean(data_float)
        std = np.std(data_float)
        
        # Create a result array of the same shape as data
        result = np.zeros_like(data_float)
        
        # Only calculate z-scores if std is not zero
        if std > 0:
            result = (data_float - mean) / std
            
        return result
else:
    def _calculate_zscore(data):
        """Calculate z-score using scipy.stats."""
        return stats.zscore(data, nan_policy='omit')

class FinancialDataProcessor:
    def __init__(self, input_file, output_file=None, n_jobs=-1, date_column=None):
        """
        Initialize the data processor with input and output file paths.
        
        Args:
            input_file (str): Path to the input CSV file
            output_file (str): Path to the output CSV file
            n_jobs (int): Number of jobs for parallel processing. -1 means using all processors.
            date_column (str): Name of the date/datetime column. If None, will auto-detect.
        """
        self.input_file = input_file
        
        if output_file is None:
            # Create output file name based on input file
            basename = os.path.basename(input_file)
            name, ext = os.path.splitext(basename)
            self.output_file = os.path.join('data/processed', f"{name}_processed{ext}")
        else:
            self.output_file = output_file
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # For parallel processing
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        # Initialize data
        self.data = None
        self.feature_importance = None
        self.selected_features = None
        self.date_column = date_column
        
        # Try to load the data to check headers and auto-detect date column
        try:
            # First, peek at the CSV headers without parsing dates
            data_peek = pd.read_csv(input_file, nrows=0)
            columns = data_peek.columns.tolist()
            
            # Auto-detect date column if not specified
            if self.date_column is None:
                # Check for common date column names
                date_column_options = ['datetime', 'date', 'timestamp', 'time', 'Date', 'DateTime', 'Timestamp', 'Time']
                for col in date_column_options:
                    if col in columns:
                        self.date_column = col
                        print(f"Auto-detected date column: '{self.date_column}'")
                        break
                
                if self.date_column is None:
                    print("WARNING: No date column detected. Using the first column as index.")
                    self.date_column = columns[0]
            
            print(f"Loading data from {input_file}")
            self.data = pd.read_csv(input_file, parse_dates=[self.date_column])
            
            # Set date column as index and handle timezone information
            self.data.set_index(self.date_column, inplace=True)
            
            # Check if the index is already a DatetimeIndex
            if not isinstance(self.data.index, pd.DatetimeIndex):
                print("Converting index to DatetimeIndex")
                # Convert to DatetimeIndex with proper timezone handling
                self.data.index = pd.to_datetime(self.data.index, utc=True)
            
            # Handle timezone information
            eastern = pytz.timezone('US/Eastern')
            if self.data.index.tzinfo is not None:
                # Convert to Eastern Time for financial data
                print("Converting timezone from UTC to Eastern Time")
                self.data.index = self.data.index.tz_convert(eastern)
            else:
                # Localize naive timestamps
                print("Localizing timestamps to Eastern Time")
                self.data.index = self.data.index.tz_localize(eastern, ambiguous='infer')
                
            print(f"Data loaded with shape: {self.data.shape}")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            print("Will attempt to load data again during pipeline execution.")
            self.data = None
        
    def filter_trading_hours(self):
        """
        Filter data to include only standard trading hours (9:30 AM to 4:00 PM Eastern Time).
        """
        print(f"Original data shape: {self.data.shape}")
        
        # Make sure we're working with Eastern Time
        eastern = pytz.timezone('US/Eastern')
        if self.data.index.tzinfo is None:
            print("Warning: Index has no timezone info. Assuming Eastern Time.")
            self.data.index = self.data.index.tz_localize(eastern, ambiguous='raise')
        elif str(self.data.index.tzinfo) != 'US/Eastern':
            print(f"Converting timezone from {self.data.index.tzinfo} to Eastern Time")
            self.data.index = self.data.index.tz_convert(eastern)
        
        # Extract time component using datetime properties
        hours = self.data.index.hour
        minutes = self.data.index.minute
        
        # Define trading hours in total minutes
        market_open_minutes = 9*60 + 30  # 9:30 AM
        market_close_minutes = 16*60     # 4:00 PM
        
        # Calculate total minutes for each timestamp
        total_minutes = hours * 60 + minutes
        
        # Filter for trading hours
        self.data = self.data[(total_minutes >= market_open_minutes) & 
                             (total_minutes <= market_close_minutes)]
        
        print(f"Data shape after trading hours filter: {self.data.shape}")
        return self
        
    def handle_missing_data(self, strategy='auto', max_missing_pct=0.3, fill_method='ffill_bfill'):
        """
        Handle missing data in the dataset using various strategies.
        
        Args:
            strategy (str): Strategy to handle missing data:
                - 'auto': Automatically choose best strategy based on data
                - 'drop_rows': Drop rows with any missing values
                - 'drop_cols': Drop columns with too many missing values
                - 'fill': Fill missing values using the specified method
                - 'interpolate': Use interpolation to fill missing values
            max_missing_pct (float): Maximum percentage of missing values allowed in a column
                before it gets dropped (only used with 'drop_cols' strategy)
            fill_method (str): Method to fill missing values:
                - 'ffill': Forward fill (use previous value)
                - 'bfill': Backward fill (use next value)
                - 'ffill_bfill': Forward fill then backward fill
                - 'mean': Use column mean
                - 'median': Use column median
                - 'mode': Use column mode
                - 'zero': Fill with zeros
                - 'value': Fill with a constant value
                
        Returns:
            self for method chaining
        """
        if self.data is None or self.data.empty:
            print("No data available to handle missing values")
            return self
            
        print("Handling missing data...")
        
        # Check for missing values
        missing_count = self.data.isna().sum()
        missing_pct = missing_count / len(self.data)
        
        total_missing = missing_count.sum()
        
        if total_missing == 0:
            print("No missing values found in the dataset")
            return self
            
        print(f"Found {total_missing} missing values across {sum(missing_count > 0)} columns")
        print("\nMissing value distribution:")
        for col in missing_count[missing_count > 0].index:
            print(f"  {col}: {missing_count[col]} values ({missing_pct[col]:.2%})")
        
        # Auto strategy selection based on data characteristics
        if strategy == 'auto':
            # If small percentage of rows have missing values, drop rows
            rows_with_missing = self.data.isna().any(axis=1).sum()
            pct_rows_missing = rows_with_missing / len(self.data)
            
            if pct_rows_missing < 0.05:
                strategy = 'drop_rows'
                print(f"\nAuto-selected strategy: 'drop_rows' ({pct_rows_missing:.2%} of rows have missing values)")
            elif any(missing_pct > max_missing_pct):
                strategy = 'drop_cols'
                print(f"\nAuto-selected strategy: 'drop_cols' (some columns exceed {max_missing_pct:.2%} missing values)")
            else:
                # For time series data, interpolation is usually best
                strategy = 'interpolate'
                print("\nAuto-selected strategy: 'interpolate' (best for time series data)")
        
        # Apply selected strategy
        original_shape = self.data.shape
        
        if strategy == 'drop_rows':
            self.data = self.data.dropna()
            rows_dropped = original_shape[0] - self.data.shape[0]
            print(f"Dropped {rows_dropped} rows with missing values")
            
        elif strategy == 'drop_cols':
            # Identify columns with too many missing values
            cols_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
            
            if cols_to_drop:
                print(f"Dropping {len(cols_to_drop)} columns with >{max_missing_pct:.2%} missing values:")
                for col in cols_to_drop:
                    print(f"  {col}: {missing_pct[col]:.2%} missing")
                
                self.data = self.data.drop(columns=cols_to_drop)
            
            # For remaining columns with missing values, use interpolation
            if self.data.isna().sum().sum() > 0:
                print("Interpolating remaining missing values...")
                self.data = self.data.interpolate(method='time').bfill().ffill()
            
        elif strategy == 'fill':
            if fill_method == 'ffill':
                self.data = self.data.fillna(method='ffill')
                # Handle missing values at the beginning that can't be forward filled
                self.data = self.data.fillna(method='bfill')
                
            elif fill_method == 'bfill':
                self.data = self.data.fillna(method='bfill')
                # Handle missing values at the end that can't be back filled
                self.data = self.data.fillna(method='ffill')
                
            elif fill_method == 'ffill_bfill':
                self.data = self.data.fillna(method='ffill').fillna(method='bfill')
                
            elif fill_method == 'mean':
                for col in missing_count[missing_count > 0].index:
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        col_mean = self.data[col].mean()
                        self.data[col] = self.data[col].fillna(col_mean)
                    else:
                        self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
                        
            elif fill_method == 'median':
                for col in missing_count[missing_count > 0].index:
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        col_median = self.data[col].median()
                        self.data[col] = self.data[col].fillna(col_median)
                    else:
                        self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
                        
            elif fill_method == 'mode':
                for col in missing_count[missing_count > 0].index:
                    col_mode = self.data[col].mode()
                    if not col_mode.empty:
                        self.data[col] = self.data[col].fillna(col_mode[0])
                        
            elif fill_method == 'zero':
                self.data = self.data.fillna(0)
                
            print(f"Filled missing values using '{fill_method}' method")
            
        elif strategy == 'interpolate':
            # Time-based interpolation for time series data
            self.data = self.data.interpolate(method='time')
            
            # Handle any remaining missing values at the edges
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            print("Interpolated missing values based on time index")
            
        else:
            raise ValueError(f"Unknown missing data handling strategy: {strategy}")
            
        # Check if any missing values remain
        remaining_missing = self.data.isna().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values remain after processing")
        else:
            print("All missing values have been handled")
            
        print(f"Data shape after handling missing values: {self.data.shape}")
        return self
    
    def detect_outliers(self, method='zscore', threshold=3.0, action='winsorize'):
        """
        Detect and optionally handle outliers in the data.
        
        Args:
            method (str): Method to detect outliers:
                - 'zscore': Use Z-score (assumes normal distribution)
                - 'iqr': Use Interquartile Range (robust to non-normal data)
                - 'quantile': Use quantile-based detection
            threshold (float): Threshold for outlier detection (e.g., 3.0 for z-score)
            action (str): Action to take for outliers:
                - 'none': Just identify outliers without modifying
                - 'remove': Remove rows with outliers
                - 'clip': Clip values at the threshold
                - 'winsorize': Replace with threshold values
                - 'mean': Replace with column mean
                - 'median': Replace with column median
                
        Returns:
            self for method chaining
        """
        print(f"Detecting outliers using '{method}' method with threshold {threshold}")
        
        # Only check numeric columns
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        print(f"Checking {len(numeric_cols)} numeric columns for outliers")
        
        # Create mask to track all outliers
        all_outliers_mask = pd.Series(False, index=self.data.index)
        
        for col in numeric_cols:
            # Skip columns with all zeros or all identical values
            if self.data[col].nunique() <= 1:
                continue
                
            # Get outlier mask for this column
            outliers_mask = None
            
            if method == 'zscore':
                # Z-score method
                if NUMBA_AVAILABLE:
                    # Get z-scores using optimized Numba function
                    z_scores = _calculate_zscore(self.data[col].values)
                    outliers_mask = abs(z_scores) > threshold
                else:
                    # Fallback to scipy.stats
                    z_scores = stats.zscore(self.data[col].values, nan_policy='omit')
                    outliers_mask = abs(z_scores) > threshold
            
            elif method == 'iqr':
                # IQR method 
                q1 = self.data[col].quantile(0.25)
                q3 = self.data[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                outliers_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                
            elif method == 'quantile':
                # Quantile method
                lower_bound = self.data[col].quantile(0.01)  # Bottom 1%
                upper_bound = self.data[col].quantile(0.99)  # Top 1%
                
                outliers_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            # Count and report outliers
            outlier_count = outliers_mask.sum()
            if outlier_count > 0:
                outlier_pct = outlier_count / len(self.data) * 100
                print(f"  {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
                
                # Update the global outlier mask
                all_outliers_mask = all_outliers_mask | outliers_mask
                
                # Handle outliers based on the specified action
                if action == 'clip':
                    # Clip values at the threshold
                    if method == 'zscore':
                        std = self.data[col].std()
                        mean = self.data[col].mean()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                    elif method == 'iqr':
                        q1 = self.data[col].quantile(0.25)
                        q3 = self.data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                    elif method == 'quantile':
                        lower_bound = self.data[col].quantile(0.01)
                        upper_bound = self.data[col].quantile(0.99)
                        
                    # Convert bounds to same dtype as column to avoid warnings
                    col_dtype = self.data[col].dtype
                    if pd.api.types.is_integer_dtype(col_dtype):
                        lower_bound = int(lower_bound)
                        upper_bound = int(upper_bound)
                        
                    self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                    self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                
                elif action == 'winsorize':
                    # Replace with threshold values
                    if method == 'zscore':
                        std = self.data[col].std()
                        mean = self.data[col].mean()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                    elif method == 'iqr':
                        q1 = self.data[col].quantile(0.25)
                        q3 = self.data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                    elif method == 'quantile':
                        lower_bound = self.data[col].quantile(0.01)
                        upper_bound = self.data[col].quantile(0.99)
                    
                    # Convert bounds to same dtype as column to avoid warnings
                    col_dtype = self.data[col].dtype
                    if pd.api.types.is_integer_dtype(col_dtype):
                        lower_bound = int(lower_bound)
                        upper_bound = int(upper_bound)
                        
                    self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                    self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                
                elif action == 'mean':
                    # Replace with column mean
                    col_mean = self.data[col].mean()
                    self.data.loc[outliers_mask, col] = col_mean
                
                elif action == 'median':
                    # Replace with column median
                    col_median = self.data[col].median()
                    self.data.loc[outliers_mask, col] = col_median
                    
                elif action == 'none':
                    # Do nothing, just identify
                    pass
                    
                elif action != 'remove':  # 'remove' is handled globally after all columns
                    raise ValueError(f"Unknown outlier action: {action}")
        
        # Report on overall outliers
        total_outlier_rows = all_outliers_mask.sum()
        if total_outlier_rows > 0:
            print(f"\nFound {total_outlier_rows} rows ({total_outlier_rows/len(self.data):.2%}) with outliers in at least one column")
            
            # Remove rows with outliers if requested
            if action == 'remove':
                original_shape = self.data.shape
                self.data = self.data[~all_outliers_mask]
                print(f"Removed {original_shape[0] - self.data.shape[0]} rows with outliers")
        else:
            print("No outliers detected in any columns")
        
        return self
    
    def compute_features(self):
        """
        Compute technical indicators and features for time series analysis.
        
        Returns:
            self for method chaining
        """
        print("Computing technical features...")
        
        if self.data is None or self.data.empty:
            print("No data available for feature computation")
            return self
            
        # Check if required columns are present
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns for feature calculation: {missing_cols}")
            print("Some features may not be computed correctly.")
        
        # Copy data to avoid modifying the original
        data = self.data.copy()
        
        try:
            # 1. Price and Return Features
            # Simple returns
            data['return_1d'] = data['close'].pct_change()
            
            # Multi-period returns (useful for momentum indicators)
            for period in [5, 10, 21, 63]:  # 1-week, 2-week, 1-month, 3-month
                data[f'return_{period}d'] = data['close'].pct_change(period)
            
            # Log returns (better for statistical analysis)
            data['log_return_1d'] = np.log(data['close'] / data['close'].shift(1))
            
            # Cumulative returns
            data['cum_return'] = (1 + data['return_1d']).cumprod()
            
            # 2. Moving Averages
            # Simple Moving Averages (SMAs)
            for period in [5, 10, 20, 50, 200]:
                data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
                
                # Distance from moving average (percent)
                data[f'close_to_sma_{period}'] = (data['close'] / data[f'sma_{period}'] - 1) * 100
            
            # Exponential Moving Averages (EMAs)
            for period in [5, 10, 20, 50, 200]:
                data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
                
                # Distance from EMA (percent)
                data[f'close_to_ema_{period}'] = (data['close'] / data[f'ema_{period}'] - 1) * 100
            
            # 3. Volatility Indicators
            # Rolling standard deviation of returns
            for period in [5, 10, 21, 63]:
                data[f'volatility_{period}d'] = data['return_1d'].rolling(window=period).std() * np.sqrt(252)  # Annualized
            
            # Average True Range (ATR)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            data['atr_14'] = true_range.rolling(14).mean()
            data['atr_percent_14'] = data['atr_14'] / data['close'] * 100
            
            # 4. Momentum Indicators
            # Relative Strength Index (RSI)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            data['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = data['close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['close'].ewm(span=26, adjust=False).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # 5. Volume Indicators
            if 'volume' in data.columns:
                # Volume change
                data['volume_change_1d'] = data['volume'].pct_change()
                
                # Rolling average volume
                for period in [5, 10, 20, 50]:
                    data[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()
                    data[f'volume_ratio_sma_{period}'] = data['volume'] / data[f'volume_sma_{period}']
                
                # On-Balance Volume (OBV)
                data['obv'] = np.nan
                data.loc[0, 'obv'] = 0
                
                for i in range(1, len(data)):
                    if data.iloc[i]['close'] > data.iloc[i-1]['close']:
                        data.loc[data.index[i], 'obv'] = data.iloc[i-1]['obv'] + data.iloc[i]['volume']
                    elif data.iloc[i]['close'] < data.iloc[i-1]['close']:
                        data.loc[data.index[i], 'obv'] = data.iloc[i-1]['obv'] - data.iloc[i]['volume']
                    else:
                        data.loc[data.index[i], 'obv'] = data.iloc[i-1]['obv']
            
            # 6. Bollinger Bands
            for period in [20]:
                # Calculate SMA and standard deviation
                sma = data['close'].rolling(window=period).mean()
                std = data['close'].rolling(window=period).std()
                
                # Create Bollinger Bands
                data[f'bollinger_upper_{period}'] = sma + (std * 2)
                data[f'bollinger_lower_{period}'] = sma - (std * 2)
                data[f'bollinger_middle_{period}'] = sma
                data[f'bollinger_width_{period}'] = (data[f'bollinger_upper_{period}'] - data[f'bollinger_lower_{period}']) / sma
                
                # Bollinger Band %B (shows where the price is relative to the bands)
                data[f'bollinger_pctb_{period}'] = (data['close'] - data[f'bollinger_lower_{period}']) / (data[f'bollinger_upper_{period}'] - data[f'bollinger_lower_{period}'])
            
            # 7. Price Patterns and Formations
            # Price gap detection
            data['gap_up'] = (data['low'] > data['high'].shift(1)).astype(int)
            data['gap_down'] = (data['high'] < data['low'].shift(1)).astype(int)
            
            # Doji pattern (open and close are very close)
            doji_threshold = 0.1  # 0.1% difference between open and close
            data['doji'] = (np.abs(data['close'] / data['open'] - 1) < doji_threshold).astype(int)
            
            # 8. Day of Week, Month, Year features (calendar effects)
            # Extract calendar features from the index safely
            # First check if the index is a DatetimeIndex
            if isinstance(data.index, pd.DatetimeIndex):
                # Safe to extract datetime features
                data['day_of_week'] = data.index.dayofweek
                data['day_of_month'] = data.index.day
                data['month'] = data.index.month
                data['quarter'] = data.index.quarter
            else:
                # We'll skip the datetime features
                print("Warning: Index is not a DatetimeIndex. Skipping datetime features.")
            
            # 9. Statistical Features
            # Rolling skewness and kurtosis of returns
            for period in [10, 21]:
                data[f'return_skew_{period}d'] = data['return_1d'].rolling(window=period).skew()
                data[f'return_kurt_{period}d'] = data['return_1d'].rolling(window=period).kurt()
            
            # 10. Candle stick patterns (simple ones)
            # Hammer: Long lower shadow, small body, little or no upper shadow
            data['lower_shadow'] = data['open'].combine(data['close'], min) - data['low']
            data['upper_shadow'] = data['high'] - data['open'].combine(data['close'], max)
            data['body'] = np.abs(data['close'] - data['open'])
            
            # Replace 0 values to avoid division by zero
            data['body'] = data['body'].replace(0, 0.0001)
            
            data['lower_shadow_ratio'] = data['lower_shadow'] / data['body']
            data['upper_shadow_ratio'] = data['upper_shadow'] / data['body']
            
            # Detect hammer pattern (long lower shadow, small upper shadow)
            data['hammer'] = ((data['lower_shadow_ratio'] > 2) & 
                            (data['upper_shadow_ratio'] < 0.5)).astype(int)
            
            # Fill missing values in technical features with appropriate defaults
            # First, identify the original columns vs the new feature columns
            original_cols = self.data.columns.tolist()
            feature_cols = [col for col in data.columns if col not in original_cols]
            
            # Fill NaN values in technical features
            # For most indicators, 0 is a reasonable default
            # Count NaN values before filling
            nan_count = data[feature_cols].isna().sum().sum()
            if nan_count > 0:
                print(f"Filling {nan_count} NaN values in technical features with appropriate defaults")
                
                # Different defaults for different types of features
                # Indicators like RSI have a natural midpoint at 50
                rsi_cols = [col for col in feature_cols if 'rsi' in col.lower()]
                if rsi_cols:
                    data[rsi_cols] = data[rsi_cols].fillna(50)
                
                # Percentage and ratio features default to 0
                pct_cols = [col for col in feature_cols if any(x in col.lower() for x in ['pct', 'ratio', 'percent'])]
                if pct_cols:
                    data[pct_cols] = data[pct_cols].fillna(0)
                
                # Fill other technical features with forward-backward fill first, then 0
                remaining_cols = [col for col in feature_cols if col not in rsi_cols and col not in pct_cols]
                if remaining_cols:
                    data[remaining_cols] = data[remaining_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Print summary of the features added
            print(f"Added {len(data.columns) - len(self.data.columns)} technical features")
                            
            # Update self.data with the new features
            self.data = data
            
        except Exception as e:
            print(f"Error computing features: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return self
    
    def validate_data(self):
        """
        Validate the data for completeness and quality.
        Check for missing values, inconsistencies, and data quality issues.
        
        Returns:
            self for method chaining
        """
        print("Validating data quality...")
        
        # Check for missing values in the original columns (OHLCV data)
        # First, identify the original price columns
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume']
        existing_price_cols = [col for col in price_volume_cols if col in self.data.columns]
        
        # Check for missing values in important columns
        missing_values = self.data[existing_price_cols].isnull().sum()
        
        if missing_values.sum() > 0:
            print("Missing values detected in important price columns:")
            print(missing_values[missing_values > 0])
            
            # Only drop rows with missing values in important columns
            original_shape = self.data.shape
            self.data = self.data.dropna(subset=existing_price_cols)
            rows_dropped = original_shape[0] - self.data.shape[0]
            
            print(f"Dropped {rows_dropped} rows with missing values in important columns")
            print(f"Data shape after dropping: {self.data.shape}")
        else:
            print("No missing values in important price and volume columns")
            
        # Check for any remaining missing values in other columns
        remaining_missing = self.data.isnull().sum()
        if remaining_missing.sum() > 0:
            print(f"Remaining missing values in other columns:")
            print(remaining_missing[remaining_missing > 0])
            
            # Fill missing values in technical indicator columns
            # Identify technical feature columns (non-price columns)
            tech_cols = [col for col in self.data.columns if col not in existing_price_cols]
            
            # Fill these with appropriate defaults
            if tech_cols:
                # For simplicity, forward fill then backward fill, then fill with 0
                self.data[tech_cols] = self.data[tech_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
                print(f"Filled missing values in technical indicators with appropriate values")
        
        # Verify no missing values remain
        final_missing = self.data.isnull().sum().sum()
        if final_missing > 0:
            print(f"WARNING: {final_missing} missing values still remain after filling")
        else:
            print("No missing values remain in the dataset")
            
        # Check for trading session completeness
        if isinstance(self.data.index, pd.DatetimeIndex):
            # Create a date column for grouping
            self.data['date'] = self.data.index.date
            
            session_completeness = {}
            expected_minutes = 390  # 6.5 hours * 60 minutes
            
            for date, group in self.data.groupby('date'):
                session_completeness[date] = len(group) / expected_minutes
            
            incomplete_sessions = {date: completeness for date, completeness in session_completeness.items() 
                                  if completeness < 1.0}
            
            if incomplete_sessions:
                print(f"Warning: {len(incomplete_sessions)} trading sessions are incomplete")
                
                # Sort sessions by completeness (ascending)
                sorted_incomplete = sorted(incomplete_sessions.items(), key=lambda x: x[1])
                
                # Show the 5 most incomplete sessions
                if len(sorted_incomplete) > 5:
                    print("Most incomplete sessions:")
                    for date, completeness in sorted_incomplete[:5]:
                        print(f"  {date}: {completeness:.2%} complete ({int(completeness * expected_minutes)} minutes)")
            else:
                print("All trading sessions are complete")
                
            # Remove the temporary date column if it wasn't in the original data
            if 'date' not in self.data.columns and 'date' in self.data.columns:
                self.data = self.data.drop(columns=['date'])
        else:
            print("Warning: Index is not a DatetimeIndex. Skipping trading session completeness check.")
            
        # Check for logical consistency in price data
        if all(col in self.data.columns for col in ['low', 'close', 'high', 'open']):
            price_consistency = (self.data['low'] <= self.data['close']) & \
                               (self.data['low'] <= self.data['open']) & \
                               (self.data['high'] >= self.data['close']) & \
                               (self.data['high'] >= self.data['open'])
            
            inconsistent_prices = ~price_consistency
            
            if inconsistent_prices.any():
                print(f"Warning: {inconsistent_prices.sum()} rows have inconsistent price data")
                print("Example of inconsistent price data:")
                print(self.data[inconsistent_prices].head())
        else:
            print("Skipping price consistency check - missing required columns")
            
        return self
    
    def save_processed_data(self):
        """
        Save the processed data to CSV.
        """
        # Reset index to make datetime a column again
        output_data = self.data.reset_index()
        
        # Save to CSV
        output_data.to_csv(self.output_file, index=False)
        print(f"Processed data saved to {self.output_file}")
        return self
    
    def decompose_seasonal(self, column='close', model='additive', period=None):
        """
        Apply classical seasonal decomposition to separate time series into trend, 
        seasonal, and residual components.
        
        Args:
            column (str): Column name to decompose
            model (str): Type of decomposition - 'additive' or 'multiplicative'
            period (int): Number of time points in a seasonal cycle, if None will be estimated
            
        Returns:
            self for method chaining
        """
        print(f"Performing seasonal decomposition on {column} using {model} model")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for decomposition")
            return self
        
        # Ensure the column exists
        if column not in self.data.columns:
            print(f"Column {column} not found in data")
            return self
        
        # Ensure no missing values in the column
        if self.data[column].isnull().any():
            print(f"Column {column} contains missing values. Filling with forward fill.")
            self.data[column] = self.data[column].fillna(method='ffill')
        
        # If period is not provided, try to estimate based on data frequency
        if period is None:
            # Try to infer the frequency and set appropriate period
            if self.data.index.freq is not None:
                freq_str = self.data.index.freq.name
                if 'D' in freq_str:  # Daily data
                    period = 5  # Trading week (5 days)
                elif 'H' in freq_str:  # Hourly data
                    period = 24  # Daily cycle (24 hours)
                elif 'T' in freq_str or 'min' in freq_str.lower():  # Minute data
                    period = 60  # Hourly cycle (60 minutes)
            else:
                # Default to daily trading week if we can't determine
                period = 5
                print(f"Could not determine data frequency. Using default period of {period}")
        
        print(f"Using period: {period} for decomposition")
        
        try:
            # Apply the decomposition
            result = seasonal_decompose(self.data[column], model=model, period=period)
            
            # Add the components as new columns
            self.data[f'{column}_trend'] = result.trend
            self.data[f'{column}_seasonal'] = result.seasonal
            self.data[f'{column}_residual'] = result.resid
            
            print(f"Decomposition successful. Added columns: {column}_trend, {column}_seasonal, {column}_residual")
            
        except Exception as e:
            print(f"Error during decomposition: {str(e)}")
        
        return self
    
    def decompose_stl(self, column='close', period=None, robust=True):
        """
        Apply STL (Seasonal-Trend decomposition using LOESS) to separate
        time series into trend, seasonal, and residual components.
        STL is more flexible than classical decomposition and can handle
        changing seasonal patterns better.
        
        Args:
            column (str): Column name to decompose
            period (int): Number of time points in a seasonal cycle, if None will be estimated
            robust (bool): Whether to use robust fitting to reduce outlier influence
            
        Returns:
            self for method chaining
        """
        print(f"Performing STL decomposition on {column}")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for decomposition")
            return self
        
        # Ensure the column exists
        if column not in self.data.columns:
            print(f"Column {column} not found in data")
            return self
        
        # Ensure no missing values in the column
        if self.data[column].isnull().any():
            print(f"Column {column} contains missing values. Filling with forward fill.")
            self.data[column] = self.data[column].fillna(method='ffill')
        
        # If period is not provided, try to estimate based on data frequency
        if period is None:
            # Try to infer the frequency and set appropriate period
            if self.data.index.freq is not None:
                freq_str = self.data.index.freq.name
                if 'D' in freq_str:  # Daily data
                    period = 5  # Trading week (5 days)
                elif 'H' in freq_str:  # Hourly data
                    period = 24  # Daily cycle (24 hours)
                elif 'T' in freq_str or 'min' in freq_str.lower():  # Minute data
                    period = 60  # Hourly cycle (60 minutes)
            else:
                # Default to daily trading week if we can't determine
                period = 5
                print(f"Could not determine data frequency. Using default period of {period}")
        
        print(f"Using period: {period} for STL decomposition")
        
        try:
            # Apply STL decomposition
            stl = STL(self.data[column], period=period, robust=robust)
            result = stl.fit()
            
            # Add the components as new columns
            self.data[f'{column}_stl_trend'] = result.trend
            self.data[f'{column}_stl_seasonal'] = result.seasonal
            self.data[f'{column}_stl_residual'] = result.resid
            
            # Calculate seasonality strength (optional)
            var_season = np.var(result.seasonal)
            var_resid = np.var(result.resid)
            if var_season + var_resid > 0:
                self.data[f'{column}_seasonality_strength'] = var_season / (var_season + var_resid)
            
            print(f"STL decomposition successful. Added columns: {column}_stl_trend, {column}_stl_seasonal, {column}_stl_residual")
        
        except Exception as e:
            print(f"Error during STL decomposition: {str(e)}")
        
        return self
    
    def decompose_mstl(self, column='close', periods=None):
        """
        Apply MSTL (Multiple Seasonal-Trend decomposition using LOESS) to separate
        time series with multiple seasonal patterns.
        Ideal for financial data that may have daily, weekly, and monthly patterns.
        
        Args:
            column (str): Column name to decompose
            periods (list): List of seasonal periods to decompose
                           e.g. [5, 22] for weekly and monthly trading patterns
            
        Returns:
            self for method chaining
        """
        print(f"Performing MSTL decomposition on {column}")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for decomposition")
            return self
        
        # Ensure the column exists
        if column not in self.data.columns:
            print(f"Column {column} not found in data")
            return self
        
        # Ensure no missing values in the column
        if self.data[column].isnull().any():
            print(f"Column {column} contains missing values. Filling with forward fill.")
            self.data[column] = self.data[column].fillna(method='ffill')
        
        # Set default periods for financial data if none provided
        if periods is None:
            # Default periods for financial data: 
            # 5 for weekly patterns (trading days)
            # 22 for monthly patterns (trading days)
            periods = [5, 22]
            print(f"Using default periods for financial data: {periods}")
        
        try:
            # Apply MSTL decomposition
            mstl = MSTL(self.data[column], periods=periods)
            result = mstl.fit()
            
            # Add the trend component
            self.data[f'{column}_mstl_trend'] = result.trend
            
            # Add each seasonal component
            for i, period in enumerate(periods):
                self.data[f'{column}_mstl_seasonal_{period}'] = result.seasonal[:, i]
            
            # Add the residual component
            self.data[f'{column}_mstl_residual'] = result.resid
            
            print(f"MSTL decomposition successful. Added trend, seasonal components for periods {periods}, and residual")
        
        except Exception as e:
            print(f"Error during MSTL decomposition: {str(e)}")
        
        return self
    
    def apply_fourier_transform(self, column='close', n_highest=10, min_period=5, max_period=None):
        """
        Apply Fast Fourier Transform (FFT) to identify cyclical patterns in the time series.
        Extract dominant frequency components and create features based on these components.
        
        Args:
            column (str): Column name to transform
            n_highest (int): Number of highest amplitude frequencies to extract as features
            min_period (int): Minimum period (in minutes) to consider for frequency analysis
            max_period (int): Maximum period (in minutes) to consider for frequency analysis
            
        Returns:
            self for method chaining
        """
        print(f"Applying Fourier Transform on {column}")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for Fourier Transform")
            return self
        
        # Ensure the column exists
        if column not in self.data.columns:
            print(f"Column {column} not found in data")
            return self
        
        # Handle missing values
        series = self.data[column].copy()
        if series.isnull().any():
            print(f"Column {column} contains missing values. Filling with forward fill.")
            series = series.fillna(method='ffill')
        
        # Remove trend to focus on cyclical patterns (optional)
        series_detrended = series - series.rolling(window=20).mean().fillna(method='bfill')
        
        try:
            # Apply FFT
            # For real-valued time series, rfft is more efficient than fft
            n = len(series_detrended)
            fft_values = rfft(series_detrended.values)
            fft_freqs = rfftfreq(n, d=1.0)  # Assuming 1-minute intervals
            
            # Calculate amplitudes
            amplitudes = np.abs(fft_values) / n  # Normalize
            
            # Convert frequencies to periods (in minutes)
            periods = np.zeros_like(fft_freqs)
            non_zero_freq = fft_freqs != 0
            periods[non_zero_freq] = 1.0 / fft_freqs[non_zero_freq]  # Period = 1/frequency
            
            # Filter periods based on min and max periods
            valid_indices = np.ones_like(periods, dtype=bool)
            if min_period is not None:
                valid_indices = valid_indices & (periods >= min_period)
            if max_period is not None:
                valid_indices = valid_indices & (periods <= max_period)
            
            # Filter out DC component (zero frequency)
            valid_indices = valid_indices & (fft_freqs > 0)
            
            # Get periods and amplitudes only for valid indices
            filtered_periods = periods[valid_indices]
            filtered_amplitudes = amplitudes[valid_indices]
            
            # Store results in the dataframe
            if len(filtered_periods) > 0:
                # Find the n_highest amplitudes and their corresponding periods
                if len(filtered_amplitudes) > n_highest:
                    top_indices = np.argsort(filtered_amplitudes)[-n_highest:]
                else:
                    top_indices = np.argsort(filtered_amplitudes)[-len(filtered_amplitudes):]
                
                top_periods = filtered_periods[top_indices]
                top_amplitudes = filtered_amplitudes[top_indices]
                
                # Create features for dominant cycles
                for i, (period, amplitude) in enumerate(zip(top_periods, top_amplitudes)):
                    # Round period to nearest minute for more intuitive naming
                    period_round = round(period)
                    
                    # Store period and amplitude as metadata or features
                    self.data[f'{column}_fft_period_{i+1}'] = period_round
                    self.data[f'{column}_fft_amplitude_{i+1}'] = amplitude
                    
                    # Create sine and cosine features for the top frequencies
                    # These capture the phase information and are useful for ML models
                    time_points = np.arange(len(series))
                    freq = 1.0 / period
                    self.data[f'{column}_fft_sin_{period_round}m'] = np.sin(2 * np.pi * freq * time_points)
                    self.data[f'{column}_fft_cos_{period_round}m'] = np.cos(2 * np.pi * freq * time_points)
                
                # Store the power spectrum for each frequency band
                # Common financial time frames
                freq_bands = {
                    'ultra_short': (1, 10),      # 1-10 minute cycles
                    'short': (10, 60),           # 10-60 minute cycles
                    'medium': (60, 240),         # 1-4 hour cycles
                    'long': (240, 1440),         # 4-24 hour cycles
                }
                
                for band_name, (lower, upper) in freq_bands.items():
                    band_mask = (filtered_periods >= lower) & (filtered_periods <= upper)
                    if np.any(band_mask):
                        band_power = np.sum(filtered_amplitudes[band_mask] ** 2)
                        self.data[f'{column}_fft_power_{band_name}'] = band_power
                
                # Reconstruct the signal using only the dominant frequencies
                if len(top_indices) > 0:
                    # Create a clean FFT array with only the dominant frequencies
                    clean_fft = np.zeros_like(fft_values, dtype=complex)
                    
                    for idx in top_indices:
                        # Find the original index in the full FFT array
                        orig_idx = np.where(valid_indices)[0][idx]
                        clean_fft[orig_idx] = fft_values[orig_idx]
                    
                    # Inverse FFT to reconstruct the signal
                    reconstructed = irfft(clean_fft, n=n)
                    self.data[f'{column}_fft_reconstructed'] = reconstructed
                    
                    # Calculate the reconstruction error
                    reconstruction_error = np.mean((series_detrended.values - reconstructed) ** 2)
                    self.data[f'{column}_fft_error'] = reconstruction_error
                
                print(f"Fourier Transform successful. Identified {len(top_periods)} dominant cycles.")
            else:
                print("No valid frequencies found within the specified period range.")
            
        except Exception as e:
            print(f"Error during Fourier Transform: {str(e)}")
        
        return self
    
    def apply_wavelet_transform(self, column='close', wavelet='db8', level=None, mode='symmetric'):
        """
        Apply Wavelet Transform to decompose the time series into different frequency components.
        Creates features based on the wavelet coefficients at different levels.
        
        Args:
            column (str): Column name to transform
            wavelet (str): Wavelet type to use (e.g., 'db4', 'sym8', 'haar')
            level (int): Decomposition level, if None will be automatically determined
            mode (str): Signal extension mode for boundary handling
            
        Returns:
            self for method chaining
        """
        print(f"Applying Wavelet Transform on {column} using {wavelet} wavelet")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for Wavelet Transform")
            return self
        
        # Ensure the column exists
        if column not in self.data.columns:
            print(f"Column {column} not found in data")
            return self
        
        # Handle missing values
        series = self.data[column].copy()
        if series.isnull().any():
            print(f"Column {column} contains missing values. Filling with forward fill.")
            series = series.fillna(method='ffill')
        
        try:
            # Determine the maximum decomposition level if not provided
            if level is None:
                # Maximum level depends on data length and wavelet
                level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)
                # For 1-minute data, limit to reasonable levels
                level = min(level, 6)  # Limit to 6 levels for performance
                print(f"Using automatic wavelet decomposition level: {level}")
            
            # Apply multilevel wavelet decomposition
            coeffs = pywt.wavedec(series.values, wavelet, mode=mode, level=level)
            
            # Extract approximation and detail coefficients
            approx = coeffs[0]  # Approximation coefficients (lowest frequency)
            details = coeffs[1:]  # Detail coefficients (higher frequencies)
            
            # Create features based on wavelet coefficients
            
            # 1. Add the approximation coefficients as a feature (smoothed signal)
            # This requires upsampling the coefficients to match the original length
            self.data[f'{column}_wavelet_approx'] = pywt.upcoef('a', approx, wavelet, level=level, take=len(series))
            
            # 2. Add each level of detail coefficients as features
            for i, detail in enumerate(details):
                detail_level = level - i
                # Upsample the coefficients to match the original length
                detail_upsampled = pywt.upcoef('d', detail, wavelet, level=detail_level, take=len(series))
                self.data[f'{column}_wavelet_detail_{detail_level}'] = detail_upsampled
            
            # 3. Calculate energy of the coefficients at each level
            # Approximation energy
            self.data[f'{column}_wavelet_energy_approx'] = np.sum(approx**2) / len(approx)
            
            # Detail energies
            for i, detail in enumerate(details):
                detail_level = level - i
                self.data[f'{column}_wavelet_energy_detail_{detail_level}'] = np.sum(detail**2) / len(detail)
            
            # 4. Relative energies (percentage of total energy)
            total_energy = self.data[f'{column}_wavelet_energy_approx']
            for i in range(level):
                detail_level = level - i
                total_energy += self.data[f'{column}_wavelet_energy_detail_{detail_level}']
            
            self.data[f'{column}_wavelet_rel_energy_approx'] = self.data[f'{column}_wavelet_energy_approx'] / total_energy
            for i in range(level):
                detail_level = level - i
                self.data[f'{column}_wavelet_rel_energy_detail_{detail_level}'] = self.data[f'{column}_wavelet_energy_detail_{detail_level}'] / total_energy
            
            # 5. Entropy of the wavelet coefficients
            # Higher entropy indicates more disorder/complexity
            def calculate_entropy(coeffs):
                coeffs = np.abs(coeffs)
                norm_coeffs = coeffs / np.sum(coeffs)
                entropy = -np.sum(norm_coeffs * np.log2(norm_coeffs + np.finfo(float).eps))
                return entropy
            
            self.data[f'{column}_wavelet_entropy_approx'] = calculate_entropy(approx)
            for i, detail in enumerate(details):
                detail_level = level - i
                self.data[f'{column}_wavelet_entropy_detail_{detail_level}'] = calculate_entropy(detail)
            
            # 6. Denoised signal using wavelet thresholding
            # Using soft thresholding to remove noise while preserving signal features
            denoised_coeffs = coeffs.copy()
            # Apply threshold only to detail coefficients
            for i in range(1, len(denoised_coeffs)):
                denoised_coeffs[i] = pywt.threshold(denoised_coeffs[i], value=np.std(denoised_coeffs[i])/2, mode='soft')
            
            # Reconstruct the denoised signal
            denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
            # Handle potential length mismatch due to wavelet transform
            if len(denoised_signal) > len(series):
                denoised_signal = denoised_signal[:len(series)]
            self.data[f'{column}_wavelet_denoised'] = denoised_signal
            
            # 7. Calculate wavelet-based volatility measures
            # Higher details (higher frequencies) often correlate with volatility
            for i in range(min(3, level)):  # Use top 3 detail levels or fewer if level < 3
                detail_level = level - i
                detail_name = f'{column}_wavelet_detail_{detail_level}'
                detail_values = self.data[detail_name]
                
                # Rolling standard deviation of the detail coefficients
                window_sizes = [5, 10, 20]  # For 1-minute data: 5min, 10min, 20min windows
                for window in window_sizes:
                    self.data[f'{detail_name}_volatility_{window}m'] = detail_values.rolling(window=window).std().fillna(0)
            
            print(f"Wavelet Transform successful. Created {level + 1} decomposition levels.")
            
        except Exception as e:
            print(f"Error during Wavelet Transform: {str(e)}")
        
        return self
    
    def add_microstructure_features(self, window_sizes=[5, 10, 20, 30]):
        """
        Add market microstructure features that are particularly useful for high-frequency (1-min) data.
        These features capture order flow imbalance, price impact, and other microstructure effects.
        
        Args:
            window_sizes (list): List of window sizes for rolling calculations
            
        Returns:
            self for method chaining
        """
        print("Computing market microstructure features")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for feature computation")
            return self
            
        # Check if required OHLCV columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            print(f"Missing required columns: {missing}")
            return self
            
        try:
            # 1. Order Flow Imbalance (OFI)
            # This estimates buy/sell pressure by looking at price changes relative to a reference price
            for w in window_sizes:
                mid_price = (self.data['high'] + self.data['low']) / 2
                self.data[f'price_direction_{w}'] = np.sign(self.data['close'].diff(1)).fillna(0)
                self.data[f'ofi_{w}'] = (self.data[f'price_direction_{w}'] * 
                                      self.data['volume']).rolling(window=w).sum() / \
                                      self.data['volume'].rolling(window=w).sum()
                
            # 2. Volume Imbalance
            # Compares the volume when price is rising vs falling
            for w in window_sizes:
                up_volume = (self.data['close'] > self.data['close'].shift(1)) * self.data['volume']
                down_volume = (self.data['close'] < self.data['close'].shift(1)) * self.data['volume']
                self.data[f'volume_imbalance_{w}'] = ((up_volume - down_volume) / 
                                                   (up_volume + down_volume).replace(0, 1)
                                                  ).rolling(window=w).mean()
                
            # 3. Weighted Price Range
            # High-low range weighted by volume
            for w in window_sizes:
                self.data[f'weighted_range_{w}'] = ((self.data['high'] - self.data['low']) / 
                                                 self.data['close']) * np.log1p(self.data['volume'])
                self.data[f'weighted_range_{w}'] = self.data[f'weighted_range_{w}'].rolling(window=w).mean()
                
            # 4. Intrabar Intensity
            # Measures the trading intensity within bars
            self.data['bar_range'] = self.data['high'] - self.data['low']
            for w in window_sizes:
                self.data[f'intrabar_intensity_{w}'] = (self.data['bar_range'] * 
                                                     self.data['volume']).rolling(window=w).mean()
                                                     
            # 5. Price impact
            # How much volume is needed to move the price
            for w in window_sizes:
                self.data[f'price_impact_{w}'] = (np.abs(self.data['close'].pct_change()) / 
                                               np.log1p(self.data['volume'])).rolling(window=w).mean()
                                               
            print(f"Added market microstructure features with window sizes: {window_sizes}")
            
        except Exception as e:
            print(f"Error computing market microstructure features: {str(e)}")
            
        return self
        
    def add_adaptive_features(self, column='close', n_clusters=5, window_sizes=[20, 60, 120, 240, 480]):
        """
        Add regime-adaptive features that automatically adjust to different market conditions
        using clustering techniques to identify distinct market regimes.
        
        Args:
            column (str): Column name to use as primary input
            n_clusters (int): Number of regimes/clusters to identify
            window_sizes (list): List of window sizes for rolling calculations
            
        Returns:
            self for method chaining
        """
        print(f"Computing adaptive regime-based features on {column}")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for feature computation")
            return self
            
        # Ensure the column exists
        if column not in self.data.columns:
            print(f"Column {column} not found in data")
            return self
            
        try:
            # Create regime detection features
            feature_set = pd.DataFrame(index=self.data.index)
            
            # 1. Extract features for regime identification
            for w in window_sizes:
                # Returns
                feature_set[f'return_{w}'] = self.data[column].pct_change(w).fillna(0)
                
                # Volatility
                feature_set[f'volatility_{w}'] = self.data[column].pct_change().rolling(w).std().fillna(0)
                
                # Range ratio
                if 'high' in self.data.columns and 'low' in self.data.columns:
                    high_low_ratio = (self.data['high'].rolling(w).max() - 
                                      self.data['low'].rolling(w).min()) / self.data[column]
                    feature_set[f'range_ratio_{w}'] = high_low_ratio.fillna(0)
                    
                # Volume acceleration if available
                if 'volume' in self.data.columns:
                    feature_set[f'volume_change_{w}'] = (self.data['volume'].pct_change(w).fillna(0))
            
            # Remove NaN values for clustering
            regime_features = feature_set.fillna(0)
            
            # 2. Normalize features for clustering
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(regime_features)
            
            # 3. Identify market regimes using clustering
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                regimes = kmeans.fit_predict(normalized_features)
                
            # 4. Add regime labels to dataframe
            self.data['market_regime'] = regimes
            
            # 5. Create regime-specific features
            for i in range(n_clusters):
                # Regime indicator (one-hot encoding)
                self.data[f'regime_{i}'] = (self.data['market_regime'] == i).astype(int)
                
                # Interaction features - important price-volume interactions for each regime
                if 'volume' in self.data.columns:
                    self.data[f'pv_ratio_regime_{i}'] = self.data[f'regime_{i}'] * \
                                                     (self.data[column] / self.data['volume'].replace(0, 1))
                                                     
            # 6. PCA-based features from each regime
            pca = PCA(n_components=min(3, len(regime_features.columns)))
            pca_features = pca.fit_transform(normalized_features)
            
            for i in range(pca_features.shape[1]):
                self.data[f'market_pca_{i+1}'] = pca_features[:, i]
                
            print(f"Added adaptive regime-based features with {n_clusters} market regimes")
            
        except Exception as e:
            print(f"Error computing adaptive features: {str(e)}")
            
        return self
        
    def add_information_theory_features(self, column='close', windows=[20, 60, 120, 240]):
        """
        Add features based on information theory and complexity measures,
        which are particularly useful for capturing non-linear patterns in financial time series.
        
        Args:
            column (str): Column name to use as primary input
            windows (list): List of window sizes for rolling calculations
            
        Returns:
            self for method chaining
        """
        print(f"Computing information theory features on {column}")
        
        # Ensure data exists
        if self.data is None or self.data.empty:
            print("No data available for feature computation")
            return self
            
        # Ensure the column exists
        if column not in self.data.columns:
            print(f"Column {column} not found in data")
            return self
            
        try:
            # 1. Approximate Entropy (measure of regularity and unpredictability)
            for w in windows:
                returns = self.data[column].pct_change().fillna(0)
                
                # Calculate entropy for rolling windows
                def rolling_entropy(x):
                    # Discretize the values into bins
                    try:
                        hist, _ = np.histogram(x, bins=10, density=True)
                        # Filter out zeros
                        hist = hist[hist > 0]
                        return entropy(hist)
                    except:
                        return 0
                
                self.data[f'return_entropy_{w}'] = returns.rolling(window=w).apply(
                    rolling_entropy, raw=True).fillna(0)
                
            # 2. Higher-order moments (capture distribution shape)
            for w in windows:
                # Skewness - measures asymmetry
                self.data[f'skew_{w}'] = self.data[column].rolling(window=w).apply(
                    lambda x: skew(x), raw=True).fillna(0)
                
                # Kurtosis - measures "tailedness"
                self.data[f'kurtosis_{w}'] = self.data[column].rolling(window=w).apply(
                    lambda x: kurtosis(x), raw=True).fillna(0)
            
            # 3. Permutation Entropy (captures order patterns)
            for w in windows:
                if w >= 5:  # Minimum window size for meaningful permutation entropy
                    def perm_entropy(x, order=3):
                        """Calculate permutation entropy"""
                        try:
                            if len(x) < order + 1:
                                return 0
                            
                            # Extract all sequential patterns of length 'order'
                            patterns = []
                            for i in range(len(x) - order):
                                pattern = x[i:i+order]
                                # Convert to ranks
                                ranked = np.argsort(np.argsort(pattern))
                                patterns.append(''.join(map(str, ranked)))
                            
                            # Count unique patterns
                            _, counts = np.unique(patterns, return_counts=True)
                            probs = counts / len(patterns)
                            return entropy(probs)
                        except:
                            return 0
                    
                    self.data[f'perm_entropy_{w}'] = self.data[column].rolling(window=w).apply(
                        lambda x: perm_entropy(x), raw=True).fillna(0)
            
            # 4. Mutual Information between price and volume (if available)
            if 'volume' in self.data.columns:
                for w in windows:
                    def rolling_mutual_info(window_data):
                        try:
                            # Extract price and volume from the window
                            price = window_data[column].values.reshape(-1, 1)
                            volume = window_data['volume'].values.reshape(-1, 1)
                            if len(price) < 5:  # Minimum required samples
                                return 0
                            return mutual_info_regression(price, volume.ravel())[0]
                        except:
                            return 0
                    
                    # Use rolling apply on the entire dataframe
                    mi_values = []
                    for i in range(len(self.data)):
                        if i < w:
                            mi_values.append(0)
                        else:
                            window_data = self.data.iloc[i-w:i]
                            try:
                                mi = rolling_mutual_info(window_data)
                                mi_values.append(mi)
                            except:
                                mi_values.append(0)
                    
                    self.data[f'price_volume_mi_{w}'] = mi_values
            
            # 5. Peak detection for local extrema
            returns = self.data[column].pct_change().fillna(0)
            for w in windows:
                def peak_density(x):
                    try:
                        peaks, _ = find_peaks(x)
                        valleys, _ = find_peaks(-x)
                        return (len(peaks) + len(valleys)) / len(x)
                    except:
                        return 0
                
                self.data[f'peak_density_{w}'] = returns.rolling(window=w).apply(
                    peak_density, raw=True).fillna(0)
            
            print(f"Added information theory features with window sizes: {windows}")
            
        except Exception as e:
            print(f"Error computing information theory features: {str(e)}")
            
        return self
    
    def select_best_features(self, target='close', n_features=20, method='ensemble', future_horizon=1, cv=5):
        """
        Select the best features for predicting the target variable.
        
        Args:
            target (str): Target column to predict (typically 'close')
            n_features (int): Number of features to select
            method (str): Feature selection method to use:
                - 'correlation': Use correlation with target
                - 'mutual_info': Use mutual information
                - 'random_forest': Use Random Forest feature importance
                - 'lasso': Use Lasso regression (L1 regularization)
                - 'rfe': Recursive Feature Elimination
                - 'ensemble': Use an ensemble of methods (recommended)
            future_horizon (int): Number of periods ahead to predict
            cv (int): Number of cross-validation folds for time series
            
        Returns:
            pd.DataFrame: Data with only the selected features
        """
        print(f"Starting feature selection using '{method}' method...")
        
        # Create future target if needed
        if future_horizon > 0:
            future_target = f'{target}_future_{future_horizon}'
            if future_target not in self.data.columns:
                print(f"Creating future target {future_target}...")
                self.data[future_target] = self.data[target].shift(-future_horizon)
        else:
            future_target = target
        
        # Remove rows with NaN in target
        data_no_na = self.data.dropna(subset=[future_target])
        
        # Check if we have enough data for feature selection
        if len(data_no_na) < n_features * 2:
            warnings.warn(f"Not enough samples ({len(data_no_na)}) for feature selection with {n_features} features!")
            return self.data
        
        # Exclude non-feature columns
        exclude_cols = []
        
        # Exclude date column if present
        if 'date' in data_no_na.columns:
            exclude_cols.append('date')
            
        # Exclude the target and future target
        exclude_cols.append(target)
        if future_horizon > 0:
            exclude_cols.append(future_target)
            
        # Get feature names
        feature_names = [col for col in data_no_na.columns if col not in exclude_cols]
        
        # Handle case where we have fewer features than requested
        if len(feature_names) <= n_features:
            print(f"Only {len(feature_names)} features available, using all of them")
            self.selected_features = feature_names
            return self.data
            
        # Prepare data for feature selection
        X = data_no_na[feature_names]
        y = data_no_na[future_target]
        
        # Replace any infinity values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values
        X = X.fillna(X.mean()).fillna(0)  # First try mean, then 0 for columns that are all NaN
        
        # Check for any remaining problematic values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: Data still contains NaN or infinity values after cleaning")
            # Replace any remaining problematic values with 0
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select features based on method
        selected_cols = []
        
        if method == 'correlation':
            # Correlation-based selection
            corr_scores = []
            for col_idx, col_name in enumerate(X.columns):
                correlation = abs(np.corrcoef(X_scaled[:, col_idx], y)[0, 1])
                if not np.isnan(correlation):
                    corr_scores.append((col_name, correlation))
            
            # Sort by correlation (descending)
            corr_scores.sort(key=lambda x: x[1], reverse=True)
            selected_cols = [item[0] for item in corr_scores[:n_features]]
            
            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'feature': [item[0] for item in corr_scores],
                'importance': [item[1] for item in corr_scores]
            }).sort_values('importance', ascending=False)
            
        elif method == 'mutual_info':
            # Mutual information-based selection
            selector = SelectKBest(mutual_info_regression, k=n_features)
            selector.fit(X_scaled, y)
            selected_mask = selector.get_support()
            selected_cols = X.columns[selected_mask].tolist()
            
            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns.tolist(),
                'importance': selector.scores_
            }).sort_values('importance', ascending=False)
            
        elif method == 'random_forest':
            # Random Forest-based selection
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=self.n_jobs)
            rf.fit(X_scaled, y)
            
            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns.tolist(),
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_cols = self.feature_importance['feature'].iloc[:n_features].tolist()
            
        elif method == 'lasso':
            # Lasso-based selection (finds optimal alpha using cross-validation)
            lasso_cv = LassoCV(cv=tscv, random_state=42, max_iter=10000)
            lasso_cv.fit(X_scaled, y)
            
            # Use optimal alpha for feature selection
            lasso = Lasso(alpha=lasso_cv.alpha_, random_state=42, max_iter=10000)
            lasso.fit(X_scaled, y)
            
            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns.tolist(),
                'importance': np.abs(lasso.coef_)
            }).sort_values('importance', ascending=False)
            
            selected_cols = self.feature_importance['feature'].iloc[:n_features].tolist()
            
        elif method == 'rfe':
            # Recursive Feature Elimination
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=self.n_jobs)
            rfe = RFE(estimator=rf, n_features_to_select=n_features, step=3)
            rfe.fit(X_scaled, y)
            
            # Store feature importance (binary mask)
            feature_ranking = pd.DataFrame({
                'feature': X.columns.tolist(),
                'ranking': rfe.ranking_
            }).sort_values('ranking')
            
            self.feature_importance = feature_ranking.copy()
            self.feature_importance['importance'] = 1 / self.feature_importance['ranking']
            self.feature_importance = self.feature_importance.sort_values('importance', ascending=False)
            
            selected_cols = X.columns[rfe.support_].tolist()
            
        elif method == 'ensemble':
            # Ensemble method - combine multiple feature selection techniques
            # 1. Get features from correlation
            corr_scores = []
            for col_idx, col_name in enumerate(X.columns):
                correlation = abs(np.corrcoef(X_scaled[:, col_idx], y)[0, 1])
                if not np.isnan(correlation):
                    corr_scores.append((col_name, correlation))
            corr_scores.sort(key=lambda x: x[1], reverse=True)
            corr_features = set([item[0] for item in corr_scores[:n_features]])
            
            # 2. Get features from mutual information
            mi_selector = SelectKBest(mutual_info_regression, k=n_features)
            mi_selector.fit(X_scaled, y)
            mi_features = set(X.columns[mi_selector.get_support()].tolist())
            
            # 3. Get features from random forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=self.n_jobs)
            rf.fit(X_scaled, y)
            rf_importance = pd.DataFrame({
                'feature': X.columns.tolist(),
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            rf_features = set(rf_importance['feature'].iloc[:n_features].tolist())
            
            # Combine results using a voting system
            feature_votes = {}
            for feature in feature_names:
                feature_votes[feature] = 0
                if feature in corr_features:
                    feature_votes[feature] += 1
                if feature in mi_features:
                    feature_votes[feature] += 1
                if feature in rf_features:
                    feature_votes[feature] += 1
            
            # Sort by votes and then by random forest importance (as tiebreaker)
            feature_rank = []
            for feature, votes in feature_votes.items():
                if feature in rf_importance['feature'].values:
                    rf_imp = rf_importance.loc[rf_importance['feature'] == feature, 'importance'].values[0]
                else:
                    rf_imp = 0
                feature_rank.append((feature, votes, rf_imp))
            
            # Sort by votes (descending) and then by RF importance (descending)
            feature_rank.sort(key=lambda x: (x[1], x[2]), reverse=True)
            selected_cols = [item[0] for item in feature_rank[:n_features]]
            
            # Store feature importance
            self.feature_importance = pd.DataFrame({
                'feature': [item[0] for item in feature_rank],
                'votes': [item[1] for item in feature_rank],
                'rf_importance': [item[2] for item in feature_rank]
            })
            self.feature_importance['normalized_importance'] = (
                self.feature_importance['votes'] + 
                self.feature_importance['rf_importance'] / self.feature_importance['rf_importance'].max()
            )
            # Set 'importance' column for consistent interface with other methods
            self.feature_importance['importance'] = self.feature_importance['normalized_importance']
            self.feature_importance = self.feature_importance.sort_values('normalized_importance', ascending=False)
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Store selected feature names for reference
        self.selected_features = selected_cols
        
        # Keep only the selected features (plus date and target columns)
        keep_cols = selected_cols.copy()
        
        # Always keep date and target columns
        if 'date' in self.data.columns:
            keep_cols.append('date')
        
        # Keep original target and future target
        keep_cols.append(target)
        if future_horizon > 0:
            keep_cols.append(future_target)
        
        # Make sure we don't have duplicates
        keep_cols = list(dict.fromkeys(keep_cols))
        
        # Create a new dataframe with only selected features
        selected_data = self.data[keep_cols].copy()
        
        # Print feature importance ranking
        print("\nTop 10 Selected Features:")
        for i, (feature, importance) in enumerate(zip(
            self.feature_importance['feature'].iloc[:min(10, len(self.feature_importance))],
            self.feature_importance['importance'].iloc[:min(10, len(self.feature_importance))]
        )):
            print(f"{i+1}. {feature}: {importance:.6f}")
        
        print(f"\nSelected {len(selected_cols)} features out of {X.shape[1]} original features.")
        
        # Update the data with selected features
        self.data = selected_data
        
        return self.data
    
    def analyze_feature_importance(self, plot=True, save_path=None):
        """
        Analyze and optionally plot feature importance from the last feature selection.
        
        Args:
            plot (bool): Whether to generate a plot
            save_path (str): Path to save the plot (if None, will display instead)
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance data available. Run select_best_features first.")
        
        # Print the full feature importance ranking
        print("\nFeature Importance Ranking:")
        for i, row in self.feature_importance.head(20).iterrows():
            print(f"{i+1}. {row['feature']}: {row['importance']:.6f}")
        
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Get top 20 features
            top_features = self.feature_importance.head(20).copy()
            
            # Create horizontal bar plot
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Top 20 Features by Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
        
        return self.feature_importance
    
    def evaluate_feature_selection(self, target='close', future_horizon=1, cv=5):
        """
        Evaluate the feature selection by comparing prediction performance
        before and after feature selection.
        
        Args:
            target (str): Target column to predict (typically 'close')
            future_horizon (int): Number of periods ahead to predict
            cv (int): Number of cross-validation folds for time series
            
        Returns:
            dict: Performance metrics before and after feature selection
        """
        if self.selected_features is None:
            print("No feature selection has been performed yet.")
            return {}
        
        print("Evaluating feature selection performance...")
        
        try:
            # Create target variable for future prediction
            future_target = f'{target}_future_{future_horizon}'
            if future_target not in self.data.columns:
                # Calculate future target (shift target column backward)
                self.data[future_target] = self.data[target].shift(-future_horizon)
            
            # Exclude non-feature columns like 'date' if present
            exclude_cols = ['date']
            exclude_cols = [col for col in exclude_cols if col in self.data.columns]
            
            # Ensure we exclude the future target as well
            exclude_cols.append(future_target)
            
            # Get X and y for modeling
            all_features = [col for col in self.data.columns if col not in exclude_cols and col != target]
            
            # Filter out rows with NaN in target
            valid_data = self.data.dropna(subset=[future_target])
            
            X_all = valid_data[all_features]
            y = valid_data[future_target]
            
            # Get the selected features
            X_selected = valid_data[self.selected_features]
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv)
            
            # Model to use for evaluation
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=self.n_jobs)
            
            # Evaluate with all features
            all_scores = []
            for train_idx, test_idx in tscv.split(X_all):
                X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                all_scores.append(score)
            
            # Evaluate with selected features
            selected_scores = []
            for train_idx, test_idx in tscv.split(X_selected):
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                selected_scores.append(score)
            
            # Report results
            avg_all = np.mean(all_scores)
            avg_selected = np.mean(selected_scores)
            
            print(f"Average R² with all features ({len(all_features)}): {avg_all:.4f}")
            print(f"Average R² with selected features ({len(self.selected_features)}): {avg_selected:.4f}")
            
            if avg_selected >= avg_all:
                print(f"Feature selection improved performance by {(avg_selected - avg_all) / abs(avg_all):.2%}")
            else:
                print(f"Feature selection reduced performance by {(avg_all - avg_selected) / abs(avg_all):.2%}")
                print("Consider using more features or a different selection method.")
            
            return {
                'all_features_r2': avg_all,
                'selected_features_r2': avg_selected,
                'improvement': avg_selected - avg_all,
                'relative_improvement': (avg_selected - avg_all) / abs(avg_all) if avg_all != 0 else 0
            }
            
        except Exception as e:
            print(f"Warning: Could not evaluate feature selection: {str(e)}")
            return {}
    
    def run_pipeline(self, filter_hours=True, handle_missing=True, detect_outliers=True, 
                    compute_features=True, validate=True, feature_selection=True,
                    target='close', n_features=20, feature_method='ensemble',
                    future_horizon=1, save=True):
        """
        Run the complete data processing pipeline.
        
        Args:
            filter_hours (bool): Whether to filter for trading hours
            handle_missing (bool): Whether to handle missing data
            detect_outliers (bool): Whether to detect and handle outliers
            compute_features (bool): Whether to compute technical features
            validate (bool): Whether to validate the data
            feature_selection (bool): Whether to perform feature selection
            target (str): Target column for feature selection
            n_features (int): Number of features to select
            feature_method (str): Feature selection method
            future_horizon (int): Number of periods ahead to predict
            save (bool): Whether to save the processed data
            
        Returns:
            pd.DataFrame: Processed data
        """
        print("Starting data processing pipeline...")
        
        # Load data if not already loaded
        if self.data is None:
            try:
                # First, peek at the CSV headers without parsing dates
                data_peek = pd.read_csv(self.input_file, nrows=0)
                columns = data_peek.columns.tolist()
                
                # Auto-detect date column if not specified
                if self.date_column is None:
                    # Check for common date column names
                    date_column_options = ['datetime', 'date', 'timestamp', 'time', 'Date', 'DateTime', 'Timestamp', 'Time']
                    for col in date_column_options:
                        if col in columns:
                            self.date_column = col
                            print(f"Auto-detected date column: '{self.date_column}'")
                            break
                    
                    if self.date_column is None:
                        print("WARNING: No date column detected. Using the first column as index.")
                        self.date_column = columns[0]
                
                print(f"Loading data from {self.input_file}")
                self.data = pd.read_csv(self.input_file, parse_dates=[self.date_column])
                
                # Set date column as index and handle timezone information
                self.data.set_index(self.date_column, inplace=True)
                
                # Check if the index is already a DatetimeIndex
                if not isinstance(self.data.index, pd.DatetimeIndex):
                    print("Converting index to DatetimeIndex")
                    # Convert to DatetimeIndex with proper timezone handling
                    self.data.index = pd.to_datetime(self.data.index, utc=True)
                
                # Handle timezone information
                eastern = pytz.timezone('US/Eastern')
                if self.data.index.tzinfo is not None:
                    # Convert to Eastern Time for financial data
                    print("Converting timezone from UTC to Eastern Time")
                    self.data.index = self.data.index.tz_convert(eastern)
                else:
                    # Localize naive timestamps
                    print("Localizing timestamps to Eastern Time")
                    self.data.index = self.data.index.tz_localize(eastern, ambiguous='infer')
                    
                print(f"Data loaded with shape: {self.data.shape}")
                
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                return None
        
        # Filter for trading hours
        if filter_hours:
            self.filter_trading_hours()
        
        # Handle missing data
        if handle_missing:
            self.handle_missing_data()
        
        # Detect outliers
        if detect_outliers:
            self.detect_outliers()
        
        # Compute features
        if compute_features:
            self.compute_features()
        
        # Validate data
        if validate:
            self.validate_data()
        
        # Feature selection
        if feature_selection:
            self.select_best_features(
                target=target,
                n_features=n_features,
                method=feature_method,
                future_horizon=future_horizon
            )
            
            # Evaluate feature selection
            try:
                self.evaluate_feature_selection(
                    target=target,
                    future_horizon=future_horizon
                )
            except Exception as e:
                print(f"Warning: Could not evaluate feature selection: {e}")
        
        # Save processed data
        if save:
            self.save_processed_data()
        
        print("Data processing pipeline completed successfully.")
        return self.data

# Optimized functions for the existing methods (examples)
def _parallel_entropy(data_chunks):
    """Helper function for parallel entropy calculation"""
    results = []
    for chunk in data_chunks:
        # Discretize and calculate entropy
        hist, _ = np.histogram(chunk, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zeros
        results.append(entropy(hist))
    return results

def _chunk_data(data, n_chunks):
    """Split data into chunks for parallel processing"""
    chunk_size = len(data) // n_chunks
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

# Example usage
if __name__ == "__main__":
    # Define input and output files
    input_file = "data/processed/SPY.csv"
    output_file = "data/processed/SPY_featured.csv"
    
    # Create processor and run pipeline
    processor = FinancialDataProcessor(input_file, output_file)
    processor.run_pipeline(target='close', n_features=20) 