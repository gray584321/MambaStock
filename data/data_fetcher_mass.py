import os
import csv
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
from tqdm import tqdm
import pandas as pd
import argparse

"""
data_fetcher_mass.py

This script is responsible for fetching historical aggregated stock data
for each active ticker retrieved from Polygon.io's REST API.

Key Features:
  - Fetches active tickers from Polygon.io.
  - For each active ticker, it retrieves 5 years of historical data in interval-optimized chunks.
  - Supports multiple time intervals: 1min, 5min, 15min, 1hour, 1day.
  - Dynamically adjusts chunk sizes and API rate limits based on the selected interval.
  - Converts API timestamps (provided in UTC) to Eastern Time (America/New_York) before saving.
  - Saves the data to CSV files based on the requested interval.
  - If a CSV file for a ticker already exists, the script resumes fetching from one time unit after the last
    recorded timestamp.
  - Provides functionality to check for and fill missing intraday data.
  
Ensure that a .env file with POLYGON_API_KEY is present
"""

# Initialize logging and load environment once
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
load_dotenv()

# Constants
API_BASE_URL = "https://api.polygon.io/v2"
EASTERN_TZ = pytz.timezone("America/New_York")
UTC_TZ = pytz.UTC
REQUEST_TIMEOUT = 30  # seconds

# Valid time intervals with optimized settings
VALID_INTERVALS = {
    "1min": {
        "multiplier": 1, 
        "timespan": "minute", 
        "folder_suffix": "",
        "chunk_size": 30,        # 30 days for 1min data (avg ~7K-10K records per chunk)
        "rate_limit_sleep": 12,  # seconds to sleep after a rate limit hit
        "recent_days_check": 7,  # check last 7 days for recent data
        "max_workers": 12        # fewer parallel workers due to more API calls
    },
    "5min": {
        "multiplier": 5, 
        "timespan": "minute", 
        "folder_suffix": "_5min",
        "chunk_size": 90,        # 90 days for 5min data (avg ~3K-4K records per chunk)
        "rate_limit_sleep": 10,
        "recent_days_check": 14,
        "max_workers": 16
    },
    "15min": {
        "multiplier": 15, 
        "timespan": "minute", 
        "folder_suffix": "_15min",
        "chunk_size": 180,       # 180 days for 15min data (avg ~2K-3K records per chunk)
        "rate_limit_sleep": 8,
        "recent_days_check": 21,
        "max_workers": 20
    },
    "1hour": {
        "multiplier": 1, 
        "timespan": "hour", 
        "folder_suffix": "_1hour",
        "chunk_size": 365,       # 365 days for 1hour data (avg ~1.5K-2K records per chunk)
        "rate_limit_sleep": 5,
        "recent_days_check": 30,
        "max_workers": 24
    },
    "1day": {
        "multiplier": 1, 
        "timespan": "day", 
        "folder_suffix": "_1day",
        "chunk_size": 1825,      # Full 5 years for daily data (only ~1K-1.3K records total)
        "rate_limit_sleep": 3,
        "recent_days_check": 60,
        "max_workers": 30
    }
}

def get_date_range():
    """Calculate date range for data fetching."""
    today = datetime.now(UTC_TZ).date()
    end_date = today  # use today's date to get the absolute latest data available
    start_date = end_date - timedelta(days=365*5)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

START_DATE, END_DATE = get_date_range()

class PolygonAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        # Optimize connection pool based on the maximum number of workers we might use
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=200,
            pool_maxsize=200,
            max_retries=3,
            pool_block=True
        )
        self.session = requests.Session()
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "polygon-data-fetcher/1.0"
        })

    def get_active_tickers(self):
        """Fetch active American tickers using session for connection pooling."""
        url = 'https://api.polygon.io/v3/reference/tickers'
        params = {
            'market': 'stocks',
            'active': 'true',
            'locale': 'us',           # Ensure we target American stocks
            'limit': 1000,
            'apiKey': self.api_key
        }
        
        all_tickers = []
        with tqdm(desc="Fetching American tickers", unit=" tickers") as pbar:
            while True:
                try:
                    response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                    if response.status_code == 429:
                        time.sleep(12)  # Sleep for 12 seconds for rate limits during ticker fetching
                        continue
                    
                    data = response.json()
                    results = data.get('results', [])
                    if not results:
                        break
                        
                    new_tickers = [ticker['ticker'] for ticker in results]
                    all_tickers.extend(new_tickers)
                    pbar.update(len(new_tickers))
                    
                    next_url = data.get('next_url')
                    if not next_url:
                        break
                        
                    url = next_url.split('&apiKey=')[0]
                    params = {'apiKey': self.api_key}
                    
                except Exception as e:
                    logger.error(f"Error fetching tickers: {e}")
                    time.sleep(5)
                    
        return all_tickers

    def fetch_ticker_data(self, ticker, start_date, end_date, interval="1min"):
        """Fetch data for a single ticker chunk with improved error handling."""
        interval_info = VALID_INTERVALS.get(interval, VALID_INTERVALS["1min"])
        multiplier = interval_info["multiplier"]
        timespan = interval_info["timespan"]
        rate_limit_sleep = interval_info["rate_limit_sleep"]
        
        url = f"{API_BASE_URL}/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            "limit": 50000,  # Maximum allowed by Polygon
            "sort": "asc",
            "apiKey": self.api_key
        }
        
        retries = 3
        while retries > 0:
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                
                if response.status_code == 429:  # Rate limit
                    time.sleep(rate_limit_sleep)
                    continue
                elif response.status_code != 200:
                    logger.warning(f"Error {response.status_code} fetching {ticker} ({interval}): {response.text}")
                    retries -= 1
                    time.sleep(5)
                    continue
                
                data = response.json()
                
                # Check if we need to handle pagination
                results = data.get('results', [])
                if len(results) == 50000:  # We hit the limit, might need pagination
                    logger.info(f"Hit 50K limit for {ticker} ({interval}) from {start_date} to {end_date}, consider smaller chunks")
                
                return results
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {ticker} ({interval}): {str(e)}")
                retries -= 1
                time.sleep(5)
                
        return []  # Return empty list if all retries failed

class DataManager:
    def __init__(self, api, interval="1min"):
        self.api = api
        self.interval = interval
        interval_info = VALID_INTERVALS.get(interval, VALID_INTERVALS["1min"])
        folder_suffix = interval_info["folder_suffix"]
        
        # For 1min data, keep using the original "raw" folder
        if folder_suffix:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"raw{folder_suffix}")
        else:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
        
        os.makedirs(self.output_dir, exist_ok=True)

    def get_last_datetime(self, ticker):
        """Get the last datetime from CSV by reading only the last line."""
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            return None

        try:
            # Try to use pandas to read the last row (faster for large files)
            try:
                df = pd.read_csv(file_path, nrows=1, skiprows=lambda x: x > 0 and x < sum(1 for _ in open(file_path)) - 1)
                if not df.empty:
                    dt_str = df["datetime"].iloc[0]
                    last_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    return EASTERN_TZ.localize(last_dt)
            except Exception:
                # Fall back to direct file reading if pandas approach fails
                with open(file_path, 'rb') as f:
                    # Move to the end of file
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    
                    # Handle small files
                    if file_size < 200:
                        f.seek(0)
                    else:
                        # Read last 100 bytes
                        f.seek(max(file_size - 200, 0))
                    
                    # Read to the end to get the last lines
                    lines = f.read().decode().splitlines()
                    if not lines:
                        return None
                    
                    last_line = lines[-1]
                    if not last_line.strip():
                        if len(lines) > 1:
                            last_line = lines[-2]
                        else:
                            return None
                
                # Parse CSV line using csv.reader
                reader = csv.reader([last_line])
                row = next(reader)
                dt_str = row[0]  # Assumes the first column is "datetime"
                last_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                return EASTERN_TZ.localize(last_dt)
                
        except Exception as e:
            logger.error(f"Error reading last datetime for {ticker}: {e}")
            return None

    def save_data(self, ticker, data, min_datetime=None):
        """Save data to CSV efficiently."""
        if not data:
            return
            
        file_path = os.path.join(self.output_dir, f"{ticker}.csv")
        file_exists = os.path.exists(file_path)
        
        try:
            new_records = []
            for entry in data:
                utc_dt = datetime.fromtimestamp(entry["t"] / 1000, UTC_TZ)
                eastern_dt = utc_dt.astimezone(EASTERN_TZ)
                
                if min_datetime and eastern_dt <= min_datetime:
                    continue
                    
                new_records.append({
                    "datetime": eastern_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": entry.get("o"),
                    "high": entry.get("h"),
                    "low": entry.get("l"),
                    "close": entry.get("c"),
                    "volume": entry.get("v"),
                    "vwap": entry.get("vw"),
                    "num_trades": entry.get("n")
                })
            
            if new_records:
                df = pd.DataFrame(new_records)
                df.to_csv(file_path, mode='a', header=not file_exists, index=False)
                return len(new_records)
            return 0
                
        except Exception as e:
            logger.error(f"Error saving {ticker} data: {e}")
            return 0

def process_ticker(ticker, api, data_manager, interval, pbar=None):
    """Process individual ticker and update it to the absolute latest data available."""
    try:
        now = datetime.now(UTC_TZ)
        last_dt = data_manager.get_last_datetime(ticker)
        interval_info = VALID_INTERVALS.get(interval, VALID_INTERVALS["1min"])
        chunk_size = interval_info["chunk_size"]
        recent_days_check = interval_info["recent_days_check"]
        
        if last_dt:
            start_date = last_dt.date()  # resume from the last recorded timestamp regardless of staleness
        else:
            # No CSV exists; perform a preliminary check of recent data
            # Adjust the recent days check based on interval
            recent_start = (now - timedelta(days=recent_days_check)).strftime("%Y-%m-%d")
            recent_data = api.fetch_ticker_data(ticker, recent_start, END_DATE, interval)
            
            if not recent_data:
                tqdm.write(f"⏩ {ticker}: No recent {interval} data in the last {recent_days_check} days. Skipping.")
                if pbar:
                    pbar.update(1)
                return
            
            start_date = datetime.strptime(START_DATE, "%Y-%m-%d").date()

        # Calculate chunks for the date range
        end_date = datetime.strptime(END_DATE, "%Y-%m-%d").date()
        current_start = start_date
        total_records = 0
        
        # For daily data, we can fetch the entire 5-year history in one API call
        if interval == "1day":
            chunk_data = api.fetch_ticker_data(
                ticker,
                current_start.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                interval
            )
            
            if chunk_data:
                records_added = data_manager.save_data(ticker, chunk_data, last_dt)
                total_records += records_added
                tqdm.write(f"✓ {ticker}: Saved {interval} data from {current_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({records_added} records)")
        else:
            # For other intervals, process in optimal chunks
            while current_start <= end_date:
                current_end = min(current_start + timedelta(days=chunk_size), end_date)
                
                chunk_data = api.fetch_ticker_data(
                    ticker,
                    current_start.strftime("%Y-%m-%d"),
                    current_end.strftime("%Y-%m-%d"),
                    interval
                )
                
                if chunk_data:
                    records_added = data_manager.save_data(ticker, chunk_data, last_dt)
                    total_records += records_added
                    tqdm.write(f"✓ {ticker}: Saved {interval} chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} ({records_added} records)")
                
                # Use adaptive delay based on the number of records retrieved
                # More records = more likely to hit rate limits
                if len(chunk_data) > 10000:
                    time.sleep(0.8)  # Longer delay for large chunks
                elif len(chunk_data) > 5000:
                    time.sleep(0.5)  # Medium delay
                elif len(chunk_data) > 0:
                    time.sleep(0.2)  # Short delay for small chunks
                else:
                    time.sleep(0.1)  # Minimal delay if no data (likely a gap)
                
                current_start = current_end + timedelta(days=1)
        
        if pbar:
            pbar.update(1)
            
        return total_records
            
    except Exception as e:
        logger.error(f"Error processing {ticker} ({interval}): {e}")
        if pbar:
            pbar.update(1)
        return 0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch historical stock data from Polygon.io")
    parser.add_argument(
        "--interval", 
        type=str, 
        default="1min",
        choices=list(VALID_INTERVALS.keys()),
        help="Time interval for the data (default: 1min)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of specific tickers to fetch (optional)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    interval = args.interval
    specific_tickers = args.tickers.split(',') if args.tickers else None
    
    if interval not in VALID_INTERVALS:
        logger.error(f"Invalid interval: {interval}. Valid options are: {', '.join(VALID_INTERVALS.keys())}")
        return
    
    # Set appropriate logging level
    if interval in ["1hour", "1day"]:
        logging.getLogger().setLevel(logging.INFO)
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.error("Missing API key. Please set POLYGON_API_KEY in .env file.")
        return

    api = PolygonAPI(api_key)
    data_manager = DataManager(api, interval)
    
    if specific_tickers:
        tickers = specific_tickers
        logger.info(f"Fetching data for {len(tickers)} specified tickers with {interval} interval")
    else:
        tickers = api.get_active_tickers()
        if not tickers:
            logger.error("No tickers found")
            return
    
    interval_info = VALID_INTERVALS.get(interval)
    max_workers = interval_info["max_workers"]
    
    total_records = 0
    with tqdm(total=len(tickers), desc=f"Processing tickers ({interval})") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_ticker, ticker, api, data_manager, interval, pbar)
                for ticker in tickers
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        total_records += result
                except Exception as e:
                    logger.error(f"Future error: {e}")
    
    logger.info(f"Completed fetching {interval} data for {len(tickers)} tickers. Total records: {total_records}")

if __name__ == "__main__":
    main() 