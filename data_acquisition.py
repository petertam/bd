"""
Data Acquisition Script for News Arbitrage AI
Fetches stock prices and news data using Alpha Vantage API, saves to CSV files to avoid API rate limits.

Usage:
    python3 data_acquisition.py [TICKER] [START_DATE] [END_DATE]
    
Examples:
    python3 data_acquisition.py PYPL 20250501 20250510
    python3 data_acquisition.py TSLA 20240101 20240131
    python3 data_acquisition.py  # Uses default ticker from .env, 600 days ago to yesterday
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import argparse
import sys
from config import STOCK_TICKER, DATA_PERIOD, ALPHA_VANTAGE_API_KEY
from news_providers import fetch_news_data
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_stock_data(ticker, start_date=None, end_date=None):
    """Fetch historical stock data using Alpha Vantage API."""
    print(f"Fetching stock data for {ticker} using Alpha Vantage...")
    
    try:
        # Alpha Vantage Time Series Daily endpoint
        base_url = "https://www.alphavantage.co/query"
        
        # Format dates for date range filtering
        if start_date and end_date:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            print(f"   Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        else:
            # Default to last 600 days if no dates specified
            end_dt = datetime.now() - timedelta(days=1)  # Use yesterday
            start_dt = end_dt - timedelta(days=600)
            print(f"   Default date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': 'full',  # Get full historical data
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        print(f"🔄 Making API request to Alpha Vantage...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            print(f"❌ Alpha Vantage API Error: {data['Error Message']}")
            return None
            
        if 'Note' in data:
            print(f"❌ Alpha Vantage API Rate Limit: {data['Note']}")
            return None
            
        if 'Time Series (Daily)' not in data:
            print(f"❌ No stock data found for {ticker}")
            return None
        
        # Convert to DataFrame
        time_series = data['Time Series (Daily)']
        
        # Create DataFrame with proper column names
        stock_data = []
        for date_str, values in time_series.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            stock_data.append({
                'Date': date,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Adj Close': float(values['5. adjusted close']),
                'Volume': int(values['6. volume'])
            })
        
        df = pd.DataFrame(stock_data)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter by date range if specified
        if start_date and end_date:
            start_dt_date = datetime.strptime(start_date, '%Y%m%d').date()
            end_dt_date = datetime.strptime(end_date, '%Y%m%d').date()
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[
                (df['Date'].dt.date >= start_dt_date) & 
                (df['Date'].dt.date <= end_dt_date)
            ]
            
            print(f"   Filtered to {len(df)} days within date range")
        
        # Save to CSV
        stock_filename = f"{ticker.lower()}_stock_data.csv"
        df.to_csv(stock_filename, index=False)
        
        print(f"✅ Stock data saved to {stock_filename}")
        print(f"   Shape: {df.shape}")
        if len(df) > 0:
            print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"   Latest close price: ${df['Close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return None

def fetch_news_data_main(ticker, start_date=None, end_date=None):
    """Fetch news data using the configured news provider."""
    print(f"Fetching news data for {ticker} using configured news provider...")
    
    try:
        # Use the unified news provider interface
        from news_providers import fetch_news_data
        
        if start_date and end_date:
            # Calculate days between start and end date
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            total_days = (end_dt - start_dt).days + 1
            print(f"   Total date range: {start_date} to {end_date} ({total_days} days)")
            
            # Check if date range is more than 30 days
            if total_days > 30:
                print(f"   Date range > 30 days, splitting into chunks...")
                all_news_data = []
                
                # Split the date range into chunks of 30 days or less
                current_start = start_dt
                chunk_num = 1
                
                while current_start <= end_dt:
                    # Calculate end date for this chunk (30 days from current start or end_dt, whichever is smaller)
                    chunk_end = min(current_start + timedelta(days=29), end_dt)
                    chunk_days = (chunk_end - current_start).days + 1
                    
                    print(f"   Chunk {chunk_num}: {current_start.strftime('%Y%m%d')} to {chunk_end.strftime('%Y%m%d')} ({chunk_days} days)")
                    
                    # Fetch news for this chunk using start_date and end_date
                    chunk_news_data = fetch_news_data(ticker, start_date=current_start, end_date=chunk_end)
                    
                    if not chunk_news_data.empty:
                        # Filter by the exact chunk date range
                        chunk_start_date = current_start.date()
                        chunk_end_date = chunk_end.date()
                        
                        # Convert Date column to datetime for filtering
                        chunk_news_data['Date'] = pd.to_datetime(chunk_news_data['Date']).dt.date
                        
                        # Filter by chunk date range
                        chunk_filtered = chunk_news_data[
                            (chunk_news_data['Date'] >= chunk_start_date) & 
                            (chunk_news_data['Date'] <= chunk_end_date)
                        ]
                        
                        if not chunk_filtered.empty:
                            all_news_data.append(chunk_filtered)
                            print(f"     Found {len(chunk_filtered)} articles in chunk {chunk_num}")
                        else:
                            print(f"     No articles found in chunk {chunk_num}")
                    else:
                        print(f"     No articles found in chunk {chunk_num}")
                    
                    # Move to next chunk
                    current_start = chunk_end + timedelta(days=1)
                    chunk_num += 1
                
                # Combine all chunks
                if all_news_data:
                    news_data = pd.concat(all_news_data, ignore_index=True)
                    # Remove duplicates based on title and publishedAt
                    news_data = news_data.drop_duplicates(subset=['title', 'publishedAt'], keep='first')
                    # Sort by date
                    news_data = news_data.sort_values('Date').reset_index(drop=True)
                    print(f"   Combined {len(news_data)} unique articles from {chunk_num-1} chunks")
                else:
                    print("   No articles found in any chunk")
                    news_data = pd.DataFrame()
            else:
                # Date range is 30 days or less, fetch normally
                print(f"   Date range ≤ 30 days, fetching normally...")
                news_data = fetch_news_data(ticker, start_date=start_dt, end_date=end_dt)
                
                if not news_data.empty:
                    # Filter news data by date range
                    start_dt_date = start_dt.date()
                    end_dt_date = end_dt.date()
                    
                    # Convert Date column to datetime for filtering
                    news_data['Date'] = pd.to_datetime(news_data['Date']).dt.date
                    
                    # Filter by date range
                    news_data = news_data[
                        (news_data['Date'] >= start_dt_date) & 
                        (news_data['Date'] <= end_dt_date)
                    ]
                    
                    print(f"   Filtered to {len(news_data)} articles within date range")
        else:
            print(f"   Fetching last 600 days of news")
            news_data = fetch_news_data(ticker, days_back=600)
        
        if not news_data.empty:
            # Save to CSV in append mode
            news_filename = f"{ticker.lower()}_news_data.csv"
            
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(news_filename)
            
            if file_exists:
                # Load existing data to check for duplicates
                try:
                    existing_data = pd.read_csv(news_filename)
                    # Remove duplicates based on title and publishedAt
                    if 'title' in existing_data.columns and 'publishedAt' in existing_data.columns:
                        combined_data = pd.concat([existing_data, news_data], ignore_index=True)
                        combined_data = combined_data.drop_duplicates(subset=['title', 'publishedAt'], keep='first')
                        # Get only the new data (data not in existing file)
                        new_data_only = combined_data[len(existing_data):]
                        if not new_data_only.empty:
                            # Append new data only (without header)
                            new_data_only.to_csv(news_filename, mode='a', header=False, index=False)
                            print(f"✅ News data appended to {news_filename}")
                            print(f"   Added {len(new_data_only)} new articles (duplicates removed)")
                        else:
                            print(f"✅ No new articles to add (all duplicates)")
                    else:
                        # If columns don't match, just append (fallback)
                        news_data.to_csv(news_filename, mode='a', header=False, index=False)
                        print(f"✅ News data appended to {news_filename}")
                        print(f"   Added {len(news_data)} articles")
                except Exception as e:
                    print(f"⚠️  Error reading existing file, overwriting: {e}")
                    news_data.to_csv(news_filename, index=False)
                    print(f"✅ News data saved to {news_filename}")
            else:
                # File doesn't exist, create new with header
                news_data.to_csv(news_filename, index=False)
                print(f"✅ News data saved to {news_filename}")
            
            print(f"   Shape: {news_data.shape}")
            if len(news_data) > 0:
                print(f"   Date range: {news_data['Date'].min()} to {news_data['Date'].max()}")
            
            return news_data
        else:
            print("❌ No news articles found")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching news data: {e}")
        return None

def load_existing_data(ticker):
    """Load existing data from CSV files if they exist."""
    stock_filename = f"{ticker.lower()}_stock_data.csv"
    news_filename = f"{ticker.lower()}_news_data.csv"
    
    stock_data = None
    news_data = None
    
    if os.path.exists(stock_filename):
        stock_data = pd.read_csv(stock_filename)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        print(f"✅ Loaded existing stock data from {stock_filename}")
    
    if os.path.exists(news_filename):
        news_data = pd.read_csv(news_filename)
        news_data['publishedAt'] = pd.to_datetime(news_data['publishedAt'])
        news_data['Date'] = pd.to_datetime(news_data['Date']).dt.date
        print(f"✅ Loaded existing news data from {news_filename}")
    
    return stock_data, news_data

def parse_arguments():
    """Parse command line arguments."""
    # Calculate default date range: 600 days ago to yesterday
    yesterday = datetime.now() - timedelta(days=1)
    six_hundred_days_ago = yesterday - timedelta(days=600)
    
    default_start = six_hundred_days_ago.strftime('%Y%m%d')
    default_end = yesterday.strftime('%Y%m%d')
    
    parser = argparse.ArgumentParser(
        description="Fetch stock and news data for News Arbitrage AI using Alpha Vantage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python3 data_acquisition.py PYPL 20250501 20250510
  python3 data_acquisition.py TSLA 20240101 20240131
  python3 data_acquisition.py  # Uses {STOCK_TICKER}, {default_start} to {default_end}
        """
    )
    
    parser.add_argument(
        'ticker', 
        nargs='?', 
        default=STOCK_TICKER,
        help=f'Stock ticker symbol (default: {STOCK_TICKER})'
    )
    
    parser.add_argument(
        'start_date', 
        nargs='?', 
        default=default_start,
        help=f'Start date in YYYYMMDD format (default: {default_start})'
    )
    
    parser.add_argument(
        'end_date', 
        nargs='?', 
        default=default_end,
        help=f'End date in YYYYMMDD format (default: {default_end})'
    )
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        start_dt = datetime.strptime(args.start_date, '%Y%m%d')
        end_dt = datetime.strptime(args.end_date, '%Y%m%d')
        
        if start_dt >= end_dt:
            print("❌ Error: Start date must be before end date")
            sys.exit(1)
            
    except ValueError as e:
        print(f"❌ Error: Invalid date format. Use YYYYMMDD format (e.g., 20250501)")
        print(f"   Details: {e}")
        sys.exit(1)
    
    return args

def main():
    """Main function to fetch all data."""
    args = parse_arguments()
    
    print("=== News Arbitrage AI - Data Acquisition ===\n")
    print(f"Configuration:")
    print(f"  Ticker: {args.ticker}")
    print(f"  Date Range: {args.start_date} to {args.end_date}")
    
    # Calculate and show the date range span
    start_dt = datetime.strptime(args.start_date, '%Y%m%d')
    end_dt = datetime.strptime(args.end_date, '%Y%m%d')
    days_span = (end_dt - start_dt).days + 1
    years_span = days_span / 365.25
    print(f"  Duration: {days_span} days (~{years_span:.1f} years)")
    print(f"  Data Source: Alpha Vantage API (rate limited)")
    print()
    
    # Validate API key
    if not ALPHA_VANTAGE_API_KEY:
        print("❌ Error: ALPHA_VANTAGE_API_KEY not found in environment variables")
        print("Please set your Alpha Vantage API key in the .env file:")
        print("ALPHA_VANTAGE_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Always fetch fresh data since we're using specific date ranges
    # Fetch stock data
    stock_data = fetch_stock_data(args.ticker, args.start_date, args.end_date)
    
    # Fetch news data
    news_data = fetch_news_data_main(args.ticker, args.start_date, args.end_date)
    
    if stock_data is not None and news_data is not None:
        print("\n✅ Data acquisition complete!")
        print(f"Next step: Run 'python3 data_processing.py {args.ticker}' to merge and process the data.")
        print(f"Files created:")
        print(f"  - {args.ticker.lower()}_stock_data.csv")
        print(f"  - {args.ticker.lower()}_news_data.csv")
    else:
        print("\n❌ Data acquisition failed. Please check your API key and try again.")

if __name__ == "__main__":
    main() 