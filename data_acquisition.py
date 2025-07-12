"""
Data Acquisition Script for News Arbitrage AI
Fetches stock prices and news data using Polygon.io API, saves to CSV files to avoid API rate limits.

Usage:
    python3 data_acquisition.py [TICKER] [START_DATE] [END_DATE]
    
Examples:
    python3 data_acquisition.py PYPL 20250501 20250510
    python3 data_acquisition.py TSLA 20240101 20240131
    python3 data_acquisition.py  # Uses PYPL, 3 months ago to yesterday
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import argparse
import sys
from config import STOCK_TICKER, DATA_PERIOD, POLYGON_API_KEY
from news_providers import fetch_news_data
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_stock_data(ticker, start_date=None, end_date=None):
    """Fetch historical stock data using Polygon.io."""
    print(f"Fetching stock data for {ticker} using Polygon.io...")
    
    try:
        # Polygon.io Aggregates (Bars) endpoint for daily data
        base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
        
        # Format dates for Polygon.io API
        if start_date and end_date:
            start_formatted = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
            end_formatted = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            url = f"{base_url}/{start_formatted}/{end_formatted}"
            print(f"   Date range: {start_formatted} to {end_formatted}")
        else:
            # Default to last 3 months if no dates specified
            end_dt = datetime.now() - timedelta(days=1)  # Use yesterday
            start_dt = end_dt - timedelta(days=3*30)
            start_formatted = start_dt.strftime('%Y-%m-%d')
            end_formatted = end_dt.strftime('%Y-%m-%d')
            url = f"{base_url}/{start_formatted}/{end_formatted}"
            print(f"   Default date range: {start_formatted} to {end_formatted}")
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,  # Maximum results
            'apikey': POLYGON_API_KEY
        }
        
        print(f"🔄 Making API request to Polygon.io...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors (accept both OK and DELAYED status)
        if data.get('status') not in ['OK', 'DELAYED']:
            print(f"❌ Polygon.io API Error: {data.get('error', data.get('message', 'Unknown error'))}")
            return None
        
        if 'results' not in data or not data['results']:
            print(f"❌ No stock data found for {ticker}")
            return None
        
        # Convert to DataFrame
        results = data['results']
        
        # Create DataFrame with proper column names
        stock_data = []
        for bar in results:
            # Convert timestamp to datetime
            date = datetime.fromtimestamp(bar['t'] / 1000)  # Polygon timestamps are in milliseconds
            
            stock_data.append({
                'Date': date,
                'Open': float(bar['o']),
                'High': float(bar['h']),
                'Low': float(bar['l']),
                'Close': float(bar['c']),
                'Adj Close': float(bar['c']),  # Polygon returns adjusted close by default
                'Volume': int(bar['v'])
            })
        
        df = pd.DataFrame(stock_data)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter by date range if specified (additional filtering for precision)
        if start_date and end_date:
            start_dt = datetime.strptime(start_date, '%Y%m%d').date()
            end_dt = datetime.strptime(end_date, '%Y%m%d').date()
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[
                (df['Date'].dt.date >= start_dt) & 
                (df['Date'].dt.date <= end_dt)
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
    """Fetch news data using Polygon.io (via the configured news provider)."""
    print(f"Fetching news data for {ticker} using Polygon.io...")
    
    try:
        # Use the unified news provider interface (which now defaults to Polygon.io)
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
            print(f"   Fetching last 90 days of news (3 months)")
            news_data = fetch_news_data(ticker, days_back=90)
        
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
    # Calculate default date range: 3 months ago to yesterday
    yesterday = datetime.now() - timedelta(days=1)
    three_months_ago = yesterday - timedelta(days=3*30)
    
    default_start = three_months_ago.strftime('%Y%m%d')
    default_end = yesterday.strftime('%Y%m%d')
    
    parser = argparse.ArgumentParser(
        description="Fetch stock and news data for News Arbitrage AI using Polygon.io",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python3 data_acquisition.py PYPL 20250501 20250510
  python3 data_acquisition.py TSLA 20240101 20240131
  python3 data_acquisition.py  # Uses PYPL, {default_start} to {default_end}
        """
    )
    
    parser.add_argument(
        'ticker', 
        nargs='?', 
        default='PYPL',
        help='Stock ticker symbol (default: PYPL)'
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
    print(f"  Data Source: Polygon.io API (no rate limiting)")
    print()
    
    # Validate API key
    if not POLYGON_API_KEY:
        print("❌ Error: POLYGON_API_KEY not found in environment variables")
        print("Please set your Polygon.io API key in the .env file:")
        print("POLYGON_API_KEY=your_api_key_here")
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