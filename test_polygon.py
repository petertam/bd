#!/usr/bin/env python3
"""
Test script for Polygon.io News API
Tests fetching news data for a given ticker and date range.
"""

import requests
import sys
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_polygon_news(ticker, start_date, end_date, api_key):
    """
    Fetch news data from Polygon.io API
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        api_key (str): Polygon.io API key
    
    Returns:
        dict: API response data
    """
    base_url = "https://api.polygon.io/v2/reference/news"
    
    params = {
        'ticker': ticker,
        'published_utc.gte': start_date,
        'published_utc.lte': end_date,
        'order': 'desc',
        'limit': 50,  # Maximum articles per request
        'apikey': api_key
    }
    
    try:
        print(f"Fetching news for {ticker} from {start_date} to {end_date}...")
        print(f"API URL: {base_url}")
        print(f"Parameters: {params}")
        print("-" * 60)
        
        response = requests.get(base_url, params=params)
        
        print(f"HTTP Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 60)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def format_date(date_str):
    """
    Convert YYYYMMDD format to YYYY-MM-DD format
    
    Args:
        date_str (str): Date in YYYYMMDD format
    
    Returns:
        str: Date in YYYY-MM-DD format
    """
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str

def print_news_summary(data):
    """
    Print a summary of the news data
    
    Args:
        data (dict): API response data
    """
    if not data:
        print("No data to display")
        return
    
    print("=== POLYGON.IO NEWS API RESPONSE ===")
    print(f"Status: {data.get('status', 'Unknown')}")
    print(f"Request ID: {data.get('request_id', 'N/A')}")
    print(f"Count: {data.get('count', 0)}")
    print(f"Next URL: {data.get('next_url', 'N/A')}")
    print("-" * 60)
    
    results = data.get('results', [])
    if not results:
        print("No news articles found for the specified criteria.")
        return
    
    print(f"Found {len(results)} articles:")
    print("-" * 60)
    
    for i, article in enumerate(results, 1):
        print(f"Article {i}:")
        print(f"  Title: {article.get('title', 'N/A')}")
        print(f"  Published: {article.get('published_utc', 'N/A')}")
        print(f"  Author: {article.get('author', 'N/A')}")
        print(f"  Publisher: {article.get('publisher', {}).get('name', 'N/A')}")
        print(f"  URL: {article.get('article_url', 'N/A')}")
        print(f"  Tickers: {article.get('tickers', [])}")
        
        # Print first 60 characters of news with date first
        published_date = article.get('published_utc', '')
        # Convert ISO datetime to YYYY-MM-DD format
        if published_date:
            try:
                # Parse ISO format and extract date
                date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                formatted_date = date_obj.strftime('%Y-%m-%d')
            except:
                formatted_date = published_date[:10]  # Fallback to first 10 chars
        else:
            formatted_date = 'N/A'
        
        # Get title and description for content
        title = article.get('title', '')
        description = article.get('description', '')
        content = f"{title} {description}".strip()
        
        # Print first 60 characters with date first
        if content:
            first_60_chars = content[:60]
            print(f"  First 60 chars: {formatted_date} - {first_60_chars}")
        else:
            print(f"  First 60 chars: {formatted_date} - [No content]")
        
        # Print full description (truncated)
        if description:
            # Truncate long descriptions
            if len(description) > 200:
                description = description[:200] + "..."
            print(f"  Description: {description}")
        
        print("-" * 40)

def main():
    """
    Main function to test Polygon.io News API
    """
    # Default values
    default_ticker = "PYPL"
    default_start_date = "20240301"
    default_end_date = "20240305"
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        ticker = default_ticker
        start_date = default_start_date
        end_date = default_end_date
    elif len(sys.argv) == 4:
        ticker = sys.argv[1].upper()
        start_date = sys.argv[2]
        end_date = sys.argv[3]
    else:
        print("Usage: python3 test_polygon.py [TICKER START_DATE END_DATE]")
        print("Example: python3 test_polygon.py PYPL 20240301 20240305")
        print(f"Default: python3 test_polygon.py (uses {default_ticker} {default_start_date} {default_end_date})")
        sys.exit(1)
    
    # Validate and format dates
    try:
        start_date_formatted = format_date(start_date)
        end_date_formatted = format_date(end_date)
        
        # Validate date format
        datetime.strptime(start_date_formatted, '%Y-%m-%d')
        datetime.strptime(end_date_formatted, '%Y-%m-%d')
        
    except ValueError as e:
        print(f"Invalid date format: {e}")
        print("Please use YYYYMMDD format (e.g., 20240301)")
        sys.exit(1)
    
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY not found in environment variables")
        print("Please set your Polygon.io API key in the .env file:")
        print("POLYGON_API_KEY=your_api_key_here")
        sys.exit(1)
    
    print("=== POLYGON.IO NEWS API TEST ===")
    print(f"Ticker: {ticker}")
    print(f"Start Date: {start_date_formatted}")
    print(f"End Date: {end_date_formatted}")
    print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '****'}")
    print("=" * 60)
    
    # Fetch news data
    data = get_polygon_news(ticker, start_date_formatted, end_date_formatted, api_key)
    
    # Print results
    print_news_summary(data)
    
    # Save raw response to file for inspection
    if data:
        filename = f"polygon_news_{ticker}_{start_date}_{end_date}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nRaw API response saved to: {filename}")

if __name__ == "__main__":
    main() 