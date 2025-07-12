"""
Data Processing Script for News Arbitrage AI
Merges stock and news data, engineers features, and prepares the final dataset.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import argparse
import sys
from config import STOCK_TICKER, SHARP_MOVE_THRESHOLD
import os

def load_data(ticker=None):
    """Load stock and news data from CSV files."""
    if ticker is None:
        ticker = STOCK_TICKER
    
    stock_filename = f"{ticker.lower()}_stock_data.csv"
    news_filename = f"{ticker.lower()}_news_data.csv"
    
    if not os.path.exists(stock_filename):
        raise FileNotFoundError(f"Stock data file not found: {stock_filename}")
    
    if not os.path.exists(news_filename):
        raise FileNotFoundError(f"News data file not found: {news_filename}")
    
    # Load stock data
    stock_data = pd.read_csv(stock_filename)
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    
    # Load news data
    news_data = pd.read_csv(news_filename)
    news_data['publishedAt'] = pd.to_datetime(news_data['publishedAt'])
    news_data['Date'] = pd.to_datetime(news_data['Date']).dt.date
    
    print(f"✅ Loaded stock data: {stock_data.shape}")
    print(f"✅ Loaded news data: {news_data.shape}")
    
    return stock_data, news_data

def engineer_stock_features(stock_data):
    """Engineer features from stock data and create target variable."""
    print("🔧 Engineering stock features...")
    
    # Calculate daily change
    stock_data['daily_change'] = stock_data['Close'].pct_change()
    
    # Define the target variable: 1 for sharp up, -1 for sharp down, 0 for no major move
    def define_sharp_move(change):
        if pd.isna(change):
            return 0
        elif change > SHARP_MOVE_THRESHOLD:
            return 1  # Sharp Up
        elif change < -SHARP_MOVE_THRESHOLD:
            return -1  # Sharp Down
        else:
            return 0  # No Sharp Move
    
    stock_data['sharp_move'] = stock_data['daily_change'].apply(define_sharp_move)
    
    # We are predicting tomorrow's move based on today's news, so shift the target
    stock_data['target'] = stock_data['sharp_move'].shift(-1)
    
    # Add additional technical indicators
    stock_data['volume_change'] = stock_data['Volume'].pct_change()
    stock_data['high_low_spread'] = (stock_data['High'] - stock_data['Low']) / stock_data['Close']
    
    # Drop rows with NaN values
    stock_data = stock_data.dropna()
    
    print(f"   Target distribution:")
    print(f"   Sharp Down (-1): {sum(stock_data['target'] == -1)}")
    print(f"   No Move (0): {sum(stock_data['target'] == 0)}")
    print(f"   Sharp Up (1): {sum(stock_data['target'] == 1)}")
    
    return stock_data

def engineer_news_features(news_data):
    """Engineer features from news data."""
    print("🔧 Engineering news features...")
    
    # Combine title and description for richer text
    news_data['full_text'] = (
        news_data['title'].fillna('') + ' ' + 
        news_data['description'].fillna('')
    )
    
    # Calculate sentiment for each article
    print("   Calculating sentiment scores...")
    news_data['sentiment'] = news_data['full_text'].apply(
        lambda text: TextBlob(text).sentiment.polarity if text.strip() else 0
    )
    
    # Group news by date and aggregate
    print("   Aggregating daily news...")
    daily_news = news_data.groupby('Date').agg({
        'full_text': ' '.join,  # Combine all text for the day
        'sentiment': ['mean', 'std', 'count'],  # Sentiment statistics
        'title': 'count'  # Number of articles
    }).reset_index()
    
    # Flatten column names
    daily_news.columns = [
        'Date', 'combined_text', 'avg_sentiment', 'sentiment_std', 'sentiment_count', 'article_count'
    ]
    
    # Fill NaN values
    daily_news['sentiment_std'].fillna(0, inplace=True)
    daily_news['combined_text'].fillna('', inplace=True)
    
    print(f"   Daily news aggregated: {daily_news.shape}")
    
    return daily_news

def merge_data(stock_data, daily_news):
    """Merge stock and news data."""
    print("🔗 Merging stock and news data...")
    
    # Merge the datasets
    final_df = pd.merge(stock_data, daily_news, on='Date', how='left')
    
    # Fill missing news data with neutral values
    final_df['combined_text'].fillna('', inplace=True)
    final_df['avg_sentiment'].fillna(0, inplace=True)
    final_df['sentiment_std'].fillna(0, inplace=True)
    final_df['sentiment_count'].fillna(0, inplace=True)
    final_df['article_count'].fillna(0, inplace=True)
    
    print(f"✅ Final merged dataset: {final_df.shape}")
    
    return final_df

def save_processed_data(final_df, ticker=None):
    """Save the processed data to CSV."""
    if ticker is None:
        ticker = STOCK_TICKER
    
    filename = f"{ticker.lower()}_processed_data.csv"
    final_df.to_csv(filename, index=False)
    print(f"✅ Processed data saved to {filename}")
    
    return filename

def main():
    """Main function to process all data."""
    parser = argparse.ArgumentParser(
        description="Process stock and news data for News Arbitrage AI"
    )
    parser.add_argument(
        'ticker', 
        nargs='?', 
        default='PYPL',
        help='Stock ticker symbol (default: PYPL)'
    )
    
    args = parser.parse_args()
    
    print("=== News Arbitrage AI - Data Processing ===\n")
    print(f"Processing data for: {args.ticker}")
    
    try:
        # Load data
        stock_data, news_data = load_data(args.ticker)
        
        # Engineer features
        stock_data = engineer_stock_features(stock_data)
        daily_news = engineer_news_features(news_data)
        
        # Merge data
        final_df = merge_data(stock_data, daily_news)
        
        # Save processed data
        filename = save_processed_data(final_df, args.ticker)
        
        # Display summary
        print("\n📊 Data Processing Summary:")
        print(f"   Ticker: {args.ticker}")
        print(f"   Total samples: {len(final_df)}")
        print(f"   Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
        print(f"   Features: {final_df.columns.tolist()}")
        
        print("\n✅ Data processing complete!")
        print("Next step: Run 'python3 modeling.py' to train the model.")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        return None

if __name__ == "__main__":
    main() 