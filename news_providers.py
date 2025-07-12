"""
News API Providers Module
Unified interface for multiple news API providers including Alpha Vantage, Stock News API, and NewsAPI.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from config import (
    ALPHA_VANTAGE_API_KEY, STOCK_TICKER, NEWS_FETCH_LIMIT, NEWS_MAX_DAYS_BACK
)

# Rate limiting for Alpha Vantage API (shared across modules)
_last_api_call_time = 0
_min_call_interval = 1.0  # 1 second between calls (60 calls per minute max)

def rate_limit_alpha_vantage():
    """Ensure we don't exceed Alpha Vantage rate limits (70 requests per minute)."""
    global _last_api_call_time
    current_time = time.time()
    time_since_last_call = current_time - _last_api_call_time
    
    if time_since_last_call < _min_call_interval:
        sleep_time = _min_call_interval - time_since_last_call
        print(f"⏳ Wait for Alpha Vantage request limit, sleep for {sleep_time:.1f} seconds")
        time.sleep(sleep_time)
    
    _last_api_call_time = time.time()

class NewsProvider:
    """Base class for news providers."""
    
    def __init__(self):
        self.provider_name = "Base Provider"
    
    def fetch_news(self, symbol, start_date=None, end_date=None, days_back=30):
        """Fetch news articles for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date as datetime object (optional)
            end_date: End date as datetime object (optional)
            days_back: Number of days back from now (used if start_date/end_date not provided)
        """
        raise NotImplementedError("Subclasses must implement fetch_news method")
    
    def format_articles(self, raw_data):
        """Format raw API response into standardized format."""
        raise NotImplementedError("Subclasses must implement format_articles method")

class AlphaVantageProvider(NewsProvider):
    """Alpha Vantage News API provider."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "Alpha Vantage"
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = ALPHA_VANTAGE_API_KEY
    
    def fetch_news(self, symbol, start_date=None, end_date=None, days_back=30):
        """Fetch news from Alpha Vantage News API."""
        try:
            # Calculate date range
            if start_date and end_date:
                from_date = start_date
                to_date = end_date
                days_back = (end_date - start_date).days + 1
            else:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days_back)
            
            # Alpha Vantage News API limitation: may not support very old dates
            # Limit to configured maximum days back for better results
            if days_back > NEWS_MAX_DAYS_BACK:
                print(f"⚠️  Alpha Vantage News API: Limiting {days_back} days to {NEWS_MAX_DAYS_BACK} days for better results")
                days_back = NEWS_MAX_DAYS_BACK
                from_date = to_date - timedelta(days=days_back)
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'time_from': from_date.strftime('%Y%m%dT%H%M'),
                'time_to': to_date.strftime('%Y%m%dT%H%M'),
                'limit': NEWS_FETCH_LIMIT,  # Configurable limit for testing
                'apikey': self.api_key
            }
            
            print(f"   Actual date range for news: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')} ({days_back} days)")
            
            print(f"Fetching news from {self.provider_name} for {symbol}...")
            
            # Apply rate limiting for Alpha Vantage
            rate_limit_alpha_vantage()
            print(f"🔄 Making API request to Alpha Vantage...")
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' in data:
                articles = self.format_articles(data['feed'])
                print(f"✅ Found {len(articles)} articles from {self.provider_name}")
                return articles
            else:
                print(f"⚠️  No articles found in {self.provider_name} response")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching from {self.provider_name}: {e}")
            return []
    
    def format_articles(self, raw_articles):
        """Format Alpha Vantage articles into standardized format."""
        formatted_articles = []
        
        print(f"🔍 DEBUG: Formatting {len(raw_articles)} articles from {self.provider_name}")
        
        for i, article in enumerate(raw_articles, 1):
            try:
                title = article.get('title', '')
                description = article.get('summary', '')
                
                # Debug: Print first line of each article
                first_line = title[:60] + "..." if len(title) > 60 else title
                print(f"📰 Article {i} {article.get('time_published', '')}: {first_line}")
                
                formatted_article = {
                    'title': title,
                    'description': description,
                    'url': article.get('url', ''),
                    'publishedAt': article.get('time_published', ''),
                    'source': article.get('source', ''),
                    'sentiment': float(article.get('overall_sentiment_score', 0)),
                    'relevance': float(article.get('relevance_score', 0))
                }
                formatted_articles.append(formatted_article)
            except Exception as e:
                print(f"⚠️  Error formatting article {i}: {e}")
                continue
        
        return formatted_articles

class StockNewsProvider(NewsProvider):
    """Stock News API provider."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "Stock News API"
        self.base_url = "https://stocknewsapi.com/api/v1"
        self.api_key = STOCK_NEWS_API_KEY
    
    def fetch_news(self, symbol, start_date=None, end_date=None, days_back=30):
        """Fetch news from Stock News API."""
        try:
            # Calculate date range
            if start_date and end_date:
                from_date = start_date
                to_date = end_date
                days_back = (end_date - start_date).days + 1
            else:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days_back)
            
            params = {
                'tickers': symbol,
                'items': 500,  # Maximum per request
                'date': f"{from_date.strftime('%m%d%Y')}-{to_date.strftime('%m%d%Y')}",
                'token': self.api_key
            }
            
            print(f"Fetching news from {self.provider_name} for {symbol}...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data:
                articles = self.format_articles(data['data'])
                print(f"✅ Found {len(articles)} articles from {self.provider_name}")
                return articles
            else:
                print(f"⚠️  No articles found in {self.provider_name} response")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching from {self.provider_name}: {e}")
            return []
    
    def format_articles(self, raw_articles):
        """Format Stock News API articles into standardized format."""
        formatted_articles = []
        
        print(f"🔍 DEBUG: Formatting {len(raw_articles)} articles from {self.provider_name}")
        
        for i, article in enumerate(raw_articles, 1):
            try:
                title = article.get('title', '')
                description = article.get('text', '')
                
                # Debug: Print first line of each article
                first_line = title[:100] + "..." if len(title) > 100 else title
                print(f"📰 Article {i}: {first_line}")
                
                formatted_article = {
                    'title': title,
                    'description': description,
                    'url': article.get('news_url', ''),
                    'publishedAt': article.get('date', ''),
                    'source': article.get('source_name', ''),
                    'sentiment': float(article.get('sentiment', 0)),
                    'relevance': 1.0  # Stock News API doesn't provide relevance score
                }
                formatted_articles.append(formatted_article)
            except Exception as e:
                print(f"⚠️  Error formatting article {i}: {e}")
                continue
        
        return formatted_articles

class PolygonProvider(NewsProvider):
    """Polygon.io News API provider."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "Polygon.io"
        self.base_url = "https://api.polygon.io/v2/reference/news"
        self.api_key = POLYGON_API_KEY
    
    def fetch_news(self, symbol, start_date=None, end_date=None, days_back=30):
        """Fetch news from Polygon.io News API."""
        try:
            # Calculate date range
            if start_date and end_date:
                from_date = start_date
                to_date = end_date
                days_back = (end_date - start_date).days + 1
            else:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days_back)
            
            params = {
                'ticker': symbol,
                'published_utc.gte': from_date.strftime('%Y-%m-%d'),
                'published_utc.lte': to_date.strftime('%Y-%m-%d'),
                'order': 'desc',
                'limit': NEWS_FETCH_LIMIT,  # Configurable limit for testing
                'apikey': self.api_key
            }
            
            print(f"   Actual date range for news: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')} ({days_back} days)")
            
            print(f"Fetching news from {self.provider_name} for {symbol}...")
            print(f"🔄 Making API request to Polygon.io...")
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' in data and data['results']:
                articles = self.format_articles(data['results'])
                print(f"✅ Found {len(articles)} articles from {self.provider_name}")
                return articles
            else:
                print(f"⚠️  No articles found in {self.provider_name} response")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching from {self.provider_name}: {e}")
            return []
    
    def format_articles(self, raw_articles):
        """Format Polygon.io articles into standardized format."""
        formatted_articles = []
        
        print(f"🔍 DEBUG: Formatting {len(raw_articles)} articles from {self.provider_name}")
        
        for i, article in enumerate(raw_articles, 1):
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                
                # Convert ISO datetime to readable format for debug message
                published_date = article.get('published_utc', '')
                formatted_date = 'N/A'
                if published_date:
                    try:
                        # Parse ISO format and convert to YYYY-MM-DD format
                        date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                        published_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        formatted_date = published_date[:10] if len(published_date) >= 10 else published_date
                        pass  # Keep original format if parsing fails
                
                # Debug: Print first line of each article with date after index
                first_line = title[:60] + "..." if len(title) > 60 else title
                print(f"📰 Article {i} ({formatted_date}): {first_line}")
                
                # Convert ISO datetime to readable format for data storage
                if published_date == 'N/A':
                    published_date = article.get('published_utc', '')
                
                formatted_article = {
                    'title': title,
                    'description': description,
                    'url': article.get('article_url', ''),
                    'publishedAt': published_date,
                    'source': article.get('publisher', {}).get('name', '') if article.get('publisher') else '',
                    'sentiment': 0.0,  # Polygon.io doesn't provide sentiment by default
                    'relevance': 1.0   # Polygon.io doesn't provide relevance score
                }
                formatted_articles.append(formatted_article)
            except Exception as e:
                print(f"⚠️  Error formatting article {i}: {e}")
                continue
        
        return formatted_articles

class NewsAPIProvider(NewsProvider):
    """NewsAPI.org provider (fallback)."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "NewsAPI.org"
        self.base_url = "https://newsapi.org/v2/everything"
        self.api_key = NEWS_API_KEY
    
    def fetch_news(self, symbol, start_date=None, end_date=None, days_back=30):
        """Fetch news from NewsAPI.org."""
        try:
            # Calculate date range (NewsAPI free tier limited to 30 days)
            if start_date and end_date:
                from_date = start_date
                to_date = end_date
                days_back = (end_date - start_date).days + 1
                # NewsAPI free tier limitation
                if days_back > 30:
                    print(f"⚠️  NewsAPI.org: Limiting {days_back} days to 30 days (free tier limitation)")
                    from_date = to_date - timedelta(days=30)
            else:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=min(days_back, 30))
            
            params = {
                'q': symbol,
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'pageSize': 100,
                'apiKey': self.api_key
            }
            
            print(f"Fetching news from {self.provider_name} for {symbol}...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'articles' in data:
                articles = self.format_articles(data['articles'])
                print(f"✅ Found {len(articles)} articles from {self.provider_name}")
                return articles
            else:
                print(f"⚠️  No articles found in {self.provider_name} response")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching from {self.provider_name}: {e}")
            return []
    
    def format_articles(self, raw_articles):
        """Format NewsAPI.org articles into standardized format."""
        formatted_articles = []
        
        print(f"🔍 DEBUG: Formatting {len(raw_articles)} articles from {self.provider_name}")
        
        for i, article in enumerate(raw_articles, 1):
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                
                # Debug: Print first line of each article
                first_line = title[:100] + "..." if len(title) > 100 else title
                print(f"📰 Article {i}: {first_line}")
                
                formatted_article = {
                    'title': title,
                    'description': description,
                    'url': article.get('url', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'sentiment': 0.0,  # NewsAPI doesn't provide sentiment
                    'relevance': 1.0   # NewsAPI doesn't provide relevance
                }
                formatted_articles.append(formatted_article)
            except Exception as e:
                print(f"⚠️  Error formatting article {i}: {e}")
                continue
        
        return formatted_articles

def get_news_provider():
    return AlphaVantageProvider()

def fetch_news_data(symbol=None, start_date=None, end_date=None, days_back=30):
    """Unified function to fetch news data from the configured provider.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date as datetime object (optional)
        end_date: End date as datetime object (optional)
        days_back: Number of days back from now (used if start_date/end_date not provided)
    """
    if symbol is None:
        symbol = STOCK_TICKER
    
    provider = get_news_provider()
    articles = provider.fetch_news(symbol, start_date, end_date, days_back)
    
    if articles:
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # Standardize date format
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df['Date'] = df['publishedAt'].dt.date
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['publishedAt'])
        
        # Reorder columns to put Date first
        column_order = ['Date', 'title', 'description', 'url', 'publishedAt', 'source', 'sentiment', 'relevance']
        df = df[column_order]
        
        print(f"📊 Processed {len(df)} articles with valid dates")
        
        # Debug: Show first few article titles in summary
        if len(df) > 0:
            print(f"🔍 DEBUG: Sample article titles:")
            for i, title in enumerate(df['title'].head(3), 1):
                title_preview = title[:80] + "..." if len(title) > 80 else title
                print(f"   {i}. {title_preview}")
        
        return df
    else:
        print("❌ No articles retrieved from any provider")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the news providers
    print("=== Testing News Providers ===")
    
    # Test current provider
    df = fetch_news_data(STOCK_TICKER, days_back=7)
    if not df.empty:
        print(f"\n📈 Sample articles:")
        print(df[['title', 'source', 'publishedAt', 'sentiment']].head())
    else:
        print("No articles found for testing") 