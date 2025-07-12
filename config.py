import os

try:
    import pandas  # noqa: F401
except ImportError:
    print("Warning: pandas could not be resolved. Please ensure pandas is installed.")

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv is not installed. Environment variables from .env will not be loaded.")

# News API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')  # For NewsAPI.org (fallback)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')  # For Alpha Vantage
STOCK_NEWS_API_KEY = os.getenv('STOCK_NEWS_API_KEY')  # For Stock News API (premium)
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')  # For Polygon.io (recommended)

# News API Selection (priority order)
NEWS_API_PROVIDER = os.getenv('NEWS_API_PROVIDER', 'polygon')  # polygon, alpha_vantage, stock_news, newsapi

# Stock Configuration
STOCK_TICKER = os.getenv('STOCK_TICKER', 'TSLA')
SHARP_MOVE_THRESHOLD = float(os.getenv('SHARP_MOVE_THRESHOLD', '0.03'))
DATA_PERIOD = os.getenv('DATA_PERIOD', '2y')

# Model Configuration
MAX_FEATURES = int(os.getenv('MAX_FEATURES', '100'))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', '0.8'))

# News API Limits (for testing)
NEWS_FETCH_LIMIT = int(os.getenv('NEWS_FETCH_LIMIT', '10'))  # Limit API requests for testing
NEWS_MAX_DAYS_BACK = int(os.getenv('NEWS_MAX_DAYS_BACK', '90'))  # Maximum days to look back for news

# Validate required environment variables based on provider
if NEWS_API_PROVIDER == 'polygon' and not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY not found in environment variables. Please check your .env file.")
elif NEWS_API_PROVIDER == 'alpha_vantage' and not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables. Please check your .env file.")
elif NEWS_API_PROVIDER == 'stock_news' and not STOCK_NEWS_API_KEY:
    raise ValueError("STOCK_NEWS_API_KEY not found in environment variables. Please check your .env file.")
elif NEWS_API_PROVIDER == 'newsapi' and not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY not found in environment variables. Please check your .env file.")

print(f"Configuration loaded:")
print(f"  Stock Ticker: {STOCK_TICKER}")
print(f"  Sharp Move Threshold: {SHARP_MOVE_THRESHOLD}")
print(f"  Data Period: {DATA_PERIOD}")
print(f"  News API Provider: {NEWS_API_PROVIDER}")
print(f"  News Fetch Limit: {NEWS_FETCH_LIMIT}")
print(f"  News Max Days Back: {NEWS_MAX_DAYS_BACK}")

# Show appropriate API key status
if NEWS_API_PROVIDER == 'polygon' and POLYGON_API_KEY:
    print(f"  Polygon.io Key: {'*' * len(POLYGON_API_KEY[:-4]) + POLYGON_API_KEY[-4:]}")
elif NEWS_API_PROVIDER == 'alpha_vantage' and ALPHA_VANTAGE_API_KEY:
    print(f"  Alpha Vantage Key: {'*' * len(ALPHA_VANTAGE_API_KEY[:-4]) + ALPHA_VANTAGE_API_KEY[-4:]}")
elif NEWS_API_PROVIDER == 'stock_news' and STOCK_NEWS_API_KEY:
    print(f"  Stock News API Key: {'*' * len(STOCK_NEWS_API_KEY[:-4]) + STOCK_NEWS_API_KEY[-4:]}")
elif NEWS_API_PROVIDER == 'newsapi' and NEWS_API_KEY:
    print(f"  NewsAPI Key: {'*' * len(NEWS_API_KEY[:-4]) + NEWS_API_KEY[-4:]}")
else:
    print(f"  API Key: NOT SET") 