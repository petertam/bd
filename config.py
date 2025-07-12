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
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')  # For Alpha Vantage

# News API Selection (priority order)
NEWS_API_PROVIDER = os.getenv('NEWS_API_PROVIDER', 'alpha_vantage')  # alpha_vantage

# Stock Configuration
STOCK_TICKER = os.getenv('STOCK_TICKER', 'PYPL')
SHARP_MOVE_THRESHOLD = float(os.getenv('SHARP_MOVE_THRESHOLD', '0.03'))
DATA_PERIOD = os.getenv('DATA_PERIOD', '2y')

# Model Configuration
MAX_FEATURES = int(os.getenv('MAX_FEATURES', '2500'))
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', '0.8'))

# News API Limits (for testing)
NEWS_FETCH_LIMIT = int(os.getenv('NEWS_FETCH_LIMIT', '10'))  # Limit API requests for testing
NEWS_MAX_DAYS_BACK = int(os.getenv('NEWS_MAX_DAYS_BACK', '90'))  # Maximum days to look back for news

# Validate required environment variables based on provider
if NEWS_API_PROVIDER == 'alpha_vantage' and not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables. Please check your .env file.")

print(f"Configuration loaded:")
print(f"  Stock Ticker: {STOCK_TICKER}")
print(f"  Sharp Move Threshold: {SHARP_MOVE_THRESHOLD}")
print(f"  Data Period: {DATA_PERIOD}")
print(f"  News API Provider: {NEWS_API_PROVIDER}")
print(f"  News Fetch Limit: {NEWS_FETCH_LIMIT}")
print(f"  News Max Days Back: {NEWS_MAX_DAYS_BACK}")

# Show appropriate API key status
if NEWS_API_PROVIDER == 'alpha_vantage' and ALPHA_VANTAGE_API_KEY:
    print(f"  Alpha Vantage Key: {'*' * len(ALPHA_VANTAGE_API_KEY[:-4]) + ALPHA_VANTAGE_API_KEY[-4:]}")
else:
    print(f"  API Key: NOT SET") 