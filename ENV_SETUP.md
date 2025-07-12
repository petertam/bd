# 🔧 Environment Setup Guide

## Alpha Vantage API Key Setup (Recommended - FREE)

### Step 1: Get Your Free API Key
1. Go to [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Fill out the form with your email and basic information
3. You'll receive your API key immediately (no email verification needed)

### Step 2: Configure .env File

After running `python3 setup.py`, edit the `.env` file and replace:

```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

With your actual API key:

```env
ALPHA_VANTAGE_API_KEY=ABC123XYZ789
```

### Step 3: Verify Configuration

Run any of the main scripts to verify your setup:

```bash
python3 data_acquisition.py
```

You should see:
```
Configuration loaded:
  Stock Ticker: TSLA
  News API Provider: alpha_vantage
  News Fetch Limit: 10
  Alpha Vantage Key: ****XYZ789
```

## Complete .env File Format

```env
# News API Configuration (choose one)
NEWS_API_PROVIDER=alpha_vantage

# Alpha Vantage (FREE - Recommended)
# Get free API key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Stock News API (Paid - Premium features)
# Get API key from: https://stocknewsapi.com/register
STOCK_NEWS_API_KEY=your_stock_news_api_key_here

# NewsAPI.org (Fallback - Limited free tier)
# Get API key from: https://newsapi.org/register
NEWS_API_KEY=your_newsapi_key_here

# Stock Configuration
STOCK_TICKER=TSLA
SHARP_MOVE_THRESHOLD=0.03
DATA_PERIOD=2y

# Model Configuration
MAX_FEATURES=100
TRAIN_TEST_SPLIT=0.8

# News API Limits (for testing)
NEWS_FETCH_LIMIT=10
```

## Notes

- **Free Tier**: Alpha Vantage provides 500 requests per day for free
- **No Credit Card**: No payment information required for free tier
- **Instant Access**: API key works immediately after signup
- **Testing Mode**: Currently limited to 10 articles per request (`NEWS_FETCH_LIMIT=10`)

## Troubleshooting

If you see "API Key not found" errors:
1. Check that your `.env` file is in the project root directory
2. Verify there are no extra spaces around the `=` sign
3. Make sure your API key doesn't have quotes around it
4. Restart any running Python processes after editing `.env` 