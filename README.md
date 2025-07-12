# 📈 News Arbitrage AI

A hackathon-ready machine learning pipeline that predicts sharp stock movements based on news sentiment analysis.

## 🎯 Project Overview

This project demonstrates how to:
1. **Collect** historical stock prices and news data
2. **Process** and merge the data with sentiment analysis
3. **Train** a machine learning model to predict sharp price movements
4. **Deploy** an interactive web app for real-time predictions

**Target**: Predict when a stock will have a sharp move (±4% by default) based on news sentiment.

## 📰 **News API Providers**

This project supports multiple news API providers with different pricing and features:

### 🎯 **Alpha Vantage (FREE - Recommended)**  (MUST get PREIMUM USD49.99)
- **Cost**: FREE up to 500 requests/day
- **Volume**: Unlimited articles per request
- **Features**: Built-in sentiment analysis, financial focus, 100+ other endpoints
- **Get API Key** FREE: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
- **GET PREMIUM key** [https://www.alphavantage.co/premium/](https://www.alphavantage.co/premium/)



## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/download the project
# Navigate to project directory

# Run setup script
python3 setup.py
```

### 2. Get API Key (Choose One)

**🎯 RECOMMENDED: Alpha Vantage (FREE)**
1. Go to [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. Copy your API key
4. Edit the `.env` file and replace `your_alpha_vantage_api_key_here` with your actual key

**💰 PREMIUM: Stock News API**
1. Go to [https://stocknewsapi.com/register](https://stocknewsapi.com/register)
2. Sign up and choose a plan
3. Copy your API key
4. Edit the `.env` file and set `NEWS_API_PROVIDER=stock_news`
5. Replace `your_stock_news_api_key_here` with your actual key

**📰 FALLBACK: NewsAPI.org**
1. Go to [https://newsapi.org/register](https://newsapi.org/register)
2. Sign up for a free account
3. Copy your API key
4. Edit the `.env` file and set `NEWS_API_PROVIDER=newsapi`
5. Replace `your_newsapi_key_here` with your actual key

### 3. Run the Pipeline

```bash
# Step 1: Fetch stock and news data
python3 data_acquisition.py

# Step 2: Process and merge data
python3 data_processing.py

# Step 3: Train the model
python3 modeling.py

# Step 4: Launch the demo app
streamlit run app.py
```

## 📊 How It Works

### Data Pipeline

1. **Stock Data**: Uses `yfinance` to fetch historical OHLCV data
2. **News Data**: Uses configurable news providers (Alpha Vantage, Stock News API, or NewsAPI.org)
3. **Feature Engineering**: 
   - Sentiment analysis using TextBlob and provider-native sentiment
   - TF-IDF vectorization of news text
   - Technical indicators (volume change, price spread)
4. **Target Variable**: Sharp moves defined as ±4% daily change

### Multi-Provider Architecture

The system uses a unified interface that supports multiple news providers:

```python
# Automatically uses the configured provider
from news_providers import fetch_news_data

# Fetch news data for any stock
news_df = fetch_news_data('TSLA', days_back=30)
```

### Model

- **Algorithm**: Logistic Regression with balanced class weights
- **Features**: TF-IDF vectors + sentiment scores + technical indicators
- **Classes**: Sharp Down (-1), No Move (0), Sharp Up (1)
- **Evaluation**: Chronological train/test split (80/20)

### Demo App

Interactive Streamlit interface that:
- Accepts custom news text or fetches latest news
- Provides real-time predictions with confidence scores
- Shows sentiment analysis and model statistics
- Displays recent stock performance charts

## 📁 Project Structure

```
bd/
├── requirements.txt          # Python dependencies
├── config.py                # Configuration from .env file
├── news_providers.py        # Multi-provider news API interface
├── setup.py                 # Automated setup script
├── data_acquisition.py      # Fetch stock and news data
├── data_processing.py       # Process and merge data
├── modeling.py              # Train and evaluate model
├── app.py                   # Streamlit demo app
├── .env                     # Environment variables (created by setup)
└── README.md               # This file
```

## 🔧 Configuration

Edit the `.env` file to customize:

```env
# News API Configuration (choose one)
NEWS_API_PROVIDER=alpha_vantage

# Alpha Vantage (FREE - Recommended)
ALPHA_VANTAGE_API_KEY=your_actual_api_key_here

# Stock News API (Paid - Premium features)
STOCK_NEWS_API_KEY=your_actual_api_key_here

# NewsAPI.org (Fallback - Limited free tier)
NEWS_API_KEY=your_actual_api_key_here

# Stock Configuration
STOCK_TICKER=TSLA              # Stock to analyze
SHARP_MOVE_THRESHOLD=0.03      # 3% threshold for "sharp" moves
DATA_PERIOD=2y                 # Historical data period

# Model Configuration
MAX_FEATURES=100               # Max TF-IDF features
TRAIN_TEST_SPLIT=0.8          # Train/test split ratio
```

## 📈 Demo Features

The Streamlit app includes:

- **📰 News Input**: Custom text or latest news from API
- **🔮 Predictions**: Real-time sharp move predictions
- **📊 Confidence Scores**: Probability for each outcome
- **😊 Sentiment Analysis**: News sentiment visualization
- **📈 Performance Charts**: Recent stock performance
- **📋 Model Stats**: Dataset and model information

## 🔄 API Provider Comparison

| Provider | Cost | Volume | Sentiment | Stock Focus | Free Tier |
|----------|------|--------|-----------|-------------|-----------|
| **Alpha Vantage** | FREE | Unlimited | ✅ Built-in | ✅ Financial | 500 req/day |
| **Stock News API** | $19.99/mo | 500/req | ✅ Built-in | ✅ Stock-specific | ❌ |
| **NewsAPI.org** | FREE/Paid | 100/req | ❌ Manual | ❌ General | 1000 req/day |

## ⚠️ Limitations & Disclaimers

- **Educational Purpose**: This is a demo/POC, not for actual trading
- **API Limits**: Each provider has different rate and volume limitations
- **Data Quality**: Limited to publicly available news and stock data
- **Model Accuracy**: Simple model for demonstration purposes
- **No Financial Advice**: Do not use for actual investment decisions

## 🛠️ Technical Requirements

- Python 3.7+
- Internet connection for data fetching
- API key from chosen news provider
- ~500MB disk space for data and models

## 🎯 Hackathon Tips

1. **Focus on Demo**: The Streamlit app is your main deliverable
2. **Choose Right Provider**: Alpha Vantage for free, Stock News API for premium features
3. **Prepare Examples**: Have interesting news examples ready
4. **Know Your Numbers**: Understand model performance metrics
5. **Tell a Story**: Explain the problem, solution, and potential impact
6. **Be Honest**: Acknowledge limitations and areas for improvement

## 🔄 Extending the Project

Potential improvements:
- Multiple stock tickers
- More sophisticated NLP models
- Real-time news streaming
- Advanced technical indicators
- Ensemble methods
- Risk management features

## 📞 Support

If you encounter issues:
1. Check that your API key is correctly set in `.env`
2. Ensure all dependencies are installed (`pip install -r requirements.txt`)
3. Verify internet connection for data fetching
4. Check the console output for specific error messages
5. Try switching to a different news provider in `.env`

---

**Built for hackathons with ❤️ and ☕**

**Powered by Alpha Vantage, Stock News API, and NewsAPI.org** 