"""
Streamlit Demo App for News Arbitrage AI
Interactive web interface for predicting stock moves based on news sentiment.
"""

import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
import os
from datetime import datetime
from config import STOCK_TICKER, SHARP_MOVE_THRESHOLD

# Page configuration
st.set_page_config(
    page_title=f"News Arbitrage AI - {STOCK_TICKER}",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to reduce top margin
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    model_filename = f"{STOCK_TICKER.lower()}_model.pkl"
    vectorizer_filename = f"{STOCK_TICKER.lower()}_vectorizer.pkl"
    
    if not os.path.exists(model_filename) or not os.path.exists(vectorizer_filename):
        return None, None
    
    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    
    return model, vectorizer



def prepare_prediction_features(text, vectorizer):
    """Prepare features for prediction."""
    # Calculate sentiment
    sentiment = TextBlob(text).sentiment.polarity
    
    # Vectorize text
    tfidf_features = vectorizer.transform([text]).toarray()
    
    # Create feature names for TF-IDF features (matching training)
    tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    
    # Create DataFrame with TF-IDF features
    features_df = pd.DataFrame(tfidf_features, columns=tfidf_feature_names)
    
    # Add sentiment and other numerical features (matching training order)
    features_df['avg_sentiment'] = sentiment
    features_df['sentiment_std'] = 0  # Default for single article
    features_df['article_count'] = 1  # Single article
    features_df['volume_change'] = 0  # Default (no stock data during prediction)
    features_df['high_low_spread'] = 0  # Default (no stock data during prediction)
    
    return features_df

def main():
    """Main Streamlit app."""
    st.title(f"📈 News Arbitrage AI for {STOCK_TICKER}")
    st.markdown("*Predicting sharp stock moves based on news sentiment*")
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        st.error("❌ Model or vectorizer not found. Please run the training pipeline first:")
        st.code("""
        1. python3 data_acquisition.py
        2. python3 data_processing.py
        3. python3 modeling.py
        """)
        return
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Stock ticker with full company name
    company_names = {
        'PYPL': 'PayPal Holdings Inc.',
        'TSLA': 'Tesla Inc.',
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'GOOG': 'Alphabet Inc.(C)',
        'MSFT': 'Microsoft Corporation',
        'AMZN': 'Amazon.com Inc.',
        'META': 'Meta Platforms Inc.',
        'NFLX': 'Netflix Inc.',
        'NVDA': 'NVIDIA Corporation',
        'AMD': 'Advanced Micro Devices Inc.'
    }
    
    company_name = company_names.get(STOCK_TICKER, 'Unknown Company')
    st.sidebar.write(f"**Stock:** {STOCK_TICKER}")
    st.sidebar.write(f"({company_name})")
    st.sidebar.write(f"**Threshold:** ±{SHARP_MOVE_THRESHOLD*100:.1f}%")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📰 News Input")
        
        # Custom text input only
        news_text = st.text_area(
            "Enter news text:",
            height=200,
            placeholder=f"Enter news about {STOCK_TICKER} here...",
            value="Google is going to invest Votee."
        )
        
        # Prediction button
        if st.button("🔮 Make Prediction", type="primary") and news_text.strip():
            with st.spinner("Analyzing news..."):
                # Prepare features
                features = prepare_prediction_features(news_text, vectorizer)
                
                # Make prediction
                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]
                
                # Display results
                st.header("🎯 Prediction Results")
                
                # Main prediction
                if prediction == 1:
                    st.success(f"🚀 **SHARP UP MOVE LIKELY** (+{SHARP_MOVE_THRESHOLD*100:.1f}%)")
                    st.balloons()
                elif prediction == -1:
                    st.error(f"📉 **SHARP DOWN MOVE LIKELY** (-{SHARP_MOVE_THRESHOLD*100:.1f}%)")
                else:
                    st.info("📊 **NO SIGNIFICANT MOVE EXPECTED**")
                
                # Confidence scores
                st.subheader("Confidence Scores")
                col_down, col_none, col_up = st.columns(3)
                
                with col_down:
                    st.metric("Sharp Down \u21E9", f"{prediction_proba[0]:.1%} \u21E9")
                
                with col_none:
                    st.metric("No Move", f"{prediction_proba[1]:.1%}")
                
                with col_up:
                    st.metric("Sharp Up \u21E7", f"{prediction_proba[2]:.1%} \u21E7")
                
                # Sentiment analysis
                sentiment = TextBlob(news_text).sentiment.polarity
                st.subheader("News Sentiment Analysis")
                
                if sentiment > 0.1:
                    st.success(f"😊 Positive sentiment: {sentiment:.3f}")
                elif sentiment < -0.1:
                    st.error(f"😟 Negative sentiment: {sentiment:.3f}")
                else:
                    st.info(f"😐 Neutral sentiment: {sentiment:.3f}")
                
                # Progress bar for sentiment
                st.progress((sentiment + 1) / 2)  # Convert -1,1 to 0,1 range
    
    with col2:
        st.header("📊 Model Info")
        
        # Load and display model performance
        try:
            processed_filename = f"{STOCK_TICKER.lower()}_processed_data.csv"
            if os.path.exists(processed_filename):
                df = pd.read_csv(processed_filename)
                
                st.subheader("Dataset Stats")
                st.write(f"**Total samples:** {len(df)}")
                st.write(f"**Date range:** {df['Date'].min()} to {df['Date'].max()}")
                
                # Target distribution
                target_counts = df['target'].value_counts().sort_index()
                st.subheader("Historical Move Distribution")
                
                for target, count in target_counts.items():
                    label = {-1: "Sharp Down", 0: "No Move", 1: "Sharp Up"}.get(target, str(target))
                    percentage = count / len(df) * 100
                    st.write(f"**{label}:** {count} ({percentage:.1f}%)")
                
                # Recent performance visualization
                st.subheader("Recent Stock Performance")
                recent_data = df.tail(30)
                
                if not recent_data.empty:
                    chart_data = pd.DataFrame({
                        'Date': pd.to_datetime(recent_data['Date']),
                        'Daily Change': recent_data['daily_change'] * 100
                    })
                    chart_data.set_index('Date', inplace=True)
                    st.line_chart(chart_data)
        
        except Exception as e:
            st.warning(f"Could not load model stats: {e}")
        
        # Instructions
        st.subheader("📋 How to Use")
        st.write("""
        1. **Enter news text** about the stock in the text box
        2. **Click 'Make Prediction'** to analyze the news
        3. **View the prediction** and confidence scores
        4. **Check sentiment analysis** for additional insights
        """)
        
        st.subheader("⚠️ Disclaimer")
        st.warning("""
        This is a demo for educational purposes only. 
        Do not use for actual trading decisions. 
        Past performance does not guarantee future results.
        """)

if __name__ == "__main__":
    main() 