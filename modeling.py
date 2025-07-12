"""
Modeling Script for News Arbitrage AI
Trains a machine learning model to predict sharp stock moves based on news sentiment.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
from config import STOCK_TICKER, MAX_FEATURES, TRAIN_TEST_SPLIT
from dotenv import load_dotenv
load_dotenv()

def load_processed_data(ticker):
    """Load the processed data from CSV file."""
    filename = f"{ticker.lower()}_processed_data.csv"
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Processed data file not found: {filename}")
    
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✅ Loaded processed data: {df.shape}")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df

def prepare_features(df):
    """Prepare features for machine learning."""
    print("🔧 Preparing features...")
    
    # Handle NaN values in combined_text
    print("   Cleaning text data...")
    df['combined_text'] = df['combined_text'].fillna('')  # Replace NaN with empty string
    
    # Check how many rows have actual text content
    non_empty_text = df['combined_text'].str.len() > 0
    print(f"   Rows with text content: {non_empty_text.sum()}/{len(df)} ({non_empty_text.mean()*100:.1f}%)")
    
    # Text vectorization using TF-IDF
    print("   Vectorizing text features...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=1  # Reduce min_df since we have limited text data
    )
    
    # Fit and transform the combined text
    tfidf_features = tfidf.fit_transform(df['combined_text']).toarray()
    
    # Create feature names for TF-IDF features
    tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    
    # Combine all features
    features = pd.DataFrame(tfidf_features, columns=tfidf_feature_names)
    
    # Add sentiment and other numerical features (handle NaN values)
    features['avg_sentiment'] = df['avg_sentiment'].fillna(0).values
    features['sentiment_std'] = df['sentiment_std'].fillna(0).values
    features['article_count'] = df['article_count'].fillna(0).values
    features['volume_change'] = df['volume_change'].fillna(0).values
    features['high_low_spread'] = df['high_low_spread'].fillna(0).values
    
    # Target variable
    target = df['target'].values
    
    print(f"   Feature matrix shape: {features.shape}")
    print(f"   Target distribution:")
    unique, counts = np.unique(target, return_counts=True)
    for val, count in zip(unique, counts):
        label = {-1: 'Sharp Down', 0: 'No Move', 1: 'Sharp Up'}.get(val, str(val))
        print(f"     {label}: {count} ({count/len(target)*100:.1f}%)")
    
    return features, target, tfidf

def train_model(X_train, y_train):
    """Train the machine learning model."""
    print("🤖 Training model...")
    
    # Use LogisticRegression with balanced class weights
    model = LogisticRegression(
        multi_class='ovr',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Model training complete!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Classification report
    target_names = ['Sharp Down (-1)', 'No Move (0)', 'Sharp Up (1)']
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted ->")
    print("Actual ↓   ", "  Down", "  None", "    Up")
    for i, row in enumerate(cm):
        label = ['Down', 'None', '  Up'][i]
        print(f"{label:>8}   ", " ".join(f"{val:>5}" for val in row))
    
    # Feature importance (for logistic regression, we can look at coefficients)
    if hasattr(model, 'coef_'):
        print("\nTop 10 Most Important Features:")
        feature_names = [f'tfidf_{i}' for i in range(len(model.coef_[0])-5)] + \
                       ['avg_sentiment', 'sentiment_std', 'article_count', 'volume_change', 'high_low_spread']
        
        # Get average absolute coefficients across all classes
        avg_coef = np.mean(np.abs(model.coef_), axis=0)
        top_indices = np.argsort(avg_coef)[-10:][::-1]
        
        for idx in top_indices:
            if idx < len(feature_names):
                print(f"  {feature_names[idx]}: {avg_coef[idx]:.4f}")
    
    return y_pred, y_pred_proba, report

def save_model_and_vectorizer(model, tfidf, ticker):
    """Save the trained model and vectorizer."""
    model_filename = f"{ticker.lower()}_model.pkl"
    vectorizer_filename = f"{ticker.lower()}_vectorizer.pkl"
    
    joblib.dump(model, model_filename)
    joblib.dump(tfidf, vectorizer_filename)
    
    print(f"✅ Model saved to {model_filename}")
    print(f"✅ Vectorizer saved to {vectorizer_filename}")
    
    return model_filename, vectorizer_filename

def main(ticker=None):
    """Main function to train and evaluate the model."""
    # Use provided ticker or default from config
    if ticker is None:
        ticker = STOCK_TICKER
    
    print(f"=== News Arbitrage AI - Model Training for {ticker} ===\n")
    
    try:
        # Load processed data
        df = load_processed_data(ticker)
        
        # Prepare features
        features, target, tfidf = prepare_features(df)
        
        # Split data chronologically (important for time series)
        split_point = int(len(features) * TRAIN_TEST_SPLIT)
        X_train, X_test = features[:split_point], features[split_point:]
        y_train, y_test = target[:split_point], target[split_point:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        y_pred, y_pred_proba, report = evaluate_model(model, X_test, y_test)
        
        # Save model and vectorizer
        model_filename, vectorizer_filename = save_model_and_vectorizer(model, tfidf, ticker)
        
        print(f"\n✅ Model training and evaluation complete for {ticker}!")
        print("Next step: Run 'streamlit run app.py' to launch the demo app.")
        
        return model, tfidf, report
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return None, None, None

if __name__ == "__main__":
    # Parse command line arguments
    ticker = None
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    
    main(ticker) 