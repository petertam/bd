
Excellent! A hackathon is all about speed, focus, and a demonstrable Proof of Concept (POC). We will make strategic cuts to the "ideal" plan to deliver something impressive in 48 hours.

Here is your concrete, step-by-step implementation plan.

### **Hackathon Goal: The "Demonstrable Pipeline"**

Our goal is *not* a profitable trading bot. It's to create a pipeline that:
1.  Takes a stock ticker (e.g., `TSLA`) as input.
2.  Pulls its historical price and news data.
3.  Trains a simple model to find correlations.
4.  Shows a prediction: "Based on today's news, a sharp move is [LIKELY / UNLIKELY]."
5.  Presents the results in a clean, visual way (like a Jupyter Notebook or a simple web app).

---

### **The Tech Stack (Keep it Simple!)**

*   **Language:** Python 3
*   **Core Libraries:** `pandas`, `numpy`, `scikit-learn`
*   **Data Acquisition:** `yfinance` (for stock prices), `newsapi-python` (for news)
*   **NLP:** `TextBlob` (for easy sentiment analysis), `scikit-learn`'s `TfidfVectorizer`
*   **Presentation (Choose one):**
    *   **Easy:** Jupyter Notebook (for showing the step-by-step process).
    *   **Impressive:** Streamlit (for building a simple interactive web app in pure Python).

### **Day 1: Data Wrangling & Feature Engineering (The Hard Part)**

**Objective:** Create a single, clean `DataFrame` with features and a target variable.

#### **Step 1: Setup Your Environment (30 mins)**

1.  Create a project folder.
2.  Set up a Python virtual environment.
3.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn yfinance newsapi-python textblob streamlit
    # Download necessary NLTK data for TextBlob
    python -m textblob.download_corpora
    ```
4.  Get a **free API key** from [NewsAPI.org](https://newsapi.org/). You'll need this.

#### **Step 2: Scope & Configuration (15 mins)**

In your Python script or notebook, define your parameters. **Do NOT try to analyze 500 stocks.** Pick one or two volatile, well-known stocks.

```python
# config.py
STOCK_TICKER = 'TSLA' # Tesla is great for this due to volatility and news volume
SHARP_MOVE_THRESHOLD = 0.03 # Let's use 3% to find more impactful events
NEWS_API_KEY = 'YOUR_NEWS_API_KEY_HERE'
DATA_PERIOD = '2y' # Get 2 years of data
```

#### **Step 3: Data Acquisition (2 hours)**

1.  **Get Stock Data:** `yfinance` makes this incredibly easy.

    ```python
    import yfinance as yf
    import pandas as pd

    # Download historical stock data
    stock_data = yf.download(STOCK_TICKER, period=DATA_PERIOD, interval='1d')
    stock_data.reset_index(inplace=True)
    print("Stock Data Head:")
    print(stock_data.head())
    ```

2.  **Get News Data:** Use the `NewsAPI` client. The free plan is limited, so be smart. Fetch news for your time period.

    ```python
    from newsapi import NewsApiClient

    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # Fetch news articles
    # Note: Free tier has limitations, so you might need to paginate or run this once and save to CSV
    all_articles = newsapi.get_everything(q=STOCK_TICKER,
                                          language='en',
                                          sort_by='publishedAt',
                                          page_size=100) # Max page_size

    # Convert to a DataFrame
    news_data = pd.DataFrame(all_articles['articles'])
    news_data['publishedAt'] = pd.to_datetime(news_data['publishedAt'])
    # Extract just the date for merging
    news_data['Date'] = news_data['publishedAt'].dt.date
    print("News Data Head:")
    print(news_data.head())
    ```
    **Hackathon Pro-Tip:** Run the news query once and **save the result to a CSV file** to avoid hitting API rate limits during development. `news_data.to_csv('tsla_news.csv')`

#### **Step 4: Data Processing & Merging (Remaining Day 1)**

This is the most critical step.

1.  **Define the Target Variable:** Create the `sharp_move` column in your `stock_data`.

    ```python
    # Ensure Date is datetime type for merging later
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date

    # Calculate daily change
    stock_data['daily_change'] = stock_data['Close'].pct_change()

    # Define the target variable: 1 for sharp up, -1 for sharp down, 0 for no major move
    def define_sharp_move(change):
        if change > SHARP_MOVE_THRESHOLD:
            return 1 # Sharp Up
        elif change < -SHARP_MOVE_THRESHOLD:
            return -1 # Sharp Down
        else:
            return 0 # No Sharp Move

    stock_data['sharp_move'] = stock_data['daily_change'].apply(define_sharp_move)

    # We are predicting tomorrow's move based on today's news, so shift the target
    stock_data['target'] = stock_data['sharp_move'].shift(-1)
    stock_data.dropna(inplace=True) # Drop last row with NaN target
    ```

2.  **Aggregate News by Day:** You have multiple news articles per day. We need to combine them into one daily "news summary".

    ```python
    # Combine title and description for richer text
    news_data['full_text'] = news_data['title'].fillna('') + ' ' + news_data['description'].fillna('')

    # Group all news text for a single day
    daily_news = news_data.groupby('Date')['full_text'].apply(' '.join).reset_index()
    ```

3.  **Merge Stock and News Data:** Combine the two data sources on the `Date` column.

    ```python
    # Merge the datasets
    final_df = pd.merge(stock_data, daily_news, on='Date', how='left')
    final_df['full_text'].fillna('', inplace=True) # Fill days with no news with an empty string

    print("Final Merged DataFrame:")
    print(final_df[['Date', 'Close', 'daily_change', 'target', 'full_text']].head())
    ```

**End of Day 1 Goal:** You have a `final_df` DataFrame ready for machine learning.

---

### **Day 2: Modeling, Evaluation & Presentation**

**Objective:** Build a model, evaluate it, and create a compelling demo.

#### **Step 5: Feature Engineering from Text (2 hours)**

1.  **Sentiment Analysis:** Use `TextBlob` for a quick and easy sentiment score.

    ```python
    from textblob import TextBlob

    # Apply sentiment analysis
    final_df['sentiment'] = final_df['full_text'].apply(lambda text: TextBlob(text).sentiment.polarity)
    ```

2.  **Text-to-Vector:** Convert the news text into numerical features using `TfidfVectorizer`.

    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=100, stop_words='english') # Limit to top 100 words
    tfidf_features = tfidf.fit_transform(final_df['full_text']).toarray()
    ```

#### **Step 6: Model Training & Evaluation (3 hours)**

1.  **Prepare Final Feature Set:**

    ```python
    # Combine sentiment and TF-IDF features
    # For simplicity, we'll just use the NLP features for this POC
    X = pd.concat([final_df['sentiment'].reset_index(drop=True),
                   pd.DataFrame(tfidf_features)], axis=1)

    y = final_df['target']
    ```

2.  **Train-Test Split (Chronological):** This is crucial for time-series data. DO NOT shuffle.

    ```python
    from sklearn.model_selection import train_test_split

    # Split data chronologically
    split_point = int(len(X) * 0.8) # 80% for training, 20% for testing
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    ```

3.  **Train a Simple Model:** `LogisticRegression` is fast and perfect for this classification task.

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    model = LogisticRegression(multi_class='ovr', class_weight='balanced') # 'balanced' helps with rare events
    model.fit(X_train, y_train)
    ```

4.  **Evaluate:** Look at the `classification_report`. **Precision is your key metric.**

    ```python
    predictions = model.predict(X_test)
    print("Evaluation Results on Test Set:")
    print(classification_report(y_test, predictions, target_names=['Sharp Down (-1)', 'No Move (0)', 'Sharp Up (1)']))
    ```
    *Interpret this:* "When our model predicted 'Sharp Up', it was correct X% of the time (Precision)." This is your money-making metric.

#### **Step 7: Build the Demo (Remaining Day 2)**

This is how you win the hackathon. A live demo is better than a static notebook.

**Option: Streamlit App**

Create a new file `app.py`.

```python
# app.py
import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Assume you have already trained and saved your model and vectorizer
# import joblib
# model = joblib.load('my_model.pkl')
# tfidf = joblib.load('my_tfidf.pkl')
# For hackathon, we can just retrain it on the fly:
# --- (paste all your data loading and model training code here) ---

st.title(f'News Arbitrage AI for {STOCK_TICKER}')

st.header("Today's Prediction")

# Fetch latest news for today
# --- (use newsapi to get today's news) ---
todays_news_text = "Tesla announces new battery technology, shares expected to soar. Elon Musk tweets about dogecoin again." # Example

# Preprocess today's news
todays_sentiment = TextBlob(todays_news_text).sentiment.polarity
todays_tfidf = tfidf.transform([todays_news_text]).toarray()
todays_features = pd.concat([pd.Series([todays_sentiment]),
                             pd.DataFrame(todays_tfidf)], axis=1)

# Make a prediction
prediction = model.predict(todays_features)[0]
prediction_proba = model.predict_proba(todays_features)

st.write(f"**Today's News:** *{todays_news_text}*")

if prediction == 1:
    st.success(f"Prediction: SHARP UP MOVE LIKELY (+{SHARP_MOVE_THRESHOLD*100}%)")
elif prediction == -1:
    st.error(f"Prediction: SHARP DOWN MOVE LIKELY (-{SHARP_MOVE_THRESHOLD*100}%)")
else:
    st.info("Prediction: No significant move expected.")

st.write("Confidence Scores:")
st.write(f"Sharp Down: {prediction_proba[0][0]:.2%}")
st.write(f"No Move: {prediction_proba[0][1]:.2%}")
st.write(f"Sharp Up: {prediction_proba[0][2]:.2%}")


st.header("Model Performance (on historical test data)")
st.text("Classification Report:")
# You can capture the classification_report output as a string and display it
report = classification_report(y_test, predictions, target_names=['Sharp Down', 'No Move', 'Sharp Up'])
st.text(report)
```

**To Run the App:**
`streamlit run app.py`

This gives you an interactive webpage you can show the judges, where you can type in hypothetical news and see the model's prediction in real-time. It's incredibly powerful for a demo.
