"""
Setup script for News Arbitrage AI
Automates environment setup and downloads necessary data.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("=== News Arbitrage AI Setup ===\n")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""# News API Configuration (choose one)
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
""")
        print("✅ .env file created. Please add your API key!")
        print("   🎯 RECOMMENDED: Alpha Vantage (FREE)")
        print("      Get free API key from: https://www.alphavantage.co/support/#api-key")
        print("      Then edit .env and replace 'your_alpha_vantage_api_key_here' with your actual key.")
        print("   💰 PREMIUM: Stock News API ($19.99/month)")
        print("      Get API key from: https://stocknewsapi.com/register")
        print("   📰 FALLBACK: NewsAPI.org (Limited free tier)")
        print("      Get API key from: https://newsapi.org/register")
        return
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return
    
    # Download TextBlob corpora
    if not run_command("python3 -m textblob.download_corpora", "Downloading TextBlob corpora"):
        return
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Make sure your API key is set in the .env file")
    print("   🎯 For Alpha Vantage: Edit ALPHA_VANTAGE_API_KEY in .env")
    print("2. Run the pipeline:")
    print("   python3 data_acquisition.py")
    print("   python3 data_processing.py")
    print("   python3 modeling.py")
    print("3. Launch the demo app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main() 