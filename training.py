import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import ta
import yfinance as yf
import joblib
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Model save path
save_dir = "./models"
model_path = f"{save_dir}/sp500_random_forest_2years_top100.pkl"

# Ensure model directory exists
os.makedirs(save_dir, exist_ok=True)

# Manually define top 100 S&P 500 stocks
sp500_top100 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "JNJ", "V", "NVDA",
    "XOM", "UNH", "WMT", "JPM", "PG", "CVX", "MA", "HD", "ABBV", "LLY",
    "BAC", "KO", "PFE", "PEP", "MRK", "COST", "DIS", "CSCO", "AVGO", "MCD",
    "ADBE", "TMO", "VZ", "CMCSA", "ACN", "DHR", "NFLX", "TXN", "INTC", "ABT",
    "WFC", "NKE", "NEE", "PM", "LIN", "MS", "RTX", "MDT", "UNP", "HON",
    "AMGN", "IBM", "LOW", "SBUX", "ORCL", "CVS", "INTU", "ELV", "GS", "CAT",
    "AMAT", "BLK", "SPGI", "PLD", "ADP", "AXP", "LMT", "DE", "T", "PYPL",
    "C", "BKNG", "QCOM", "GE", "SCHW", "NOW", "BMY", "USB", "MO", "ADI",
    "ISRG", "ZTS", "MDLZ", "GILD", "SYK", "TGT", "REGN", "PNC", "BA", "EQIX",
    "CB", "SO", "BDX", "APD", "MU", "DUK", "MMC", "EW", "SLB", "CL"
]

def fetch_stock_data(symbol):
    """Fetch historical stock data for 2 years."""
    try:
        print(f"Fetching data for {symbol}...")
        stock = yf.Ticker(symbol)

        # Fetch data for the last 2 years
        df = stock.history(start="2022-01-01", end="2023-12-31")

        if df.empty or len(df) < 100:
            print(f"Insufficient data for {symbol}")
            return None

        # Add technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
        df['MACD'] = ta.trend.MACD(close=df['Close']).macd()
        df['MACD_Signal'] = ta.trend.MACD(close=df['Close']).macd_signal()

        # Add target labels
        df['Future_Returns'] = df['Close'].shift(-5) / df['Close'] - 1
        df['label'] = np.select(
            [df['Future_Returns'] > 0.02, df['Future_Returns'] < -0.02],
            [1, -1],
            default=0
        )
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def prepare_features(dataframes):
    """Combine all stock dataframes and prepare features and labels."""
    combined_data = pd.concat(dataframes, ignore_index=True)
    feature_columns = ['RSI', 'SMA_50', 'SMA_200', 'MACD', 'MACD_Signal']
    X = combined_data[feature_columns]
    y = combined_data['label']
    return X, y

def train_random_forest(X, y):
    """Train a Random Forest Classifier."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

    return model, scaler

def main():
    # Fetch and process data for each symbol
    dataframes = []
    for symbol in sp500_top100:
        df = fetch_stock_data(symbol)
        if df is not None:
            dataframes.append(df)

    if not dataframes:
        print("No data available for training.")
        return

    # Prepare features and labels
    print("Preparing features and labels...")
    X, y = prepare_features(dataframes)

    # Train Random Forest model
    print("Training Random Forest model...")
    model, scaler = train_random_forest(X, y)

    # Save the model and scaler
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
