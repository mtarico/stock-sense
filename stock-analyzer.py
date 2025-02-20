import pandas as pd
import numpy as np
import ta
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from transformers import pipeline
from datetime import datetime, timedelta
import pytz
import warnings
import threading
from colorama import init, Fore, Back, Style
from tabulate import tabulate
import mplfinance as mpf
import logging
import os 

def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')
# Initialize colorama for colored output
init(autoreset=True)

# Setup logging
logging.basicConfig(
    filename='stock_analyzer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = "./models/sp500_random_forest_2years_top100.pkl"
FEATURE_COLUMNS = ['RSI', 'SMA_50', 'SMA_200', 'MACD', 'MACD_Signal']

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

# Load the pre-trained model and scaler
print("Loading model from saved file...")
try:
    saved_data = joblib.load(MODEL_PATH)
    model = saved_data['model']
    scaler = saved_data['scaler']
    print(f"{Fore.GREEN}Model loaded successfully!{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Error loading the model. Ensure the model is properly trained and saved.{Style.RESET_ALL}")
    model, scaler = None, None

class MarketData:
    def __init__(self):
        self.data = {
            'SPY': {'price': None, 'change': None, 'prev_close': None},
            'DOW': {'price': None, 'change': None, 'prev_close': None},
            'VIX': {'price': None, 'change': None},
            'sectors': {},
            'last_update': None
        }
        self.alerts = []
        self.trading_log = []

market = MarketData()

def get_greeting():
    """Generate a greeting based on the current time."""
    est = pytz.timezone("US/Central")
    current_time = datetime.now(est)
    hour = current_time.hour
    
    if hour < 12:
        return f"{Fore.CYAN}Good morning{Style.RESET_ALL}"
    elif hour < 18:
        return f"{Fore.CYAN}Good afternoon{Style.RESET_ALL}"
    else:
        return f"{Fore.CYAN}Good evening{Style.RESET_ALL}"

def fetch_market_status():
    """Fetch comprehensive market status."""
    try:
        # Fetch main indices with 2-day data for change calculation
        for symbol, ticker_symbol in [('SPY', 'SPY'), ('DOW', '^DJI'), ('VIX', '^VIX')]:
            try:
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                    
                    market.data[symbol] = {
                        'price': current,
                        'prev_close': prev,
                        'change': change,
                        'volume': volume,
                        'direction': '🟢' if change >= 0 else '🔴'
                    }
                else:
                    logging.warning(f"No data returned for {symbol}")
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                market.data[symbol] = {
                    'price': 0,
                    'prev_close': 0,
                    'change': 0,
                    'volume': 0,
                    'direction': '-'
                }
        
        # Initialize sectors dict if it doesn't exist
        if 'sectors' not in market.data:
            market.data['sectors'] = {}
        
        # Fetch sector performance
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLY': 'Consumer'
        }
        
        for symbol, name in sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist else 0
                    
                    market.data['sectors'][name] = {
                        'price': current,
                        'prev_close': prev,
                        'change': change,
                        'volume': volume,
                        'direction': '🟢' if change >= 0 else '🔴'
                    }
                else:
                    logging.warning(f"No data returned for sector {name} ({symbol})")
            except Exception as e:
                logging.error(f"Error fetching sector {name}: {e}")
                market.data['sectors'][name] = {
                    'price': 0,
                    'prev_close': 0,
                    'change': 0,
                    'volume': 0,
                    'direction': '-'
                }
        
        market.data['last_update'] = datetime.now(pytz.timezone('US/Eastern'))
        return True
        
    except Exception as e:
        logging.error(f"Market status fetch error: {e}")
        market.data = {
            'SPY': {'price': 0, 'prev_close': 0, 'change': 0, 'volume': 0, 'direction': '-'},
            'DOW': {'price': 0, 'prev_close': 0, 'change': 0, 'volume': 0, 'direction': '-'},
            'VIX': {'price': 0, 'prev_close': 0, 'change': 0, 'volume': 0, 'direction': '-'},
            'sectors': {},
            'last_update': datetime.now(pytz.timezone('US/Eastern'))
        }
        return False

def print_market_status():
    """Print formatted market status."""
    print("\nMarket Overview:")
    print("=" * 50)
    
    # Print main indices
    for index in ['SPY', 'DOW', 'VIX']:
        if market.data[index]['price'] is not None:
            price = market.data[index]['price']
            change = market.data[index]['change']
            direction = market.data[index]['direction']
            color = Fore.GREEN if change >= 0 else Fore.RED
            print(f"{index}: {color}${price:.2f} ({change:+.2f}%) {direction}")
    
    # Print sector performance
    if market.data['sectors']:
        print("\nSector Performance:")
        print("-" * 40)
        for sector, data in market.data['sectors'].items():
            color = Fore.GREEN if data['change'] >= 0 else Fore.RED
            print(f"{sector}: {color}{data['change']:+.2f}% {data['direction']}")
    
    if market.data['last_update']:
        print(f"\nLast Updated: {market.data['last_update'].strftime('%I:%M:%S %p ET')}")
    
    print("=" * 50)

def background_refresh():
    """Background market data refresh."""
    while True:
        fetch_market_status()
        threading.Event().wait(60)  # Refresh every minute

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators."""
    try:
        # Price-based indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
        
        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Trend indicators
        df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
        
        return df
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        return df

def fetch_stock_data(symbol):
    """Fetch and process stock data."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        
        if df.empty:
            print(f"{Fore.RED}No data found for {symbol}{Style.RESET_ALL}")
            return pd.DataFrame()
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        
        # Keep last 6 months and handle missing values
        df = df.ffill().bfill().last("6M")
        return df
        
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_options_data(symbol):
    """Fetch and analyze options data with strike recommendations."""
    try:
        stock = yf.Ticker(symbol)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        
        if not stock.options:
            return None
            
        # Get nearest expiration
        exp_date = stock.options[0]
        opt_chain = stock.option_chain(exp_date)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Filter for options with volume
        calls = calls[calls['volume'] > 10].copy()
        puts = puts[puts['volume'] > 10].copy()
        
        # Analyze call options
        calls['strike_diff'] = abs(calls['strike'] - current_price)
        best_call = calls.nsmallest(3, 'strike_diff').sort_values('volume', ascending=False).iloc[0]
        
        # Analyze put options
        puts['strike_diff'] = abs(puts['strike'] - current_price)
        best_put = puts.nsmallest(3, 'strike_diff').sort_values('volume', ascending=False).iloc[0]
        
        # Calculate put/call ratio
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        return {
            'expiration': exp_date,
            'current_price': current_price,
            'call_volume': call_volume,
            'put_volume': put_volume,
            'put_call_ratio': put_call_ratio,
            'sentiment': 'Bullish' if put_call_ratio < 0.7 else 'Bearish' if put_call_ratio > 1.3 else 'Neutral',
            'best_call': {
                'strike': best_call['strike'],
                'price': best_call['lastPrice'],
                'volume': best_call['volume'],
                'implied_volatility': best_call['impliedVolatility']
            },
            'best_put': {
                'strike': best_put['strike'],
                'price': best_put['lastPrice'],
                'volume': best_put['volume'],
                'implied_volatility': best_put['impliedVolatility']
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching options data: {e}")
        return None
    
def print_options_analysis(options_data, sentiment_score):
    """Print detailed options analysis and recommendations."""
    if not options_data:
        return
        
    print(f"\n{Style.BRIGHT}Options Analysis:{Style.RESET_ALL}")
    print("-" * 50)
    
    current_price = options_data['current_price']
    exp_date = options_data['expiration']
    
    print(f"Expiration Date: {exp_date}")
    print(f"Put/Call Ratio: {options_data['put_call_ratio']:.2f}")
    print(f"Options Sentiment: {options_data['sentiment']}\n")
    
    # Call Options Analysis
    print(f"{Style.BRIGHT}Call Options Recommendation:{Style.RESET_ALL}")
    call = options_data['best_call']
    call_price = call['price']
    call_strike = call['strike']
    call_profit = call_strike - current_price - call_price
    
    print(f"Strike Price: ${call_strike:.2f}")
    print(f"Option Premium: ${call_price:.2f}")
    print(f"Break-even Price: ${(call_strike + call_price):.2f}")
    print(f"Volume: {call['volume']}")
    print(f"Implied Volatility: {(call['implied_volatility'] * 100):.1f}%")
    
    # Put Options Analysis
    print(f"\n{Style.BRIGHT}Put Options Recommendation:{Style.RESET_ALL}")
    put = options_data['best_put']
    put_price = put['price']
    put_strike = put['strike']
    put_profit = current_price - put_strike - put_price
    
    print(f"Strike Price: ${put_strike:.2f}")
    print(f"Option Premium: ${put_price:.2f}")
    print(f"Break-even Price: ${(put_strike - put_price):.2f}")
    print(f"Volume: {put['volume']}")
    print(f"Implied Volatility: {(put['implied_volatility'] * 100):.1f}%")
    
    # Trading Recommendation
    print(f"\n{Style.BRIGHT}Options Trading Recommendation:{Style.RESET_ALL}")
    if sentiment_score > 0.3 and options_data['sentiment'] == 'Bullish':
        print(f"{Fore.GREEN}Consider CALL Option:")
        print(f"Buy Call @ Strike ${call_strike:.2f} for ${call_price:.2f}")
        print(f"Target Profit: ${call_profit:.2f} per contract if price rises above ${(call_strike + call_price):.2f}")
    elif sentiment_score < -0.3 and options_data['sentiment'] == 'Bearish':
        print(f"{Fore.RED}Consider PUT Option:")
        print(f"Buy Put @ Strike ${put_strike:.2f} for ${put_price:.2f}")
        print(f"Target Profit: ${put_profit:.2f} per contract if price falls below ${(put_strike - put_price):.2f}")
    else:
        print(f"{Fore.YELLOW}Market signals are mixed. Consider waiting for clearer direction.")

def fetch_news_sentiment(symbol):
    """Fetch and analyze news sentiment."""
    try:
        stock = yf.Ticker(symbol)
        news_data = stock.news
        
        if not news_data:
            print(f"{Fore.YELLOW}No recent news found for {symbol}{Style.RESET_ALL}")
            return {'avg_sentiment': 0.0, 'news_items': []}
            
        # Process up to 10 most recent news items
        news_items = []
        sentiments = []
        
        for item in news_data[:10]:
            try:
                # Get sentiment for the headline
                sentiment_result = sentiment_pipeline(item['title'])[0]
                sentiment_value = 1 if sentiment_result['label'] == 'POSITIVE' else -1
                sentiments.append(sentiment_value)
                
                # Format timestamp
                news_date = datetime.fromtimestamp(item['providerPublishTime'])
                
                # Create news item
                news_item = {
                    'title': item['title'],
                    'link': item['link'],
                    'date': news_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': sentiment_value,
                    'label': "🟢 POSITIVE" if sentiment_value > 0 else "🔴 NEGATIVE",
                    'publisher': item.get('publisher', 'Unknown')
                }
                news_items.append(news_item)
                
                # Print formatted news (keeping your console output)
                color = Fore.GREEN if sentiment_value > 0 else Fore.RED
                print(f"{color}{news_item['label']}: {news_item['title']}")
                print(f"   Link: {news_item['link']}")
                print(f"   Date: {news_item['date']}\n")
                
            except Exception as e:
                logging.error(f"Error processing news item: {e}")
                continue
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        # Print overall sentiment (keeping your console output)
        sentiment_label = "🟢 Bullish" if avg_sentiment > 0 else "🔴 Bearish"
        print(f"\nOverall Sentiment: {sentiment_label} ({avg_sentiment:.2f})")
        
        # Return both the sentiment data and news items
        return {
            'avg_sentiment': float(avg_sentiment),
            'news_items': news_items,
            'sentiment_label': sentiment_label,
            'last_update': datetime.now(pytz.timezone('US/Eastern')).strftime('%I:%M:%S %p ET')
        }
        
    except Exception as e:
        logging.error(f"Error in news sentiment analysis: {e}")
        return {
            'avg_sentiment': 0.0,
            'news_items': [],
            'sentiment_label': "Neutral",
            'last_update': datetime.now(pytz.timezone('US/Eastern')).strftime('%I:%M:%S %p ET')
        }

def visualize_stock_data(df, symbol):
    """Enhanced stock visualization with user control."""
    show_charts = input("\nWould you like to see technical charts? (y/n): ").lower().strip()
    
    if show_charts == 'y':
        try:
            # Create figure with subplots for different indicators
            fig = plt.figure(figsize=(15, 10))
            
            # Price and Moving Averages
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(df.index, df['Close'], label='Price', color='blue')
            ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='orange')
            ax1.plot(df.index, df['SMA_200'], label='SMA 200', color='red')
            ax1.set_title(f'{symbol} Price Movement')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI
            ax2 = plt.subplot(3, 1, 2)
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_title('RSI Indicator')
            ax2.set_ylabel('RSI')
            ax2.grid(True, alpha=0.3)
            
            # MACD
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='orange')
            ax3.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.3)
            ax3.set_title('MACD')
            ax3.set_ylabel('MACD')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            print("\nPress 'Q' to close the charts and continue...")
            plt.show()
            
        except Exception as e:
            logging.error(f"Error in visualization: {e}")
            print(f"{Fore.RED}Error displaying charts. Check the log for details.{Style.RESET_ALL}")
    else:
        print("\nSkipping technical charts...")

def calculate_ai_confidence(features, sentiment_score, analysis, options_data):
    """Calculate AI confidence score based on multiple factors"""
    try:
        confidence_factors = {
            'technical': 0,
            'sentiment': 0,
            'options': 0,
            'model': 0
        }

        # Technical Analysis Confidence (0-25)
        if analysis:
            # RSI confidence
            rsi_value = analysis['rsi']['value']
            if rsi_value < 30 or rsi_value > 70:  # Strong signal
                confidence_factors['technical'] += 8
            elif 35 < rsi_value < 65:  # Neutral
                confidence_factors['technical'] += 4

            # Trend confidence
            if analysis['trend']['price'] != "Sideways":
                confidence_factors['technical'] += 8
            if analysis['trend']['macd'] in ["Bullish", "Bearish"]:
                confidence_factors['technical'] += 9

        # Sentiment Confidence (0-25)
        if abs(sentiment_score) > 0.7:
            confidence_factors['sentiment'] = 25
        elif abs(sentiment_score) > 0.3:
            confidence_factors['sentiment'] = 15
        else:
            confidence_factors['sentiment'] = 10

        # Options Flow Confidence (0-25)
        if options_data:
            put_call_ratio = options_data.get('put_call_ratio', 1)
            if abs(1 - put_call_ratio) > 0.5:  # Strong options signal
                confidence_factors['options'] = 25
            elif abs(1 - put_call_ratio) > 0.2:
                confidence_factors['options'] = 15
            else:
                confidence_factors['options'] = 10

        # ML Model Confidence (0-25)
        scaled_features = scaler.transform(features)
        model_proba = model.predict_proba(scaled_features)[0]
        max_proba = max(model_proba)
        confidence_factors['model'] = int(max_proba * 25)

        # Calculate total confidence (0-100)
        total_confidence = sum(confidence_factors.values())

        return {
            'total': min(total_confidence, 100),
            'factors': confidence_factors,
            'strength': 'High' if total_confidence > 75 else 'Medium' if total_confidence > 50 else 'Low'
        }

    except Exception as e:
        logging.error(f"Error calculating AI confidence: {e}")
        return {
            'total': 0,
            'factors': confidence_factors,
            'strength': 'Low'
        }

def analyze_stock(symbol, df):
    """Comprehensive stock analysis."""
    try:
        current_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1]
        
        # Calculate trends
        price_trend = "Uptrend" if current_price > sma_50 > sma_200 else "Downtrend" if current_price < sma_50 < sma_200 else "Sideways"
        
        # RSI analysis
        rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        
        # MACD analysis
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        macd_trend = "Bullish" if macd > macd_signal else "Bearish"
        
        # Volume analysis
        avg_volume = df['Volume'].mean()
        current_volume = df['Volume'].iloc[-1]
        volume_trend = "High" if current_volume > avg_volume * 1.5 else "Low" if current_volume < avg_volume * 0.5 else "Normal"
        
        analysis = {
            'price': current_price,
            'rsi': {'value': rsi, 'signal': rsi_signal},
            'trend': {'price': price_trend, 'macd': macd_trend},
            'volume': volume_trend
        }
        
        return analysis
        
    except Exception as e:
        logging.error(f"Error in stock analysis: {e}")
        return None
    
def make_trading_decision(features, sentiment_score, analysis, options_data):
    """Enhanced trading decision logic"""
    try:
        scaled_features = scaler.transform(features)
        model_prediction = model.predict(scaled_features)[0]
        model_proba = model.predict_proba(scaled_features)[0]
        max_proba = max(model_proba)
        
        # Combine multiple signals
        signals = {
            'model': model_prediction,
            'rsi': 1 if analysis['rsi']['value'] < 30 else -1 if analysis['rsi']['value'] > 70 else 0,
            'trend': 1 if analysis['trend']['price'] == "Uptrend" else -1 if analysis['trend']['price'] == "Downtrend" else 0,
            'macd': 1 if analysis['trend']['macd'] == "Bullish" else -1,
            'sentiment': 1 if sentiment_score > 0.3 else -1 if sentiment_score < -0.3 else 0
        }
        
        if options_data and 'sentiment' in options_data:
            signals['options'] = 1 if options_data['sentiment'] == 'Bullish' else -1 if options_data['sentiment'] == 'Bearish' else 0
        
        # Calculate weighted decision
        bullish_signals = sum(1 for signal in signals.values() if signal == 1)
        bearish_signals = sum(1 for signal in signals.values() if signal == -1)
        
        # Calculate confidence factors
        confidence_factors = {
            'technical': int((abs(analysis['rsi']['value'] - 50) / 30) * 25),  # RSI confidence
            'sentiment': int(abs(sentiment_score) * 25),  # Sentiment confidence
            'model': int(max_proba * 25),  # ML model confidence
            'trend': 25 if abs(signals['trend']) > 0 else 0  # Trend confidence
        }
        
        # Calculate total confidence
        total_confidence = sum(confidence_factors.values())
        confidence_level = {
            'total': min(total_confidence, 100),
            'factors': confidence_factors,
            'strength': 'High' if total_confidence > 75 else 'Medium' if total_confidence > 50 else 'Low'
        }
        
        return {
            'signal': "BUY" if bullish_signals >= 3 else "SELL" if bearish_signals >= 3 else "HOLD",
            'confidence': confidence_level,
            'signals': signals,
            'bullish_count': bullish_signals,
            'bearish_count': bearish_signals
        }
            
    except Exception as e:
        logging.error(f"Error in trading decision: {e}")
        return {
            'signal': "HOLD",
            'confidence': {
                'total': 0,
                'factors': {
                    'technical': 0,
                    'sentiment': 0,
                    'model': 0,
                    'trend': 0
                },
                'strength': 'Low'
            },
            'signals': {},
            'bullish_count': 0,
            'bearish_count': 0
        }
        

def main():
    if not model or not scaler:
        print(f"{Fore.RED}Model not loaded. Please ensure the model is properly trained and saved.{Style.RESET_ALL}")
        return
    
    clear_screen()
    print(f"\n{get_greeting()}, Welcome to the Stock Analysis Tool!")
    print(f"Current Time: {datetime.now(pytz.timezone('US/Eastern')).strftime('%I:%M:%S %p ET')}")
    
    # Start background market data refresh
    threading.Thread(target=background_refresh, daemon=True).start()
    fetch_market_status()  # Initial fetch
    
    while True:
        print_market_status()
        print("\nOptions:")
        print("1. Analyze Stock")
        print("2. Refresh Market Data")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == '3':
            clear_screen()
            print(f"\n{Fore.CYAN}Thank you for using the Stock Analysis Tool!{Style.RESET_ALL}")
            break
        elif choice == '2':
            clear_screen()
            fetch_market_status()
            continue
        elif choice == '1':
            clear_screen()
            symbol = input("\nEnter a stock symbol to analyze: ").strip().upper()
            if not symbol:
                print("Please enter a valid stock symbol.")
                continue
                
            stock_data = fetch_stock_data(symbol)
            if stock_data.empty:
                continue
                
            # Display stock analysis
            analysis = analyze_stock(symbol, stock_data)
            if analysis:
                print(f"\n{Style.BRIGHT}Technical Analysis for {symbol}:{Style.RESET_ALL}")
                print(f"Current Price: ${analysis['price']:.2f}")
                print(f"RSI: {analysis['rsi']['value']:.2f} ({analysis['rsi']['signal']})")
                print(f"Trend: {analysis['trend']['price']} (MACD: {analysis['trend']['macd']})")
                print(f"Volume: {analysis['volume']}")
                
            # Get and display sentiment
            sentiment_score = fetch_news_sentiment(symbol)
            
            # Get options data
            options_data = fetch_options_data(symbol)
            if options_data:
                print_options_analysis(options_data, sentiment_score)
            
            # Make prediction with enhanced decision logic
            features = stock_data[FEATURE_COLUMNS].iloc[-1:].copy()
            
            if not features.empty:
                decision = make_trading_decision(features, sentiment_score, analysis, options_data)
                print(f"\n{Style.BRIGHT}Trading Signal:{Style.RESET_ALL} {decision['signal']}")
                print(f"AI Confidence: {decision['confidence']['total']}% ({decision['confidence']['strength']})")
                print("\nConfidence Breakdown:")
                for factor, value in decision['confidence']['factors'].items():
                    print(f"- {factor.title()}: {value}%")
            
            # Show visualization
            visualize_stock_data(stock_data, symbol)
            
            input("\nPress Enter to continue...")
            clear_screen()
        else:
            print("Invalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()