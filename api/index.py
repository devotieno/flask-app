from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path so we can import the forex_analyzer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forex_analyzer import EnhancedForexAnalyzer

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available currency pairs and their display names
AVAILABLE_PAIRS = ["CAD=X", "BTC-USD", "EURJPY=X", "GBPUSD=X", "GBPJPY=X", "AUDUSD=X", "EURUSD=X"]
PAIR_DISPLAY_NAMES = {
    "CAD=X": "USD/CAD",
    "BTC-USD": "BTC/USD",
    "EURJPY=X": "EUR/JPY",
    "GBPUSD=X": "GBP/USD",
    "GBPJPY=X": "GBP/JPY",
    "AUDUSD=X": "AUD/USD",
    "EURUSD=X": "EUR/USD"
}

def fetch_and_analyze_data(selected_pair):
    current_date = datetime.now()  # Use exact current time
    start_date = current_date - timedelta(days=30)
    intervals = ["5m", "15m", "30m", "60m"]
    results = {}

    for interval in intervals:
        logger.info(f"Fetching data for {selected_pair} - {interval} interval...")
        try:
            # Fetch historical data up to the current date
            data = yf.download(selected_pair,
                              start=start_date.strftime('%Y-%m-%d'),
                              end=current_date.strftime('%Y-%m-%d'),
                              interval=interval,
                              auto_adjust=False,
                              group_by='column',
                              prepost=True)
            if data.empty:
                logger.warning(f"Initial data fetch failed for {interval}. Attempting fallback fetch.")
                data = yf.download(selected_pair,
                                  period="1mo",
                                  interval=interval,
                                  auto_adjust=False,
                                  group_by='column',
                                  prepost=True)
                if data.empty:
                    logger.error(f"Fallback fetch failed for {selected_pair} on {interval} interval.")
                    results[interval] = {"error": f"No data retrieved for {selected_pair} on {interval} interval after fallback."}
                    continue

            # Ensure single-level columns
            if isinstance(data.columns, pd.MultiIndex):
                logger.warning(f"MultiIndex detected despite group_by='column'. Flattening manually.")
                price_level = data.columns.names.index('Price') if 'Price' in data.columns.names else 1
                data.columns = [col[price_level] for col in data.columns]

            # Debug: Log DataFrame structure
            logger.info(f"Data fetched for {selected_pair} on {interval} interval. Shape: {data.shape}, Columns: {list(data.columns)}")

            # Ensure historical data index is timezone-naive
            if data.index.tz is not None:
                data.index = data.index.tz_convert('UTC').tz_localize(None)

            # Fetch the current price and timestamp using yfinance.Ticker
            ticker = yf.Ticker(selected_pair)
            current_info = ticker.history(period="1d", interval="1m")  # Get the most recent 1-minute data
            if not current_info.empty:
                current_time = current_info.index[-1]
                # Ensure current_time is timezone-naive to match historical data
                if current_time.tzinfo is not None:
                    current_time = current_time.tz_convert('UTC').tz_localize(None)
                current_price = current_info['Close'].iloc[-1]

                # Create a new row with the current price and timestamp
                new_row = pd.DataFrame({
                    'Open': [current_price],
                    'High': [current_price],
                    'Low': [current_price],
                    'Close': [current_price],
                    'Volume': [0],  # Volume might not be available; set to 0
                }, index=[current_time])

                # Append the new row to the historical data
                data = pd.concat([data, new_row])
                data = data.sort_index()  # Ensure chronological order
                logger.info(f"Appended current price {current_price} at timestamp {current_time} for {interval} interval.")
            else:
                logger.warning(f"Could not fetch current price for {selected_pair}. Using last available price.")
                current_time = datetime.now()  # Fallback to current time
                current_price = data['Close'].iloc[-1]  # Use last available price
                new_row = pd.DataFrame({
                    'Open': [current_price],
                    'High': [current_price],
                    'Low': [current_price],
                    'Close': [current_price],
                    'Volume': [0],
                }, index=[current_time])
                data = pd.concat([data, new_row])
                data = data.sort_index()
                logger.info(f"Appended last available price {current_price} at timestamp {current_time} for {interval} interval.")

            # Final check to ensure the entire index is timezone-naive
            if data.index.tz is not None:
                data.index = data.index.tz_convert('UTC').tz_localize(None)

            analyzer = EnhancedForexAnalyzer(adapt_weights=True, optimize_params=False)
            analyzer.load_data(data)
            analyzer.calculate_indicators()
            analyzer.generate_signals()
            analyzer.calculate_take_profit_stop_loss()
            backtest = analyzer.backtest(initial_capital=10000)

            results[interval] = {
                'backtest': backtest['stats'],
                'latest_signal': analyzer.data['Combined_Signal'].iloc[-1],
                'latest_time': analyzer.data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': analyzer.data['Close'].iloc[-1],
                'stop_loss_long': analyzer.data['Stop_Loss_Long'].iloc[-1],
                'take_profit_long': analyzer.data['Take_Profit_Long'].iloc[-1],
                'stop_loss_short': analyzer.data['Stop_Loss_Short'].iloc[-1],
                'take_profit_short': analyzer.data['Take_Profit_Short'].iloc[-1],
                'signal_interpretation': analyzer.interpret_signal(analyzer.data['Combined_Signal'].iloc[-1]),
                'market_regime': analyzer.market_regime
            }
        except Exception as e:
            logger.error(f"Error processing {selected_pair} on {interval} interval: {str(e)}")
            results[interval] = {"error": f"Failed to process data for {selected_pair}: {str(e)}"}
    
    return results

@app.route('/')
def index():
    selected_pair = request.args.get('pair', 'AUDUSD=X')
    if selected_pair not in AVAILABLE_PAIRS:
        logger.warning(f"Invalid pair selected: {selected_pair}. Falling back to default 'AUDUSD=X'.")
        selected_pair = "AUDUSD=X"  # Fallback to default if invalid pair
    
    # Get the display name for the selected pair
    display_pair = PAIR_DISPLAY_NAMES.get(selected_pair, selected_pair)
    
    analysis_results = fetch_and_analyze_data(selected_pair)
    return render_template('index.html', 
                         results=analysis_results, 
                         intervals=["5m", "15m", "30m", "60m"],
                         available_pairs=AVAILABLE_PAIRS,
                         selected_pair=selected_pair,
                         display_pair=display_pair,
                         PAIR_DISPLAY_NAMES=PAIR_DISPLAY_NAMES)

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# For Vercel serverless functions
def handler(request, context):
    return app(request)