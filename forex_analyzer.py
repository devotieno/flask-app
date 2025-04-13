import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Union, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
from datetime import datetime, timedelta
import ccxt
import requests
from scipy.optimize import minimize

class EnhancedForexAnalyzer:
    def __init__(self, adapt_weights: bool = True, optimize_params: bool = False):
        self.data: Optional[pd.DataFrame] = None
        self.adapt_weights = adapt_weights
        self.optimize_params = optimize_params
        self.default_settings: Dict = {
            'sma_short': 10,
            'sma_long': 30,
            'rsi_period': 10,
            'macd_fast': 8,
            'macd_slow': 21,
            'macd_signal': 5,
            'bb_period': 15,
            'bb_std': 2,
            'atr_period': 10,
            'stoch_k': 10,
            'stoch_d': 3,
            'adx_period': 10
        }
        self.settings = self.default_settings.copy()
        self.market_regime: str = "unknown"
        self.indicator_weights: Dict[str, float] = {
            'SMA_Cross': 0.3,
            'RSI_Signal': 0.2,
            'MACD_Signal': 0.3,
            'BB_Signal': 0.1,
            'ADX_Signal': 0.1,
        }
        self.risk_params: Dict = {
            'max_risk_per_trade': 0.02,
            'risk_reward_ratio': 2,
            'trailing_stop': True,
            'trailing_pct': 0.5,
            'max_drawdown_limit': 0.15,
            'max_correlated_trades': 3,
            'volatility_adjustment': True
        }
        self.validation_results: Dict = {}

    def load_data(self, data: pd.DataFrame) -> None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Debug: Log the DataFrame's columns and index type
        print(f"DataFrame columns: {data.columns}")
        print(f"Columns type: {type(data.columns)}")
        print(f"Index type: {type(data.index)}")
        print(f"First few index values: {data.index[:5]}")
        print(f"Last few index values: {data.index[-5:]}")
        
        # Check if data has a MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            print("MultiIndex detected. Flattening to single-level columns using price field.")
            data = data.copy()
            price_level = data.columns.names.index('Price') if 'Price' in data.columns.names else 1
            data.columns = [col[price_level] if isinstance(col, tuple) else col for col in data.columns]
        
        # Ensure columns are capitalized
        data.columns = [col.capitalize() for col in data.columns]
        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])

        # Validate required columns
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Ensure index is DatetimeIndex and timezone-naive
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            print("Timezone-aware index detected. Converting to timezone-naive.")
            data.index = data.index.tz_convert('UTC').tz_localize(None)
        
        # Handle missing values
        if data.isnull().any().any():
            missing_pct = data.isnull().mean().mean() * 100
            print(f"Warning: Data contains {missing_pct:.2f}% missing values. Applying interpolation.")
            data = self._preprocess_missing_data(data)
        
        # Validate data length
        min_required_points = max(self.settings.values()) * 3
        if len(data) < min_required_points:
            raise ValueError(f"Insufficient data points. Need at least {min_required_points}.")
        
        self.data = data.copy()
        self._detect_market_regime()
        if self.optimize_params:
            self._optimize_parameters()

    def _preprocess_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data_ffill = data.fillna(method='ffill', limit=3)
        data_clean = data_ffill.interpolate(method='time')
        data_clean = data_clean.fillna(method='bfill')
        if data_clean.isnull().any().any():
            warnings.warn("Some missing values remain after preprocessing.")
        return data_clean

    def _detect_market_regime(self) -> None:
        if self.data is None or len(self.data) < 100:
            self.market_regime = "unknown"
            return
        returns = self.data['Close'].pct_change().dropna()
        recent_vol = returns[-20:].std()
        historical_vol = returns[:-20].std()
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        price = self.data['Close']
        sma_short = price.rolling(window=self.settings['sma_short']).mean()
        sma_long = price.rolling(window=self.settings['sma_long']).mean()
        sma_short_slope = (sma_short.iloc[-1] - sma_short.iloc[-20]) / sma_short.iloc[-20] if len(sma_short) >= 20 else 0
        sma_long_slope = (sma_long.iloc[-1] - sma_long.iloc[-20]) / sma_long.iloc[-20] if len(sma_long) >= 20 else 0
        if vol_ratio > 1.5:
            self.market_regime = "volatile"
        elif abs(sma_short_slope) > 0.03 and abs(sma_long_slope) > 0.01 and np.sign(sma_short_slope) == np.sign(sma_long_slope):
            self.market_regime = "trending"
            if sma_short_slope > 0:
                self.market_regime = "uptrend"
            else:
                self.market_regime = "downtrend"
        else:
            self.market_regime = "ranging"
        print(f"Detected market regime: {self.market_regime}")
        if self.adapt_weights:
            self._adapt_indicator_weights()

    def _adapt_indicator_weights(self) -> None:
        if self.market_regime in ["uptrend", "downtrend"]:
            self.indicator_weights = {'SMA_Cross': 0.4, 'RSI_Signal': 0.1, 'MACD_Signal': 0.4, 'BB_Signal': 0.05, 'ADX_Signal': 0.05}
        elif self.market_regime == "ranging":
            self.indicator_weights = {'SMA_Cross': 0.2, 'RSI_Signal': 0.3, 'MACD_Signal': 0.2, 'BB_Signal': 0.25, 'ADX_Signal': 0.05}
        elif self.market_regime == "volatile":
            self.indicator_weights = {'SMA_Cross': 0.3, 'RSI_Signal': 0.2, 'MACD_Signal': 0.3, 'BB_Signal': 0.1, 'ADX_Signal': 0.1}
        else:
            self.indicator_weights = {'SMA_Cross': 0.3, 'RSI_Signal': 0.2, 'MACD_Signal': 0.3, 'BB_Signal': 0.1, 'ADX_Signal': 0.1}
        print(f"Adjusted indicator weights for {self.market_regime} regime: {self.indicator_weights}")

    def _optimize_parameters(self) -> None:
        if self.data is None or len(self.data) < 200:
            print("Cannot optimize parameters - insufficient data")
            return
        def objective_function(params):
            sma_short, sma_long, rsi_period, macd_fast, macd_slow = map(int, params)
            if sma_short >= sma_long or macd_fast >= macd_slow:
                return 1000000
            temp_settings = self.settings.copy()
            temp_settings.update({'sma_short': sma_short, 'sma_long': sma_long, 'rsi_period': rsi_period, 'macd_fast': macd_fast, 'macd_slow': macd_slow})
            original_settings = self.settings.copy()
            self.settings = temp_settings
            try:
                self.calculate_indicators()
                signals_df = self.generate_signals()
                balance = 10000
                position = 0
                entry_price = 0
                for i in range(100, len(signals_df)):
                    signal = signals_df['Combined_Signal'].iloc[i]
                    close_price = self.data['Close'].iloc[i]
                    if signal > 0.5 and position == 0:
                        position = balance / close_price
                        entry_price = close_price
                        balance = 0
                    elif signal < -0.5 and position > 0:
                        balance = position * close_price
                        position = 0
                if position > 0:
                    balance = position * self.data['Close'].iloc[-1]
                self.settings = original_settings
                return -balance
            except Exception:
                self.settings = original_settings
                return 1000000
        initial_params = [self.settings['sma_short'], self.settings['sma_long'], self.settings['rsi_period'], self.settings['macd_fast'], self.settings['macd_slow']]
        bounds = [(5, 30), (20, 200), (7, 21), (5, 20), (15, 40)]
        print("Optimizing parameters...")
        result = minimize(objective_function, initial_params, method='SLSQP', bounds=bounds, options={'maxiter': 50})
        if result.success:
            optimized_params = list(map(int, result.x))
            self.settings.update({'sma_short': optimized_params[0], 'sma_long': optimized_params[1], 'rsi_period': optimized_params[2], 'macd_fast': optimized_params[3], 'macd_slow': optimized_params[4]})
            print(f"Optimized settings: {self.settings}")
        else:
            print(f"Optimization failed: {result.message}")

    def calculate_indicators(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded.")
        self.data['SMA_short'] = self.data['Close'].rolling(window=self.settings['sma_short']).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=self.settings['sma_long']).mean()
        self.data['EMA_short'] = self.data['Close'].ewm(span=self.settings['sma_short'], adjust=False).mean()
        self.data['EMA_long'] = self.data['Close'].ewm(span=self.settings['sma_long'], adjust=False).mean()
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.settings['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.settings['rsi_period']).mean()
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        exp1 = self.data['Close'].ewm(span=self.settings['macd_fast'], adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=self.settings['macd_slow'], adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=self.settings['macd_signal'], adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']
        self.data['BB_middle'] = self.data['Close'].rolling(window=self.settings['bb_period']).mean()
        bb_std = self.data['Close'].rolling(window=self.settings['bb_period']).std(ddof=0)
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * self.settings['bb_std'])
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * self.settings['bb_std'])
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.data['ATR'] = true_range.rolling(window=self.settings['atr_period']).mean()
        low_min = self.data['Low'].rolling(window=self.settings['stoch_k']).min()
        high_max = self.data['High'].rolling(window=self.settings['stoch_k']).max()
        self.data['Stoch_K'] = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=self.settings['stoch_d']).mean()
        plus_dm = self.data['High'].diff()
        minus_dm = self.data['Low'].diff(-1).abs()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr1 = self.data['High'] - self.data['Low']
        tr2 = (self.data['High'] - self.data['Close'].shift()).abs()
        tr3 = (self.data['Low'] - self.data['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.settings['adx_period']).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.settings['adx_period']).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.settings['adx_period']).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        self.data['ADX'] = dx.rolling(window=self.settings['adx_period']).mean()
        self.data['+DI'] = plus_di
        self.data['-DI'] = minus_di
        if 'Volume' in self.data.columns and not self.data['Volume'].isnull().all():
            self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
            self.data['OBV'] = (np.sign(self.data['Close'].diff()) * self.data['Volume']).fillna(0).cumsum()
        self.data['Volatility_20d'] = self.data['Close'].pct_change().rolling(window=20).std() * np.sqrt(20)
        return self.data

    def calculate_take_profit_stop_loss(self, atr_multiplier: float = 1.5, risk_reward_ratio: float = None) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded.")
        if risk_reward_ratio is None:
            risk_reward_ratio = self.risk_params['risk_reward_ratio']
        adjusted_atr_multiplier = atr_multiplier
        if self.risk_params['volatility_adjustment'] and 'Volatility_20d' in self.data.columns:
            vol = self.data['Volatility_20d'].iloc[-1]
            hist_vol = self.data['Volatility_20d'].mean()
            vol_z = (vol - hist_vol) / self.data['Volatility_20d'].std() if self.data['Volatility_20d'].std() > 0 else 0
            volatility_factor = max(0.5, min(1.5, 1 - (vol_z * 0.2)))
            adjusted_atr_multiplier *= volatility_factor
        if self.market_regime == "volatile":
            adjusted_atr_multiplier *= 1.2
        elif self.market_regime == "ranging":
            adjusted_atr_multiplier *= 0.8
        self.data['Stop_Loss_Long'] = self.data['Close'] - (self.data['ATR'] * adjusted_atr_multiplier)
        self.data['Take_Profit_Long'] = self.data['Close'] + (self.data['ATR'] * adjusted_atr_multiplier * risk_reward_ratio)
        self.data['Stop_Loss_Short'] = self.data['Close'] + (self.data['ATR'] * adjusted_atr_multiplier)
        self.data['Take_Profit_Short'] = self.data['Close'] - (self.data['ATR'] * adjusted_atr_multiplier * risk_reward_ratio)
        return self.data

    def generate_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded.")
        self.data['Signal'] = 0
        self.data['SMA_Cross'] = np.where(
            (self.data['Close'] > self.data['SMA_short']) & (self.data['SMA_short'] > self.data['SMA_long']), 1,
            np.where((self.data['Close'] < self.data['SMA_short']) & (self.data['SMA_short'] < self.data['SMA_long']), -1, 0)
        )
        rsi_lower = 40 if self.market_regime == "uptrend" else 20 if self.market_regime == "downtrend" else 30
        rsi_upper = 80 if self.market_regime == "uptrend" else 60 if self.market_regime == "downtrend" else 70
        self.data['RSI_Signal'] = np.where(self.data['RSI'] < rsi_lower, 1, np.where(self.data['RSI'] > rsi_upper, -1, 0))
        self.data['MACD_Signal'] = np.where(
            (self.data['MACD'] > self.data['Signal_Line']) & (self.data['MACD_Histogram'] > self.data['MACD_Histogram'].shift()), 1,
            np.where((self.data['MACD'] < self.data['Signal_Line']) & (self.data['MACD_Histogram'] < self.data['MACD_Histogram'].shift()), -1, 0)
        )
        if self.market_regime == "ranging":
            self.data['BB_Signal'] = np.where(self.data['Close'] < self.data['BB_lower'], 1, np.where(self.data['Close'] > self.data['BB_upper'], -1, 0))
        else:
            self.data['BB_Signal'] = np.where(
                (self.data['Close'] < self.data['BB_lower']) | 
                ((self.data['Close'] > self.data['BB_upper']) & (self.data['Close'] > self.data['Close'].shift()) & 
                 (self.data['Close'].shift() > self.data['Close'].shift(2)) & (self.market_regime == "uptrend")), 1,
                np.where(
                    (self.data['Close'] > self.data['BB_upper']) | 
                    ((self.data['Close'] < self.data['BB_lower']) & (self.data['Close'] < self.data['Close'].shift()) & 
                     (self.data['Close'].shift() < self.data['Close'].shift(2)) & (self.market_regime == "downtrend")), -1, 0)
            )
        self.data['ADX_Signal'] = np.where(
            (self.data['ADX'] > 25) & (self.data['+DI'] > self.data['-DI']), 1,
            np.where((self.data['ADX'] > 25) & (self.data['+DI'] < self.data['-DI']), -1, 0)
        )
        volume_factor = np.where(
            self.data['Volume_Ratio'] > 1.5, 1.2, 
            np.where(self.data['Volume_Ratio'] < 0.5, 0.8, 1.0)
        ) if 'Volume_Ratio' in self.data.columns else 1.0
        self.data['Combined_Signal'] = (
            self.data['SMA_Cross'] * self.indicator_weights['SMA_Cross'] +
            self.data['RSI_Signal'] * self.indicator_weights['RSI_Signal'] +
            self.data['MACD_Signal'] * self.indicator_weights['MACD_Signal'] +
            self.data['BB_Signal'] * self.indicator_weights['BB_Signal'] +
            self.data['ADX_Signal'] * self.indicator_weights['ADX_Signal']
        ) * volume_factor
        if self.market_regime == "uptrend":
            self.data.loc[self.data['Combined_Signal'] < 0, 'Combined_Signal'] *= 0.9
        elif self.market_regime == "downtrend":
            self.data.loc[self.data['Combined_Signal'] > 0, 'Combined_Signal'] *= 0.9
        self.data['Timing_Quality'] = self._calculate_timing_quality()
        return self.data

    def _calculate_timing_quality(self) -> pd.Series:
        timing = pd.Series(index=self.data.index, data=1.0)
        indicators_agree = (np.sign(self.data['SMA_Cross']) == np.sign(self.data['MACD_Signal'])).astype(int)
        in_pullback = ((self.data['Close'] < self.data['SMA_short']) & (self.data['Close'] > self.data['SMA_long']) & 
                       (self.data['SMA_short'] > self.data['SMA_long'])) | \
                      ((self.data['Close'] > self.data['SMA_short']) & (self.data['Close'] < self.data['SMA_long']) & 
                       (self.data['SMA_short'] < self.data['SMA_long']))
        near_support = (self.data['Close'] - self.data['BB_lower']).abs() < (0.2 * self.data['ATR'])
        near_resistance = (self.data['Close'] - self.data['BB_upper']).abs() < (0.2 * self.data['ATR'])
        timing = timing + indicators_agree * 0.2
        timing = np.where(in_pullback, timing + 0.2, timing)
        timing = np.where(near_support & (self.data['Combined_Signal'] > 0), timing + 0.2, timing)
        timing = np.where(near_resistance & (self.data['Combined_Signal'] < 0), timing + 0.2, timing)
        return timing

    def interpret_signal(self, signal_value: float) -> str:
        if signal_value > 0.5:
            return "Strong Buy"
        elif 0.15 < signal_value <= 0.5:
            return "Weak Buy"
        elif -0.15 <= signal_value <= 0.15:
            return "Neutral"
        elif -0.5 <= signal_value < -0.15:
            return "Weak Sell"
        else:
            return "Strong Sell"

    def analyze_volatility(self) -> Dict[str, float]:
        if self.data is None:
            raise ValueError("No data loaded.")
        volatility_metrics = {
            'std_dev': self.data['Close'].std(),
            'avg_atr': self.data['ATR'].mean(),
            'max_drawdown': self._calculate_max_drawdown(),
            'daily_range': ((self.data['High'] - self.data['Low']) / self.data['Low']).mean() * 100
        }
        returns = self.data['Close'].pct_change().dropna()
        if len(returns) >= 40:
            volatility_metrics['recent_vol'] = returns[-20:].std() * np.sqrt(252)
            volatility_metrics['historical_vol'] = returns[-40:-20].std() * np.sqrt(252)
            volatility_metrics['vol_ratio'] = volatility_metrics['recent_vol'] / volatility_metrics['historical_vol']
        try:
            from scipy.stats import skew, kurtosis
            volatility_metrics['skewness'] = skew(returns)
            volatility_metrics['kurtosis'] = kurtosis(returns)
            volatility_metrics['is_fat_tailed'] = volatility_metrics['kurtosis'] > 3
            volatility_metrics['is_asymmetric'] = abs(volatility_metrics['skewness']) > 0.5
        except ImportError:
            print("SciPy not available for skew/kurtosis calculation")
        squared_returns = returns ** 2
        volatility_metrics['volatility_clustering'] = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        return volatility_metrics

    def _calculate_max_drawdown(self) -> float:
        if self.data is None or len(self.data) < 2:
            return 0.0
        rolling_max = self.data['Close'].cummax()
        drawdown = (self.data['Close'] / rolling_max - 1.0)
        return drawdown.min()

    def calculate_correlation(self, other_data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded.")
        if 'Close' not in other_data.columns:
            raise ValueError("Other data must contain a 'Close' column")
        merged = pd.DataFrame({'asset1': self.data['Close'], 'asset2': other_data['Close']})
        rolling_corr = merged['asset1'].rolling(window=window).corr(merged['asset2'])
        self.data['correlation'] = rolling_corr
        return self.data

    def backtest(self, initial_capital: float = 10000, position_size: float = 0.1, commission_pct: float = 0.001) -> Dict:
        if self.data is None:
            raise ValueError("No data loaded.")
        if 'Combined_Signal' not in self.data.columns:
            self.generate_signals()
        if 'Stop_Loss_Long' not in self.data.columns:
            self.calculate_take_profit_stop_loss()
        backtest_results = self.data.copy()
        backtest_results['position'] = 0
        backtest_results['capital'] = initial_capital
        backtest_results['equity'] = initial_capital
        backtest_results['entry_price'] = 0.0
        backtest_results['stop_loss'] = 0.0
        backtest_results['take_profit'] = 0.0
        backtest_results['trade_active'] = False
        backtest_results['trade_pnl'] = 0.0
        for i in range(1, len(backtest_results)):
            backtest_results.iloc[i, backtest_results.columns.get_loc('capital')] = backtest_results.iloc[i-1, backtest_results.columns.get_loc('capital')]
            backtest_results.iloc[i, backtest_results.columns.get_loc('position')] = backtest_results.iloc[i-1, backtest_results.columns.get_loc('position')]
            backtest_results.iloc[i, backtest_results.columns.get_loc('entry_price')] = backtest_results.iloc[i-1, backtest_results.columns.get_loc('entry_price')]
            backtest_results.iloc[i, backtest_results.columns.get_loc('stop_loss')] = backtest_results.iloc[i-1, backtest_results.columns.get_loc('stop_loss')]
            backtest_results.iloc[i, backtest_results.columns.get_loc('take_profit')] = backtest_results.iloc[i-1, backtest_results.columns.get_loc('take_profit')]
            backtest_results.iloc[i, backtest_results.columns.get_loc('trade_active')] = backtest_results.iloc[i-1, backtest_results.columns.get_loc('trade_active')]
            current_price = backtest_results['Close'].iloc[i]
            current_position = backtest_results['position'].iloc[i]
            current_capital = backtest_results['capital'].iloc[i]
            entry_price = backtest_results['entry_price'].iloc[i]
            stop_loss = backtest_results['stop_loss'].iloc[i]
            take_profit = backtest_results['take_profit'].iloc[i]
            if current_position != 0:
                if current_position == 1:
                    if backtest_results['Low'].iloc[i] <= stop_loss:
                        trade_pnl = (stop_loss / entry_price - 1) * current_position * current_capital * position_size
                        commission = abs(current_capital * position_size * commission_pct)
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_pnl')] = trade_pnl
                        backtest_results.iloc[i, backtest_results.columns.get_loc('capital')] = current_capital + trade_pnl - commission
                        backtest_results.iloc[i, backtest_results.columns.get_loc('position')] = 0
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_active')] = False
                    elif backtest_results['High'].iloc[i] >= take_profit:
                        trade_pnl = (take_profit / entry_price - 1) * current_position * current_capital * position_size
                        commission = abs(current_capital * position_size * commission_pct)
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_pnl')] = trade_pnl
                        backtest_results.iloc[i, backtest_results.columns.get_loc('capital')] = current_capital + trade_pnl - commission
                        backtest_results.iloc[i, backtest_results.columns.get_loc('position')] = 0
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_active')] = False
                elif current_position == -1:
                    if backtest_results['High'].iloc[i] >= stop_loss:
                        trade_pnl = (1 - stop_loss / entry_price) * abs(current_position) * current_capital * position_size
                        commission = abs(current_capital * position_size * commission_pct)
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_pnl')] = trade_pnl
                        backtest_results.iloc[i, backtest_results.columns.get_loc('capital')] = current_capital + trade_pnl - commission
                        backtest_results.iloc[i, backtest_results.columns.get_loc('position')] = 0
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_active')] = False
                    elif backtest_results['Low'].iloc[i] <= take_profit:
                        trade_pnl = (1 - take_profit / entry_price) * abs(current_position) * current_capital * position_size
                        commission = abs(current_capital * position_size * commission_pct)
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_pnl')] = trade_pnl
                        backtest_results.iloc[i, backtest_results.columns.get_loc('capital')] = current_capital + trade_pnl - commission
                        backtest_results.iloc[i, backtest_results.columns.get_loc('position')] = 0
                        backtest_results.iloc[i, backtest_results.columns.get_loc('trade_active')] = False
            if not backtest_results['trade_active'].iloc[i]:
                combined_signal = backtest_results['Combined_Signal'].iloc[i]
                if combined_signal > 0.5:
                    stop_price = backtest_results['Stop_Loss_Long'].iloc[i]
                    target_price = backtest_results['Take_Profit_Long'].iloc[i]
                    backtest_results.iloc[i, backtest_results.columns.get_loc('position')] = 1
                    backtest_results.iloc[i, backtest_results.columns.get_loc('entry_price')] = current_price
                    backtest_results.iloc[i, backtest_results.columns.get_loc('stop_loss')] = stop_price
                    backtest_results.iloc[i, backtest_results.columns.get_loc('take_profit')] = target_price
                    backtest_results.iloc[i, backtest_results.columns.get_loc('trade_active')] = True
                elif combined_signal < -0.5:
                    stop_price = backtest_results['Stop_Loss_Short'].iloc[i]
                    target_price = backtest_results['Take_Profit_Short'].iloc[i]
                    backtest_results.iloc[i, backtest_results.columns.get_loc('position')] = -1
                    backtest_results.iloc[i, backtest_results.columns.get_loc('entry_price')] = current_price
                    backtest_results.iloc[i, backtest_results.columns.get_loc('stop_loss')] = stop_price
                    backtest_results.iloc[i, backtest_results.columns.get_loc('take_profit')] = target_price
                    backtest_results.iloc[i, backtest_results.columns.get_loc('trade_active')] = True
            if backtest_results['position'].iloc[i] != 0:
                position_value = (current_price / entry_price - 1) * current_capital * position_size if backtest_results['position'].iloc[i] == 1 else \
                                 (1 - current_price / entry_price) * current_capital * position_size
                backtest_results.iloc[i, backtest_results.columns.get_loc('equity')] = current_capital + position_value
            else:
                backtest_results.iloc[i, backtest_results.columns.get_loc('equity')] = current_capital
        backtest_stats = self._calculate_backtest_statistics(backtest_results)
        return {'results': backtest_results, 'stats': backtest_stats}

    def _calculate_backtest_statistics(self, results: pd.DataFrame) -> Dict:
        stats = {}
        initial_capital = results['capital'].iloc[0]
        final_capital = results['capital'].iloc[-1]
        stats['total_return'] = (final_capital / initial_capital - 1) * 100
        stats['total_trades'] = (results['trade_pnl'] != 0).sum()
        completed_trades = results[results['trade_pnl'] != 0]
        if len(completed_trades) > 0:
            stats['win_rate'] = (completed_trades['trade_pnl'] > 0).mean() * 100
            stats['avg_win'] = completed_trades.loc[completed_trades['trade_pnl'] > 0, 'trade_pnl'].mean()
            stats['avg_loss'] = abs(completed_trades.loc[completed_trades['trade_pnl'] < 0, 'trade_pnl'].mean())
            stats['profit_factor'] = stats['avg_win'] * stats['win_rate'] / (stats['avg_loss'] * (100 - stats['win_rate'])) if stats['avg_loss'] > 0 else float('inf')
        else:
            stats['win_rate'] = stats['avg_win'] = stats['avg_loss'] = stats['profit_factor'] = 0
        if len(results) > 1:
            equity_returns = results['equity'].pct_change().dropna()
            stats['sharpe_ratio'] = np.sqrt(252) * (equity_returns.mean() / equity_returns.std()) if equity_returns.std() > 0 else 0
            stats['max_drawdown'] = self._calculate_max_drawdown_from_equity(results['equity'])
            stats['calmar_ratio'] = (stats['total_return'] / 100) / abs(stats['max_drawdown']) if stats['max_drawdown'] != 0 else float('inf')
        else:
            stats['sharpe_ratio'] = stats['max_drawdown'] = stats['calmar_ratio'] = 0
        return stats

    def _calculate_max_drawdown_from_equity(self, equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        drawdown = (equity / rolling_max - 1)
        return drawdown.min()

    def validate(self, test_size: float = 0.3, n_splits: int = 5) -> Dict:
        if self.data is None:
            raise ValueError("No data loaded.")
        if len(self.data) < 300:
            print("Warning: Limited data for validation.")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_metrics = []
        for train_idx, test_idx in tscv.split(self.data):
            train_data = self.data.iloc[train_idx].copy()
            test_data = self.data.iloc[test_idx].copy()
            temp_analyzer = EnhancedForexAnalyzer(adapt_weights=self.adapt_weights, optimize_params=False)
            temp_analyzer.load_data(train_data)
            temp_analyzer.calculate_indicators()
            temp_analyzer.settings = self.settings.copy()
            temp_analyzer.load_data(test_data)
            temp_analyzer.calculate_indicators()
            temp_analyzer.generate_signals()
            temp_analyzer.calculate_take_profit_stop_loss()
            backtest_results = temp_analyzer.backtest()
            all_metrics.append(backtest_results['stats'])
        validation_results = {
            'avg_total_return': np.mean([m['total_return'] for m in all_metrics]),
            'avg_win_rate': np.mean([m['win_rate'] for m in all_metrics]),
            'avg_sharpe': np.mean([m['sharpe_ratio'] for m in all_metrics]),
            'avg_max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
            'consistency': np.std([m['total_return'] for m in all_metrics]),
            'all_periods': all_metrics
        }
        self.validation_results = validation_results
        return validation_results