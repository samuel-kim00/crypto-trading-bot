import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
import logging
from binance.client import Client
import talib
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_predictor import EnhancedPredictor

class VLMBacktester:
    def __init__(self, initial_balance: float = 10000):
        """
        VLM (Volatility-Liquidity-Momentum) Strategy Backtester
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # symbol: {'quantity', 'entry_price', 'entry_time', 'type'}
        self.trade_history = []
        self.daily_balances = []
        
        # Initialize Binance client
        try:
            from binance.client import Client
            # Use public endpoints for historical data
            self.client = Client()
        except Exception as e:
            print(f"Warning: Could not initialize Binance client: {e}")
            self.client = None
        
        # Strategy parameters
        self.transaction_fee = 0.001  # 0.1% per trade
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.stop_loss_pct = 0.03     # 3% stop loss
        self.take_profit_pct = 0.06   # 6% take profit
        
        # ML Predictor for enhanced signals
        self.predictor = EnhancedPredictor()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical OHLCV data from Binance"""
        try:
            if not self.client:
                # Use sample data if client unavailable
                return self._generate_sample_data(symbol, start_date, end_date)
            
            klines = self.client.get_historical_klines(
                symbol, interval, start_date, end_date
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df[numeric_columns]
        
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return self._generate_sample_data(symbol, start_date, end_date)

    def _generate_sample_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate sample data for testing when API is unavailable"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='1H')
        
        # Generate realistic price data with trend and volatility
        np.random.seed(42)  # For reproducible results
        n_points = len(dates)
        
        # Base price movement
        base_price = 45000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1
        trend = np.linspace(0, 0.2, n_points)  # 20% upward trend over period
        volatility = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
        
        price_changes = trend + volatility
        prices = base_price * (1 + price_changes).cumprod()
        
        # Generate OHLC from close prices
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = np.maximum(df['open'], df['close']) * (1 + np.random.uniform(0, 0.01, n_points))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - np.random.uniform(0, 0.01, n_points))
        df['volume'] = np.random.uniform(1000, 10000, n_points)
        
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for VLM strategy"""
        data = df.copy()
        
        # Volatility indicators
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['volatility'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean() * 100
        
        # Momentum indicators
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        data['rsi_fast'] = talib.RSI(data['close'], timeperiod=7)
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(data['close'])
        
        # Trend indicators
        data['sma_20'] = talib.SMA(data['close'], timeperiod=20)
        data['sma_50'] = talib.SMA(data['close'], timeperiod=50)
        data['ema_12'] = talib.EMA(data['close'], timeperiod=12)
        data['ema_26'] = talib.EMA(data['close'], timeperiod=26)
        
        # Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(data['close'])
        
        # Liquidity proxy (volume analysis)
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['price_volume'] = data['close'] * data['volume']
        
        # Support/Resistance
        data['support'] = data['low'].rolling(window=20).min()
        data['resistance'] = data['high'].rolling(window=20).max()
        
        return data

    def generate_vlm_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate VLM (Volatility-Liquidity-Momentum) trading signals"""
        signals = data.copy()
        
        # Initialize signal columns
        signals['vlm_signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
        signals['vlm_strength'] = 0  # Signal strength 0-100
        signals['signal_reason'] = ''
        
        for i in range(50, len(signals)):  # Start after enough data for indicators
            row = signals.iloc[i]
            prev_row = signals.iloc[i-1]
            
            # Volatility component (25% weight)
            volatility_score = 0
            if row['atr'] > data['atr'].rolling(window=20).mean().iloc[i] * 1.2:
                volatility_score += 25  # High volatility favors entry
            
            # Liquidity component (25% weight)
            liquidity_score = 0
            if row['volume_ratio'] > 1.5:  # Volume 50% above average
                liquidity_score += 25
            
            # Momentum component (50% weight)
            momentum_score = 0
            
            # RSI momentum
            if 30 <= row['rsi'] <= 40:  # Oversold but recovering
                momentum_score += 15
            elif 60 <= row['rsi'] <= 70:  # Overbought, potential sell
                momentum_score -= 15
            
            # MACD momentum
            if row['macd'] > row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal']:
                momentum_score += 20  # MACD bullish crossover
            elif row['macd'] < row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal']:
                momentum_score -= 20  # MACD bearish crossover
            
            # Price action momentum
            if row['close'] > row['sma_20'] and row['sma_20'] > row['sma_50']:
                momentum_score += 15  # Uptrend
            elif row['close'] < row['sma_20'] and row['sma_20'] < row['sma_50']:
                momentum_score -= 15  # Downtrend
            
            # Combine VLM scores
            total_score = volatility_score + liquidity_score + momentum_score
            
            # Generate signals
            if total_score >= 60:
                signals.iloc[i, signals.columns.get_loc('vlm_signal')] = 1  # Buy
                signals.iloc[i, signals.columns.get_loc('signal_reason')] = f"VLM Buy (V:{volatility_score} L:{liquidity_score} M:{momentum_score})"
            elif total_score <= 30:
                signals.iloc[i, signals.columns.get_loc('vlm_signal')] = -1  # Sell
                signals.iloc[i, signals.columns.get_loc('signal_reason')] = f"VLM Sell (V:{volatility_score} L:{liquidity_score} M:{momentum_score})"
            
            signals.iloc[i, signals.columns.get_loc('vlm_strength')] = max(0, min(100, total_score))
        
        return signals

    def execute_trade(self, symbol: str, signal: int, price: float, timestamp: pd.Timestamp, reason: str):
        """Execute a trade based on signal"""
        position_value = self.current_balance * self.max_position_size
        
        if signal == 1 and symbol not in self.positions:  # Buy signal and no existing position
            quantity = position_value / price
            cost = quantity * price * (1 + self.transaction_fee)
            
            if cost <= self.current_balance:
                self.current_balance -= cost
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_time': timestamp,
                    'type': 'long'
                }
                
                self.trade_history.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'timestamp': timestamp,
                    'balance_after': self.current_balance,
                    'reason': reason
                })
        
        elif signal == -1 and symbol in self.positions:  # Sell signal and existing position
            position = self.positions[symbol]
            proceeds = position['quantity'] * price * (1 - self.transaction_fee)
            self.current_balance += proceeds
            
            # Calculate P&L
            entry_value = position['quantity'] * position['entry_price']
            pnl = proceeds - entry_value
            pnl_pct = (pnl / entry_value) * 100
            
            self.trade_history.append({
                'symbol': symbol,
                'action': 'sell',
                'quantity': position['quantity'],
                'price': price,
                'timestamp': timestamp,
                'balance_after': self.current_balance,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'hold_time': (timestamp - position['entry_time']).total_seconds() / 3600,  # hours
                'reason': reason
            })
            
            del self.positions[symbol]

    def check_stop_loss_take_profit(self, symbol: str, current_price: float, timestamp: pd.Timestamp):
        """Check and execute stop loss or take profit orders"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # Calculate price levels
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        take_profit_price = entry_price * (1 + self.take_profit_pct)
        
        if current_price <= stop_loss_price:
            self.execute_trade(symbol, -1, current_price, timestamp, f"Stop Loss at {stop_loss_price:.2f}")
        elif current_price >= take_profit_price:
            self.execute_trade(symbol, -1, current_price, timestamp, f"Take Profit at {take_profit_price:.2f}")

    async def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Run complete backtest on multiple symbols"""
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        self.logger.info(f"Testing symbols: {symbols}")
        self.logger.info(f"Initial balance: ${self.initial_balance:,.2f}")
        
        # Store daily balances for performance tracking
        all_dates = pd.date_range(start_date, end_date, freq='D')
        daily_balance_tracking = {date: self.current_balance for date in all_dates}
        
        for symbol in symbols:
            self.logger.info(f"Processing {symbol}...")
            
            # Get historical data
            df = self.get_historical_data(symbol, '1h', start_date, end_date)
            if df.empty:
                continue
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Generate VLM signals
            df = self.generate_vlm_signals(df)
            
            # Execute trades based on signals
            for timestamp, row in df.iterrows():
                if pd.isna(row['vlm_signal']):
                    continue
                
                # Check stop loss/take profit first
                self.check_stop_loss_take_profit(symbol, row['close'], timestamp)
                
                # Execute new signals
                if row['vlm_signal'] != 0:
                    self.execute_trade(
                        symbol, 
                        int(row['vlm_signal']), 
                        row['close'], 
                        timestamp, 
                        row['signal_reason']
                    )
                
                # Track daily balance
                date = timestamp.date()
                if date in daily_balance_tracking:
                    daily_balance_tracking[date] = self.current_balance + self._calculate_unrealized_pnl(row['close'], symbol)
        
        # Close all remaining positions at end
        for symbol in list(self.positions.keys()):
            last_price = df['close'].iloc[-1] if not df.empty else self.positions[symbol]['entry_price']
            self.execute_trade(symbol, -1, last_price, df.index[-1], "End of backtest")
        
        return self._generate_backtest_results(daily_balance_tracking)

    def _calculate_unrealized_pnl(self, current_price: float, symbol: str) -> float:
        """Calculate unrealized P&L for current positions"""
        if symbol not in self.positions:
            return 0
        
        position = self.positions[symbol]
        current_value = position['quantity'] * current_price
        entry_value = position['quantity'] * position['entry_price']
        return current_value - entry_value

    def _generate_backtest_results(self, daily_balances: Dict) -> Dict:
        """Generate comprehensive backtest results"""
        final_balance = self.current_balance
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trade_history)
        completed_trades = trades_df[trades_df['action'] == 'sell'] if not trades_df.empty else pd.DataFrame()
        
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        total_trades = 0
        winning_trades_count = 0
        losing_trades_count = 0
        
        if not completed_trades.empty:
            total_trades = len(completed_trades)
            winning_trades = completed_trades[completed_trades['pnl'] > 0]
            losing_trades = completed_trades[completed_trades['pnl'] <= 0]
            
            winning_trades_count = len(winning_trades)
            losing_trades_count = len(losing_trades)
            win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
            avg_win = float(winning_trades['pnl_pct'].mean()) if not winning_trades.empty else 0
            avg_loss = float(losing_trades['pnl_pct'].mean()) if not losing_trades.empty else 0
        
        # Calculate metrics
        daily_returns = []
        balance_values = list(daily_balances.values())
        for i in range(1, len(balance_values)):
            daily_return = (balance_values[i] - balance_values[i-1]) / balance_values[i-1]
            daily_returns.append(daily_return)
        
        daily_returns = np.array(daily_returns)
        
        # Risk metrics
        volatility = float(np.std(daily_returns) * np.sqrt(365) * 100) if len(daily_returns) > 0 else 0
        sharpe_ratio = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)) if np.std(daily_returns) > 0 else 0
        
        # Maximum drawdown
        peak = self.initial_balance
        max_drawdown = 0
        for balance in balance_values:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate total fees and average holding time safely
        total_fees = 0
        avg_hold_time = 0
        try:
            total_fees = sum([float(trade.get('quantity', 0)) * float(trade.get('price', 0)) * self.transaction_fee 
                             for trade in self.trade_history])
            if not completed_trades.empty and 'hold_time' in completed_trades.columns:
                avg_hold_time = float(completed_trades['hold_time'].mean())
        except:
            pass

        # Return flattened structure for easier JSON serialization
        results = {
            # Main metrics (flattened from summary)
            'initial_balance': float(self.initial_balance),
            'final_balance': float(final_balance),
            'total_return': float(total_return),
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades_count),
            'losing_trades': int(losing_trades_count),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'profit_factor': float(abs(avg_win / avg_loss) if avg_loss != 0 else 1.0),
            'total_fees': float(total_fees),
            'avg_hold_time': float(avg_hold_time),
            
            # Trade data (cleaned)
            'trade_history': self.trade_history,
            'daily_balances': daily_balances
        }
        
        # Apply JSON serialization to clean everything
        return self._make_json_serializable(results)

    def save_backtest_results(self, results: Dict, filename: str = None):
        """Save backtest results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        results_copy = self._make_json_serializable(results)
        
        filepath = f"src/analysis/reports/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        self.logger.info(f"Backtest results saved to {filepath}")
        return filepath

    def _make_json_serializable(self, obj):
        """Convert pandas/numpy objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return str(obj)
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Try to convert to string as fallback
            return str(obj)

async def main():
    """Run example backtest"""
    # Initialize backtester
    backtester = VLMBacktester(initial_balance=10000)
    
    # Define test parameters
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
    start_date = '2024-01-01'
    end_date = '2024-06-01'
    
    print("🚀 Starting VLM Strategy Backtest...")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Initial Balance: ${backtester.initial_balance:,.2f}")
    print(f"📊 Testing symbols: {', '.join(symbols)}")
    print()
    
    # Run backtest
    results = await backtester.run_backtest(symbols, start_date, end_date)
    
    # Display results
    summary = results['summary']
    print("📈 BACKTEST RESULTS")
    print("=" * 50)
    print(f"Initial Balance:     ${summary['initial_balance']:,.2f}")
    print(f"Final Balance:       ${summary['final_balance']:,.2f}")
    print(f"Total Return:        {summary['total_return_pct']:.2f}%")
    print(f"Total Trades:        {summary['total_trades']}")
    print(f"Win Rate:           {summary['win_rate_pct']:.1f}%")
    print(f"Average Win:        {summary['avg_win_pct']:.2f}%")
    print(f"Average Loss:       {summary['avg_loss_pct']:.2f}%")
    print(f"Sharpe Ratio:       {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {summary['max_drawdown_pct']:.2f}%")
    print(f"Volatility:         {summary['volatility_pct']:.2f}%")
    
    # Save results
    filepath = backtester.save_backtest_results(results)
    print(f"\n💾 Results saved to: {filepath}")

if __name__ == "__main__":
    asyncio.run(main()) 