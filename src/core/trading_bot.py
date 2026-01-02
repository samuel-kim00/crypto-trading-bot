import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
from dotenv import load_dotenv
import ccxt
import talib
import asyncio
import aiohttp
from typing import Dict, List, Tuple

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.performance_tracker import PerformanceTracker
from analysis.media_analyzer_v2 import MediaAnalyzer
from analysis.self_learning_integration import SelfLearningIntegration

# Load environment variables
load_dotenv()

print("Starting trading bot...")

class TradingBot:
    def __init__(self):
        print("Initializing trading bot...")
        self.config_file = 'config/strategy_config.json'
        self.status_file = 'config/strategy_status.json'
        self.positions_file = 'config/active_positions.json'
        self.performance_file = 'data/performance_data.json'
        self.heartbeat_file = 'logs/trading_bot_heartbeat.json'
        self.running = True
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}  # Use spot market (changed from futures)
        })
        
        print("Loading configuration...")
        self.load_config()
        
        print("Initializing components...")
        self.performance_tracker = PerformanceTracker()
        self.media_analyzer = MediaAnalyzer()
        
        # Initialize AI integration
        print("Initializing AI learning system...")
        self.ai_integration = SelfLearningIntegration()
        
        # Initialize strategy parameters
        self.active_positions = {}
        self.symbol_data = {}  # Store OHLCV and indicator data
        self.trailing_stops = {}
        
        print("Updating heartbeat...")
        self.update_heartbeat()
        
        print("Trading bot initialized successfully")

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                print(f"Loaded config: {self.config}")
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            raise

    def update_heartbeat(self):
        """Update heartbeat file"""
        try:
            heartbeat_data = {
                'timestamp': time.time(),
                'status': 'running' if self.running else 'stopped',
                'pid': os.getpid(),
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                'error_count': 0
            }
            
            with open(self.heartbeat_file, 'w') as f:
                json.dump(heartbeat_data, f)
            print("Updated heartbeat file")
        except Exception as e:
            print(f"Error updating heartbeat: {str(e)}")
            raise

    async def fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch and process market data for a symbol"""
        try:
            # Fetch OHLCV data (1-minute candles) - ccxt calls are synchronous
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=30)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate indicators
            closes = df['close'].values
            volumes = df['volume'].values
            
            # RSI
            df['rsi'] = talib.RSI(closes, timeperiod=14)
            
            # EMAs
            df['ema9'] = talib.EMA(closes, timeperiod=9)
            df['ema21'] = talib.EMA(closes, timeperiod=21)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macdsignal
            df['macd_hist'] = macdhist
            
            # Volume spike detection
            df['volume_avg'] = df['volume'].rolling(window=10).mean()
            df['volume_spike'] = df['volume'] / df['volume_avg']
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    async def check_entry_conditions(self, symbol: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if we should enter a position with AI integration"""
        try:
            if len(df) < 20:  # Need enough data
                return False, 'long'
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check for valid indicators
            if pd.isna(latest['rsi']) or pd.isna(latest['ema9']) or pd.isna(latest['ema21']):
                return False, 'long'
            
            # Get AI prediction (convert symbol format for AI integration)
            ai_symbol = symbol.replace('/', '')  # Convert BTC/USDT to BTCUSDT
            ai_prediction = await self.ai_integration.get_live_ai_prediction(ai_symbol)
            ai_confidence = ai_prediction.get('confidence', 0) if 'error' not in ai_prediction else 0
            ai_recommendation = ai_prediction.get('recommendation', 'HOLD') if 'error' not in ai_prediction else 'HOLD'
            
            # Technical indicators
            rsi = latest['rsi']
            ema9 = latest['ema9']
            ema21 = latest['ema21']
            volume_spike = latest['volume_spike']
            
            # Technical conditions
            bullish_ema = ema9 > ema21
            bearish_ema = ema9 < ema21
            oversold_rsi = rsi < 40
            overbought_rsi = rsi > 60
            volume_confirmation = volume_spike > 1.2
            
            # AI-enhanced decision making
            ai_buy_signal = ai_recommendation == 'BUY' and ai_confidence > 0.6
            ai_sell_signal = ai_recommendation == 'SELL' and ai_confidence > 0.6
            
            # Combined signals (AI + Technical)
            long_signal = (bullish_ema and oversold_rsi and volume_confirmation) or ai_buy_signal
            short_signal = (bearish_ema and overbought_rsi and volume_confirmation) or ai_sell_signal
            
            # Log AI decision
            if ai_confidence > 0.5:
                print(f"AI Analysis for {symbol}: {ai_recommendation} ({ai_confidence:.2f} confidence)")
            
            if long_signal:
                return True, 'long'
            elif short_signal:
                return True, 'short'
                
            return False, 'long'
            
        except Exception as e:
            print(f"Error checking entry conditions for {symbol}: {str(e)}")
            return False, 'long'

    async def execute_trade(self, symbol: str, side: str, size: float):
        """Execute a trade with position sizing and risk management"""
        try:
            # Calculate position size based on risk
            balance = float(self.exchange.fetch_balance()['total']['USDT'])
            risk_per_trade = balance * 0.02  # 2% risk per trade
            
            # Place the order
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=size,
                params={'reduceOnly': False}
            )
            
            # Record the position
            entry_price = float(order['price'])
            self.active_positions[symbol] = {
                'side': side,
                'entry_price': entry_price,
                'size': size,
                'stop_loss': entry_price * (0.985 if side == 'LONG' else 1.015),  # 1.5% stop loss
                'take_profits': [
                    entry_price * (1.03 if side == 'LONG' else 0.97),   # +3% (50% size)
                    entry_price * (1.05 if side == 'LONG' else 0.95),   # +5% (30% size)
                    entry_price * (1.06 if side == 'LONG' else 0.94),   # +6% (20% size)
                ],
                'sizes': [size * 0.5, size * 0.3, size * 0.2],
                'entry_time': datetime.now(),
                'trailing_stop': None
            }
            
            print(f"Entered {side} position in {symbol} at {entry_price}")
            return True
            
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return False

    async def manage_positions(self):
        """Manage open positions (stop loss, take profit, trailing stop)"""
        try:
            for symbol in list(self.active_positions.keys()):
                position = self.active_positions[symbol]
                current_price = float(self.exchange.fetch_ticker(symbol)['last'])
                
                # Check stop loss
                if (position['side'] == 'LONG' and current_price <= position['stop_loss']) or \
                   (position['side'] == 'SHORT' and current_price >= position['stop_loss']):
                    await self.close_position(symbol, 'Stop loss hit')
                    continue
                
                # Check time-based stop (3 minutes)
                time_in_trade = (datetime.now() - position['entry_time']).total_seconds()
                if time_in_trade > 180:  # 3 minutes
                    price_change = ((current_price - position['entry_price']) / position['entry_price']) * 100
                    if (position['side'] == 'LONG' and price_change < 1) or \
                       (position['side'] == 'SHORT' and price_change > -1):
                        await self.close_position(symbol, 'Time-based stop')
                        continue
                
                # Check take profits
                for i, (tp_price, tp_size) in enumerate(zip(position['take_profits'], position['sizes'])):
                    if position['sizes'][i] > 0:  # If this level hasn't been taken yet
                        if (position['side'] == 'LONG' and current_price >= tp_price) or \
                           (position['side'] == 'SHORT' and current_price <= tp_price):
                            # Take partial profit
                            await self.take_partial_profit(symbol, tp_size, i)
                            
                            # If this is the last take profit, activate trailing stop
                            if i == 2:
                                position['trailing_stop'] = current_price * (0.98 if position['side'] == 'LONG' else 1.02)
                
                # Update trailing stop if active
                if position['trailing_stop']:
                    if position['side'] == 'LONG':
                        position['trailing_stop'] = max(position['trailing_stop'], current_price * 0.98)
                        if current_price <= position['trailing_stop']:
                            await self.close_position(symbol, 'Trailing stop hit')
                    else:  # SHORT
                        position['trailing_stop'] = min(position['trailing_stop'], current_price * 1.02)
                        if current_price >= position['trailing_stop']:
                            await self.close_position(symbol, 'Trailing stop hit')
                            
        except Exception as e:
            print(f"Error managing positions: {str(e)}")

    async def take_partial_profit(self, symbol: str, size: float, level: int):
        """Take partial profit at specified level"""
        try:
            side = 'sell' if self.active_positions[symbol]['side'] == 'LONG' else 'buy'
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=size,
                params={'reduceOnly': True}
            )
            
            # Update position sizes
            self.active_positions[symbol]['sizes'][level] = 0
            print(f"Took partial profit ({size} units) at level {level+1} for {symbol}")
            
            # Record the profit
            self.performance_tracker.record_trade({
                'symbol': symbol,
                'type': 'partial_tp',
                'profit': float(order['info']['realizedPnl']),
                'quantity': size
            })
            
        except Exception as e:
            print(f"Error taking partial profit: {str(e)}")

    async def close_position(self, symbol: str, reason: str):
        """Close an entire position"""
        try:
            position = self.active_positions[symbol]
            remaining_size = sum(position['sizes'])
            
            if remaining_size > 0:
                side = 'sell' if position['side'] == 'LONG' else 'buy'
                order = await self.exchange.create_order(
                symbol=symbol,
                    type='market',
                    side=side,
                    amount=remaining_size,
                    params={'reduceOnly': True}
                )
                
                # Record the trade
                self.performance_tracker.record_trade({
                    'symbol': symbol,
                    'type': reason,
                    'profit': float(order['info']['realizedPnl']),
                    'quantity': remaining_size
                })
                
                print(f"Closed position in {symbol} ({reason})")
            
            # Remove the position
            del self.active_positions[symbol]
            
        except Exception as e:
            print(f"Error closing position: {str(e)}")

    async def scan_markets(self):
        """Scan all markets for opportunities"""
        try:
            # Get top trading pairs to avoid too many API calls
            top_symbols = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
                'XRP/USDT', 'SOL/USDT', 'DOGE/USDT', 'MATIC/USDT'
            ]
            
            for symbol in top_symbols:
                if symbol not in self.active_positions:  # Only check if we don't have a position
                    try:
                        # Get market data
                        df = await self.fetch_market_data(symbol)
                        if df is not None and len(df) > 20:
                            # Check entry conditions
                            should_enter, side = await self.check_entry_conditions(symbol, df)
                            
                            if should_enter:
                                # Calculate position size
                                try:
                                    balance_info = self.exchange.fetch_balance()
                                    usdt_balance = float(balance_info['USDT']['free'])
                                    position_size = usdt_balance * 0.1  # 10% of balance per trade
                                    
                                    if position_size >= 10:  # Minimum $10 trade
                                        # Execute the trade
                                        success = await self.execute_trade(symbol, side.upper(), position_size)
                                        if success:
                                            print(f"✅ Opened {side} position in {symbol}")
                                            break  # Only one trade per scan
                                except Exception as e:
                                    print(f"Error calculating position size: {str(e)}")
                                    
                    except Exception as e:
                        print(f"Error processing {symbol}: {str(e)}")
                        continue
                
        except Exception as e:
            print(f"Error scanning markets: {str(e)}")

    async def run(self):
        """Main bot loop"""
        print("Starting main loop...")
        
        while self.running:
            try:
                # Update heartbeat
                self.update_heartbeat()
                
                # Scan markets for opportunities
                await self.scan_markets()
                
                # Manage open positions
                await self.manage_positions()
                
                # Sleep for a bit (5 seconds)
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                print("Received keyboard interrupt, stopping...")
                self.running = False
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    try:
        bot = TradingBot()
        print("Bot created successfully")
        asyncio.run(bot.run())
    except Exception as e:
        print(f"Error creating bot: {str(e)}")
        sys.exit(1)