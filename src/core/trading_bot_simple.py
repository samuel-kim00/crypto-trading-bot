import os
import sys
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

print("Starting SIMPLE trading bot...")

class TradingBotSimple:
    def __init__(self):
        print("Initializing SIMPLE trading bot...")
        self.config_file = 'config/strategy_config.json'
        self.heartbeat_file = 'logs/trading_bot_heartbeat.json'
        self.running = True
        
        # Binance API endpoints
        self.base_url = "https://api.binance.com/api/v3"
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        print("Loading configuration...")
        self.load_config()
        
        # Initialize strategy parameters
        self.active_positions = {}
        self.last_scan_time = 0
        self.scan_interval = 5  # Scan every 5 seconds for ultra-fast trading
        
        print("Updating heartbeat...")
        self.update_heartbeat()
        
        print("SIMPLE Trading bot initialized successfully")

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                print(f"Loaded config: {self.config}")
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            # Use default config if file doesn't exist
            self.config = {
                'risk_per_trade': 0.02,
                'max_positions': 1,
                'min_trade_size': 10
            }

    def update_heartbeat(self):
        """Update heartbeat file"""
        try:
            # Get real balance
            try:
                balance = self.get_account_balance()
                usdt_balance = float(balance.get('USDT', 0))
            except:
                usdt_balance = 9.77  # Default balance
            
            heartbeat_data = {
                'timestamp': time.time(),
                'status': 'running' if self.running else 'stopped',
                'pid': os.getpid(),
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                'balance': usdt_balance,
                'error_count': 0,
                'mode': 'simple_http'
            }
            
            # Ensure logs directory exists
            os.makedirs('logs', exist_ok=True)
            
            with open(self.heartbeat_file, 'w') as f:
                json.dump(heartbeat_data, f)
        except Exception as e:
            print(f"Error updating heartbeat: {str(e)}")

    def get_market_data(self, symbol: str) -> List:
        """Get market data using direct HTTP requests"""
        try:
            # Convert symbol format (BTC/USDT -> BTCUSDT)
            binance_symbol = symbol.replace('/', '')
            
            # Get klines (candlestick data)
            url = f"{self.base_url}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': '1m',
                'limit': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to OHLCV format
            ohlcv = []
            for candle in data:
                ohlcv.append([
                    int(candle[0]),  # timestamp
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5])   # volume
                ])
            
            return ohlcv
            
        except Exception as e:
            print(f"⚠️  Market data error for {symbol}: {str(e)}")
            return []

    def get_account_balance(self) -> Dict:
        """Get account balance using direct HTTP requests"""
        try:
            if not self.api_key or not self.api_secret:
                return {'USDT': 9.77}  # Default balance if no API keys
            
            # For now, return default balance
            # In production, you'd implement signed requests here
            return {'USDT': 9.77}
            
        except Exception as e:
            print(f"⚠️  Balance fetch error: {str(e)}")
            return {'USDT': 9.77}

    def calculate_simple_indicators(self, ohlcv_data):
        """Calculate simple indicators"""
        if len(ohlcv_data) < 20:
            return None
            
        closes = [float(candle[4]) for candle in ohlcv_data]
        volumes = [float(candle[5]) for candle in ohlcv_data]
        
        # Simple moving averages
        sma_9 = sum(closes[-9:]) / 9 if len(closes) >= 9 else closes[-1]
        sma_21 = sum(closes[-21:]) / 21 if len(closes) >= 21 else closes[-1]
        
        # Simple RSI calculation
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Volume spike
        avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else volumes[-1]
        volume_spike = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        return {
            'sma_9': sma_9,
            'sma_21': sma_21,
            'rsi': rsi,
            'volume_spike': volume_spike,
            'current_price': closes[-1]
        }

    def check_entry_conditions(self, indicators):
        """Check entry conditions with ULTRA-AGGRESSIVE scalping logic"""
        if not indicators:
            return False, 'long'
        
        # Ultra-aggressive scalping conditions
        bullish_sma = indicators['sma_9'] > indicators['sma_21']
        oversold_rsi = indicators['rsi'] < 45  # Very aggressive
        overbought_rsi = indicators['rsi'] > 55  # Very aggressive
        volume_confirmation = indicators['volume_spike'] > 1.0  # Lower threshold
        
        # Quick momentum check - much lower threshold
        price_momentum = (indicators['current_price'] - indicators['sma_9']) / indicators['sma_9']
        strong_momentum = abs(price_momentum) > 0.001  # 0.1% momentum (vs 0.3%)
        
        # Ultra-fast entry signals - ANY signal triggers trade
        # Long: bullish trend + ANY condition
        long_signal = bullish_sma and (oversold_rsi or strong_momentum or volume_confirmation)
        
        # Short: bearish trend + ANY condition
        short_signal = not bullish_sma and (overbought_rsi or strong_momentum or volume_confirmation)
        
        # Quick reversal opportunities - more aggressive
        reversal_long = oversold_rsi and (volume_confirmation or strong_momentum)
        reversal_short = overbought_rsi and (volume_confirmation or strong_momentum)
        
        # Pure momentum trades - catch any movement
        momentum_long = strong_momentum and volume_confirmation
        momentum_short = strong_momentum and volume_confirmation
        
        if long_signal or reversal_long or momentum_long:
            return True, 'long'
        elif short_signal or reversal_short or momentum_short:
            return True, 'short'
            
        return False, 'long'

    def execute_trade(self, symbol: str, side: str, size: float):
        """Execute a trade (simulated for now)"""
        try:
            # For now, just simulate the trade
            entry_price = self.get_current_price(symbol)
            
            # Record the position
            self.active_positions[symbol] = {
                'side': side,
                'entry_price': entry_price,
                'size': size,
                'stop_loss': entry_price * (0.985 if side == 'buy' else 1.015),
                'take_profit': entry_price * (1.03 if side == 'buy' else 0.97),
                'entry_time': datetime.now()
            }
            
            print(f"✅ Simulated {side} position in {symbol} at {entry_price}")
            return True
            
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return False

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            binance_symbol = symbol.replace('/', '')
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': binance_symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
            
        except Exception as e:
            print(f"Error getting price for {symbol}: {str(e)}")
            return 0.0

    def close_position(self, symbol: str, reason: str):
        """Close a position (simulated)"""
        try:
            position = self.active_positions[symbol]
            current_price = self.get_current_price(symbol)
            
            if current_price > 0:
                profit = current_price - position['entry_price']
                if position['side'] == 'sell':
                    profit = -profit
                    
                print(f"Closed position in {symbol} ({reason}) - Profit: ${profit:.2f}")
            
            # Remove the position
            del self.active_positions[symbol]
            
        except Exception as e:
            print(f"Error closing position: {str(e)}")

    def manage_positions(self):
        """Manage open positions"""
        try:
            for symbol in list(self.active_positions.keys()):
                position = self.active_positions[symbol]
                current_price = self.get_current_price(symbol)
                
                if current_price <= 0:
                    continue
                
                # Check stop loss
                if (position['side'] == 'buy' and current_price <= position['stop_loss']) or \
                   (position['side'] == 'sell' and current_price >= position['stop_loss']):
                    self.close_position(symbol, 'Stop loss hit')
                    continue
                
                # Check take profit
                if (position['side'] == 'buy' and current_price >= position['take_profit']) or \
                   (position['side'] == 'sell' and current_price <= position['take_profit']):
                    self.close_position(symbol, 'Take profit hit')
                    continue
                
                # Time-based stop (1 minute for ultra-fast trading)
                if (datetime.now() - position['entry_time']).total_seconds() > 60:
                    self.close_position(symbol, 'Time-based stop (1 min)')
                    continue
                            
        except Exception as e:
            print(f"Error managing positions: {str(e)}")

    def scan_markets(self):
        """Scan markets for opportunities"""
        try:
            # Only scan if enough time has passed
            current_time = time.time()
            if current_time - self.last_scan_time < self.scan_interval:
                return
                
            self.last_scan_time = current_time
            
            # Only scan if we don't have max positions
            if len(self.active_positions) >= self.config.get('max_positions', 1):
                return
            
            # Focus on top 4 pairs for speed
            top_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
            
            for symbol in top_symbols:
                if symbol not in self.active_positions:
                    try:
                        # Get market data
                        ohlcv = self.get_market_data(symbol)
                        
                        if len(ohlcv) >= 20:
                            # Calculate indicators
                            indicators = self.calculate_simple_indicators(ohlcv)
                            
                            if indicators:
                                # Check entry conditions
                                should_enter, side = self.check_entry_conditions(indicators)
                                
                                if should_enter:
                                    # Calculate position size
                                    try:
                                        balance_info = self.get_account_balance()
                                        usdt_balance = float(balance_info.get('USDT', 9.77))
                                        position_size = usdt_balance * self.config.get('risk_per_trade', 0.02)
                                        
                                        if position_size >= self.config.get('min_trade_size', 10):
                                            # Execute the trade
                                            success = self.execute_trade(symbol, side, position_size)
                                            if success:
                                                print(f"✅ Trade executed: {side} {symbol}")
                                                break  # Only one trade per scan
                                    except Exception as e:
                                        print(f"⚠️  Balance fetch error: {str(e)}")
                                        continue
                                        
                    except Exception as e:
                        # Don't spam error messages
                        if time.time() % 60 < 5:  # Only log errors every minute
                            print(f"⚠️  Market data error for {symbol}: {str(e)}")
                        continue
                
        except Exception as e:
            print(f"Error scanning markets: {str(e)}")

    def run(self):
        """Main bot loop - optimized for speed"""
        print("Starting SIMPLE main loop...")
        
        while self.running:
            try:
                # Update heartbeat (less frequently)
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    self.update_heartbeat()
                
                # Scan markets for opportunities
                self.scan_markets()
                
                # Manage open positions
                self.manage_positions()
                
                # Sleep for a bit (1 second for ultra-fast response)
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("Received keyboard interrupt, stopping...")
                self.running = False
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    try:
        bot = TradingBotSimple()
        print("SIMPLE Bot created successfully")
        bot.run()
    except Exception as e:
        print(f"Error creating SIMPLE bot: {str(e)}")
        sys.exit(1)
