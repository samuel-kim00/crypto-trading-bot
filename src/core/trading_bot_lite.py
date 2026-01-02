#!/usr/bin/env python3
"""
Lightweight Trading Bot - Optimized for low CPU usage with REAL balance
"""

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
import asyncio
from typing import Dict, List, Tuple

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

print("Starting LITE trading bot...")

class TradingBotLite:
    def __init__(self):
        print("Initializing LITE trading bot...")
        self.config_file = 'config/strategy_config.json'
        self.heartbeat_file = 'logs/trading_bot_heartbeat.json'
        self.running = True
        
        # Initialize exchange with rate limiting
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'rateLimit': 200,  # 200ms between requests
            'options': {'defaultType': 'spot'},
            'sandbox': False,
            'verbose': False
        })
        
        # Test connection
        try:
            print("Testing Binance connection...")
            self.exchange.load_markets()
            print("✅ Binance connection successful")
        except Exception as e:
            print(f"❌ Binance connection failed: {str(e)}")
            # Try without API keys for public data
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 200,
                'options': {'defaultType': 'spot'},
                'sandbox': False,
                'verbose': False
            })
            print("🔄 Using public API mode (no trading)")
        
        print("Loading configuration...")
        self.load_config()
        
        # Initialize strategy parameters
        self.active_positions = {}
        self.last_scan_time = 0
        self.scan_interval = 10  # Scan every 10 seconds instead of 5
        
        print("Updating heartbeat...")
        self.update_heartbeat()
        
        print("LITE Trading bot initialized successfully")

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
                balance = self.exchange.fetch_balance()
                usdt_balance = float(balance['USDT']['free'])
            except:
                usdt_balance = 0
            
            heartbeat_data = {
                'timestamp': time.time(),
                'status': 'running' if self.running else 'stopped',
                'pid': os.getpid(),
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                'balance': usdt_balance,
                'error_count': 0
            }
            
            # Ensure logs directory exists
            os.makedirs('logs', exist_ok=True)
            
            with open(self.heartbeat_file, 'w') as f:
                json.dump(heartbeat_data, f)
        except Exception as e:
            print(f"Error updating heartbeat: {str(e)}")

    def calculate_simple_indicators(self, ohlcv_data):
        """Calculate simple indicators without talib for speed"""
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
        """Check entry conditions with simplified logic"""
        if not indicators:
            return False, 'long'
        
        # Simple conditions without AI
        bullish_sma = indicators['sma_9'] > indicators['sma_21']
        oversold_rsi = indicators['rsi'] < 30
        overbought_rsi = indicators['rsi'] > 70
        volume_confirmation = indicators['volume_spike'] > 1.5
        
        # Entry signals
        long_signal = bullish_sma and oversold_rsi and volume_confirmation
        short_signal = not bullish_sma and overbought_rsi and volume_confirmation
        
        if long_signal:
            return True, 'long'
        elif short_signal:
            return True, 'short'
            
        return False, 'long'

    def execute_trade(self, symbol: str, side: str, size: float):
        """Execute a trade"""
        try:
            # Place the order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=size
            )
            
            # Record the position
            entry_price = float(order['price'])
            self.active_positions[symbol] = {
                'side': side,
                'entry_price': entry_price,
                'size': size,
                'stop_loss': entry_price * (0.985 if side == 'buy' else 1.015),  # 1.5% stop loss
                'take_profit': entry_price * (1.03 if side == 'buy' else 0.97),  # 3% take profit
                'entry_time': datetime.now()
            }
            
            print(f"✅ Opened {side} position in {symbol} at {entry_price}")
            return True
            
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return False

    def close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.active_positions[symbol]
            side = 'sell' if position['side'] == 'buy' else 'buy'
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position['size']
            )
            
            profit = float(order['price']) - position['entry_price']
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
                
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = float(ticker['last'])
                
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
                
                # Check time-based stop (2 minutes for faster exits)
                time_in_trade = (datetime.now() - position['entry_time']).total_seconds()
                if time_in_trade > 120:  # 2 minutes
                    price_change = ((current_price - position['entry_price']) / position['entry_price']) * 100
                    if (position['side'] == 'buy' and price_change < 0.5) or \
                       (position['side'] == 'sell' and price_change > -0.5):
                        self.close_position(symbol, 'Time-based stop')
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
                        # Fetch OHLCV data (1-minute candles, limit 20 for speed)
                        ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=20)
                        
                        if len(ohlcv) >= 20:
                            # Calculate indicators
                            indicators = self.calculate_simple_indicators(ohlcv)
                            
                            if indicators:
                                # Check entry conditions
                                should_enter, side = self.check_entry_conditions(indicators)
                                
                                if should_enter:
                                    # Calculate position size
                                    try:
                                        balance_info = self.exchange.fetch_balance()
                                        usdt_balance = float(balance_info['USDT']['free'])
                                        position_size = usdt_balance * self.config.get('risk_per_trade', 0.02)
                                        
                                        if position_size >= self.config.get('min_trade_size', 10):
                                            # Execute the trade
                                            success = self.execute_trade(symbol, side, position_size)
                                            if success:
                                                print(f"✅ Trade executed: {side} {symbol}")
                                                break  # Only one trade per scan
                                    except Exception as e:
                                        print(f"⚠️  Balance fetch error: {str(e)}")
                                        # Skip trading if can't get balance
                                        continue
                                        
                    except Exception as e:
                        # Don't spam error messages - only log occasionally
                        if time.time() % 60 < 5:  # Only log errors every minute
                            print(f"⚠️  Market data error for {symbol}: {str(e)}")
                        continue
                
        except Exception as e:
            print(f"Error scanning markets: {str(e)}")

    def run(self):
        """Main bot loop - optimized for speed"""
        print("Starting LITE main loop...")
        
        while self.running:
            try:
                # Update heartbeat (less frequently)
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    self.update_heartbeat()
                
                # Scan markets for opportunities
                self.scan_markets()
                
                # Manage open positions
                self.manage_positions()
                
                # Sleep for a bit (2 seconds for faster response)
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("Received keyboard interrupt, stopping...")
                self.running = False
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    try:
        bot = TradingBotLite()
        print("LITE Bot created successfully")
        bot.run()
    except Exception as e:
        print(f"Error creating LITE bot: {str(e)}")
        sys.exit(1) 