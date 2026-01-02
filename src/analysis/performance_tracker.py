import logging
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class PerformanceTracker:
    def __init__(self):
        self.logger = logging.getLogger('performance_tracker')
        self.logger.setLevel(logging.INFO)
        self.target_amount = 10000
        self.performance_file = 'data/performance_data.json'
        self.positions_file = 'config/active_positions.json'
        self.initial_balance = self._get_initial_balance()
        self.current_balance = self.initial_balance
        self.start_date = datetime.now()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        self.ensure_files_exist()
        self.data = self.load_performance_data()
    
    def _get_initial_balance(self) -> float:
        """Get initial balance from environment or default to 1000"""
        return float(os.getenv('INITIAL_BALANCE', 1000))
    
    def _load_performance_history(self) -> Dict:
        """Load performance history from file or create new"""
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        return {
            'total_profit': 0.0,
            'daily_profit': 0.0,
            'trades': [],
            'last_update': datetime.now().isoformat()
        }
    
    def _save_performance_history(self):
        """Save performance history to file"""
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_history, f, indent=4)
    
    def update_balance(self, new_balance: float):
        """Update current balance"""
        self.current_balance = new_balance
    
    def record_trade(self, trade_data: Dict):
        """Record a new trade"""
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        # Calculate profit/loss
        trade_data['profit_loss'] = trade_data.get('profit_loss', 0.0)
        
        # Update trade history
        self.performance_history['trades'].append(trade_data)
        
        # Update daily stats
        date = datetime.fromisoformat(trade_data['timestamp']).date().isoformat()
        if date not in self.performance_history['daily_stats']:
            self.performance_history['daily_stats'][date] = {
                'trades': 0,
                'profit_loss': 0.0,
                'volume': 0.0
            }
        self.performance_history['daily_stats'][date]['trades'] += 1
        self.performance_history['daily_stats'][date]['profit_loss'] += trade_data['profit_loss']
        self.performance_history['daily_stats'][date]['volume'] += trade_data['quantity']
        
        # Update hourly stats
        hour = datetime.fromisoformat(trade_data['timestamp']).strftime('%H:00')
        if hour not in self.performance_history['hourly_stats']:
            self.performance_history['hourly_stats'][hour] = {
                'trades': 0,
                'profit_loss': 0.0,
                'volume': 0.0
            }
        self.performance_history['hourly_stats'][hour]['trades'] += 1
        self.performance_history['hourly_stats'][hour]['profit_loss'] += trade_data['profit_loss']
        self.performance_history['hourly_stats'][hour]['volume'] += trade_data['quantity']
        
        # Update symbol stats
        symbol = trade_data['symbol']
        if symbol not in self.performance_history['symbol_stats']:
            self.performance_history['symbol_stats'][symbol] = {
                'trades': 0,
                'profit_loss': 0.0,
                'volume': 0.0,
                'win_rate': 0.0
            }
        self.performance_history['symbol_stats'][symbol]['trades'] += 1
        self.performance_history['symbol_stats'][symbol]['profit_loss'] += trade_data['profit_loss']
        self.performance_history['symbol_stats'][symbol]['volume'] += trade_data['quantity']
        
        # Update strategy stats
        strategy = trade_data.get('strategy', 'unknown')
        if strategy not in self.performance_history['strategy_stats']:
            self.performance_history['strategy_stats'][strategy] = {
                'trades': 0,
                'profit_loss': 0.0,
                'win_rate': 0.0
            }
        self.performance_history['strategy_stats'][strategy]['trades'] += 1
        self.performance_history['strategy_stats'][strategy]['profit_loss'] += trade_data['profit_loss']
        
        # Update profit/loss stats
        self.performance_history['total_profit'] += trade_data['profit_loss']
        self.performance_history['daily_profit'] += trade_data['profit_loss']
        self.performance_history['profit_loss']['by_symbol'][symbol] = \
            self.performance_history['profit_loss']['by_symbol'].get(symbol, 0.0) + trade_data['profit_loss']
        self.performance_history['profit_loss']['by_strategy'][strategy] = \
            self.performance_history['profit_loss']['by_strategy'].get(strategy, 0.0) + trade_data['profit_loss']
        self.performance_history['profit_loss']['by_hour'][hour] = \
            self.performance_history['profit_loss']['by_hour'].get(hour, 0.0) + trade_data['profit_loss']
        self.performance_history['profit_loss']['by_day'][date] = \
            self.performance_history['profit_loss']['by_day'].get(date, 0.0) + trade_data['profit_loss']
        
        # Save updated history
        self._save_performance_history()
    
    def get_progress_stats(self):
        """Get basic progress statistics"""
        return {
            'total_trades': len(self.performance_history['trades']),
            'winning_trades': sum(1 for trade in self.performance_history['trades'] 
                               if trade.get('profit_loss', 0) > 0),
            'losing_trades': len(self.performance_history['trades']) - sum(1 for trade in self.performance_history['trades'] 
                                                                           if trade.get('profit_loss', 0) > 0),
            'total_profit_loss': self.performance_history['total_profit'],
            'current_balance': self.current_balance
        }
    
    def get_detailed_stats(self):
        """Get detailed performance statistics"""
        return {
            'trades': self.performance_history['trades'],
            'daily_profit_loss': self.performance_history['daily_stats'],
            'win_rate': sum(1 for trade in self.performance_history['trades'] 
                           if trade.get('profit_loss', 0) > 0) / len(self.performance_history['trades']) * 100,
            'average_profit': self.performance_history['total_profit'] / len(self.performance_history['trades']),
            'average_loss': -self.performance_history['total_profit'] / len(self.performance_history['trades'])
        }
    
    def _analyze_symbols(self):
        """Analyze performance by symbol"""
        result = {}
        for symbol, stats in self.performance_history['symbol_stats'].items():
            winning_trades = sum(1 for trade in self.performance_history['trades'] 
                               if trade['symbol'] == symbol and trade.get('profit_loss', 0) > 0)
            win_rate = (winning_trades / stats['trades'] * 100) if stats['trades'] > 0 else 0
            
            result[symbol] = {
                'total_trades': stats['trades'],
                'total_profit': stats['profit_loss'],
                'total_volume': stats['volume'],
                'win_rate': win_rate
            }
        return result
    
    def _analyze_strategies(self):
        """Analyze performance by strategy"""
        result = {}
        for strategy, stats in self.performance_history['strategy_stats'].items():
            winning_trades = sum(1 for trade in self.performance_history['trades'] 
                               if trade.get('strategy', 'unknown') == strategy and 
                               trade.get('profit_loss', 0) > 0)
            win_rate = (winning_trades / stats['trades'] * 100) if stats['trades'] > 0 else 0
            
            result[strategy] = {
                'total_trades': stats['trades'],
                'total_profit': stats['profit_loss'],
                'win_rate': win_rate
            }
        return result
    
    def _analyze_time_patterns(self):
        """Analyze performance by time patterns"""
        result = {
            'best_hours': [],
            'worst_hours': [],
            'best_days': [],
            'worst_days': []
        }
        
        # Analyze hourly patterns
        for hour, stats in self.performance_history['hourly_stats'].items():
            if stats['profit_loss'] > 0:
                result['best_hours'].append(hour)
            else:
                result['worst_hours'].append(hour)
        
        # Analyze daily patterns
        for day, stats in self.performance_history['daily_stats'].items():
            if stats['profit_loss'] > 0:
                result['best_days'].append(day)
            else:
                result['worst_days'].append(day)
        
        return result 

    def ensure_files_exist(self):
        """Ensure necessary files exist with default content"""
        if not os.path.exists(self.performance_file):
            self.save_performance_data({
                'total_profit': 0.0,
                'daily_profit': 0.0,
                'trades': [],
                'last_update': datetime.now().isoformat()
            })
        
        if not os.path.exists(self.positions_file):
            with open(self.positions_file, 'w') as f:
                json.dump({'positions': []}, f)

    def save_performance_data(self, data):
        """Save performance data to file"""
        with open(self.performance_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_performance_data(self):
        """Load performance data from file"""
        try:
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'total_profit': 0.0,
                'daily_profit': 0.0,
                'trades': [],
                'last_update': datetime.now().isoformat()
            }

    def update_performance(self, trade_result):
        """Update performance data with new trade result"""
        data = self.load_performance_data()
        
        # Update profits
        data['total_profit'] += trade_result['profit']
        data['daily_profit'] += trade_result['profit']
        
        # Add trade to history
        data['trades'].append({
            'timestamp': datetime.now().isoformat(),
            'pair': trade_result['pair'],
            'type': trade_result['type'],
            'entry_price': trade_result['entry_price'],
            'exit_price': trade_result['exit_price'],
            'quantity': trade_result['quantity'],
            'profit': trade_result['profit']
        })
        
        # Keep only last 1000 trades
        if len(data['trades']) > 1000:
            data['trades'] = data['trades'][-1000:]
        
        # Reset daily profit if it's a new day
        last_update = datetime.fromisoformat(data['last_update'])
        if datetime.now().date() > last_update.date():
            data['daily_profit'] = trade_result['profit']
        
        data['last_update'] = datetime.now().isoformat()
        self.save_performance_data(data)

    def get_performance_data(self):
        """Get current performance metrics"""
        data = self.load_performance_data()
        
        # Calculate additional metrics
        if data['trades']:
            df = pd.DataFrame(data['trades'])
            data['win_rate'] = (df['profit'] > 0).mean() * 100
            data['avg_profit'] = df['profit'].mean()
            data['max_profit'] = df['profit'].max()
            data['max_loss'] = df['profit'].min()
        else:
            data['win_rate'] = 0
            data['avg_profit'] = 0
            data['max_profit'] = 0
            data['max_loss'] = 0
        
        return data

    def get_active_positions(self):
        """Get current active positions"""
        try:
            with open(self.positions_file, 'r') as f:
                return json.load(f)['positions']
        except (FileNotFoundError, json.JSONDecodeError):
            return [] 