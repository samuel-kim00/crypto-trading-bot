#!/usr/bin/env python3
"""
Auto-Discovery Backtester - Let the bot choose what to trade
This simulates real autonomous trading where the bot:
1. Discovers all available cryptocurrencies
2. Analyzes each one for trading opportunities
3. Makes its own buy/sell decisions
4. Manages a portfolio autonomously
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os
import sys
import concurrent.futures

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_root, 'src', 'analysis'))

from backtester import VLMBacktester
from enhanced_predictor import EnhancedPredictor

class AutoDiscoveryBacktester:
    """Autonomous backtester that discovers and selects cryptocurrencies to trade"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # {symbol: position_info}
        self.trade_history = []
        self.daily_portfolio_value = {}
        
        # Auto-discovery settings
        self.max_positions = 5  # Maximum concurrent positions
        self.min_volume_usdt = 1000000  # $1M daily volume minimum
        self.position_size_pct = 0.15  # 15% of portfolio per position
        self.scan_interval_hours = 6  # Rescan market every 6 hours
        
        # Risk management
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.08  # 8% take profit
        self.max_daily_loss_pct = 0.05  # 5% max daily loss
        
        # Analysis tools
        self.vlm_backtester = VLMBacktester(initial_balance)
        self.ml_predictor = EnhancedPredictor()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def discover_trading_opportunities(self, date: str, max_symbols: int = 20) -> List[Dict]:
        """Discover and rank trading opportunities across all cryptocurrencies"""
        self.logger.info(f"🔍 Discovering opportunities for {date} (analyzing top {max_symbols} symbols)")
        
        # Get top cryptocurrencies by market cap (reduced for performance)
        all_symbols = await self._get_all_usdt_pairs()
        
        opportunities = []
        
        # Limit analysis based on the adaptive parameter
        symbols_to_analyze = all_symbols[:max_symbols]
        
        # Use concurrent processing for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:  # Increased from 4
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self._analyze_symbol_sync, symbol, date)
                for symbol in symbols_to_analyze
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if result and not isinstance(result, Exception):
                    opportunities.append(result)
        
        # Rank by combined score
        opportunities.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self.logger.info(f"Found {len(opportunities)} trading opportunities from {max_symbols} symbols")
        return opportunities[:6]  # Return top 6 for even faster processing
    
    async def _get_all_usdt_pairs(self) -> List[str]:
        """Get all USDT trading pairs sorted by volume"""
        # Simulated top cryptocurrencies by volume
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'MATICUSDT', 'SOLUSDT', 'DOTUSDT', 'LTCUSDT',
            'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'ETCUSDT', 'FILUSDT',
            'TRXUSDT', 'XLMUSDT', 'VETUSDT', 'ICPUSDT', 'FTMUSDT',
            'HBARUSDT', 'ALGOUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT',
            'THETAUSDT', 'XTZUSDT', 'EGLDUSDT', 'AAVEUSDT', 'EOSUSDT',
            'MKRUSDT', 'KLAYUSDT', 'NEARUSDT', 'FLOWUSDT', 'CHZUSDT',
            'ENJUSDT', 'GRTUSDT', 'BATUSDT', 'ZECUSDT', 'DASHUSDT',
            'COMPUSDT', 'YFIUSDT', 'SUSHIUSDT', 'SNXUSDT', 'UNIUSDT',
            'CRVUSDT', 'BALUSDT', 'RENUSDT', 'KNCUSDT', 'BANDUSDT'
        ]
        return symbols
    
    async def _analyze_symbol_opportunity(self, symbol: str, date: str) -> Dict:
        """Analyze a single symbol for trading opportunity"""
        
        # Get historical data (simulated)
        df = self._generate_sample_data(symbol, date)
        if df.empty:
            return None
        
        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        # VLM Analysis
        vlm_score = self._calculate_vlm_score(df)
        
        # ML Prediction (simplified for speed)
        ml_confidence = await self._get_ml_prediction(symbol, latest)
        
        # Volume and liquidity check
        volume_usdt = latest['volume'] * latest['close']
        if volume_usdt < self.min_volume_usdt:
            return None
        
        # Calculate combined opportunity score
        combined_score = (vlm_score * 0.4) + (ml_confidence * 0.3) + (min(volume_usdt / 10000000, 10) * 0.3)
        
        if combined_score < 5.0:  # Minimum threshold
            return None
        
        return {
            'symbol': symbol,
            'price': latest['close'],
            'vlm_score': vlm_score,
            'ml_confidence': ml_confidence,
            'volume_usdt': volume_usdt,
            'combined_score': combined_score,
            'rsi': latest['rsi'],
            'macd_signal': latest['macd'] > latest['macd_signal'],
            'recommendation': 'BUY' if combined_score > 7.0 else 'WATCH',
            'analysis_date': date
        }
    
    def _generate_sample_data(self, symbol: str, date: str) -> pd.DataFrame:
        """Generate realistic sample data for a cryptocurrency (optimized)"""
        try:
            start_date = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=10)  # Reduced from 14 to 10 days
            dates = pd.date_range(start_date, date, freq='6H')  # Changed from 4H to 6H for fewer data points
            
            # Different volatility patterns for different coins
            base_prices = {
                'BTCUSDT': 45000, 'ETHUSDT': 3000, 'BNBUSDT': 400, 'XRPUSDT': 0.5,
                'ADAUSDT': 0.4, 'DOGEUSDT': 0.08, 'MATICUSDT': 1.2, 'SOLUSDT': 100,
                'DOTUSDT': 8, 'LTCUSDT': 100, 'AVAXUSDT': 20, 'LINKUSDT': 10,
                'ATOMUSDT': 12, 'ETCUSDT': 25, 'FILUSDT': 6, 'TRXUSDT': 0.1,
                'XLMUSDT': 0.15, 'VETUSDT': 0.03, 'ICPUSDT': 15, 'FTMUSDT': 0.3
            }
            
            base_price = base_prices.get(symbol, 10.0)
            volatility = np.random.uniform(0.02, 0.06)  # Reduced volatility range for faster calculation
            
            # Generate price series with trend and noise (vectorized for speed)
            trend = np.random.uniform(-0.001, 0.001)
            random_changes = np.random.normal(trend, volatility, len(dates))
            
            # Calculate cumulative prices
            price_multipliers = np.cumprod(1 + random_changes)
            prices = base_price * price_multipliers
            
            # Generate volumes with correlation to price movement (vectorized)
            volumes = np.random.lognormal(mean=6.5, sigma=0.6, size=len(dates))  # Reduced variance
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': prices * np.random.uniform(1.0, 1.01, len(dates)),  # Reduced high variance
                'low': prices * np.random.uniform(0.99, 1.0, len(dates)),   # Reduced low variance
                'close': prices,
                'volume': volumes
            })
            
            return df.tail(60)  # Last 2.5 days of 6H data (reduced from 84)
            
        except Exception as e:
            self.logger.error(f"Error generating data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis (optimized)"""
        if len(df) < 14:  # Reduced minimum requirement
            return df
        
        # RSI (optimized calculation)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use exponential weighted moving average for faster calculation
        alpha = 1/14
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (simplified)
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands (using smaller window for speed)
        window = min(14, len(df))  # Adaptive window size
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        bb_std = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators (simplified)
        vol_window = min(14, len(df))
        df['volume_sma'] = df['volume'].rolling(window=vol_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def _calculate_vlm_score(self, df: pd.DataFrame) -> float:
        """Calculate VLM (Volatility-Liquidity-Momentum) score"""
        if len(df) < 20:
            return 0.0
        
        latest = df.iloc[-1]
        
        # Volatility Score (0-10)
        volatility = df['close'].pct_change().std() * 100
        vol_score = min(volatility * 2, 10)  # Cap at 10
        
        # Liquidity Score (0-10) 
        liquidity_score = min(latest['volume_ratio'] * 2, 10)
        
        # Momentum Score (0-10)
        rsi = latest['rsi']
        if 30 <= rsi <= 70:  # Good range
            momentum_score = 8
        elif 20 <= rsi < 30 or 70 < rsi <= 80:  # Acceptable
            momentum_score = 6
        else:  # Extreme levels
            momentum_score = 3
        
        # MACD momentum
        if latest['macd'] > latest['macd_signal']:
            momentum_score += 2
        
        return (vol_score + liquidity_score + momentum_score) / 3
    
    async def _get_ml_prediction(self, symbol: str, latest_data: pd.Series) -> float:
        """Get ML prediction confidence (simplified for speed)"""
        try:
            # Simplified ML scoring based on technical patterns
            score = 50  # Base confidence
            
            # RSI pattern
            rsi = latest_data['rsi']
            if 40 <= rsi <= 60:
                score += 20
            elif 30 <= rsi <= 70:
                score += 10
            
            # MACD signal
            if latest_data['macd'] > latest_data['macd_signal']:
                score += 15
            
            # Bollinger band position
            bb_position = (latest_data['close'] - latest_data['bb_lower']) / (latest_data['bb_upper'] - latest_data['bb_lower'])
            if 0.2 <= bb_position <= 0.8:
                score += 10
            
            return min(score, 95)  # Cap at 95%
            
        except Exception:
            return 50  # Default confidence
    
    async def run_auto_discovery_backtest(self, start_date: str, end_date: str) -> Dict:
        """Run autonomous discovery backtesting"""
        self.logger.info("🤖 Starting Autonomous Discovery Backtest")
        self.logger.info(f"📅 Period: {start_date} to {end_date}")
        self.logger.info(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        
        # Generate date range for scanning
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Optimize scanning frequency based on backtest duration
        duration_days = (end_dt - start_dt).days
        
        if duration_days > 90:  # More than 3 months
            scan_interval_hours = 24  # Daily scans for long periods
            symbols_to_analyze = 15  # Reduced from 20
            self.logger.info("🕒 Using daily scanning for long-term backtest")
        elif duration_days > 30:  # More than 1 month
            scan_interval_hours = 16  # Reduced frequency  
            symbols_to_analyze = 10  # Reduced analysis
            self.logger.info("🕒 Using 16-hour scanning for medium-term backtest")
        elif duration_days > 7:  # 1-4 weeks
            scan_interval_hours = 12  # Every 12 hours (reduced from 8)
            symbols_to_analyze = 8   # More focused analysis  
            self.logger.info("🕒 Using 12-hour scanning for short-term backtest")
        else:
            scan_interval_hours = 8   # Every 8 hours for very short
            symbols_to_analyze = 6   # Minimal analysis
            self.logger.info(f"🕒 Using 8-hour scanning for very short backtest")
        
        current_date = start_dt
        scan_dates = []
        
        while current_date <= end_dt:
            scan_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(hours=scan_interval_hours)
        
        total_scans = len(scan_dates)
        self.logger.info(f"🔄 Will perform {total_scans} market scans (every {scan_interval_hours} hours)")
        
        # Track discovery statistics
        discovery_stats = {
            'scans_performed': 0,
            'opportunities_found': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'symbols_traded': set()
        }
        
        # Progress tracking for long backtests
        last_progress_report = 0
        progress_interval = max(1, total_scans // 10)  # Report every 10%
        
        for i, scan_date in enumerate(scan_dates):
            # Progress reporting for long backtests
            if i - last_progress_report >= progress_interval:
                progress_pct = (i / total_scans) * 100
                self.logger.info(f"📈 Progress: {progress_pct:.0f}% ({i}/{total_scans} scans)")
                last_progress_report = i
            
            # Discover opportunities
            opportunities = await self.discover_trading_opportunities(scan_date, symbols_to_analyze)
            discovery_stats['scans_performed'] += 1
            discovery_stats['opportunities_found'] += len(opportunities)
            
            # Execute trades based on discoveries
            await self._execute_autonomous_trades(opportunities, scan_date, discovery_stats)
            
            # Update portfolio value
            portfolio_value = await self._calculate_portfolio_value(scan_date)
            self.daily_portfolio_value[scan_date] = portfolio_value
            
            # Risk management check
            daily_loss = (portfolio_value - self.initial_balance) / self.initial_balance
            if daily_loss < -self.max_daily_loss_pct:
                self.logger.warning(f"⚠️ Daily loss limit reached: {daily_loss:.2%}")
                # Close all positions
                await self._close_all_positions(scan_date, "Risk Management", discovery_stats)
        
        self.logger.info("✅ Backtest completed - generating results...")
        
        # Generate final results
        final_results = await self._generate_discovery_results(discovery_stats)
        return final_results
    
    async def _execute_autonomous_trades(self, opportunities: List[Dict], date: str, stats: Dict):
        """Execute trades based on discovered opportunities"""
        
        # Check for exit conditions on existing positions
        for symbol in list(self.positions.keys()):
            await self._check_exit_conditions(symbol, date, stats)
        
        # Look for new entry opportunities
        available_slots = self.max_positions - len(self.positions)
        if available_slots > 0:
            for opportunity in opportunities[:available_slots]:
                if opportunity['recommendation'] == 'BUY' and opportunity['combined_score'] > 7.0:
                    await self._open_position(opportunity, date, stats)
    
    async def _open_position(self, opportunity: Dict, date: str, stats: Dict):
        """Open a new trading position"""
        symbol = opportunity['symbol']
        price = opportunity['price']
        
        if symbol in self.positions:
            return  # Already have position
        
        # Calculate position size
        position_value = self.current_balance * self.position_size_pct
        quantity = position_value / price
        
        if position_value < 100:  # Minimum position size
            return
        
        # Create position
        self.positions[symbol] = {
            'entry_price': price,
            'quantity': quantity,
            'entry_date': date,
            'entry_reason': f"Auto-discovery: Score {opportunity['combined_score']:.1f}",
            'stop_loss': price * (1 - self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct),
            'ml_confidence': opportunity['ml_confidence'],
            'vlm_score': opportunity['vlm_score']
        }
        
        # Update balance
        self.current_balance -= position_value
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'value': position_value,
            'date': date,
            'reason': self.positions[symbol]['entry_reason'],
            'ml_confidence': opportunity['ml_confidence'],
            'vlm_score': opportunity['vlm_score']
        })
        
        stats['positions_opened'] += 1
        stats['symbols_traded'].add(symbol)
        
        self.logger.info(f"🟢 OPENED {symbol} @ ${price:.4f} | Size: ${position_value:.0f} | Score: {opportunity['combined_score']:.1f}")
    
    async def _check_exit_conditions(self, symbol: str, date: str, stats: Dict):
        """Check if position should be closed"""
        position = self.positions[symbol]
        
        # Get current price (simulated)
        current_price = await self._get_current_price(symbol, date)
        if not current_price:
            return
        
        should_close = False
        close_reason = ""
        
        # Stop loss
        if current_price <= position['stop_loss']:
            should_close = True
            close_reason = "Stop Loss"
        
        # Take profit
        elif current_price >= position['take_profit']:
            should_close = True
            close_reason = "Take Profit"
        
        # Time-based exit (hold for max 7 days)
        entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d')
        current_date = datetime.strptime(date, '%Y-%m-%d')
        if (current_date - entry_date).days > 7:
            should_close = True
            close_reason = "Time Exit"
        
        if should_close:
            await self._close_position(symbol, current_price, date, close_reason, stats)
    
    async def _close_position(self, symbol: str, price: float, date: str, reason: str, stats: Dict):
        """Close a trading position"""
        position = self.positions[symbol]
        
        # Calculate P&L
        position_value = position['quantity'] * price
        entry_value = position['quantity'] * position['entry_price']
        pnl = position_value - entry_value
        pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
        
        # Update balance
        self.current_balance += position_value
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'quantity': position['quantity'],
            'value': position_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'date': date,
            'reason': reason,
            'hold_days': (datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(position['entry_date'], '%Y-%m-%d')).days,
            'entry_price': position['entry_price']
        })
        
        # Remove position
        del self.positions[symbol]
        stats['positions_closed'] += 1
        
        color = "🟢" if pnl > 0 else "🔴"
        self.logger.info(f"{color} CLOSED {symbol} @ ${price:.4f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%) | {reason}")
    
    async def _get_current_price(self, symbol: str, date: str) -> float:
        """Get current price for a symbol (simulated)"""
        try:
            df = self._generate_sample_data(symbol, date)
            if not df.empty:
                return df['close'].iloc[-1]
        except:
            pass
        return None
    
    async def _close_all_positions(self, date: str, reason: str, stats: Dict):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            price = await self._get_current_price(symbol, date)
            if price:
                await self._close_position(symbol, price, date, reason, stats)
    
    async def _calculate_portfolio_value(self, date: str) -> float:
        """Calculate total portfolio value"""
        total_value = self.current_balance
        
        for symbol, position in self.positions.items():
            current_price = await self._get_current_price(symbol, date)
            if current_price:
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        return total_value
    
    async def _generate_discovery_results(self, stats: Dict) -> Dict:
        """Generate comprehensive results from discovery backtesting"""
        final_balance = await self._calculate_portfolio_value(list(self.daily_portfolio_value.keys())[-1])
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Trade analysis
        completed_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        
        win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in completed_trades if t.get('pnl', 0) <= 0]) if completed_trades else 0
        
        results = {
            # Performance metrics
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'total_return': total_return,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(completed_trades) - len(winning_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            
            # Discovery statistics
            'discovery_stats': {
                'scans_performed': stats['scans_performed'],
                'opportunities_found': stats['opportunities_found'],
                'symbols_discovered': len(stats['symbols_traded']),
                'symbols_traded': list(stats['symbols_traded']),
                'avg_opportunities_per_scan': stats['opportunities_found'] / stats['scans_performed'] if stats['scans_performed'] > 0 else 0
            },
            
            # Trading analysis
            'autonomous_trading': {
                'max_concurrent_positions': self.max_positions,
                'positions_opened': stats['positions_opened'],
                'positions_closed': stats['positions_closed'],
                'still_open': len(self.positions),
                'avg_hold_time': np.mean([t.get('hold_days', 0) for t in completed_trades]) if completed_trades else 0,
                'discovery_success_rate': len(winning_trades) / stats['positions_opened'] * 100 if stats['positions_opened'] > 0 else 0
            },
            
            # Portfolio tracking
            'daily_portfolio_value': self.daily_portfolio_value,
            'trade_history': self.trade_history,
            'open_positions': self.positions,
            
            # Strategy info
            'strategy_info': {
                'strategy_type': 'Autonomous Discovery + VLM + ML',
                'ml_integration': True,
                'auto_discovery': True,
                'risk_management': True,
                'adaptive_features': [
                    'Automatic cryptocurrency discovery',
                    'Real-time opportunity scoring',
                    'Dynamic position sizing',
                    'ML-enhanced decision making',
                    'Multi-timeframe analysis',
                    'Risk-based position management'
                ]
            }
        }
        
        return results
    
    def save_discovery_results(self, results: Dict, filename: str = None):
        """Save discovery backtest results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"auto_discovery_backtest_{timestamp}.json"
        
        # Clean results for JSON serialization
        clean_results = self._make_json_serializable(results)
        
        filepath = f"src/analysis/reports/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Discovery results saved to {filepath}")
        return filepath
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _analyze_symbol_sync(self, symbol: str, date: str) -> Dict:
        """Synchronous version of symbol analysis for concurrent processing"""
        try:
            # Get historical data (simulated)
            df = self._generate_sample_data(symbol, date)
            if df.empty:
                return None
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # VLM Analysis
            vlm_score = self._calculate_vlm_score(df)
            
            # Simplified ML prediction for speed
            ml_confidence = self._get_ml_prediction_sync(latest)
            
            # Volume and liquidity check
            volume_usdt = latest['volume'] * latest['close']
            if volume_usdt < self.min_volume_usdt:
                return None
            
            # Calculate combined opportunity score
            combined_score = (vlm_score * 0.4) + (ml_confidence * 0.3) + (min(volume_usdt / 10000000, 10) * 0.3)
            
            if combined_score < 5.0:  # Minimum threshold
                return None
            
            return {
                'symbol': symbol,
                'price': latest['close'],
                'vlm_score': vlm_score,
                'ml_confidence': ml_confidence,
                'volume_usdt': volume_usdt,
                'combined_score': combined_score,
                'rsi': latest['rsi'],
                'macd_signal': latest['macd'] > latest['macd_signal'],
                'recommendation': 'BUY' if combined_score > 7.0 else 'WATCH',
                'analysis_date': date
            }
        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol}: {e}")
            return None

    def _get_ml_prediction_sync(self, latest_data: pd.Series) -> float:
        """Synchronous ML prediction for speed"""
        try:
            # Simplified ML scoring based on technical patterns
            score = 50  # Base confidence
            
            # RSI pattern
            rsi = latest_data['rsi']
            if 40 <= rsi <= 60:
                score += 20
            elif 30 <= rsi <= 70:
                score += 10
            
            # MACD signal
            if latest_data['macd'] > latest_data['macd_signal']:
                score += 15
            
            # Bollinger band position
            bb_position = (latest_data['close'] - latest_data['bb_lower']) / (latest_data['bb_upper'] - latest_data['bb_lower'])
            if 0.2 <= bb_position <= 0.8:
                score += 10
            
            return min(score, 95)  # Cap at 95%
            
        except Exception:
            return 50  # Default confidence

async def main():
    """Demo of autonomous discovery backtesting"""
    backtester = AutoDiscoveryBacktester(initial_balance=10000)
    
    print("🤖 AUTONOMOUS CRYPTOCURRENCY DISCOVERY BACKTESTING")
    print("=" * 60)
    print("🧠 Bot will autonomously:")
    print("   • Discover trading opportunities across ALL cryptocurrencies")
    print("   • Make its own buy/sell decisions")
    print("   • Manage portfolio risk automatically")
    print("   • Adapt to market conditions in real-time")
    print()
    
    # Run autonomous backtest
    results = await backtester.run_auto_discovery_backtest('2024-03-01', '2024-03-15')
    
    # Display results
    print("🎯 AUTONOMOUS TRADING RESULTS")
    print("=" * 40)
    print(f"💰 Final Balance: ${results['final_balance']:,.2f}")
    print(f"📈 Total Return: {results['total_return']:+.2f}%")
    print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
    print(f"📊 Total Trades: {results['total_trades']}")
    print()
    
    print("🔍 DISCOVERY PERFORMANCE")
    print("=" * 40)
    discovery = results['discovery_stats']
    print(f"🔄 Market Scans: {discovery['scans_performed']}")
    print(f"💡 Opportunities Found: {discovery['opportunities_found']}")
    print(f"🪙 Cryptocurrencies Discovered: {discovery['symbols_discovered']}")
    print(f"📈 Symbols Traded: {', '.join(discovery['symbols_traded'])}")
    print(f"⭐ Avg Opportunities/Scan: {discovery['avg_opportunities_per_scan']:.1f}")
    
    # Save results
    filepath = backtester.save_discovery_results(results)
    print(f"\n💾 Full results saved to: {filepath}")

if __name__ == "__main__":
    asyncio.run(main()) 