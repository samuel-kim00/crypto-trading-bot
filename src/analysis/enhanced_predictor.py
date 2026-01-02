import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import ta
import asyncio
import aiohttp
from dotenv import load_dotenv
import logging
import warnings
import glob
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

load_dotenv()

class EnhancedPredictor:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.market_data = {}
        self.prediction_accuracy = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load prediction history for feedback learning
        self.load_prediction_history()
        
    def load_prediction_history(self):
        """Load historical predictions for feedback analysis"""
        try:
            with open('data/prediction_history.json', 'r') as f:
                self.prediction_history = json.load(f)
        except FileNotFoundError:
            self.prediction_history = []
            
    def save_prediction_history(self):
        """Save prediction history for future feedback analysis"""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/prediction_history.json', 'w') as f:
                json.dump(self.prediction_history, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving prediction history: {str(e)}")
    
    def analyze_past_predictions(self):
        """Analyze accuracy of past predictions for feedback learning"""
        if not self.prediction_history:
            return {"message": "No historical predictions to analyze"}
        
        feedback_analysis = {
            'total_predictions': len(self.prediction_history),
            'accuracy_by_timeframe': {},
            'accuracy_by_symbol': {},
            'model_performance': {},
            'lessons_learned': []
        }
        
        for pred in self.prediction_history:
            if 'actual_performance' in pred:
                symbol = pred['symbol']
                
                # Calculate accuracy for each timeframe
                for timeframe in ['1d', '3d', '7d']:
                    if f'predicted_{timeframe}' in pred and f'actual_{timeframe}' in pred['actual_performance']:
                        predicted = pred[f'predicted_{timeframe}']
                        actual = pred['actual_performance'][f'actual_{timeframe}']
                        
                        error = abs(predicted - actual)
                        accuracy = max(0, 100 - (error * 100))  # Convert to percentage accuracy
                        
                        if timeframe not in feedback_analysis['accuracy_by_timeframe']:
                            feedback_analysis['accuracy_by_timeframe'][timeframe] = []
                        feedback_analysis['accuracy_by_timeframe'][timeframe].append(accuracy)
                        
                        if symbol not in feedback_analysis['accuracy_by_symbol']:
                            feedback_analysis['accuracy_by_symbol'][symbol] = []
                        feedback_analysis['accuracy_by_symbol'][symbol].append(accuracy)
        
        # Calculate average accuracies
        for timeframe, accuracies in feedback_analysis['accuracy_by_timeframe'].items():
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                feedback_analysis['accuracy_by_timeframe'][timeframe] = {
                    'average_accuracy': avg_accuracy,
                    'sample_size': len(accuracies),
                    'best_accuracy': max(accuracies),
                    'worst_accuracy': min(accuracies)
                }
        
        # Generate lessons learned
        lessons = []
        if '1d' in feedback_analysis['accuracy_by_timeframe']:
            day1_acc = feedback_analysis['accuracy_by_timeframe']['1d']['average_accuracy']
            if day1_acc > 70:
                lessons.append("1-day predictions are highly accurate - good for day trading signals")
            elif day1_acc < 50:
                lessons.append("1-day predictions need improvement - consider reducing day trading recommendations")
        
        if '7d' in feedback_analysis['accuracy_by_timeframe']:
            day7_acc = feedback_analysis['accuracy_by_timeframe']['7d']['average_accuracy']
            if day7_acc > 60:
                lessons.append("7-day predictions are reliable - good for swing trading")
            elif day7_acc < 40:
                lessons.append("7-day predictions are challenging - increase confidence thresholds")
        
        feedback_analysis['lessons_learned'] = lessons
        
        return feedback_analysis
    
    async def fetch_market_data(self, symbol, timeframe='1d', limit=365):
        """Fetch historical market data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Price-based indicators
        df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
        df['sma_21'] = ta.trend.sma_indicator(df['close'], window=21)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_fast'] = ta.momentum.rsi(df['close'], window=7)  # For day trading
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['atr_percent'] = df['atr'] / df['close'] * 100  # ATR as percentage
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend indicators
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Price action features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        df['volume_change'] = df['volume'].pct_change()
        
        # Support/Resistance levels
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        df['distance_to_support'] = (df['close'] - df['support']) / df['close']
        df['distance_to_resistance'] = (df['resistance'] - df['close']) / df['close']
        
        # Day trading specific indicators
        df['intraday_range'] = (df['high'] - df['low']) / df['close']
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    async def get_market_sentiment(self):
        """Get market sentiment from various sources"""
        sentiment_data = {
            'fear_greed_index': 50,
            'bitcoin_dominance': 0,
            'total_market_cap': 0,
            'news_sentiment': 0
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/') as response:
                    if response.status == 200:
                        data = await response.json()
                        sentiment_data['fear_greed_index'] = int(data['data'][0]['value'])
                
                async with session.get('https://api.coingecko.com/api/v3/global') as response:
                    if response.status == 200:
                        data = await response.json()
                        sentiment_data['bitcoin_dominance'] = data['data']['market_cap_percentage']['btc']
                        sentiment_data['total_market_cap'] = data['data']['total_market_cap']['usd']
        
        except Exception as e:
            self.logger.warning(f"Could not fetch sentiment data: {str(e)}")
        
        return sentiment_data
    
    def categorize_trading_opportunity(self, analysis, predictions):
        """Categorize trading opportunity as day trading or long-term"""
        current_rsi = analysis['technical_analysis']['rsi']
        atr_percent = analysis.get('atr_percent', 2.0)
        volume_ratio = analysis['technical_analysis']['volume_ratio']
        
        # Day trading criteria
        day_trading_score = 0
        if atr_percent > 3:  # High volatility
            day_trading_score += 25
        if volume_ratio > 1.5:  # High volume
            day_trading_score += 20
        if 30 < current_rsi < 70:  # Not extreme RSI
            day_trading_score += 15
        if abs(predictions.get('target_1d', 0)) > 0.03:  # Significant 1-day movement expected
            day_trading_score += 25
        if analysis['technical_analysis']['trend'] == 'bullish':
            day_trading_score += 15
        
        # Long-term trading criteria
        long_term_score = 0
        if abs(predictions.get('target_7d', 0)) > 0.10:  # Significant 7-day movement
            long_term_score += 30
        if current_rsi < 35 or current_rsi > 65:  # Extreme RSI for reversal
            long_term_score += 20
        if analysis['technical_analysis']['trend'] == 'bullish':
            long_term_score += 25
        if atr_percent < 5:  # Not too volatile for long-term hold
            long_term_score += 15
        if volume_ratio > 1.2:  # Decent volume
            long_term_score += 10
        
        # Determine category
        if day_trading_score > 60:
            return 'day_trading'
        elif long_term_score > 60:
            return 'long_term'
        elif day_trading_score > long_term_score:
            return 'day_trading'
        else:
            return 'long_term'
    
    def generate_timeframe_specific_signals(self, df, predictions, sentiment_data, category):
        """Generate signals specific to trading timeframe"""
        signals = {
            'recommendation': 'HOLD',
            'confidence': 0,
            'category': category,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': [],
            'reasoning': [],
            'timeframe': '',
            'risk_level': 'medium'
        }
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_rsi_fast = df['rsi_fast'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        atr_percent = df['atr_percent'].iloc[-1] if 'atr_percent' in df.columns else 2.0
        
        if category == 'day_trading':
            signals['timeframe'] = '1-3 days'
            signals['risk_level'] = 'high'
            
            # Day trading signals (more aggressive)
            if 'target_1d' in predictions:
                pred_1d = predictions['target_1d']
                if pred_1d > 0.03:  # 3% upside in 1 day
                    signals['reasoning'].append(f"ML predicts +{pred_1d*100:.1f}% in 1 day")
                    signals['confidence'] += 35
                elif pred_1d < -0.03:
                    signals['reasoning'].append(f"ML predicts {pred_1d*100:.1f}% in 1 day")
                    signals['confidence'] -= 35
            
            # Fast RSI for day trading
            if current_rsi_fast < 25:
                signals['reasoning'].append("Fast RSI oversold - day trading opportunity")
                signals['confidence'] += 25
            elif current_rsi_fast > 75:
                signals['reasoning'].append("Fast RSI overbought - day trading exit signal")
                signals['confidence'] -= 25
            
            # MACD for momentum
            if current_macd > 0:
                signals['reasoning'].append("MACD bullish for short-term momentum")
                signals['confidence'] += 15
            
            # Day trading specific stops and targets
            if signals['confidence'] > 70:
                signals['recommendation'] = 'BUY'
                signals['entry_price'] = current_price * 0.995
                signals['stop_loss'] = current_price * 0.97  # 3% stop
                signals['take_profit'] = [
                    current_price * 1.02,  # 2% target
                    current_price * 1.04,  # 4% target
                    current_price * 1.06   # 6% target
                ]
            elif signals['confidence'] < -70:
                signals['recommendation'] = 'SELL'
        
        else:  # long_term
            signals['timeframe'] = '1-4 weeks'
            signals['risk_level'] = 'medium'
            
            # Long-term signals (more conservative)
            if 'target_7d' in predictions:
                pred_7d = predictions['target_7d']
                if pred_7d > 0.08:  # 8% upside in 7 days
                    signals['reasoning'].append(f"ML predicts +{pred_7d*100:.1f}% in 7 days")
                    signals['confidence'] += 40
                elif pred_7d < -0.08:
                    signals['reasoning'].append(f"ML predicts {pred_7d*100:.1f}% in 7 days")
                    signals['confidence'] -= 40
            
            # Standard RSI for long-term
            if current_rsi < 30:
                signals['reasoning'].append("RSI oversold - long-term value opportunity")
                signals['confidence'] += 30
            elif current_rsi > 70:
                signals['reasoning'].append("RSI overbought - long-term exit signal")
                signals['confidence'] -= 30
            
            # Trend analysis for long-term
            sma_trend = df['sma_7'].iloc[-1] > df['sma_21'].iloc[-1]
            if sma_trend:
                signals['reasoning'].append("Bullish trend for long-term hold")
                signals['confidence'] += 20
            
            # Long-term specific stops and targets
            if signals['confidence'] > 65:
                signals['recommendation'] = 'BUY'
                signals['entry_price'] = current_price * 0.98
                signals['stop_loss'] = current_price * 0.92  # 8% stop
                signals['take_profit'] = [
                    current_price * 1.05,  # 5% target
                    current_price * 1.12,  # 12% target
                    current_price * 1.20   # 20% target
                ]
            elif signals['confidence'] < -65:
                signals['recommendation'] = 'SELL'
        
        # Market sentiment adjustment
        fear_greed = sentiment_data.get('fear_greed_index', 50)
        if fear_greed < 25:
            signals['reasoning'].append("Extreme fear - contrarian opportunity")
            signals['confidence'] += 15
        elif fear_greed > 75:
            signals['reasoning'].append("Extreme greed - caution advised")
            signals['confidence'] -= 15
        
        signals['confidence'] = abs(signals['confidence'])
        return signals
    
    async def get_all_usdt_symbols(self):
        """Get all USDT trading pairs from Binance"""
        try:
            markets = self.exchange.load_markets()
            usdt_symbols = []
            
            for symbol, market in markets.items():
                if (symbol.endswith('/USDT') and 
                    market['active'] and 
                    market['type'] == 'spot' and
                    market['spot']):
                    usdt_symbols.append(symbol)
            
            # Filter out leveraged tokens
            filtered_symbols = []
            unwanted = ['UP/', 'DOWN/', 'BULL/', 'BEAR/', '3L/', '3S/', '5L/', '5S/']
            
            for symbol in usdt_symbols:
                if not any(unwanted_token in symbol for unwanted_token in unwanted):
                    filtered_symbols.append(symbol)
            
            self.logger.info(f"Found {len(filtered_symbols)} USDT trading pairs")
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting USDT symbols: {str(e)}")
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']

    async def filter_by_volume_and_liquidity(self, symbols):
        """Filter symbols by volume and liquidity"""
        volume_data = []
        
        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                if ticker['quoteVolume'] and ticker['quoteVolume'] > 1000000:
                    volume_data.append({
                        'symbol': symbol,
                        'volume': ticker['quoteVolume'],
                        'price': ticker['last']
                    })
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.debug(f"Error getting ticker for {symbol}: {str(e)}")
                continue
        
        volume_data.sort(key=lambda x: x['volume'], reverse=True)
        top_symbols = [item['symbol'] for item in volume_data[:50]]
        self.logger.info(f"Filtered to top {len(top_symbols)} coins by volume")
        
        return top_symbols
    
    def prepare_features(self, df, sentiment_data):
        """Prepare features for ML model"""
        feature_columns = [
            'sma_7', 'sma_21', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'rsi_fast', 'macd', 'macd_signal', 'stoch', 'williams_r',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'atr_percent',
            'volume_sma', 'mfi', 'ad', 'adx', 'cci',
            'volatility', 'volume_change',
            'distance_to_support', 'distance_to_resistance',
            'intraday_range', 'gap'
        ]
        
        # Add sentiment features
        for key, value in sentiment_data.items():
            df[key] = value
            feature_columns.append(key)
        
        # Add time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        feature_columns.extend(['day_of_week', 'month', 'quarter'])
        
        # Create target variables
        df['target_1d'] = df['close'].shift(-1) / df['close'] - 1
        df['target_3d'] = df['close'].shift(-3) / df['close'] - 1
        df['target_7d'] = df['close'].shift(-7) / df['close'] - 1
        
        return df[feature_columns], df[['target_1d', 'target_3d', 'target_7d']]
    
    def train_model(self, X, y, target_col):
        """Train ML model with feedback learning"""
        mask = ~(X.isna().any(axis=1) | y[target_col].isna())
        X_clean = X[mask]
        y_clean = y[target_col][mask]
        
        if len(X_clean) < 50:
            return None, None, {}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, shuffle=False
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Adjust model parameters based on feedback
        feedback = self.analyze_past_predictions()
        n_estimators = 100
        max_depth = 6
        learning_rate = 0.1
        
        # Adjust based on past performance
        if target_col in feedback.get('accuracy_by_timeframe', {}):
            accuracy = feedback['accuracy_by_timeframe'][target_col].get('average_accuracy', 50)
            if accuracy < 50:  # Poor performance, make model more conservative
                max_depth = 4
                learning_rate = 0.05
            elif accuracy > 70:  # Good performance, can be more aggressive
                n_estimators = 150
                max_depth = 8
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        return model, scaler, metrics
    
    async def analyze_symbol(self, symbol):
        """Comprehensive analysis for a single symbol"""
        self.logger.info(f"Analyzing {symbol}...")
        
        df = await self.fetch_market_data(symbol)
        if df is None or len(df) < 100:
            return None
        
        df = self.calculate_technical_indicators(df)
        sentiment_data = await self.get_market_sentiment()
        
        X, y = self.prepare_features(df, sentiment_data)
        
        models = {}
        predictions = {}
        
        for target in ['target_1d', 'target_3d', 'target_7d']:
            model, scaler, metrics = self.train_model(X, y, target)
            if model is not None:
                models[target] = {'model': model, 'scaler': scaler, 'metrics': metrics}
                
                latest_features = X.iloc[-1:].fillna(0)
                scaled_features = scaler.transform(latest_features)
                pred = model.predict(scaled_features)[0]
                predictions[target] = pred
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        atr_percent = df['atr_percent'].iloc[-1] if 'atr_percent' in df.columns else 2.0
        
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'predictions': predictions,
            'technical_analysis': {
                'rsi': current_rsi,
                'macd': current_macd,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'trend': 'bullish' if df['sma_7'].iloc[-1] > df['sma_21'].iloc[-1] else 'bearish',
                'support': df['support'].iloc[-1],
                'resistance': df['resistance'].iloc[-1]
            },
            'atr_percent': atr_percent,
            'model_performance': {k: v['metrics'] for k, v in models.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        # Categorize and generate timeframe-specific signals
        category = self.categorize_trading_opportunity(analysis, predictions)
        signals = self.generate_timeframe_specific_signals(df, predictions, sentiment_data, category)
        analysis['signals'] = signals
        
        # Store prediction for future feedback
        prediction_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'predicted_1d': predictions.get('target_1d', 0),
            'predicted_3d': predictions.get('target_3d', 0),
            'predicted_7d': predictions.get('target_7d', 0),
            'current_price': current_price,
            'category': category,
            'recommendation': signals['recommendation'],
            'confidence': signals['confidence']
        }
        self.prediction_history.append(prediction_record)
        
        return analysis
    
    async def generate_enhanced_report(self):
        """Generate enhanced weekly report with categorization and feedback"""
        self.logger.info("Generating enhanced weekly trading report...")
        
        # Analyze past predictions first
        feedback_analysis = self.analyze_past_predictions()
        
        all_symbols = await self.get_all_usdt_symbols()
        symbols = await self.filter_by_volume_and_liquidity(all_symbols)
        
        analyses = []
        
        for symbol in symbols:
            try:
                analysis = await self.analyze_symbol(symbol)
                if analysis:
                    analyses.append(analysis)
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        # Categorize recommendations
        day_trading_buys = [a for a in analyses if a['signals']['recommendation'] == 'BUY' and a['signals']['category'] == 'day_trading' and a['signals']['confidence'] >= 70]
        long_term_buys = [a for a in analyses if a['signals']['recommendation'] == 'BUY' and a['signals']['category'] == 'long_term' and a['signals']['confidence'] >= 65]
        
        day_trading_sells = [a for a in analyses if a['signals']['recommendation'] == 'SELL' and a['signals']['category'] == 'day_trading' and a['signals']['confidence'] >= 70]
        long_term_sells = [a for a in analyses if a['signals']['recommendation'] == 'SELL' and a['signals']['category'] == 'long_term' and a['signals']['confidence'] >= 65]
        
        # Sort by confidence
        day_trading_buys.sort(key=lambda x: x['signals']['confidence'], reverse=True)
        long_term_buys.sort(key=lambda x: x['signals']['confidence'], reverse=True)
        day_trading_sells.sort(key=lambda x: x['signals']['confidence'], reverse=True)
        long_term_sells.sort(key=lambda x: x['signals']['confidence'], reverse=True)
        
        watchlist = [a for a in analyses if a['signals']['recommendation'] == 'HOLD']
        watchlist.sort(key=lambda x: (
            x['technical_analysis']['volume_ratio'] * 
            (100 - x['technical_analysis']['rsi']) if x['technical_analysis']['rsi'] < 50 else 0
        ), reverse=True)
        
        sentiment_data = await self.get_market_sentiment()
        
        report = {
            'report_date': datetime.now().isoformat(),
            'feedback_analysis': feedback_analysis,
            'market_overview': {
                'fear_greed_index': sentiment_data.get('fear_greed_index', 50),
                'bitcoin_dominance': sentiment_data.get('bitcoin_dominance', 0),
                'market_sentiment': self.interpret_market_sentiment(sentiment_data),
                'total_analyzed': len(analyses),
                'day_trading_opportunities': len(day_trading_buys),
                'long_term_opportunities': len(long_term_buys)
            },
            'day_trading': {
                'buy_recommendations': day_trading_buys,
                'sell_recommendations': day_trading_sells,
                'characteristics': 'High volatility, 1-3 day timeframe, 3% stops, 2-6% targets'
            },
            'long_term': {
                'buy_recommendations': long_term_buys,
                'sell_recommendations': long_term_sells,
                'characteristics': '1-4 week timeframe, 8% stops, 5-20% targets'
            },
            'watchlist': watchlist[:20],
            'summary': self.generate_enhanced_summary(day_trading_buys, long_term_buys, day_trading_sells, long_term_sells, sentiment_data, len(analyses), feedback_analysis)
        }
        
        # Save report and prediction history
        os.makedirs('reports', exist_ok=True)
        report_filename = f"reports/enhanced_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=4)
        
        self.save_prediction_history()
        
        self.logger.info(f"Enhanced report saved to {report_filename}")
        return report
    
    def interpret_market_sentiment(self, sentiment_data):
        """Interpret overall market sentiment"""
        fear_greed = sentiment_data.get('fear_greed_index', 50)
        
        if fear_greed < 25:
            return "Extreme Fear - Potential buying opportunities"
        elif fear_greed < 45:
            return "Fear - Cautious optimism warranted"
        elif fear_greed < 55:
            return "Neutral - Mixed signals in the market"
        elif fear_greed < 75:
            return "Greed - Monitor for overextension"
        else:
            return "Extreme Greed - Consider taking profits"
    
    def generate_enhanced_summary(self, day_buys, long_buys, day_sells, long_sells, sentiment_data, total_analyzed, feedback):
        """Generate enhanced executive summary"""
        summary = []
        
        summary.append(f"Analyzed {total_analyzed} cryptocurrencies with >$1M daily volume")
        
        # Feedback insights
        if feedback.get('lessons_learned'):
            summary.append("📊 Learning from past predictions:")
            for lesson in feedback['lessons_learned'][:2]:
                summary.append(f"   • {lesson}")
        
        # Day trading opportunities
        if day_buys:
            summary.append(f"⚡ {len(day_buys)} high-confidence DAY TRADING opportunities")
            top_day = day_buys[0]
            summary.append(f"   Top: {top_day['symbol']} ({top_day['signals']['confidence']:.0f}% confidence, {top_day['signals']['timeframe']})")
        
        # Long-term opportunities
        if long_buys:
            summary.append(f"📈 {len(long_buys)} high-confidence LONG-TERM opportunities")
            top_long = long_buys[0]
            summary.append(f"   Top: {top_long['symbol']} ({top_long['signals']['confidence']:.0f}% confidence, {top_long['signals']['timeframe']})")
        
        if not day_buys and not long_buys:
            summary.append("🟡 No high-confidence opportunities in current market conditions")
        
        # Sell warnings
        total_sells = len(day_sells) + len(long_sells)
        if total_sells > 0:
            summary.append(f"🔴 {total_sells} high-confidence SELL warnings")
        
        fear_greed = sentiment_data.get('fear_greed_index', 50)
        summary.append(f"Market sentiment: {self.interpret_market_sentiment(sentiment_data)} (F&G: {fear_greed})")
        
        return summary

async def main():
    predictor = EnhancedPredictor()
    report = await predictor.generate_enhanced_report()
    
    print("\n=== ENHANCED WEEKLY TRADING REPORT ===")
    print(f"Generated: {report['report_date']}")
    
    # Show feedback analysis
    feedback = report['feedback_analysis']
    if feedback.get('total_predictions', 0) > 0:
        print(f"\n📊 LEARNING FROM PAST PREDICTIONS:")
        print(f"Total past predictions analyzed: {feedback['total_predictions']}")
        for timeframe, data in feedback.get('accuracy_by_timeframe', {}).items():
            if isinstance(data, dict):
                print(f"{timeframe} accuracy: {data['average_accuracy']:.1f}% (samples: {data['sample_size']})")
    
    print(f"\nMarket Overview:")
    print(f"- Fear & Greed Index: {report['market_overview']['fear_greed_index']}")
    print(f"- Bitcoin Dominance: {report['market_overview']['bitcoin_dominance']:.1f}%")
    print(f"- Sentiment: {report['market_overview']['market_sentiment']}")
    
    # Day trading opportunities
    day_buys = report['day_trading']['buy_recommendations']
    if day_buys:
        print(f"\n⚡ DAY TRADING OPPORTUNITIES ({len(day_buys)}):")
        for i, rec in enumerate(day_buys[:5], 1):
            print(f"{i}. {rec['symbol']} - {rec['signals']['confidence']:.0f}% confidence")
            print(f"   Entry: ${rec['signals']['entry_price']:.4f}, Stop: ${rec['signals']['stop_loss']:.4f}")
            print(f"   Timeframe: {rec['signals']['timeframe']}, Risk: {rec['signals']['risk_level']}")
            print(f"   Reasoning: {', '.join(rec['signals']['reasoning'][:2])}")
            print()
    
    # Long-term opportunities
    long_buys = report['long_term']['buy_recommendations']
    if long_buys:
        print(f"\n📈 LONG-TERM OPPORTUNITIES ({len(long_buys)}):")
        for i, rec in enumerate(long_buys[:5], 1):
            print(f"{i}. {rec['symbol']} - {rec['signals']['confidence']:.0f}% confidence")
            print(f"   Entry: ${rec['signals']['entry_price']:.4f}, Stop: ${rec['signals']['stop_loss']:.4f}")
            print(f"   Timeframe: {rec['signals']['timeframe']}, Risk: {rec['signals']['risk_level']}")
            print(f"   Reasoning: {', '.join(rec['signals']['reasoning'][:2])}")
            print()
    
    print(f"\n📈 EXECUTIVE SUMMARY:")
    for point in report['summary']:
        print(f"• {point}")

if __name__ == "__main__":
    asyncio.run(main()) 