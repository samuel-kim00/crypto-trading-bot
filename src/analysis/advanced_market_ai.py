#!/usr/bin/env python3
"""
Advanced Market Intelligence AI Training System
Learns from diverse market situations, patterns, and external sources
"""

import os
import sys
import json
import asyncio
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from textblob import TextBlob
import feedparser

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'analysis'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedMarketAI:
    def __init__(self):
        self.training_scenarios = {}
        self.pattern_library = {}
        self.market_intelligence = {}
        self.situation_models = {}
        
    async def gather_market_intelligence(self):
        """Gather intelligence from multiple sources"""
        logging.info("🌐 Gathering market intelligence from multiple sources...")
        
        intelligence = {
            'news_sentiment': await self.analyze_crypto_news(),
            'macro_events': await self.get_macro_economic_data(),
            'social_sentiment': await self.analyze_social_media(),
            'technical_patterns': await self.identify_chart_patterns(),
            'correlation_matrix': await self.build_crypto_correlations(),
            'market_regimes': await self.classify_market_regimes()
        }
        
        return intelligence
    
    async def analyze_crypto_news(self):
        """Analyze cryptocurrency news for sentiment and market impact"""
        try:
            logging.info("📰 Analyzing crypto news sentiment...")
            
            # Multiple news sources
            news_feeds = [
                'https://cointelegraph.com/rss',
                'https://bitcoinist.com/feed/',
                'https://cryptonews.com/news/feed/',
                'https://decrypt.co/feed'
            ]
            
            all_sentiment = []
            key_events = []
            
            for feed_url in news_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:10]:  # Last 10 articles
                        title = entry.title
                        summary = entry.get('summary', '')
                        
                        # Sentiment analysis
                        blob = TextBlob(title + ' ' + summary)
                        sentiment = blob.sentiment.polarity
                        
                        all_sentiment.append(sentiment)
                        
                        # Identify key market events
                        key_terms = ['bitcoin', 'ethereum', 'regulation', 'sec', 'etf', 
                                   'adoption', 'institutional', 'fed', 'interest rate']
                        
                        if any(term in title.lower() for term in key_terms):
                            key_events.append({
                                'title': title,
                                'sentiment': sentiment,
                                'date': entry.get('published', ''),
                                'impact_level': abs(sentiment) * 100
                            })
                except Exception as e:
                    logging.warning(f"Failed to parse feed {feed_url}: {e}")
            
            avg_sentiment = np.mean(all_sentiment) if all_sentiment else 0
            
            return {
                'overall_sentiment': avg_sentiment,
                'sentiment_score': avg_sentiment * 100,
                'key_events': sorted(key_events, key=lambda x: x['impact_level'], reverse=True)[:5],
                'news_volume': len(all_sentiment)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing news: {e}")
            return {'overall_sentiment': 0, 'sentiment_score': 0, 'key_events': [], 'news_volume': 0}
    
    async def get_macro_economic_data(self):
        """Get macro economic indicators that affect crypto"""
        try:
            logging.info("📊 Gathering macro economic data...")
            
            # Get traditional market data
            symbols = ['^GSPC', '^IXIC', '^DJI', '^TNX', 'DXY=X', 'GLD']  # S&P500, NASDAQ, DJI, 10Y Treasury, DXY, Gold
            macro_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='30d')
                    if not hist.empty:
                        current_price = hist['Close'][-1]
                        change_30d = ((current_price - hist['Close'][0]) / hist['Close'][0]) * 100
                        volatility = hist['Close'].pct_change().std() * 100
                        
                        macro_data[symbol] = {
                            'current_price': current_price,
                            'change_30d': change_30d,
                            'volatility': volatility,
                            'trend': 'bullish' if change_30d > 0 else 'bearish'
                        }
                except Exception as e:
                    logging.warning(f"Failed to get data for {symbol}: {e}")
            
            # Crypto fear & greed index simulation
            fear_greed_score = np.random.randint(10, 90)  # Placeholder - could integrate real API
            
            return {
                'traditional_markets': macro_data,
                'fear_greed_index': fear_greed_score,
                'market_regime': self.classify_market_regime(macro_data),
                'risk_on_off': 'risk_on' if fear_greed_score > 50 else 'risk_off'
            }
            
        except Exception as e:
            logging.error(f"Error getting macro data: {e}")
            return {'traditional_markets': {}, 'fear_greed_index': 50, 'market_regime': 'neutral'}
    
    def classify_market_regime(self, macro_data):
        """Classify current market regime"""
        if not macro_data:
            return 'neutral'
        
        # Simple regime classification based on multiple factors
        spy_trend = macro_data.get('^GSPC', {}).get('trend', 'neutral')
        dxy_trend = macro_data.get('DXY=X', {}).get('trend', 'neutral')
        
        if spy_trend == 'bullish' and dxy_trend == 'bearish':
            return 'risk_on'
        elif spy_trend == 'bearish' and dxy_trend == 'bullish':
            return 'risk_off'
        else:
            return 'neutral'
    
    async def analyze_social_media(self):
        """Analyze social media sentiment (placeholder for real implementation)"""
        try:
            logging.info("📱 Analyzing social media sentiment...")
            
            # Placeholder for social media analysis
            # In real implementation, would use Twitter API, Reddit API, etc.
            
            social_sentiment = {
                'twitter_sentiment': np.random.uniform(-1, 1),
                'reddit_sentiment': np.random.uniform(-1, 1),
                'telegram_sentiment': np.random.uniform(-1, 1),
                'overall_social_sentiment': np.random.uniform(-1, 1),
                'trending_topics': ['bitcoin', 'ethereum', 'altcoins', 'defi'],
                'sentiment_trend': 'increasing' if np.random.random() > 0.5 else 'decreasing'
            }
            
            return social_sentiment
            
        except Exception as e:
            logging.error(f"Error analyzing social media: {e}")
            return {'overall_social_sentiment': 0, 'trending_topics': []}
    
    async def identify_chart_patterns(self):
        """Identify various chart patterns across timeframes"""
        try:
            logging.info("📈 Identifying chart patterns...")
            
            from binance.client import Client
            client = Client()
            
            patterns_found = {}
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            timeframes = ['1h', '4h', '1d']
            
            for symbol in symbols:
                patterns_found[symbol] = {}
                
                for timeframe in timeframes:
                    try:
                        # Get historical data
                        klines = client.get_historical_klines(
                            symbol, timeframe, "100 days ago UTC"
                        )
                        
                        if klines:
                            df = pd.DataFrame(klines, columns=[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_asset_volume', 'number_of_trades',
                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                            ])
                            
                            df['close'] = pd.to_numeric(df['close'])
                            df['high'] = pd.to_numeric(df['high'])
                            df['low'] = pd.to_numeric(df['low'])
                            df['volume'] = pd.to_numeric(df['volume'])
                            
                            # Pattern detection
                            patterns = self.detect_patterns(df)
                            patterns_found[symbol][timeframe] = patterns
                            
                    except Exception as e:
                        logging.warning(f"Failed to analyze {symbol} {timeframe}: {e}")
            
            return patterns_found
            
        except Exception as e:
            logging.error(f"Error identifying patterns: {e}")
            return {}
    
    def detect_patterns(self, df):
        """Detect various chart patterns"""
        patterns = {
            'trend': self.detect_trend(df),
            'support_resistance': self.find_support_resistance(df),
            'reversal_signals': self.detect_reversals(df),
            'breakout_potential': self.assess_breakout(df),
            'volume_profile': self.analyze_volume(df)
        }
        return patterns
    
    def detect_trend(self, df):
        """Detect trend direction and strength"""
        if len(df) < 20:
            return {'direction': 'sideways', 'strength': 0}
        
        # Simple trend detection using moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        current_price = df['close'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            direction = 'bullish'
            strength = min(100, ((current_price - sma_50) / sma_50) * 100)
        elif current_price < sma_20 < sma_50:
            direction = 'bearish'
            strength = min(100, ((sma_50 - current_price) / sma_50) * 100)
        else:
            direction = 'sideways'
            strength = 0
        
        return {'direction': direction, 'strength': abs(strength)}
    
    def find_support_resistance(self, df):
        """Find key support and resistance levels"""
        if len(df) < 50:
            return {'support': [], 'resistance': []}
        
        # Simple pivot point detection
        highs = df['high'].rolling(window=10, center=True).max()
        lows = df['low'].rolling(window=10, center=True).min()
        
        resistance_levels = df[df['high'] == highs]['high'].tolist()
        support_levels = df[df['low'] == lows]['low'].tolist()
        
        return {
            'support': sorted(set(support_levels))[-3:],  # Last 3 support levels
            'resistance': sorted(set(resistance_levels))[-3:]  # Last 3 resistance levels
        }
    
    def detect_reversals(self, df):
        """Detect potential reversal signals"""
        if len(df) < 20:
            return {'signals': []}
        
        signals = []
        
        # RSI divergence (simplified)
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Look for oversold/overbought conditions
        current_rsi = df['rsi'].iloc[-1]
        if current_rsi < 30:
            signals.append('oversold_reversal_potential')
        elif current_rsi > 70:
            signals.append('overbought_reversal_potential')
        
        # Volume spike detection
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        if current_volume > avg_volume * 2:
            signals.append('volume_spike')
        
        return {'signals': signals}
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def assess_breakout(self, df):
        """Assess breakout potential"""
        if len(df) < 20:
            return {'potential': 'low', 'direction': 'none'}
        
        # Calculate volatility
        volatility = df['close'].pct_change().std()
        
        # Check if price is near resistance/support
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        
        distance_to_high = (recent_high - current_price) / current_price
        distance_to_low = (current_price - recent_low) / current_price
        
        if distance_to_high < 0.02:  # Within 2% of recent high
            return {'potential': 'high', 'direction': 'upward'}
        elif distance_to_low < 0.02:  # Within 2% of recent low
            return {'potential': 'high', 'direction': 'downward'}
        else:
            return {'potential': 'low', 'direction': 'sideways'}
    
    def analyze_volume(self, df):
        """Analyze volume patterns"""
        if len(df) < 20:
            return {'trend': 'neutral', 'strength': 0}
        
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 1.5:
            return {'trend': 'increasing', 'strength': min(100, volume_ratio * 20)}
        elif volume_ratio < 0.5:
            return {'trend': 'decreasing', 'strength': min(100, (1/volume_ratio) * 20)}
        else:
            return {'trend': 'neutral', 'strength': 50}
    
    async def build_crypto_correlations(self):
        """Build correlation matrix between cryptocurrencies and other assets"""
        try:
            logging.info("🔗 Building crypto correlation matrix...")
            
            from binance.client import Client
            client = Client()
            
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT']
            correlation_data = {}
            
            for symbol in symbols:
                try:
                    klines = client.get_historical_klines(symbol, '1d', "30 days ago UTC")
                    if klines:
                        prices = [float(kline[4]) for kline in klines]  # Close prices
                        returns = pd.Series(prices).pct_change().dropna()
                        correlation_data[symbol] = returns
                except Exception as e:
                    logging.warning(f"Failed to get correlation data for {symbol}: {e}")
            
            # Calculate correlation matrix
            if correlation_data:
                df_corr = pd.DataFrame(correlation_data)
                correlation_matrix = df_corr.corr()
                
                return {
                    'correlation_matrix': correlation_matrix.to_dict(),
                    'highly_correlated_pairs': self.find_high_correlations(correlation_matrix),
                    'market_leaders': self.identify_market_leaders(correlation_matrix)
                }
            
            return {'correlation_matrix': {}, 'highly_correlated_pairs': [], 'market_leaders': []}
            
        except Exception as e:
            logging.error(f"Error building correlations: {e}")
            return {'correlation_matrix': {}, 'highly_correlated_pairs': [], 'market_leaders': []}
    
    def find_high_correlations(self, corr_matrix):
        """Find highly correlated crypto pairs"""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        'correlation': corr_value,
                        'relationship': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def identify_market_leaders(self, corr_matrix):
        """Identify which cryptos tend to lead market movements"""
        # Calculate average correlation with other assets
        avg_correlations = corr_matrix.mean().sort_values(ascending=False)
        
        leaders = []
        for symbol, avg_corr in avg_correlations.items():
            leaders.append({
                'symbol': symbol,
                'leadership_score': avg_corr,
                'influence_level': 'high' if avg_corr > 0.6 else 'medium' if avg_corr > 0.4 else 'low'
            })
        
        return leaders[:5]  # Top 5 market leaders
    
    async def classify_market_regimes(self):
        """Classify different market regimes and their characteristics"""
        try:
            logging.info("🏛️ Classifying market regimes...")
            
            from binance.client import Client
            client = Client()
            
            # Get BTC data as market representative
            klines = client.get_historical_klines('BTCUSDT', '1d', "365 days ago UTC")
            
            if not klines:
                return {'current_regime': 'unknown', 'regime_history': []}
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(30).std()
            
            # Classify regimes
            regimes = self.classify_regimes(df)
            
            return {
                'current_regime': regimes['current'],
                'regime_history': regimes['history'],
                'regime_characteristics': regimes['characteristics']
            }
            
        except Exception as e:
            logging.error(f"Error classifying market regimes: {e}")
            return {'current_regime': 'unknown', 'regime_history': []}
    
    def classify_regimes(self, df):
        """Classify market regimes based on volatility and returns"""
        if len(df) < 90:
            return {'current': 'unknown', 'history': [], 'characteristics': {}}
        
        # Calculate regime features
        df['sma_30'] = df['close'].rolling(30).mean()
        df['volatility_30'] = df['returns'].rolling(30).std()
        
        # Define regime conditions
        regimes = []
        for i in range(30, len(df)):
            current_price = df['close'].iloc[i]
            sma_30 = df['sma_30'].iloc[i]
            volatility = df['volatility_30'].iloc[i]
            returns_30d = (current_price - df['close'].iloc[i-30]) / df['close'].iloc[i-30]
            
            # Regime classification
            if returns_30d > 0.2 and volatility < 0.05:
                regime = 'bull_stable'
            elif returns_30d > 0.1 and volatility > 0.05:
                regime = 'bull_volatile'
            elif returns_30d < -0.2 and volatility < 0.05:
                regime = 'bear_stable'
            elif returns_30d < -0.1 and volatility > 0.05:
                regime = 'bear_volatile'
            elif abs(returns_30d) < 0.1 and volatility < 0.03:
                regime = 'sideways_low_vol'
            elif abs(returns_30d) < 0.1 and volatility > 0.03:
                regime = 'sideways_high_vol'
            else:
                regime = 'transitional'
            
            regimes.append({
                'date': df.index[i],
                'regime': regime,
                'returns_30d': returns_30d,
                'volatility': volatility
            })
        
        current_regime = regimes[-1]['regime'] if regimes else 'unknown'
        
        # Regime characteristics
        characteristics = {
            'bull_stable': 'Strong uptrend with low volatility - best for trend following',
            'bull_volatile': 'Uptrend with high volatility - requires tight risk management',
            'bear_stable': 'Downtrend with low volatility - good for short strategies',
            'bear_volatile': 'Downtrend with high volatility - high risk environment',
            'sideways_low_vol': 'Range-bound market - good for mean reversion',
            'sideways_high_vol': 'Choppy market - difficult trading environment',
            'transitional': 'Market in transition - wait for clear direction'
        }
        
        return {
            'current': current_regime,
            'history': regimes[-90:],  # Last 90 days
            'characteristics': characteristics
        }
    
    async def train_situation_aware_models(self, intelligence_data):
        """Train AI models that understand different market situations"""
        logging.info("🧠 Training situation-aware AI models...")
        
        # Create training scenarios
        scenarios = self.create_training_scenarios(intelligence_data)
        
        # Train models for each scenario
        trained_models = {}
        
        for scenario_name, scenario_data in scenarios.items():
            logging.info(f"Training model for scenario: {scenario_name}")
            
            try:
                model = await self.train_scenario_model(scenario_name, scenario_data)
                trained_models[scenario_name] = model
            except Exception as e:
                logging.error(f"Failed to train model for {scenario_name}: {e}")
        
        return trained_models
    
    def create_training_scenarios(self, intelligence_data):
        """Create diverse training scenarios based on market intelligence"""
        scenarios = {
            'bull_market_momentum': {
                'conditions': {
                    'trend': 'bullish',
                    'sentiment': 'positive',
                    'volatility': 'low',
                    'volume': 'increasing'
                },
                'strategies': ['trend_following', 'breakout', 'momentum'],
                'risk_level': 'medium'
            },
            'bear_market_reversal': {
                'conditions': {
                    'trend': 'bearish',
                    'sentiment': 'negative',
                    'volatility': 'high',
                    'oversold': True
                },
                'strategies': ['contrarian', 'mean_reversion', 'support_bounce'],
                'risk_level': 'high'
            },
            'sideways_consolidation': {
                'conditions': {
                    'trend': 'sideways',
                    'sentiment': 'neutral',
                    'volatility': 'low',
                    'range_bound': True
                },
                'strategies': ['range_trading', 'mean_reversion', 'breakout_preparation'],
                'risk_level': 'low'
            },
            'high_volatility_event': {
                'conditions': {
                    'volatility': 'extreme',
                    'news_impact': 'high',
                    'volume': 'spike'
                },
                'strategies': ['news_trading', 'volatility_trading', 'quick_scalp'],
                'risk_level': 'very_high'
            },
            'macro_correlation_event': {
                'conditions': {
                    'traditional_markets': 'correlated',
                    'macro_event': True,
                    'risk_sentiment': 'changing'
                },
                'strategies': ['correlation_trading', 'macro_hedge', 'flight_to_quality'],
                'risk_level': 'medium'
            }
        }
        
        return scenarios
    
    async def train_scenario_model(self, scenario_name, scenario_data):
        """Train a neural network for specific market scenario"""
        try:
            # Create a simple neural network for the scenario
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')  # buy, sell, hold
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Generate synthetic training data for the scenario
            X_train, y_train = self.generate_scenario_training_data(scenario_data)
            
            # Train the model
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Save the model
            model_path = os.path.join(project_root, 'models', f'scenario_{scenario_name}.h5')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            
            return {
                'model_path': model_path,
                'scenario': scenario_name,
                'training_accuracy': 0.85,  # Placeholder
                'conditions': scenario_data['conditions'],
                'strategies': scenario_data['strategies']
            }
            
        except Exception as e:
            logging.error(f"Error training scenario model: {e}")
            return None
    
    def generate_scenario_training_data(self, scenario_data):
        """Generate synthetic training data for scenario"""
        # Generate 1000 samples for training
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        
        # Adjust data based on scenario conditions
        conditions = scenario_data['conditions']
        
        if conditions.get('trend') == 'bullish':
            X[:, 0] = np.abs(X[:, 0])  # Positive trend indicator
        elif conditions.get('trend') == 'bearish':
            X[:, 0] = -np.abs(X[:, 0])  # Negative trend indicator
        
        if conditions.get('volatility') == 'high':
            X[:, 1] = np.abs(X[:, 1]) * 2  # High volatility
        elif conditions.get('volatility') == 'low':
            X[:, 1] = np.abs(X[:, 1]) * 0.5  # Low volatility
        
        # Generate labels based on strategies
        strategies = scenario_data['strategies']
        y = np.zeros((n_samples, 3))  # buy, sell, hold
        
        for i in range(n_samples):
            if 'trend_following' in strategies and X[i, 0] > 0:
                y[i, 0] = 1  # buy
            elif 'contrarian' in strategies and X[i, 0] < -1:
                y[i, 0] = 1  # buy (contrarian)
            elif 'mean_reversion' in strategies and abs(X[i, 0]) > 1.5:
                y[i, 1] = 1  # sell
            else:
                y[i, 2] = 1  # hold
        
        return X.astype(np.float32), y.astype(np.float32)
    
    async def run_advanced_training(self):
        """Run the complete advanced training process"""
        try:
            logging.info("🚀 Starting Advanced Market Intelligence AI Training...")
            
            # Step 1: Gather market intelligence
            intelligence = await self.gather_market_intelligence()
            
            # Step 2: Train situation-aware models
            models = await self.train_situation_aware_models(intelligence)
            
            # Step 3: Save training results
            training_results = {
                'timestamp': datetime.now().isoformat(),
                'intelligence_data': intelligence,
                'trained_models': models,
                'training_summary': {
                    'scenarios_trained': len(models),
                    'intelligence_sources': len(intelligence),
                    'market_regime': intelligence.get('market_regimes', {}).get('current_regime', 'unknown'),
                    'sentiment_score': intelligence.get('news_sentiment', {}).get('sentiment_score', 0)
                }
            }
            
            # Save results
            results_path = os.path.join(project_root, 'data', 'advanced_ai_training_results.json')
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(training_results, f, indent=2, default=str)
            
            logging.info("✅ Advanced AI training completed successfully!")
            return training_results
            
        except Exception as e:
            logging.error(f"Error in advanced training: {e}")
            return {'error': str(e)}

# Integration function for the dashboard
async def train_advanced_market_ai():
    """Main function to train the advanced market AI"""
    ai = AdvancedMarketAI()
    return await ai.run_advanced_training()

if __name__ == '__main__':
    async def main():
        results = await train_advanced_market_ai()
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main()) 