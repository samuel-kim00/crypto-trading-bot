#!/usr/bin/env python3
"""
Self-Learning Trading Bot
=======================
This bot uses multiple AI techniques to train itself and develop trading strategies:
1. Reinforcement Learning (Q-Learning & Deep Q-Networks)
2. Genetic Algorithm for strategy evolution
3. Neural Networks for pattern recognition
4. Online learning from live market data
5. Strategy ensemble and voting
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import talib

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'analysis'))

class TradingEnvironment:
    """Simulated trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 50  # Start after enough data for indicators
        self.balance = self.initial_balance
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.position_value = 0
        self.entry_price = 0
        self.trades_count = 0
        self.returns = []
        return self._get_state()
    
    def _get_state(self):
        """Get current market state as feature vector"""
        if self.current_step >= len(self.data):
            return np.zeros(20)  # Return zero state if no more data
            
        row = self.data.iloc[self.current_step]
        
        # Technical indicators (normalized)
        state = [
            row.get('rsi', 50) / 100,  # RSI normalized 0-1
            row.get('macd', 0) / row.get('close', 1),  # MACD relative to price
            row.get('macd_signal', 0) / row.get('close', 1),
            row.get('bb_position', 0.5),  # Bollinger Band position
            row.get('volume_ratio', 1),  # Volume ratio
            row.get('price_change_1h', 0) / 100,  # 1h price change
            row.get('price_change_4h', 0) / 100,  # 4h price change
            row.get('price_change_24h', 0) / 100,  # 24h price change
            row.get('ema9', row.get('close', 0)) / row.get('close', 1),  # EMA9 relative
            row.get('ema21', row.get('close', 0)) / row.get('close', 1),  # EMA21 relative
            row.get('atr', 0) / row.get('close', 1),  # ATR relative to price
            float(self.position),  # Current position
            self.balance / self.initial_balance,  # Balance ratio
            (self.current_step % 24) / 24,  # Hour of day
            (self.current_step % (24*7)) / (24*7),  # Day of week
            len(self.returns) / 1000,  # Experience factor
            np.mean(self.returns[-10:]) if self.returns else 0,  # Recent performance
            np.std(self.returns[-10:]) if len(self.returns) > 1 else 0,  # Recent volatility
            1 if self.trades_count > 0 else 0,  # Has trading experience
            min(self.trades_count / 100, 1)  # Trading experience level
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
            
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        reward = 0
        info = {}
        
        # Actions: 0=hold, 1=buy, 2=sell
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.position_value = self.balance * 0.95  # 95% of balance (5% reserve)
            reward = -0.001  # Small penalty for transaction cost
            info['action'] = 'BUY'
            
        elif action == 2 and self.position == 1:  # Sell
            profit = (next_price - self.entry_price) / self.entry_price
            self.balance = self.position_value * (1 + profit) * 0.999  # 0.1% transaction fee
            
            reward = profit * 100  # Scale reward
            self.returns.append(profit)
            self.trades_count += 1
            
            self.position = 0
            self.position_value = 0
            self.entry_price = 0
            info['action'] = 'SELL'
            info['profit'] = profit
            
        else:  # Hold
            if self.position == 1:
                # Small reward/penalty based on unrealized P&L
                unrealized_pnl = (next_price - self.entry_price) / self.entry_price
                reward = unrealized_pnl * 0.1  # Small reward for good positions
            info['action'] = 'HOLD'
        
        # Additional rewards/penalties
        if self.position == 1:
            # Penalty for holding losing positions too long
            unrealized_pnl = (next_price - self.entry_price) / self.entry_price
            if unrealized_pnl < -0.05:  # More than 5% loss
                reward -= 0.01
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done, info

class DQNAgent:
    """Deep Q-Network agent for trading"""
    
    def __init__(self, state_size=20, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
    def _build_model(self):
        """Build neural network for Q-learning"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        target_q_values = self.target_network.predict(next_states, verbose=0)
        max_target_q_values = np.max(target_q_values, axis=1)
        
        targets = rewards + (0.95 * max_target_q_values * (1 - dones))
        
        target_full = self.q_network.predict(states, verbose=0)
        target_full[range(batch_size), actions] = targets
        
        self.q_network.fit(states, target_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())

class GeneticStrategyEvolution:
    """Genetic algorithm for evolving trading strategies"""
    
    def __init__(self, population_size=50):
        self.population_size = population_size
        self.generation = 0
        self.population = []
        self.fitness_history = []
        
    def create_individual(self):
        """Create a random trading strategy (chromosome)"""
        return {
            'rsi_buy_threshold': random.uniform(20, 40),
            'rsi_sell_threshold': random.uniform(60, 80),
            'macd_sensitivity': random.uniform(0.5, 2.0),
            'volume_threshold': random.uniform(1.5, 5.0),
            'stop_loss_pct': random.uniform(0.01, 0.05),
            'take_profit_pct': random.uniform(0.02, 0.10),
            'position_size_pct': random.uniform(0.05, 0.20),
            'holding_time_max': random.randint(10, 100),
            'risk_tolerance': random.uniform(0.1, 0.9),
            'trend_following_weight': random.uniform(0.0, 1.0),
            'mean_reversion_weight': random.uniform(0.0, 1.0),
            'momentum_weight': random.uniform(0.0, 1.0)
        }
    
    def initialize_population(self):
        """Create initial population of strategies"""
        self.population = [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual, data):
        """Evaluate strategy performance (fitness function)"""
        try:
            balance = 10000
            position = 0
            entry_price = 0
            trades = 0
            max_drawdown = 0
            peak_balance = balance
            
            for i in range(50, len(data)):
                row = data.iloc[i]
                current_price = row['close']
                
                # Calculate signals based on individual's parameters
                rsi = row.get('rsi', 50)
                macd = row.get('macd', 0)
                macd_signal = row.get('macd_signal', 0)
                volume_ratio = row.get('volume_ratio', 1)
                
                # Buy signal
                if (position == 0 and 
                    rsi < individual['rsi_buy_threshold'] and
                    macd > macd_signal * individual['macd_sensitivity'] and
                    volume_ratio > individual['volume_threshold']):
                    
                    position = balance * individual['position_size_pct'] / current_price
                    entry_price = current_price
                    balance -= position * current_price
                    trades += 1
                
                # Sell signal
                elif position > 0:
                    profit_pct = (current_price - entry_price) / entry_price
                    should_sell = (
                        rsi > individual['rsi_sell_threshold'] or
                        profit_pct >= individual['take_profit_pct'] or
                        profit_pct <= -individual['stop_loss_pct'] or
                        trades % individual['holding_time_max'] == 0
                    )
                    
                    if should_sell:
                        balance += position * current_price
                        position = 0
                        entry_price = 0
                
                # Track drawdown
                current_value = balance + (position * current_price if position > 0 else 0)
                if current_value > peak_balance:
                    peak_balance = current_value
                else:
                    drawdown = (peak_balance - current_value) / peak_balance
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Final balance calculation
            final_balance = balance + (position * data.iloc[-1]['close'] if position > 0 else 0)
            total_return = (final_balance - 10000) / 10000
            
            # Fitness function (combination of return and risk)
            risk_adjusted_return = total_return - (max_drawdown * individual['risk_tolerance'])
            trade_frequency_bonus = min(trades / 100, 0.1)  # Bonus for reasonable trading frequency
            
            fitness = risk_adjusted_return + trade_frequency_bonus
            
            return max(fitness, -1.0)  # Prevent extremely negative fitness
            
        except Exception as e:
            return -1.0  # Poor fitness for broken strategies
    
    def selection(self, fitness_scores):
        """Select parents for reproduction using tournament selection"""
        tournament_size = 5
        selected = []
        
        for _ in range(self.population_size):
            tournament = random.sample(list(zip(self.population, fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Create offspring through crossover"""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def mutate(self, individual, mutation_rate=0.1):
        """Mutate individual parameters"""
        for key, value in individual.items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    individual[key] = value * random.uniform(0.8, 1.2)
                elif isinstance(value, int):
                    individual[key] = max(1, int(value * random.uniform(0.8, 1.2)))
        return individual
    
    def evolve(self, data):
        """Evolve the population for one generation"""
        # Evaluate fitness
        fitness_scores = [self.evaluate_fitness(ind, data) for ind in self.population]
        self.fitness_history.append(max(fitness_scores))
        
        # Selection
        selected = self.selection(fitness_scores)
        
        # Create new generation
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % len(selected)]
            
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return max(fitness_scores), np.mean(fitness_scores)

class PatternRecognitionNN:
    """Neural network for recognizing trading patterns"""
    
    def __init__(self, input_size=100, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _build_model(self):
        """Build pattern recognition neural network"""
        model = keras.Sequential([
            layers.Dense(self.hidden_size, activation='relu', input_shape=(self.input_size,)),
            layers.Dropout(0.3),
            layers.Dense(self.hidden_size // 2, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.hidden_size // 4, activation='relu'),
            layers.Dense(3, activation='softmax')  # 3 classes: buy, sell, hold
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_features(self, data, lookback=50):
        """Prepare features for pattern recognition"""
        features = []
        labels = []
        
        for i in range(lookback, len(data) - 1):
            # Create feature vector from recent price/volume data
            window = data.iloc[i-lookback:i]
            
            feature_vector = []
            # Price features (normalized)
            prices = window['close'].values
            price_norm = (prices - prices.mean()) / prices.std()
            feature_vector.extend(price_norm)
            
            # Volume features (normalized)
            volumes = window['volume'].values
            volume_norm = (volumes - volumes.mean()) / volumes.std()
            feature_vector.extend(volume_norm[-10:])  # Last 10 volume points
            
            # Technical indicators
            feature_vector.extend([
                window['rsi'].iloc[-1] / 100,
                window['macd'].iloc[-1],
                window['macd_signal'].iloc[-1],
                window.get('bb_position', pd.Series([0.5])).iloc[-1],
                window.get('volume_ratio', pd.Series([1.0])).iloc[-1]
            ])
            
            # Pad or truncate to exact input size
            if len(feature_vector) > self.input_size:
                feature_vector = feature_vector[:self.input_size]
            else:
                feature_vector.extend([0] * (self.input_size - len(feature_vector)))
            
            features.append(feature_vector)
            
            # Create label (future price movement)
            current_price = data.iloc[i]['close']
            future_price = data.iloc[i + 1]['close']
            price_change = (future_price - current_price) / current_price
            
            # Classify into buy/sell/hold
            if price_change > 0.002:  # > 0.2% increase
                labels.append([1, 0, 0])  # Buy
            elif price_change < -0.002:  # > 0.2% decrease
                labels.append([0, 1, 0])  # Sell
            else:
                labels.append([0, 0, 1])  # Hold
        
        return np.array(features), np.array(labels)
    
    def train(self, data):
        """Train the pattern recognition model"""
        X, y = self.prepare_features(data)
        
        if len(X) < 100:  # Need minimum data
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        test_predictions = self.model.predict(X_test_scaled, verbose=0)
        test_accuracy = accuracy_score(
            np.argmax(y_test, axis=1),
            np.argmax(test_predictions, axis=1)
        )
        
        self.is_trained = True
        return test_accuracy > 0.4  # Minimum acceptable accuracy
    
    def predict(self, data, lookback=50):
        """Predict trading action for current market state"""
        if not self.is_trained or len(data) < lookback:
            return 2, 0.33  # Hold with low confidence
        
        # Prepare feature vector for latest data
        window = data.iloc[-lookback:]
        feature_vector = []
        
        # Price features
        prices = window['close'].values
        price_norm = (prices - prices.mean()) / prices.std()
        feature_vector.extend(price_norm)
        
        # Volume features
        volumes = window['volume'].values
        volume_norm = (volumes - volumes.mean()) / volumes.std()
        feature_vector.extend(volume_norm[-10:])
        
        # Technical indicators
        feature_vector.extend([
            window['rsi'].iloc[-1] / 100,
            window['macd'].iloc[-1],
            window['macd_signal'].iloc[-1],
            window.get('bb_position', pd.Series([0.5])).iloc[-1],
            window.get('volume_ratio', pd.Series([1.0])).iloc[-1]
        ])
        
        # Pad or truncate
        if len(feature_vector) > self.input_size:
            feature_vector = feature_vector[:self.input_size]
        else:
            feature_vector.extend([0] * (self.input_size - len(feature_vector)))
        
        # Scale and predict
        feature_scaled = self.scaler.transform([feature_vector])
        prediction = self.model.predict(feature_scaled, verbose=0)[0]
        
        action = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return action, confidence

class SelfLearningTradingBot:
    """Main self-learning trading bot class"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        
        # AI Components
        self.dqn_agent = DQNAgent()
        self.genetic_evolution = GeneticStrategyEvolution()
        self.pattern_nn = PatternRecognitionNN()
        
        # Learning parameters
        self.learning_episodes = 1000
        self.genetic_generations = 50
        self.retrain_interval = 100  # Retrain every 100 trades
        
        # Model persistence
        self.models_dir = 'src/analysis/models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.strategy_performance = {
            'dqn': [],
            'genetic': [],
            'pattern_nn': [],
            'ensemble': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_indicators(self, df):
        """Calculate technical indicators for the data"""
        data = df.copy()
        
        # Basic indicators
        data['rsi'] = talib.RSI(data['close'].values, timeperiod=14)
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(data['close'].values)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values)
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # EMAs
        data['ema9'] = talib.EMA(data['close'].values, timeperiod=9)
        data['ema21'] = talib.EMA(data['close'].values, timeperiod=21)
        
        # Volume analysis
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Price changes
        data['price_change_1h'] = data['close'].pct_change(1) * 100
        data['price_change_4h'] = data['close'].pct_change(4) * 100
        data['price_change_24h'] = data['close'].pct_change(24) * 100
        
        # ATR
        data['atr'] = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
        
        return data.fillna(method='bfill').fillna(method='ffill')
    
    def train_dqn_agent(self, data):
        """Train the DQN agent on historical data"""
        self.logger.info("🧠 Training DQN Agent...")
        
        env = TradingEnvironment(data, self.initial_balance)
        
        episode_returns = []
        best_return = -float('inf')
        
        for episode in range(self.learning_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.dqn_agent.act(state)
                next_state, reward, done, info = env.step(action)
                self.dqn_agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Train the agent
            if len(self.dqn_agent.memory) > 1000:
                self.dqn_agent.replay()
            
            # Update target network periodically
            if episode % 100 == 0:
                self.dqn_agent.update_target_network()
            
            episode_returns.append(total_reward)
            
            if total_reward > best_return:
                best_return = total_reward
                # Save best model
                self.dqn_agent.q_network.save(f'{self.models_dir}/best_dqn_model.h5')
            
            if episode % 100 == 0:
                avg_return = np.mean(episode_returns[-100:])
                self.logger.info(f"Episode {episode}, Avg Return: {avg_return:.4f}, Epsilon: {self.dqn_agent.epsilon:.4f}")
        
        self.strategy_performance['dqn'] = episode_returns
        return best_return
    
    def evolve_genetic_strategies(self, data):
        """Evolve trading strategies using genetic algorithm"""
        self.logger.info("🧬 Evolving Genetic Strategies...")
        
        self.genetic_evolution.initialize_population()
        
        best_fitness_history = []
        avg_fitness_history = []
        
        for generation in range(self.genetic_generations):
            best_fitness, avg_fitness = self.genetic_evolution.evolve(data)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            if generation % 10 == 0:
                self.logger.info(f"Generation {generation}, Best Fitness: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")
        
        # Save best strategy
        best_individual = max(self.genetic_evolution.population, 
                            key=lambda x: self.genetic_evolution.evaluate_fitness(x, data))
        
        with open(f'{self.models_dir}/best_genetic_strategy.json', 'w') as f:
            json.dump(best_individual, f, indent=2)
        
        self.strategy_performance['genetic'] = best_fitness_history
        return best_individual
    
    def train_pattern_recognition(self, data):
        """Train the pattern recognition neural network"""
        self.logger.info("🔍 Training Pattern Recognition...")
        
        success = self.pattern_nn.train(data)
        
        if success:
            self.pattern_nn.model.save(f'{self.models_dir}/pattern_recognition_model.h5')
            self.logger.info("✅ Pattern Recognition trained successfully")
        else:
            self.logger.warning("⚠️ Pattern Recognition training failed - insufficient data or poor performance")
        
        return success
    
    def ensemble_prediction(self, data):
        """Combine predictions from all AI models"""
        predictions = []
        confidences = []
        
        # DQN prediction
        env = TradingEnvironment(data, self.initial_balance)
        state = env._get_state()
        dqn_action = self.dqn_agent.act(state)
        dqn_confidence = 0.7 if self.dqn_agent.epsilon < 0.1 else 0.3
        predictions.append(dqn_action)
        confidences.append(dqn_confidence)
        
        # Pattern NN prediction
        if self.pattern_nn.is_trained:
            pattern_action, pattern_confidence = self.pattern_nn.predict(data)
            predictions.append(pattern_action)
            confidences.append(pattern_confidence)
        
        # Genetic strategy prediction (if available)
        genetic_strategy_path = f'{self.models_dir}/best_genetic_strategy.json'
        if os.path.exists(genetic_strategy_path):
            with open(genetic_strategy_path, 'r') as f:
                best_strategy = json.load(f)
            
            # Apply genetic strategy
            if len(data) > 0:
                current_row = data.iloc[-1]
                rsi = current_row.get('rsi', 50)
                macd = current_row.get('macd', 0)
                macd_signal = current_row.get('macd_signal', 0)
                volume_ratio = current_row.get('volume_ratio', 1)
                
                genetic_action = 0  # Hold by default
                if (rsi < best_strategy['rsi_buy_threshold'] and
                    macd > macd_signal * best_strategy['macd_sensitivity'] and
                    volume_ratio > best_strategy['volume_threshold']):
                    genetic_action = 1  # Buy
                elif rsi > best_strategy['rsi_sell_threshold']:
                    genetic_action = 2  # Sell
                
                predictions.append(genetic_action)
                confidences.append(0.6)
        
        # Ensemble voting
        if not predictions:
            return 0, 0.0  # Hold with no confidence
        
        # Weighted voting
        weighted_votes = {}
        for pred, conf in zip(predictions, confidences):
            if pred not in weighted_votes:
                weighted_votes[pred] = 0
            weighted_votes[pred] += conf
        
        # Final prediction
        final_action = max(weighted_votes, key=weighted_votes.get)
        final_confidence = weighted_votes[final_action] / sum(confidences)
        
        return final_action, final_confidence
    
    async def train_all_models(self, historical_data):
        """Train all AI models on historical data"""
        self.logger.info("🚀 Starting Self-Learning Training Process...")
        
        # Prepare data
        processed_data = self.calculate_indicators(historical_data)
        
        # Train all models
        results = {}
        
        # 1. Train DQN Agent
        dqn_performance = self.train_dqn_agent(processed_data)
        results['dqn_performance'] = dqn_performance
        
        # 2. Evolve Genetic Strategies
        best_genetic_strategy = self.evolve_genetic_strategies(processed_data)
        results['best_genetic_strategy'] = best_genetic_strategy
        
        # 3. Train Pattern Recognition
        pattern_success = self.train_pattern_recognition(processed_data)
        results['pattern_recognition_success'] = pattern_success
        
        self.logger.info("✅ Self-Learning Training Complete!")
        return results
    
    def save_models(self):
        """Save all trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DQN
        self.dqn_agent.q_network.save(f'{self.models_dir}/dqn_model_{timestamp}.h5')
        
        # Save Pattern NN
        if self.pattern_nn.is_trained:
            self.pattern_nn.model.save(f'{self.models_dir}/pattern_model_{timestamp}.h5')
        
        # Save performance history
        performance_data = {
            'timestamp': timestamp,
            'strategy_performance': self.strategy_performance,
            'trade_history': self.trade_history
        }
        
        with open(f'{self.models_dir}/performance_history_{timestamp}.json', 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
    
    def load_models(self, timestamp=None):
        """Load previously trained models"""
        try:
            if timestamp is None:
                # Find latest models
                model_files = os.listdir(self.models_dir)
                if not model_files:
                    return False
                
                # Load best DQN model
                if os.path.exists(f'{self.models_dir}/best_dqn_model.h5'):
                    self.dqn_agent.q_network = keras.models.load_model(f'{self.models_dir}/best_dqn_model.h5')
                    self.dqn_agent.target_network = keras.models.load_model(f'{self.models_dir}/best_dqn_model.h5')
                
                # Load Pattern NN
                if os.path.exists(f'{self.models_dir}/pattern_recognition_model.h5'):
                    self.pattern_nn.model = keras.models.load_model(f'{self.models_dir}/pattern_recognition_model.h5')
                    self.pattern_nn.is_trained = True
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    async def autonomous_trading_session(self, live_data_feed):
        """Run autonomous trading session with continuous learning"""
        self.logger.info("🤖 Starting Autonomous Trading Session...")
        
        trades_since_retrain = 0
        session_data = []
        
        async for market_data in live_data_feed:
            # Add new data to session
            session_data.append(market_data)
            
            # Keep only recent data (last 1000 candles)
            if len(session_data) > 1000:
                session_data = session_data[-1000:]
            
            # Process data
            if len(session_data) >= 100:  # Minimum data for analysis
                df = pd.DataFrame(session_data)
                processed_data = self.calculate_indicators(df)
                
                # Get ensemble prediction
                action, confidence = self.ensemble_prediction(processed_data)
                
                # Execute trade if confidence is high enough
                if confidence > 0.6:
                    current_price = processed_data.iloc[-1]['close']
                    
                    if action == 1:  # Buy
                        await self.execute_buy(current_price, confidence)
                    elif action == 2:  # Sell
                        await self.execute_sell(current_price, confidence)
                
                # Retrain models periodically
                trades_since_retrain += 1
                if trades_since_retrain >= self.retrain_interval:
                    self.logger.info("🔄 Retraining models with new data...")
                    await self.train_all_models(processed_data)
                    trades_since_retrain = 0
                
            # Small delay
            await asyncio.sleep(1)
    
    async def execute_buy(self, price, confidence):
        """Execute buy order"""
        position_size = self.balance * 0.1 * confidence  # Size based on confidence
        if position_size > 100 and 'position' not in self.positions:
            self.positions['position'] = {
                'type': 'long',
                'entry_price': price,
                'size': position_size / price,
                'entry_time': datetime.now(),
                'confidence': confidence
            }
            self.balance -= position_size
            
            self.trade_history.append({
                'action': 'BUY',
                'price': price,
                'size': position_size / price,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"🟢 BUY @ ${price:.4f} | Confidence: {confidence:.2f}")
    
    async def execute_sell(self, price, confidence):
        """Execute sell order"""
        if 'position' in self.positions:
            position = self.positions['position']
            proceeds = position['size'] * price
            profit = proceeds - (position['size'] * position['entry_price'])
            
            self.balance += proceeds
            
            self.trade_history.append({
                'action': 'SELL',
                'price': price,
                'size': position['size'],
                'profit': profit,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            del self.positions['position']
            self.logger.info(f"🔴 SELL @ ${price:.4f} | Profit: ${profit:.2f} | Confidence: {confidence:.2f}")

# Usage example
async def main():
    """Example usage of the self-learning trading bot"""
    
    # Initialize the bot
    bot = SelfLearningTradingBot(initial_balance=10000)
    
    # Load historical data (example)
    # In practice, you'd load real historical data
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='1H')
    np.random.seed(42)
    historical_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Train all models
    training_results = await bot.train_all_models(historical_data)
    print("Training Results:", training_results)
    
    # Save trained models
    bot.save_models()
    
    print("✅ Self-Learning Trading Bot trained and ready!")

if __name__ == "__main__":
    asyncio.run(main()) 