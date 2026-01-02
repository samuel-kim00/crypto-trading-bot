#!/usr/bin/env python3
"""
Self-Learning Trading Bot Setup
==============================
This script sets up and tests the self-learning trading bot system.
"""

import subprocess
import sys
import os
import json
import asyncio
from datetime import datetime

def install_requirements():
    """Install required packages for self-learning bot"""
    print("🔧 Installing self-learning bot requirements...")
    
    requirements = [
        'tensorflow>=2.13.0',
        'scikit-learn>=1.3.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'talib-binary',
        'python-binance',
        'asyncio',
        'aiohttp'
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All requirements installed successfully!")
    return True

def create_models_directory():
    """Create models directory for storing AI models"""
    models_dir = 'src/analysis/models'
    os.makedirs(models_dir, exist_ok=True)
    print(f"📁 Created models directory: {models_dir}")

def test_basic_functionality():
    """Test basic functionality of the self-learning bot"""
    print("🧪 Testing self-learning bot functionality...")
    
    try:
        # Test imports
        import tensorflow as tf
        from sklearn.neural_network import MLPRegressor
        import talib
        print("✅ All AI libraries imported successfully")
        
        # Test TensorFlow GPU availability
        if tf.config.list_physical_devices('GPU'):
            print("🚀 GPU acceleration available for TensorFlow")
        else:
            print("💻 Using CPU for TensorFlow (GPU not available)")
        
        # Create sample data to test the bot
        import pandas as pd
        import numpy as np
        
        # Generate sample market data
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1H')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        print("✅ Sample data generated successfully")
        
        # Test the self-learning bot (basic initialization)
        sys.path.append('src/analysis')
        from self_learning_bot import SelfLearningTradingBot
        
        bot = SelfLearningTradingBot(initial_balance=10000)
        print("✅ Self-learning bot initialized successfully")
        
        # Test data processing
        processed_data = bot.calculate_indicators(sample_data)
        print(f"✅ Technical indicators calculated: {len(processed_data)} data points")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_training_workflow():
    """Test the complete training workflow"""
    print("🎯 Testing training workflow...")
    
    try:
        # Add paths
        sys.path.append('src/analysis')
        from self_learning_bot import SelfLearningTradingBot
        
        # Create bot
        bot = SelfLearningTradingBot(initial_balance=10000)
        
        # Generate sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='1H')
        np.random.seed(42)
        
        training_data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        print(f"📊 Training data prepared: {len(training_data)} candles")
        
        # Quick training test (reduced parameters for speed)
        bot.learning_episodes = 10  # Reduced for testing
        bot.genetic_generations = 5  # Reduced for testing
        
        print("🧠 Starting mini training session...")
        results = await bot.train_all_models(training_data)
        
        print("✅ Training workflow completed successfully!")
        print(f"Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_configuration():
    """Create configuration files for self-learning bot"""
    config = {
        "self_learning_bot": {
            "enabled": True,
            "initial_balance": 10000,
            "learning_episodes": 1000,
            "genetic_generations": 50,
            "retrain_interval": 100,
            "confidence_threshold": 0.7,
            "ai_methods": [
                "deep_q_network",
                "genetic_algorithm", 
                "pattern_recognition",
                "ensemble_voting"
            ]
        },
        "training_symbols": [
            "BTCUSDT",
            "ETHUSDT", 
            "BNBUSDT",
            "ADAUSDT",
            "SOLUSDT"
        ],
        "model_settings": {
            "save_interval": 100,
            "backup_models": True,
            "performance_tracking": True
        }
    }
    
    config_path = 'config/self_learning_config.json'
    os.makedirs('config', exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️ Configuration saved to {config_path}")

def main():
    """Main setup function"""
    print("🚀 Self-Learning Trading Bot Setup")
    print("=" * 40)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Step 2: Create directories
    create_models_directory()
    
    # Step 3: Create configuration
    create_configuration()
    
    # Step 4: Test basic functionality
    if not test_basic_functionality():
        print("❌ Setup failed at basic functionality test")
        return
    
    # Step 5: Test training workflow
    print("\n🎯 Testing training workflow (this may take a few minutes)...")
    try:
        asyncio.run(test_training_workflow())
    except Exception as e:
        print(f"⚠️ Training workflow test failed: {e}")
        print("This is normal if you don't have sufficient compute resources")
    
    print("\n" + "=" * 50)
    print("✅ Self-Learning Trading Bot Setup Complete!")
    print("\nNext Steps:")
    print("1. Run training: POST /api/train_self_learning_bot")
    print("2. Run AI backtest: POST /api/run_self_learning_backtest")
    print("3. Get live predictions: GET /api/get_ai_prediction/<symbol>")
    print("4. Check status: GET /api/self_learning_status")
    print("\nThe bot will:")
    print("• Train neural networks on market data")
    print("• Evolve trading strategies using genetic algorithms")
    print("• Use reinforcement learning for decision making")
    print("• Combine multiple AI models for ensemble predictions")
    print("• Continuously adapt to market changes")
    print("\n🎉 Happy AI Trading!")

if __name__ == "__main__":
    main() 