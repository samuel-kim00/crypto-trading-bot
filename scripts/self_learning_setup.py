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