#!/usr/bin/env python3
"""
Test Self-Learning Trading Bot
=============================
Quick test to verify the self-learning bot is working correctly.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project paths
sys.path.append('src/analysis')

async def test_self_learning_bot():
    """Test the self-learning trading bot"""
    print("🤖 Testing Self-Learning Trading Bot")
    print("=" * 40)
    
    try:
        # Import the bot
        from self_learning_bot import SelfLearningTradingBot
        
        # Create sample data
        print("📊 Creating sample market data...")
        dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='1H')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        print(f"✅ Generated {len(sample_data)} data points")
        
        # Initialize bot
        print("🧠 Initializing self-learning bot...")
        bot = SelfLearningTradingBot(initial_balance=10000)
        
        # Reduce training time for quick test
        bot.learning_episodes = 5
        bot.genetic_generations = 3
        
        print("✅ Bot initialized successfully")
        
        # Test indicator calculation
        print("📈 Testing technical indicator calculation...")
        processed_data = bot.calculate_indicators(sample_data)
        print(f"✅ Calculated indicators for {len(processed_data)} candles")
        
        # Test ensemble prediction (without training)
        print("🔮 Testing ensemble prediction...")
        action, confidence = bot.ensemble_prediction(processed_data)
        print(f"✅ Prediction: Action={action}, Confidence={confidence:.3f}")
        
        # Quick mini-training session
        print("🎯 Running mini training session...")
        results = await bot.train_all_models(sample_data)
        
        print("✅ Training completed!")
        print(f"DQN Performance: {results.get('dqn_performance', 'N/A')}")
        print(f"Pattern Recognition: {results.get('pattern_recognition_success', False)}")
        
        # Test prediction after training
        print("🔮 Testing prediction after training...")
        action, confidence = bot.ensemble_prediction(processed_data)
        print(f"✅ Post-training prediction: Action={action}, Confidence={confidence:.3f}")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Self-learning bot is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration():
    """Test the integration layer"""
    print("\n🔗 Testing Integration Layer")
    print("=" * 30)
    
    try:
        from self_learning_integration import SelfLearningIntegration
        
        integration = SelfLearningIntegration()
        print("✅ Integration layer initialized")
        
        # Test status
        status = integration.get_training_status()
        print(f"✅ Training status: {status['current_stage']}")
        
        # Test model info
        model_info = integration.get_model_info()
        print(f"✅ Model info available: {len(model_info)} entries")
        
        print("✅ Integration layer working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Self-Learning Trading Bot Test Suite")
    print("=" * 50)
    
    # Test 1: Core bot functionality
    success1 = asyncio.run(test_self_learning_bot())
    
    # Test 2: Integration layer
    success2 = asyncio.run(test_integration())
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour self-learning trading bot is ready to use!")
        print("\nNext steps:")
        print("1. Start the dashboard: python src/dashboard/app.py")
        print("2. Access: http://localhost:8081")
        print("3. Train the bot with real market data")
        print("4. Run AI-powered backtests")
        print("5. Get live AI predictions")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 