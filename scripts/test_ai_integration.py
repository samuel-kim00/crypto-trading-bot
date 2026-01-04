#!/usr/bin/env python3
"""
Test AI Integration
==================
Test script to verify AI integration is working in the trading bot.
"""

import asyncio
import sys
import os

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.analysis.self_learning_integration import SelfLearningIntegration

async def test_ai_integration():
    """Test AI integration functionality"""
    print("🤖 Testing AI Integration...")
    
    # Initialize AI integration
    ai_integration = SelfLearningIntegration()
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing AI prediction for {symbol}...")
        
        try:
            # Get AI prediction
            prediction = await ai_integration.get_live_ai_prediction(symbol)
            
            if 'error' in prediction:
                print(f"❌ Error: {prediction['error']}")
            else:
                print(f"✅ AI Recommendation: {prediction['recommendation']}")
                print(f"   Confidence: {prediction['confidence']:.2f}")
                print(f"   Current Price: ${prediction['current_price']:.4f}")
                print(f"   AI Analysis: {prediction['ai_analysis']}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Test model info
    print(f"\n📋 Model Information:")
    model_info = ai_integration.get_model_info()
    
    print(f"   DQN Model: {'✅ Trained' if model_info['dqn_model']['trained'] else '❌ Not Trained'}")
    print(f"   Scenario Models: {model_info['scenario_models']['count']}/5 trained")
    print(f"   Total Models: {model_info['training_summary']['total_models']}")
    print(f"   Training Complete: {'✅ Yes' if model_info['training_summary']['training_complete'] else '❌ No'}")
    
    print(f"\n🎯 AI Integration Status: {'✅ ACTIVE' if model_info['training_summary']['training_complete'] else '❌ INACTIVE'}")

if __name__ == "__main__":
    asyncio.run(test_ai_integration())
