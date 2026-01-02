#!/usr/bin/env python3
"""
Comprehensive AI Training Monitor
Monitor all AI training activities including external intelligence
"""

import requests
import time
import json
import os
from datetime import datetime

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_comprehensive_status():
    """Get comprehensive AI training status"""
    try:
        # Check all AI systems
        endpoints = {
            'self_learning': 'http://localhost:8081/api/self_learning_status',
            'market_intelligence': 'http://localhost:8081/api/market_intelligence_status',
            'advanced_ai': 'http://localhost:8081/api/advanced_ai_status',
            'training_log': 'http://localhost:8081/api/ai_training_log'
        }
        
        status = {}
        for system, url in endpoints.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    status[system] = response.json()
                else:
                    status[system] = {'success': False, 'error': f'HTTP {response.status_code}'}
            except:
                status[system] = {'success': False, 'error': 'Connection failed'}
        
        return status
    except Exception as e:
        return {'error': str(e)}

def format_status_display(status):
    """Format status for display"""
    display = []
    
    # Header
    display.append("=" * 80)
    display.append("🤖 COMPREHENSIVE AI TRAINING MONITOR")
    display.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    display.append("=" * 80)
    display.append("")
    
    # Self-Learning Bot Status
    display.append("🧠 SELF-LEARNING BOT")
    display.append("-" * 40)
    
    self_learning = status.get('self_learning', {})
    if self_learning.get('success'):
        training_status = self_learning.get('training_status', {})
        model_info = self_learning.get('model_info', {})
        learning_indicators = self_learning.get('learning_indicators', {})
        
        display.append(f"✅ DQN Episodes: {model_info.get('dqn_episodes', 0)}")
        display.append(f"✅ Genetic Generations: {model_info.get('genetic_generations', 0)}")
        display.append(f"✅ Pattern Accuracy: {model_info.get('pattern_accuracy', 0):.1f}%")
        display.append(f"🎯 Ensemble Ready: {'✅' if learning_indicators.get('ensemble_ready') else '❌'}")
    else:
        display.append(f"❌ Error: {self_learning.get('error', 'Unknown error')}")
    
    display.append("")
    
    # Market Intelligence Status
    display.append("🌐 MARKET INTELLIGENCE")
    display.append("-" * 40)
    
    intelligence = status.get('market_intelligence', {})
    if intelligence.get('success') and intelligence.get('intelligence_available'):
        combined_sentiment = intelligence.get('combined_sentiment', {})
        market_consensus = intelligence.get('market_consensus', {})
        
        sentiment_score = combined_sentiment.get('overall_score', 0)
        consensus = market_consensus.get('consensus', 'unknown')
        confidence = market_consensus.get('confidence', 'unknown')
        
        display.append(f"📊 Overall Sentiment: {sentiment_score:.1f}")
        display.append(f"🎯 Market Consensus: {consensus.upper()} ({confidence} confidence)")
        display.append(f"📹 YouTube Mood: {market_consensus.get('youtube_mood', 'unknown')}")
        display.append(f"📰 Web Mood: {market_consensus.get('web_mood', 'unknown')}")
        display.append(f"⏰ Last Update: {intelligence.get('last_update', 'Unknown')}")
    else:
        display.append("⏳ Intelligence gathering in progress or not started")
    
    display.append("")
    
    # Advanced AI Status
    display.append("🔬 ADVANCED AI MODELS")
    display.append("-" * 40)
    
    advanced_ai = status.get('advanced_ai', {})
    if advanced_ai.get('success') and advanced_ai.get('training_completed'):
        training_summary = advanced_ai.get('training_summary', {})
        trained_scenarios = advanced_ai.get('trained_scenarios', [])
        
        display.append(f"✅ Scenarios Trained: {training_summary.get('scenarios_trained', 0)}")
        display.append(f"🏛️ Market Regime: {training_summary.get('market_regime', 'unknown')}")
        display.append(f"📈 Sentiment Score: {training_summary.get('sentiment_score', 0):.1f}")
        display.append(f"🎯 Trained Scenarios: {', '.join(trained_scenarios)}")
    else:
        display.append("⏳ Advanced AI training in progress or not started")
    
    display.append("")
    
    # Training Activity Log
    display.append("📝 RECENT TRAINING ACTIVITY")
    display.append("-" * 40)
    
    training_log = status.get('training_log', {})
    if training_log.get('success'):
        is_training = training_log.get('is_training', False)
        progress = training_log.get('current_progress', 0)
        training_count = training_log.get('training_count', 0)
        
        display.append(f"🔄 Currently Training: {'YES' if is_training else 'NO'}")
        display.append(f"📊 Progress: {progress}%")
        display.append(f"🔢 Training Sessions: {training_count}")
        
        recent_logs = training_log.get('training_log', [])
        if recent_logs:
            display.append("\n📋 Recent Activity:")
            for log in recent_logs[-5:]:  # Last 5 activities
                timestamp = log.get('timestamp', 'Unknown')
                message = log.get('message', 'No message')
                display.append(f"  • [{timestamp}] {message}")
    else:
        display.append("❌ Unable to fetch training logs")
    
    display.append("")
    display.append("=" * 80)
    display.append("💡 Commands:")
    display.append("  • Press Ctrl+C to exit")
    display.append("  • Refreshes every 5 seconds")
    display.append("=" * 80)
    
    return "\n".join(display)

def start_comprehensive_training():
    """Start comprehensive AI training"""
    try:
        response = requests.post('http://localhost:8081/api/start_comprehensive_ai_training')
        if response.status_code == 200:
            result = response.json()
            print("🚀 Comprehensive AI training started!")
            print(f"💡 {result.get('message', 'Training initiated')}")
            print(f"⏱️ Expected duration: {result.get('expected_duration', 'Unknown')}")
        else:
            print(f"❌ Failed to start training: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error starting training: {e}")

def main():
    """Main monitoring loop"""
    print("🤖 Comprehensive AI Training Monitor")
    print("=" * 50)
    
    # Ask if user wants to start training
    start_training = input("Start comprehensive AI training? (y/n): ").lower().strip()
    if start_training == 'y':
        start_comprehensive_training()
        print("\n⏳ Starting monitoring in 5 seconds...")
        time.sleep(5)
    
    try:
        while True:
            clear_screen()
            
            # Get status from all systems
            status = get_comprehensive_status()
            
            # Display formatted status
            display = format_status_display(status)
            print(display)
            
            # Wait before next update
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped. Have a great day!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")

if __name__ == '__main__':
    main() 