#!/usr/bin/env python3
"""
AI Learning Progress Monitor
Watch your AI bot learn in real-time!
"""

import requests
import time
import json
from datetime import datetime

def clear_screen():
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def get_ai_status():
    """Get AI training status"""
    try:
        response = requests.get('http://localhost:8081/api/ai_training_log', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {'success': False, 'error': 'Cannot connect to dashboard'}

def get_model_status():
    """Get AI model information"""
    try:
        response = requests.get('http://localhost:8081/api/self_learning_status', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {'success': False}

def format_time(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%H:%M:%S")
    except:
        return timestamp_str

def display_progress(status, model_info):
    """Display training progress"""
    clear_screen()
    
    print("🤖 AI LEARNING PROGRESS MONITOR")
    print("=" * 50)
    print(f"🕒 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    if not status.get('success'):
        print("❌ Cannot connect to AI training system")
        return
    
    # Training Status
    is_training = status.get('is_training', False)
    progress = status.get('current_progress', 0)
    training_count = status.get('training_count', 0)
    
    if is_training:
        print(f"🟢 TRAINING IN PROGRESS - {progress}%")
        print("▓" * (progress // 5) + "░" * (20 - progress // 5))
    else:
        print("🔴 TRAINING IDLE")
    
    print()
    print(f"📊 Total Training Sessions: {training_count}")
    
    # Model Status
    if model_info.get('success'):
        model_data = model_info.get('model_info', {})
        indicators = model_info.get('learning_indicators', {})
        
        print("\n🧠 AI MODEL STATUS:")
        print(f"  🎯 DQN Episodes: {model_data.get('dqn_episodes', 0)}")
        print(f"  🧬 Genetic Generations: {model_data.get('genetic_generations', 0)}")  
        print(f"  🔍 Pattern Accuracy: {model_data.get('pattern_accuracy', 0):.1f}%")
        print(f"  🤝 Ensemble Ready: {'✅' if indicators.get('ensemble_ready') else '❌'}")
    
    # Recent Training Log
    training_log = status.get('training_log', [])
    if training_log:
        print("\n📝 RECENT TRAINING ACTIVITY:")
        for entry in training_log[-5:]:  # Show last 5 entries
            time_str = format_time(entry.get('timestamp', ''))
            message = entry.get('message', '')
            progress_val = entry.get('progress')
            progress_str = f" ({progress_val}%)" if progress_val is not None else ""
            print(f"  {time_str} - {message}{progress_str}")
    
    print("\n" + "=" * 50)
    print("Press Ctrl+C to stop monitoring...")

def main():
    """Main monitoring loop"""
    print("🚀 Starting AI Learning Monitor...")
    print("Connecting to dashboard at http://localhost:8081")
    
    try:
        while True:
            ai_status = get_ai_status()
            model_status = get_model_status()
            display_progress(ai_status, model_status)
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped. AI continues learning in background!")

if __name__ == '__main__':
    main() 