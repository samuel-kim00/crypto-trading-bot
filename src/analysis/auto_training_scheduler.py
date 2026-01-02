#!/usr/bin/env python3
"""
Automatic Self-Learning Bot Training Scheduler
"""

import os
import sys
import json
import time
import asyncio
import logging
import schedule
from datetime import datetime
from threading import Thread

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src', 'analysis'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutoTrainingScheduler:
    def __init__(self):
        self.training_status = {
            'is_training': False,
            'last_training': None,
            'training_count': 0,
            'current_progress': 0,
            'training_log': []
        }
        self.status_file = os.path.join(project_root, 'data', 'ai_training_status.json')
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
        os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
        
    def save_status(self):
        """Save training status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.training_status, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save status: {e}")
    
    def log_progress(self, message, progress=None):
        """Log training progress"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'message': message,
            'progress': progress
        }
        
        self.training_status['training_log'].append(log_entry)
        if progress is not None:
            self.training_status['current_progress'] = progress
            
        # Keep only last 50 log entries
        if len(self.training_status['training_log']) > 50:
            self.training_status['training_log'] = self.training_status['training_log'][-50:]
            
        self.save_status()
        logging.info(f"🤖 AI Training: {message} {f'({progress}%)' if progress else ''}")
    
    async def train_ai_models(self):
        """Train the AI models with progress tracking"""
        if self.training_status['is_training']:
            return
            
        try:
            self.training_status['is_training'] = True
            self.save_status()
            
            self.log_progress("🚀 Starting automatic AI training", 0)
            
            from self_learning_integration import train_self_learning_bot
            
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            self.log_progress(f"📊 Training on symbols: {', '.join(symbols)}", 10)
            
            results = await train_self_learning_bot(symbols)
            
            if results['success']:
                self.log_progress("✅ AI training completed successfully!", 100)
                self.training_status['last_training'] = datetime.now().isoformat()
                self.training_status['training_count'] += 1
            else:
                self.log_progress(f"❌ Training failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.log_progress(f"💥 Training error: {str(e)}")
            
        finally:
            self.training_status['is_training'] = False
            self.save_status()
    
    def get_training_status(self):
        """Get current training status"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return self.training_status
    
    def start_training_now(self):
        """Start training immediately"""
        asyncio.run(self.train_ai_models())

if __name__ == '__main__':
    scheduler = AutoTrainingScheduler()
    scheduler.start_training_now() 