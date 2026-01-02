import subprocess
import os
import time
from datetime import datetime

def run_bot(env_file, name):
    """Run a trading bot with the specified environment file"""
    # Copy the environment file to .env
    os.system(f'cp {env_file} .env')
    
    # Run the bot
    process = subprocess.Popen(
        ['python', 'trading_bot.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    print(f"Started {name} bot with {env_file}")
    return process

def monitor_performance():
    """Monitor the performance of both bots"""
    while True:
        try:
            # Read performance data for both bots
            with open('performance_data_large.json', 'r') as f:
                large_data = eval(f.read())
            with open('performance_data_small.json', 'r') as f:
                small_data = eval(f.read())
            
            # Calculate current balances
            large_balance = large_data.get('current_balance', 10000)
            small_balance = small_data.get('current_balance', 10)
            
            # Calculate returns
            large_return = ((large_balance - 10000) / 10000) * 100
            small_return = ((small_balance - 10) / 10) * 100
            
            # Print status
            print(f"\n=== Performance Update ({datetime.now()}) ===")
            print(f"Large Bot (10,000 USDT):")
            print(f"  Current Balance: {large_balance:.2f} USDT")
            print(f"  Return: {large_return:.2f}%")
            print(f"Small Bot (10 USDT):")
            print(f"  Current Balance: {small_balance:.2f} USDT")
            print(f"  Return: {small_return:.2f}%")
            print(f"  Progress to 10,000 USDT: {(small_balance/10000)*100:.2f}%")
            
            time.sleep(60)  # Update every minute
            
        except Exception as e:
            print(f"Error monitoring performance: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    # Start both bots
    large_bot = run_bot('.env.large', 'Large')
    small_bot = run_bot('.env.small', 'Small')
    
    # Start monitoring
    monitor_performance() 