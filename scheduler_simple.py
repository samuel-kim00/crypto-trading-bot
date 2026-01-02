import subprocess
import time
import json
import os
import sys
import logging
from datetime import datetime

# --- Configuration ---
LOGS_DIR = 'logs'
CONFIG_DIR = 'config'
HEARTBEAT_FILE = os.path.join(LOGS_DIR, 'trading_bot_heartbeat.json')
HEARTBEAT_TIMEOUT = 120  # 2 minutes

# --- Setup Logging ---
os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, 'scheduler.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler(sys.stdout)])

# --- Process Management ---
processes = {}

def get_python_executable():
    """Get the python executable from the virtual environment."""
    return os.path.join(os.getcwd(), 'venv', 'bin', 'python')

def start_process(name, command, log_filename):
    """Starts a process and logs its output."""
    if name in processes and processes[name].poll() is None:
        logging.info(f"{name} is already running.")
        return

    logging.info(f"Starting {name}...")
    try:
        log_path = os.path.join(LOGS_DIR, log_filename)
        with open(log_path, 'w') as log_file:
            # Set PYTHONPATH to include the src directory
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.join(os.getcwd(), 'src') + os.pathsep + env.get('PYTHONPATH', '')
            
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True
            )
        processes[name] = process
        logging.info(f"{name} started successfully with PID {process.pid}. Log: {log_path}")
    except Exception as e:
        logging.error(f"Failed to start {name}: {e}")

def check_trading_bot_heartbeat():
    """Checks the trading bot's heartbeat file for signs of life."""
    if not os.path.exists(HEARTBEAT_FILE):
        logging.warning("Trading bot heartbeat file not found.")
        return False
    
    try:
        with open(HEARTBEAT_FILE, 'r') as f:
            heartbeat_data = json.load(f)
        
        last_heartbeat_time = heartbeat_data.get('timestamp', 0)
        time_since_heartbeat = time.time() - last_heartbeat_time
        
        if time_since_heartbeat > HEARTBEAT_TIMEOUT:
            logging.error(f"Trading bot heartbeat timeout! Last seen {time_since_heartbeat:.0f} seconds ago.")
            return False
            
        logging.info(f"Trading bot heartbeat is fresh (last seen {time_since_heartbeat:.0f}s ago).")
        return True

    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.warning(f"Could not read heartbeat file: {e}")
        return False

def monitor_processes():
    """Monitors and restarts unhealthy processes."""
    python_exec = get_python_executable()
    
    # Define processes to manage
    bot_command = [python_exec, os.path.join('src', 'core', 'trading_bot.py')]
    dashboard_command = [python_exec, os.path.join('src', 'dashboard', 'app.py')]
    
    # Initial start
    start_process('trading_bot', bot_command, 'trading_bot.log')
    start_process('dashboard', dashboard_command, 'dashboard.log')
    
    while True:
        time.sleep(30)  # Check every 30 seconds
        
        # Check trading bot
        bot_healthy = check_trading_bot_heartbeat()
        if not bot_healthy:
            logging.warning("Trading bot is unhealthy. Attempting restart...")
            if 'trading_bot' in processes:
                processes['trading_bot'].terminate()
                time.sleep(5) # Give it time to shut down
            start_process('trading_bot', bot_command, 'trading_bot.log')

        # Check dashboard process
        if 'dashboard' in processes and processes['dashboard'].poll() is not None:
            logging.warning("Dashboard process has exited. Attempting restart...")
            start_process('dashboard', dashboard_command, 'dashboard.log')
            
        logging.info("Monitoring check complete. All systems nominal.")


if __name__ == "__main__":
    try:
        monitor_processes()
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user. Shutting down all processes...")
        for name, process in processes.items():
            if process.poll() is None:
                logging.info(f"Terminating {name}...")
                process.terminate()
        logging.info("Shutdown complete.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the scheduler: {e}") 