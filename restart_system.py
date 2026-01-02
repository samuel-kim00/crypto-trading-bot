#!/usr/bin/env python3
"""
Clean restart script for the trading bot system
Handles port conflicts and process cleanup
"""

import os
import sys
import subprocess
import signal
import time
import psutil
import requests
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def kill_processes_by_pattern(patterns):
    """Kill processes matching patterns"""
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            for pattern in patterns:
                if pattern in cmdline.lower():
                    logging.info(f"Killing process {proc.info['pid']}: {cmdline[:100]}")
                    proc.kill()
                    killed.append(proc.info['pid'])
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed

def check_port_availability(ports):
    """Check which ports are available"""
    available = []
    for port in ports:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result != 0:
                available.append(port)
        except:
            pass
    return available

def cleanup_files():
    """Clean up temporary files"""
    files_to_remove = [
        'logs/trading_bot_heartbeat.json',
        'MANUAL_INTERVENTION_REQUIRED.txt'
    ]
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed {file_path}")
        except Exception as e:
            logging.warning(f"Could not remove {file_path}: {e}")

def main():
    logging.info("🔄 Starting system cleanup and restart...")
    
    # 1. Kill existing processes
    patterns = [
        'trading_bot.py',
        'dashboard/app.py',
        'scheduler.py',
        'flask',
        'socketio'
    ]
    
    killed = kill_processes_by_pattern(patterns)
    if killed:
        logging.info(f"Killed {len(killed)} processes")
        time.sleep(3)  # Wait for processes to die
    
    # 2. Check port availability
    ports = [8080, 8081, 8082, 8083, 8084, 8085]
    available_ports = check_port_availability(ports)
    logging.info(f"Available ports: {available_ports}")
    
    # 3. Clean up files
    cleanup_files()
    
    # 4. Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # 5. Check system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    logging.info(f"System status - CPU: {cpu_percent}%, Memory: {memory_percent}%")
    
    if cpu_percent > 80:
        logging.warning("High CPU usage detected, waiting for system to settle...")
        time.sleep(10)
    
    # 6. Start the system
    logging.info("🚀 Starting trading bot system...")
    
    try:
        # Try to start just the dashboard first to test
        logging.info("Testing dashboard startup...")
        dashboard_proc = subprocess.Popen(
            ['python3', 'src/dashboard/app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(5)
        
        # Check if dashboard started
        if dashboard_proc.poll() is None:
            # Try to connect
            dashboard_running = False
            for port in available_ports:
                try:
                    resp = requests.get(f'http://localhost:{port}/', timeout=3)
                    if resp.status_code == 200:
                        logging.info(f"✅ Dashboard running on port {port}")
                        dashboard_running = True
                        break
                except:
                    continue
            
            if dashboard_running:
                logging.info("Dashboard test successful, stopping and starting full system...")
                dashboard_proc.terminate()
                time.sleep(2)
                
                # Start the full scheduler
                logging.info("Starting full scheduler system...")
                scheduler_proc = subprocess.Popen(
                    ['python3', 'scheduler.py'],
                    stdout=None,
                    stderr=None
                )
                
                logging.info(f"✅ System started successfully! Scheduler PID: {scheduler_proc.pid}")
                logging.info("Monitor the system with: tail -f logs/scheduler.log")
                
            else:
                logging.error("Dashboard test failed - system may have issues")
                dashboard_proc.terminate()
                
        else:
            stdout, stderr = dashboard_proc.communicate()
            logging.error(f"Dashboard failed to start: {stderr.decode()}")
            
    except Exception as e:
        logging.error(f"Failed to start system: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 