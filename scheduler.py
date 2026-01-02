import schedule
import time
import subprocess
import signal
import os
import json
from datetime import datetime
import logging
import socket
import requests
import psutil
import sys

# Add src directory to Python path
sys.path.append('src')

from dashboard.app import app
from core.trading_bot import TradingBot
from analysis.youtube_scraper import YouTubeScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)

class TradingBotScheduler:
    def __init__(self):
        self.trading_process = None
        self.dashboard_process = None
        self.is_running = False
        self.last_restart = datetime.now()
        self.max_restarts = 10
        self.restart_count = 0
        self.restart_window = 7200
        self.heartbeat_file = "logs/trading_bot_heartbeat.json"
        self.heartbeat_timeout = 180
        self.last_youtube_scrape = None
        self.youtube_scrape_interval = 7 * 24 * 60 * 60
        self.load_schedule()

    def load_schedule(self):
        try:
            with open('config/schedule_config.json', 'r') as f:
                self.schedule_config = json.load(f)
        except FileNotFoundError:
            # Default schedule: 24/7 operation
            self.schedule_config = {
                'trading_hours': {
                    'start': '00:00',
                    'end': '23:59'
                },
                'trading_days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            }
            self.save_schedule()

    def save_schedule(self):
        with open('config/schedule_config.json', 'w') as f:
            json.dump(self.schedule_config, f, indent=4)

    def check_internet(self):
        try:
            # Try to connect to a reliable server
            requests.get('https://api.binance.com/api/v3/ping', timeout=5)
            return True
        except:
            return False

    def check_process_health(self):
        """Check if the trading bot process is healthy"""
        try:
            # Check heartbeat file
            if not os.path.exists(self.heartbeat_file):
                logging.warning("Heartbeat file not found")
                return False

            # Read heartbeat data with proper error handling
            try:
                with open(self.heartbeat_file, 'r') as f:
                    heartbeat_data = json.load(f)
            except (json.JSONDecodeError, PermissionError) as e:
                logging.warning(f"Error reading heartbeat file: {str(e)}")
                # If file exists but can't be read, try to remove it
                try:
                    os.remove(self.heartbeat_file)
                except:
                    pass
                return False

            # Check timestamp with more lenient timeout
            current_time = time.time()
            last_heartbeat = heartbeat_data.get('timestamp', 0)
            time_since_heartbeat = current_time - last_heartbeat
            
            # Progressive warning levels with more lenient thresholds
            if time_since_heartbeat > (self.heartbeat_timeout * 1.2):  # 120% of timeout
                logging.warning(f"Heartbeat delayed: {time_since_heartbeat:.2f} seconds since last heartbeat")
            if time_since_heartbeat > (self.heartbeat_timeout * 2.0):  # 200% of timeout
                logging.error(f"Heartbeat timeout: {time_since_heartbeat:.2f} seconds since last heartbeat")
                return False

            # Check process status
            status = heartbeat_data.get('status', '')
            if status == 'shutdown':
                logging.info("Trading bot reported clean shutdown")
                return False
            if status != 'running':
                logging.warning(f"Process status is {status}")
                return False

            # Check memory usage with higher threshold
            memory_usage = heartbeat_data.get('memory_usage', 0)
            if memory_usage > 2000:  # More than 2GB
                logging.warning(f"High memory usage detected: {memory_usage:.2f} MB")

            # Check if PID exists and process is responsive
            pid = heartbeat_data.get('pid')
            if pid is None:
                logging.warning("No PID in heartbeat file")
                return False
            
            try:
                process = psutil.Process(pid)
                if process.status() == psutil.STATUS_ZOMBIE:
                    logging.warning("Process is zombie")
                    return False
            
                # Check CPU usage with higher threshold
                cpu_percent = process.cpu_percent(interval=0.1)
                if cpu_percent > 95:  # High CPU usage
                    logging.warning(f"High CPU usage detected: {cpu_percent}%")
            
                # Check if process is actually our trading bot
                if not any('trading_bot.py' in cmd for cmd in process.cmdline()):
                    logging.warning("PID exists but is not trading bot")
                    return False
                
                # Check for error count in heartbeat with higher threshold
                error_count = heartbeat_data.get('error_count', 0)
                if error_count > 5:  # Increased from 3 to 5
                    logging.warning(f"High error count in heartbeat: {error_count}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logging.warning("Trading bot process not found or access denied")
                return False

            return True

        except Exception as e:
            logging.error(f"Error checking process health: {str(e)}")
            return False

    def start_trading_bot(self):
        """Start the trading bot process"""
        try:
            if self.trading_process is None or self.trading_process.poll() is not None:
                logging.info("Starting trading bot...")
                
                # Clean up any existing heartbeat file
                try:
                    if os.path.exists(self.heartbeat_file):
                        os.remove(self.heartbeat_file)
                except Exception as e:
                    logging.warning(f"Failed to remove old heartbeat file: {str(e)}")
                
                # Get the absolute path to the trading bot
                trading_bot_path = os.path.join('src', 'core', 'trading_bot.py')
                
                # Verify the file exists
                if not os.path.exists(trading_bot_path):
                    logging.error(f"Trading bot not found at {trading_bot_path}")
                    return False
                
                # Start the process with line buffering and correct Python path
                self.trading_process = subprocess.Popen(
                    ['python3', '-u', trading_bot_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True,
                    env=dict(os.environ, PYTHONPATH='src')
                )
                
                # Wait for process to start and create heartbeat file
                start_time = time.time()
                while time.time() - start_time < 10:  # Wait up to 10 seconds
                    if os.path.exists(self.heartbeat_file):
                        if self.check_process_health():
                            logging.info("Trading bot started successfully")
                            return True
                    time.sleep(0.5)
                
                # If we get here, startup failed
                logging.error("Trading bot failed to start properly")
                self.stop_trading_bot()  # Clean up
                return False
                
        except Exception as e:
            logging.error(f"Error starting trading bot: {str(e)}")
            return False

    def stop_trading_bot(self):
        """Stop the trading bot process"""
        try:
            if self.trading_process and self.trading_process.poll() is None:
                logging.info("Stopping trading bot...")
                
                # Try graceful shutdown first
                self.trading_process.terminate()
                try:
                    self.trading_process.wait(timeout=10)  # Wait up to 10 seconds
                except subprocess.TimeoutExpired:
                    logging.warning("Trading bot did not shut down gracefully, forcing...")
                    self.trading_process.kill()
                    self.trading_process.wait(timeout=5)
                
                # Clean up heartbeat file
                try:
                    if os.path.exists(self.heartbeat_file):
                        os.remove(self.heartbeat_file)
                except Exception as e:
                    logging.warning(f"Failed to remove heartbeat file: {str(e)}")
                
                self.trading_process = None
                logging.info("Trading bot stopped successfully")
                return True
                
        except Exception as e:
            logging.error(f"Error stopping trading bot: {str(e)}")
            return False

    def start_dashboard(self):
        """Start the dashboard process"""
        try:
            if self.dashboard_process is None or self.dashboard_process.poll() is not None:
                logging.info("Starting dashboard...")
                
                # Get the absolute path to the dashboard app
                dashboard_path = os.path.join('src', 'dashboard', 'app.py')
                
                # Verify the file exists
                if not os.path.exists(dashboard_path):
                    logging.error(f"Dashboard app not found at {dashboard_path}")
                    return False
                
                # Start the process with the correct path
                self.dashboard_process = subprocess.Popen(
                    ['python3', '-u', dashboard_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True,
                    env=dict(os.environ, PYTHONPATH='src')
                )
                
                # Wait briefly to check if process started
                time.sleep(3)  # Increased wait time to allow for startup
                if self.dashboard_process.poll() is None:
                    # Try multiple ports to verify it's running
                    ports = [8081, 8082, 8083, 8084, 8085]
                    dashboard_started = False
                    
                    for port in ports:
                        try:
                            requests.get(f'http://localhost:{port}/', timeout=5)
                            logging.info(f"Dashboard started successfully on port {port}")
                            dashboard_started = True
                            break
                        except requests.exceptions.RequestException:
                            continue
                    
                    if dashboard_started:
                        return True
                    else:
                        # Get process output for debugging
                        stdout, stderr = self.dashboard_process.communicate(timeout=5)
                        if "No available ports" in stderr:
                            logging.error("Dashboard failed to start: All ports are in use")
                        else:
                            logging.error(f"Dashboard started but not responding on any port: {stderr}")
                        return False
                else:
                    stdout, stderr = self.dashboard_process.communicate()
                    logging.error(f"Dashboard failed to start: {stderr}")
                    return False
                    
        except Exception as e:
            logging.error(f"Error starting dashboard: {str(e)}")
            return False

    def monitor_processes(self):
        """Monitor and manage running processes"""
        try:
            # Check if trading bot process exists
            if self.trading_process is None or self.trading_process.poll() is not None:
                logging.warning("Trading bot process is not running")
                self.handle_trading_bot_restart()
                return

            # Check trading bot health
            if not self.check_process_health():
                logging.warning("Trading bot process is not healthy")
                self.handle_trading_bot_restart()
                return
                
            # Check dashboard health
            if self.dashboard_process and self.dashboard_process.poll() is not None:
                logging.warning("Dashboard process is not running")
                self.start_dashboard()

        except Exception as e:
            logging.error(f"Error in process monitoring: {str(e)}")

    def handle_trading_bot_restart(self):
        """Handle trading bot restart logic"""
        try:
            # Check if we need to reset restart count
            current_time = datetime.now()
            if (current_time - self.last_restart).total_seconds() > self.restart_window:
                logging.info("Resetting restart count due to window expiry")
                self.restart_count = 0
            
            # Get heartbeat data if available
            try:
                with open(self.heartbeat_file, 'r') as f:
                    heartbeat_data = json.load(f)
                error_count = heartbeat_data.get('error_count', 0)
                memory_usage = heartbeat_data.get('memory_usage', 0)
            except:
                error_count = 0
                memory_usage = 0
            
            # Stop the trading bot
            self.stop_trading_bot()
            
            # Check if we should attempt restart
            if self.restart_count >= self.max_restarts:
                logging.error("Too many restart attempts. Manual intervention required.")
                self.notify_admin()
                return
            
            # Adjust restart strategy based on failure type
            if error_count > 3:
                logging.warning("High error count detected, waiting longer before restart")
                time.sleep(30)  # Wait longer if there are persistent errors
            elif memory_usage > 1000:
                logging.warning("High memory usage detected, performing garbage collection before restart")
                import gc
                gc.collect()
            
            # Clean up any existing heartbeat file
            try:
                if os.path.exists(self.heartbeat_file):
                    os.remove(self.heartbeat_file)
            except Exception as e:
                logging.warning(f"Failed to remove old heartbeat file: {str(e)}")
            
            # Attempt restart
            if self.start_trading_bot():
                self.restart_count += 1
                self.last_restart = current_time
                logging.info(f"Restart attempt {self.restart_count}/{self.max_restarts} successful")
            else:
                logging.error("Failed to restart trading bot")
                
        except Exception as e:
            logging.error(f"Error handling trading bot restart: {str(e)}")
            # Try one last time to notify admin
            try:
                self.notify_admin()
            except:
                pass

    def notify_admin(self):
        """Notify admin of critical issues"""
        try:
            # Log critical error
            logging.critical("Trading bot requires manual intervention")
            
            # Create a visible file with error details
            with open('MANUAL_INTERVENTION_REQUIRED.txt', 'w') as f:
                f.write(f"Trading bot stopped at {datetime.now().isoformat()}\n")
                f.write("Please check logs for details\n")
                
                # Add recent log entries
                f.write("\nRecent log entries:\n")
                try:
                    with open('logs/scheduler.log', 'r') as log:
                        last_lines = log.readlines()[-20:]  # Get last 20 lines
                        f.writelines(last_lines)
                except Exception as e:
                    f.write(f"Could not read log file: {str(e)}\n")
                
                # Add system info
                f.write("\nSystem Information:\n")
                try:
                    import psutil
                    f.write(f"CPU Usage: {psutil.cpu_percent()}%\n")
                    f.write(f"Memory Usage: {psutil.virtual_memory().percent}%\n")
                    f.write(f"Disk Usage: {psutil.disk_usage('/').percent}%\n")
                except Exception as e:
                    f.write(f"Could not get system info: {str(e)}\n")
            
        except Exception as e:
            logging.error(f"Failed to notify admin: {str(e)}")
            
        # Set file permissions
        try:
            os.chmod('MANUAL_INTERVENTION_REQUIRED.txt', 0o644)
        except:
            pass

    def run_youtube_scraper(self):
        """Run the YouTube scraper"""
        try:
            logging.info("Starting weekly YouTube scraping...")
            
            # Run the scraper
            scraper_process = subprocess.Popen(
                ['python3', '-u', 'src/analysis/youtube_scraper.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait for process to finish
            stdout, stderr = scraper_process.communicate()
            
            if scraper_process.returncode == 0:
                logging.info("YouTube scraping completed successfully")
                return True
            else:
                logging.error(f"YouTube scraper failed: {stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error running YouTube scraper: {str(e)}")
            return False

    def run(self):
        """Main execution loop"""
        try:
            # Start processes
            if not self.start_dashboard():
                logging.error("Failed to start dashboard")
                return
                
            logging.info("Scheduler started")
            logging.info("Running in 24/7 mode with internet connectivity check")
            
            # Start trading bot
            if not self.start_trading_bot():
                logging.error("Failed to start trading bot")
                return
                
            # Schedule process monitoring
            schedule.every(1).minutes.do(self.monitor_processes)
            
            # Schedule weekly YouTube scraping (every Monday at 00:00)
            schedule.every().monday.at("00:00").do(self.run_youtube_scraper)
            
            # Run YouTube scraper immediately if it hasn't run before
            if self.last_youtube_scrape is None:
                self.run_youtube_scraper()
            
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
            self.cleanup()
        except Exception as e:
            logging.error(f"Error in scheduler: {str(e)}")
            self.cleanup()

    def cleanup(self):
        """Clean up processes on shutdown"""
        self.stop_trading_bot()
        if self.dashboard_process:
            self.dashboard_process.terminate()
        logging.info("Scheduler shutdown complete")

if __name__ == "__main__":
    scheduler = TradingBotScheduler()
    scheduler.run() 