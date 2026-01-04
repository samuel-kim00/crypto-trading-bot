#!/usr/bin/env python3
"""
Simple AI Trading Bot Monitor
A lightweight monitoring solution to track your AI training progress
"""

import os
import json
import time
import psutil
from datetime import datetime
import subprocess

class SimpleMonitor:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
    def check_system_resources(self):
        """Check CPU, memory, and disk usage"""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    
    def check_running_processes(self):
        """Check for trading bot related processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(keyword in cmdline.lower() for keyword in ['scheduler.py', 'dashboard', 'trading_bot', 'app.py']):
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def check_ai_training_status(self):
        """Check AI training status from various sources"""
        status = {
            'self_learning_trained': False,
            'models_found': [],
            'training_files': [],
            'last_training': None
        }
        
        # Check for AI model files
        model_paths = [
            'models/',
            'src/analysis/models/',
            'data/models/'
        ]
        
        for path in model_paths:
            full_path = os.path.join(self.project_root, path)
            if os.path.exists(full_path):
                for file in os.listdir(full_path):
                    if file.endswith(('.h5', '.keras', '.pkl', '.json')):
                        status['models_found'].append(f"{path}{file}")
        
        # Check for training result files
        result_paths = [
            'data/',
            'src/analysis/reports/',
            'reports/'
        ]
        
        for path in result_paths:
            full_path = os.path.join(self.project_root, path)
            if os.path.exists(full_path):
                for file in os.listdir(full_path):
                    if 'training' in file.lower() or 'self_learning' in file.lower():
                        status['training_files'].append(f"{path}{file}")
                        # Get file modification time
                        file_path = os.path.join(full_path, file)
                        mod_time = os.path.getmtime(file_path)
                        if not status['last_training'] or mod_time > status['last_training']:
                            status['last_training'] = mod_time
        
        if status['models_found'] or status['training_files']:
            status['self_learning_trained'] = True
            
        return status
    
    def check_backtest_results(self):
        """Check for recent backtest results"""
        results = []
        
        result_paths = [
            'src/analysis/reports/',
            'reports/',
            'data/'
        ]
        
        for path in result_paths:
            full_path = os.path.join(self.project_root, path)
            if os.path.exists(full_path):
                for file in os.listdir(full_path):
                    if any(keyword in file.lower() for keyword in ['backtest', 'auto_discovery', 'optimized']):
                        file_path = os.path.join(full_path, file)
                        mod_time = os.path.getmtime(file_path)
                        results.append({
                            'file': f"{path}{file}",
                            'modified': datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S'),
                            'size_kb': os.path.getsize(file_path) / 1024
                        })
        
        # Sort by modification time (newest first)
        results.sort(key=lambda x: x['modified'], reverse=True)
        return results[:5]  # Return top 5 most recent
    
    def kill_conflicting_processes(self):
        """Kill processes that might be causing conflicts"""
        killed = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                # Look for dashboard processes on port 8080 or scheduler processes
                if ('app.py' in cmdline and '8080' in cmdline) or 'scheduler.py' in cmdline:
                    proc.terminate()
                    killed.append(f"PID {proc.info['pid']}: {proc.info['name']}")
                    time.sleep(1)
                    if proc.is_running():
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return killed
    
    def start_simple_dashboard(self):
        """Start dashboard on an available port"""
        available_ports = [8081, 8082, 8083, 8084, 8085]
        
        for port in available_ports:
            try:
                # Check if port is available
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result != 0:  # Port is available
                    print(f"🚀 Starting dashboard on http://localhost:{port}")
                    subprocess.Popen(['python3', 'src/dashboard/app.py'], 
                                   cwd=self.project_root, 
                                   env=dict(os.environ, PORT=str(port)))
                    return port
            except Exception as e:
                continue
        
        return None
    
    def display_status(self):
        """Display comprehensive status"""
        print("\n" + "="*80)
        print("🤖 AI TRADING BOT MONITOR")
        print("="*80)
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System resources
        print("\n💻 SYSTEM RESOURCES:")
        resources = self.check_system_resources()
        cpu_status = "🔴 HIGH" if resources['cpu_percent'] > 80 else "🟢 NORMAL"
        memory_status = "🔴 HIGH" if resources['memory_percent'] > 80 else "🟢 NORMAL"
        
        print(f"   CPU Usage: {resources['cpu_percent']:.1f}% {cpu_status}")
        print(f"   Memory: {resources['memory_percent']:.1f}% ({resources['memory_available_gb']:.1f}GB free) {memory_status}")
        print(f"   Disk: {resources['disk_percent']:.1f}% ({resources['disk_free_gb']:.1f}GB free)")
        
        # Running processes
        print("\n🔄 RUNNING PROCESSES:")
        processes = self.check_running_processes()
        if processes:
            for proc in processes:
                print(f"   PID {proc['pid']}: {proc['cmdline']}")
        else:
            print("   ✅ No trading bot processes currently running")
        
        # AI Training Status
        print("\n🧠 AI TRAINING STATUS:")
        ai_status = self.check_ai_training_status()
        if ai_status['self_learning_trained']:
            print("   ✅ AI models detected!")
            print(f"   📊 Models found: {len(ai_status['models_found'])}")
            for model in ai_status['models_found'][:3]:  # Show first 3
                print(f"      - {model}")
            if ai_status['last_training']:
                last_training_time = datetime.fromtimestamp(ai_status['last_training'])
                print(f"   🕐 Last training: {last_training_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("   ⏳ No AI training detected yet")
        
        # Backtest Results
        print("\n📈 RECENT BACKTEST RESULTS:")
        backtest_results = self.check_backtest_results()
        if backtest_results:
            for result in backtest_results[:3]:  # Show top 3
                print(f"   📊 {result['file']} ({result['modified']}, {result['size_kb']:.1f}KB)")
        else:
            print("   ⏳ No backtest results found")
        
        print("\n" + "="*80)
    
    def interactive_menu(self):
        """Interactive menu for bot management"""
        while True:
            self.display_status()
            print("\n🎛️  CONTROL MENU:")
            print("1. 🔄 Refresh status")
            print("2. 🛑 Kill conflicting processes")
            print("3. 🚀 Start simple dashboard")
            print("4. 📊 Start AI training")
            print("5. 🏃 Run quick backtest")
            print("6. ❌ Exit")
            
            try:
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == '1':
                    continue  # Refresh by continuing loop
                
                elif choice == '2':
                    print("\n🛑 Killing conflicting processes...")
                    killed = self.kill_conflicting_processes()
                    if killed:
                        for proc in killed:
                            print(f"   ✅ Killed: {proc}")
                    else:
                        print("   ✅ No conflicting processes found")
                    input("\nPress Enter to continue...")
                
                elif choice == '3':
                    print("\n🚀 Starting dashboard...")
                    port = self.start_simple_dashboard()
                    if port:
                        print(f"   ✅ Dashboard started on http://localhost:{port}")
                        print("   💡 You can now access the web interface")
                    else:
                        print("   ❌ Could not find available port")
                    input("\nPress Enter to continue...")
                
                elif choice == '4':
                    print("\n🧠 Starting AI training...")
                    print("   This will run in the background...")
                    try:
                        subprocess.Popen(['python3', '-c', '''
import sys
sys.path.append("src/analysis")
from self_learning_integration import train_self_learning_bot
import asyncio
asyncio.run(train_self_learning_bot(["BTCUSDT"]))
'''], cwd=self.project_root)
                        print("   ✅ AI training started in background")
                    except Exception as e:
                        print(f"   ❌ Error starting training: {e}")
                    input("\nPress Enter to continue...")
                
                elif choice == '5':
                    print("\n📊 Running quick backtest...")
                    print("   This may take a few minutes...")
                    try:
                        result = subprocess.run(['python3', '-c', '''
import sys
sys.path.append("src/analysis")
from backtester import VLMBacktester
import asyncio

async def quick_test():
    backtester = VLMBacktester(initial_balance=1000)
    results = await backtester.run_backtest(
        symbols=["BTCUSDT"], 
        start_date="2024-01-01", 
        end_date="2024-01-15"
    )
    print(f"Quick backtest completed: {results.get('total_return', 'N/A')}% return")

asyncio.run(quick_test())
'''], cwd=self.project_root, capture_output=True, text=True, timeout=120)
                        
                        if result.returncode == 0:
                            print("   ✅ Quick backtest completed")
                            print(f"   📊 Output: {result.stdout}")
                        else:
                            print(f"   ❌ Backtest failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        print("   ⏰ Backtest timed out (taking longer than expected)")
                    except Exception as e:
                        print(f"   ❌ Error running backtest: {e}")
                    input("\nPress Enter to continue...")
                
                elif choice == '6':
                    print("\n👋 Goodbye!")
                    break
                
                else:
                    print("\n❌ Invalid choice. Please select 1-6.")
                    input("Press Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

def main():
    monitor = SimpleMonitor()
    
    # Check if running with arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'status':
            monitor.display_status()
        elif sys.argv[1] == 'clean':
            killed = monitor.kill_conflicting_processes()
            print(f"Killed {len(killed)} conflicting processes")
        elif sys.argv[1] == 'dashboard':
            port = monitor.start_simple_dashboard()
            if port:
                print(f"Dashboard started on http://localhost:{port}")
            else:
                print("Could not start dashboard")
    else:
        # Interactive mode
        monitor.interactive_menu()

if __name__ == '__main__':
    main() 