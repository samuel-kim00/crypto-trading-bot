#!/usr/bin/env python3
import os
import psutil
import subprocess
from datetime import datetime

def check_status():
    print('\n' + '='*60)
    print('🤖 AI TRADING BOT MONITOR')
    print('='*60)
    print(f'📅 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # System resources
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    cpu_status = '🔴 HIGH' if cpu > 80 else '🟢 NORMAL'
    memory_status = '🔴 HIGH' if memory.percent > 80 else '🟢 NORMAL'
    
    print(f'\n💻 SYSTEM:')
    print(f'   CPU: {cpu:.1f}% {cpu_status}')
    print(f'   Memory: {memory.percent:.1f}% {memory_status}')
    
    # Check for running processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(keyword in cmdline.lower() for keyword in ['scheduler.py', 'app.py', 'dashboard']):
                processes.append((proc.info['pid'], cmdline[:60]))
        except:
            continue
    
    print(f'\n🔄 PROCESSES:')
    if processes:
        for pid, cmd in processes:
            print(f'   PID {pid}: {cmd}...')
    else:
        print('   ✅ No bot processes running')
    
    # Check for AI training files
    ai_files = []
    if os.path.exists('models'):
        ai_files.extend([f'models/{f}' for f in os.listdir('models') if f.endswith('.h5')])
    if os.path.exists('src/analysis/reports'):
        ai_files.extend([f'reports/{f}' for f in os.listdir('src/analysis/reports') if 'training' in f])
    
    print(f'\n🧠 AI STATUS:')
    if ai_files:
        print(f'   ✅ {len(ai_files)} AI files found')
        for f in ai_files[:3]:
            print(f'   📊 {f}')
    else:
        print('   ⏳ No AI training detected')
    
    print('\n' + '='*60)

def kill_processes():
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'scheduler.py' in cmdline or ('app.py' in cmdline and '8080' in cmdline):
                proc.terminate()
                killed += 1
        except:
            continue
    return killed

def main():
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'clean':
            killed = kill_processes()
            print(f'🛑 Killed {killed} conflicting processes')
        elif sys.argv[1] == 'dashboard':
            subprocess.Popen(['python3', 'src/dashboard/app.py'])
            print('🚀 Starting dashboard on port 8081')
    else:
        check_status()

if __name__ == '__main__':
    main()
