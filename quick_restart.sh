#!/bin/bash

echo "🔄 Restarting Trading Bot System..."

# Kill existing processes
echo "Stopping existing processes..."
pkill -f "python.*scheduler.py" 2>/dev/null
pkill -f "python.*dashboard" 2>/dev/null  
pkill -f "python.*trading_bot" 2>/dev/null
sleep 3

# Clean up temp files
echo "Cleaning up files..."
rm -f logs/trading_bot_heartbeat.json
rm -f MANUAL_INTERVENTION_REQUIRED.txt

# Create logs directory
mkdir -p logs

# Check system status
echo "System Status:"
echo "- CPU: $(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')"
echo "- Available ports:"
for port in 8081 8082 8083 8084 8085; do
    if ! lsof -ti:$port > /dev/null 2>&1; then
        echo "  ✅ Port $port available"
    else
        echo "  ❌ Port $port in use"
    fi
done

echo ""
echo "🚀 Starting system..."

# Start the scheduler which will start everything else
source venv/bin/activate
python3 scheduler.py &

echo "✅ System started!"
echo "📊 Dashboard will be available at: http://localhost:8081"
echo "📝 Monitor logs with: tail -f logs/scheduler.log"
echo ""
echo "To stop the system: pkill -f scheduler.py" 