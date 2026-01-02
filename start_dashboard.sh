#!/bin/bash

# Cryptocurrency Trading Bot Dashboard Startup Script

echo "🚀 Starting Cryptocurrency Trading Bot Dashboard..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Kill any existing processes on port 8080
echo "🔍 Checking for existing processes on port 8080..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

# Start the dashboard
echo "🌐 Starting dashboard on http://localhost:8080"
echo "📊 Access your trading dashboard in your web browser"
echo "⏹️  Press Ctrl+C to stop the dashboard"
echo ""

python src/dashboard/app.py 