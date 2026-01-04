# 🚀 Cryptocurrency Trading Bot - Ultra-Aggressive Scalping Strategy

A high-performance cryptocurrency trading bot with ultra-aggressive scalping strategy, designed for fast short-term trades (30 seconds to 2 minutes). Features real-time market monitoring, automated risk management, and a comprehensive web dashboard.

## ✨ Key Features

### 🎯 Ultra-Aggressive Scalping
- **Trade Duration**: 30 seconds to 2 minutes
- **Scan Interval**: Every 5 seconds
- **Risk Management**: 8% risk per trade with 1% stop loss
- **Take Profit**: 1.5-3.5% quick profit targets
- **Max Positions**: 5 simultaneous trades

### 📊 Real-Time Dashboard
- Live balance tracking
- Active positions monitoring
- Performance analytics
- Trade history with P/L analysis
- Real-time heartbeat status

### 🔧 Technical Features
- Direct HTTP API calls (bypasses CCXT issues)
- Ultra-low memory usage (~25-95MB)
- Fast indicator calculations (SMA, RSI)
- Robust error handling
- Simulated trading mode for testing

## 📁 Project Structure

```
├── src/                              # Source code
│   ├── core/                         # Core trading functionality
│   │   ├── trading_bot_simple.py    # ⭐ Main ultra-fast trading bot
│   │   ├── trading_bot.py           # Full-featured bot with AI integration
│   │   └── trading_bot_lite.py     # Lightweight monitoring bot
│   ├── dashboard/                    # Web dashboard
│   │   ├── app.py                    # Flask web application
│   │   ├── templates/                # HTML templates
│   │   │   ├── index.html
│   │   │   ├── backtest.html
│   │   │   └── ai_training.html
│   │   └── src/                      # Dashboard assets
│   ├── analysis/                     # Analysis and ML modules
│   │   ├── auto_discovery_backtester_fixed.py
│   │   ├── enhanced_predictor.py    # ML predictions
│   │   ├── self_learning_integration.py
│   │   ├── performance_tracker.py   # Performance analytics
│   │   ├── media_analyzer_v2.py     # Sentiment analysis
│   │   ├── youtube_scraper.py       # Social sentiment
│   │   └── backtester.py            # Strategy backtesting
│   └── utils/                        # Utility functions
│       ├── pdf_generator.py          # PDF report generation
│       ├── run_bots.py               # Bot runner utilities
│       └── test_connection.py        # Connection testing
├── config/                            # Configuration files
│   ├── strategy_config.json          # Trading strategy configuration
│   ├── strategy_config.json.example  # Configuration template
│   └── *.json                        # Additional config files
├── scripts/                           # Utility scripts
│   ├── monitor.py                    # Monitoring scripts
│   ├── optimize_parameters.py        # Parameter optimization
│   ├── scheduler.py                  # Task scheduler
│   └── *.py                          # Additional utility scripts
├── docs/                              # Documentation
│   ├── README.md                     # Documentation index
│   ├── PROJECT_STRUCTURE.md          # Detailed project structure
│   ├── GITHUB_SETUP.md               # GitHub setup guide
│   └── *.md                          # Additional documentation
├── tests/                             # Test files
│   ├── test_system.py                # System tests
│   ├── test_backtest.py              # Backtest tests
│   └── test_adaptive_backtest.py     # Adaptive backtest tests
├── requirements.txt                   # Python dependencies
└── README.md                          # This file

# Local directories (not in repository, generated at runtime)
├── data/                              # Trading data (excluded from git)
├── logs/                              # Application logs (excluded from git)
├── reports/                           # Generated reports (excluded from git)
├── models/                            # ML models (excluded from git)
└── venv/                              # Virtual environment (excluded from git)
```

### 📂 Key Directories

- **`src/core/`** - Main trading bot implementations
  - `trading_bot_simple.py` is the recommended ultra-fast scalping bot
- **`src/dashboard/`** - Web-based monitoring dashboard
- **`src/analysis/`** - Machine learning and analysis modules
- **`config/`** - Strategy and system configuration files
- **`scripts/`** - Utility scripts for monitoring, optimization, and testing
- **`docs/`** - Project documentation and guides
- **`tests/`** - Unit and integration tests

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd "Cursor Trading bot"
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Strategy
Edit `config/strategy_config.json` to customize:
- Risk per trade
- Stop loss/take profit levels
- Trading pairs
- Indicator parameters

### 5. Set Up API Keys (Optional)
For live trading, add your Binance API credentials. The bot works in simulated mode without API keys.

## 🚀 Usage

### Start the Trading Bot
```bash
python3 src/core/trading_bot_simple.py
```

The bot will:
- Scan markets every 5 seconds
- Execute trades based on ultra-aggressive scalping strategy
- Update heartbeat file for dashboard
- Log all activity to `logs/trading_bot.log`

### Start the Dashboard
```bash
python3 src/dashboard/app.py
```

Access the dashboard at: **http://localhost:8081**

### Run Both (Background)
```bash
python3 src/core/trading_bot_simple.py &
python3 src/dashboard/app.py &
```

## ⚙️ Configuration

### Strategy Parameters (`config/strategy_config.json`)

**Ultra-Aggressive Scalping Settings:**
```json
{
  "risk_per_trade": 0.08,           // 8% risk per trade
  "stop_loss_pct": 0.01,            // 1% stop loss
  "take_profit_levels": [0.015, 0.025, 0.035],  // 1.5-3.5% targets
  "time_based_stop": 60,            // 1 minute max hold
  "scan_interval": 5,               // 5 second scans
  "max_open_positions": 5           // 5 simultaneous trades
}
```

**Indicator Settings:**
- **RSI**: 5-period, oversold < 45, overbought > 55
- **MACD**: Fast 5, Slow 13, Signal 3
- **EMA**: Fast 3, Slow 8
- **Volume Spike**: 1.0x threshold

## 📊 Trading Strategy

### Entry Conditions (ANY triggers trade)
- **Long**: Bullish SMA + (Oversold RSI OR Momentum OR Volume)
- **Short**: Bearish SMA + (Overbought RSI OR Momentum OR Volume)
- **Reversal**: Extreme RSI + Volume confirmation
- **Momentum**: 0.1% price movement + Volume spike

### Exit Strategy
- **Take Profit**: 1.5% (80%), 2.5% (15%), 3.5% (5%)
- **Stop Loss**: 1% hard stop
- **Time Exit**: 1 minute maximum
- **Trailing Stop**: 0.5% for profit protection
- **Reverse Signal**: Exit on opposite signal

## 📈 Performance Monitoring

### Dashboard Features
- **Real Balance**: Live account balance from heartbeat
- **Active Trades**: Current positions with P/L
- **Performance Stats**: Win rate, total P/L, trade count
- **Heartbeat Status**: Bot health and memory usage

### Logs
- Trading activity: `logs/trading_bot.log`
- Dashboard: `logs/dashboard.log`
- Heartbeat: `logs/trading_bot_heartbeat.json`

## 🔒 Security Notes

- **Never commit API keys** to the repository
- Use `.env` file for sensitive credentials (not included in repo)
- The bot works in simulated mode without API keys
- All trades are logged for audit purposes

## ⚠️ Disclaimer

**This trading bot is for educational purposes only.**

- Cryptocurrency trading involves significant risk
- Never trade with money you cannot afford to lose
- Past performance does not guarantee future results
- Always test with small amounts first
- The ultra-aggressive strategy is high-risk/high-reward

## 🐛 Troubleshooting

### Bot Not Trading
1. Check `logs/trading_bot.log` for errors
2. Verify `config/strategy_config.json` is valid
3. Check heartbeat file: `logs/trading_bot_heartbeat.json`
4. Ensure sufficient balance for minimum trade size

### Dashboard Not Loading
1. Check if port 8081 is available
2. Verify dashboard is running: `ps aux | grep app.py`
3. Check dashboard logs: `logs/dashboard.log`

### API Connection Issues
- The bot uses direct HTTP requests (no CCXT dependency)
- Check internet connection
- Verify Binance API is accessible

## 📝 License

This project is for personal use only. Please ensure compliance with your local regulations regarding automated trading.

## 🤝 Contributing

Feel free to fork, modify, and use this project for your own trading needs. Remember to:
- Test thoroughly before live trading
- Start with small amounts
- Monitor performance closely
- Adjust strategy parameters based on market conditions

---

**Happy Trading! 🚀📈**
