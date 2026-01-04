# Trading Bot Project Structure

## Overview
This project implements a sophisticated cryptocurrency trading bot with VLM (Volatility-Liquidity-Momentum) Scalper strategy, featuring automated sentiment analysis, performance tracking, and a web-based dashboard.

## Directory Structure

```
/
├── scheduler.py                    # Main process manager and scheduler
├── requirements.txt               # Python dependencies
├── trading_bot.service           # Systemd service configuration
├── start_dashboard.sh            # Quick dashboard startup script
├── README.md                     # Project documentation
├── PROJECT_STRUCTURE.md          # This file
├── 
├── src/                          # Main source code
│   ├── __init__.py
│   ├── core/                     # Core trading functionality
│   │   ├── __init__.py
│   │   └── trading_bot.py        # Main trading bot implementation
│   ├── analysis/                 # Analysis and data processing
│   │   ├── __init__.py
│   │   ├── performance_tracker.py    # Performance monitoring
│   │   ├── media_analyzer_v2.py      # Sentiment analysis
│   │   ├── youtube_scraper.py         # YouTube data collection
│   │   ├── weekly_scheduler.py        # Report scheduling
│   │   ├── enhanced_predictor.py      # ML predictions
│   │   ├── backtester.py              # Strategy backtesting
│   │   ├── adaptive_backtester.py     # Advanced backtesting
│   │   ├── auto_discovery_backtester.py # Auto strategy discovery
│   │   ├── live_tracker.py            # Live performance tracking
│   │   ├── data/                      # Analysis data storage
│   │   └── reports/                   # Generated reports
│   ├── dashboard/                # Web dashboard
│   │   ├── __init__.py
│   │   ├── app.py                # Flask web application
│   │   └── templates/            # HTML templates
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── pdf_generator.py      # PDF report generation
│       ├── setup_service.sh      # System service setup
│       ├── run_bots.py          # Bot runner utilities
│       └── test_connection.py    # Connection testing
│
├── config/                       # Configuration files
│   ├── strategy_config.json      # Trading strategy parameters
│   ├── active_positions.json     # Current trading positions
│   ├── strategy_status.json      # Strategy status tracking
│   └── schedule_config.json      # Scheduling configuration
│
├── data/                        # Data storage
│   ├── performance_data.json     # Performance metrics
│   ├── youtube_sentiment.json    # Sentiment analysis results
│   ├── prediction_history.json   # ML prediction history
│   └── media_analysis/           # Media analysis cache
│
├── logs/                        # Application logs
│   ├── scheduler.log             # Main scheduler logs
│   ├── trading_bot.log          # Trading bot logs
│   ├── dashboard.log            # Dashboard logs
│   └── trading_bot_heartbeat.json # Health monitoring
│
├── tests/                       # Test files
│   ├── test_system.py           # System component tests
│   ├── test_adaptive_backtest.py # Backtest tests
│   └── test_backtest.py         # Basic backtest tests
│
├── reports/                     # Generated reports
├── templates/                   # Additional templates
└── venv/                       # Python virtual environment
```

## Key Components

### 1. Main Scheduler (`scheduler.py`)
- **Purpose**: Process manager and main entry point
- **Features**: 
  - Health monitoring with heartbeat system
  - Automatic process restart on failure
  - Weekly YouTube sentiment scraping
  - Internet connectivity checking
  - Advanced error handling and logging

### 2. Trading Bot (`src/core/trading_bot.py`)
- **Purpose**: Core trading logic implementation
- **Strategy**: VLM (Volatility-Liquidity-Momentum) Scalper
- **Features**:
  - Real-time market data processing
  - Technical indicator analysis (RSI, EMA, MACD)
  - Risk management with stop-loss and take-profit
  - Position sizing and portfolio management
  - Sentiment integration

### 3. Dashboard (`src/dashboard/app.py`)
- **Purpose**: Web-based monitoring and control interface
- **Features**:
  - Real-time performance monitoring
  - Backtest execution and results
  - Report generation and PDF download
  - Strategy configuration
  - WebSocket updates

### 4. Analysis Suite (`src/analysis/`)
- **Performance Tracker**: Trade and portfolio performance monitoring
- **Media Analyzer**: YouTube and news sentiment analysis
- **Backtester**: Historical strategy testing with multiple variants
- **Enhanced Predictor**: Machine learning predictions
- **Live Tracker**: Real-time performance tracking

## Configuration Files

### Strategy Config (`config/strategy_config.json`)
- Trading pairs and market selection
- Risk management parameters
- Technical indicator settings
- Position sizing rules

### Active Positions (`config/active_positions.json`)
- Current open positions
- Entry prices and sizes
- Stop-loss and take-profit levels

## Data Management

### Performance Data (`data/performance_data.json`)
- Trade history and results
- Daily and hourly statistics
- Win/loss ratios and profit metrics

### Sentiment Data (`data/youtube_sentiment.json`)
- YouTube video analysis results
- Weighted sentiment scores
- Historical sentiment trends

## Logging System

All components use structured logging:
- **scheduler.log**: Main process management events
- **trading_bot.log**: Trading decisions and executions
- **dashboard.log**: Web interface activities
- **trading_bot_heartbeat.json**: Real-time health status

## Testing

Comprehensive test suite in `tests/`:
- System component validation
- Import and initialization tests
- Configuration file verification
- Directory structure validation

## Dependencies

Key Python packages (see `requirements.txt`):
- `python-binance`: Exchange API integration
- `ccxt`: Multi-exchange support
- `pandas/numpy`: Data processing
- `TA-Lib`: Technical analysis
- `flask/socketio`: Web dashboard
- `schedule`: Task scheduling
- `textblob`: Sentiment analysis

## Service Management

### Systemd Service (`trading_bot.service`)
- Automatic startup on boot
- Process monitoring and restart
- Proper user and environment setup

### Manual Startup
```bash
# Start scheduler (recommended)
python scheduler.py

# Start dashboard only
bash start_dashboard.sh

# Run tests
python tests/test_system.py
```

## Security Considerations

- API keys stored in environment variables
- No sensitive data in configuration files
- Process isolation and monitoring
- Comprehensive error handling

## Maintenance

### Regular Tasks
- Monitor log files for errors
- Review performance metrics
- Update sentiment analysis data
- Backup configuration and data files

### Troubleshooting
- Check `MANUAL_INTERVENTION_REQUIRED.txt` for critical issues
- Review scheduler logs for process failures
- Verify internet connectivity and API access
- Monitor disk space and memory usage 