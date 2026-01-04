# Cryptocurrency Trading Bot (Python)

## Overview
A personal Python project focused on designing an automated trading system
with an emphasis on system architecture, data flow, and monitoring.

Rather than optimizing for profitability, this project explores how trading
logic can be structured, updated, and observed in a reliable and extensible way.
The system is built to initialize strategies from historical indicators and
iteratively adjust parameters as new data becomes available.

---

## Project Goals
- Design a modular and maintainable trading system
- Work with real-time and periodically updated external data via APIs
- Separate core execution logic from experiments and analysis
- Implement basic monitoring and visualization for system status
- Explore learning-oriented strategy updates without relying on full ML models

---

## Key Components
- **Core Trading Engine**  
  Handles strategy evaluation and trade execution logic
- **Configuration Layer**  
  Strategy parameters are defined through configuration files for easier iteration
- **Monitoring & Logging**  
  Tracks system health, execution status, and runtime behavior
- **Web Dashboard**  
  A lightweight Flask-based dashboard for observing system state and activity
- **Experimental Modules**  
  Scripts used to test alternative strategies and parameter variations

---

## Project Structure
```
src/
  core/          # main trading bot implementations
  dashboard/     # Flask-based monitoring dashboard
  analysis/      # research, backtesting, and experimental scripts
  utils/         # helper utilities
config/
  strategy_config.json
scripts/         # utility scripts for monitoring and optimization
docs/            # project documentation
tests/           # test files
requirements.txt
README.md
```

---

## How It Works
1. The system establishes an initial trading strategy using historical indicators
2. Market and external data are periodically fetched through APIs
3. Strategy parameters are adjusted over time based on updated information
4. Trades are executed locally using exchange APIs
5. System status is tracked through logs and a monitoring dashboard

The system is designed to behave in a learning-oriented manner through
iterative parameter updates, rather than through a trained machine learning model.

---

## Getting Started

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Configure Strategy
Edit the configuration file below to adjust strategy parameters for testing:
```
config/strategy_config.json
```

---

## Run the Trading Bot
```bash
python3 src/core/trading_bot_simple.py
```

---

## Run the Dashboard
```bash
python3 src/dashboard/app.py
```

Open locally at:
```
http://localhost:8081
```

---

## Safety Notes
- API keys and credentials must not be committed to this repository
- Sensitive information should be managed using environment variables or `.env` files
- Simulated or limited testing modes are recommended
- This project prioritizes system design and experimentation over financial outcomes

---

## Disclaimer
This project is for educational and engineering exploration purposes only.
It does not constitute financial advice or a production trading system.
