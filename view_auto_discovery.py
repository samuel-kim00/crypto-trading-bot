#!/usr/bin/env python3
"""
Display Auto Discovery Backtest Results
"""
import json
import glob
from datetime import datetime

def format_number(num):
    """Format large numbers for display"""
    if num > 1e12:
        return f"${num/1e12:.2f}T"
    elif num > 1e9:
        return f"${num/1e9:.2f}B"
    elif num > 1e6:
        return f"${num/1e6:.2f}M"
    elif num > 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def display_auto_discovery_results():
    """Display the latest auto discovery results"""
    # Find the latest auto discovery results
    pattern = "src/analysis/reports/auto_discovery_*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No auto discovery results found!")
        return
    
    latest_file = max(files)
    print(f"📊 Loading results from: {latest_file}")
    print("=" * 80)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Display summary
    print("🚀 AUTO DISCOVERY BACKTEST RESULTS")
    print("=" * 80)
    print(f"Initial Balance:     {format_number(results['initial_balance'])}")
    print(f"Final Balance:       {format_number(results['final_balance'])}")
    print(f"Total Return:        {results['total_return']:,.2f}%")
    print(f"Total Trades:        {results['total_trades']}")
    print(f"Winning Trades:      {results['winning_trades']}")
    print(f"Losing Trades:       {results['losing_trades']}")
    print(f"Win Rate:            {results['win_rate']:.2f}%")
    print(f"Average Win:         ${results['avg_win']:.2f}")
    print(f"Average Loss:        ${results['avg_loss']:.2f}")
    print()
    
    # Discovery stats
    discovery = results.get('discovery_stats', {})
    print("🔍 DISCOVERY STATISTICS")
    print("=" * 40)
    print(f"Scans Performed:     {discovery.get('scans_performed', 0)}")
    print(f"Opportunities Found: {discovery.get('opportunities_found', 0)}")
    print(f"Symbols Discovered:  {discovery.get('symbols_discovered', 0)}")
    print(f"Avg Opportunities:   {discovery.get('avg_opportunities_per_scan', 0):.2f}")
    print()
    
    # Symbols traded
    symbols = discovery.get('symbols_traded', [])
    print("💰 TOP SYMBOLS TRADED:")
    print("-" * 30)
    for i, symbol in enumerate(symbols[:10], 1):
        print(f"{i:2d}. {symbol}")
    print()
    
    # Autonomous trading stats
    auto = results.get('autonomous_trading', {})
    print("🤖 AUTONOMOUS TRADING")
    print("=" * 40)
    print(f"Max Positions:       {auto.get('max_concurrent_positions', 0)}")
    print(f"Positions Opened:    {auto.get('positions_opened', 0)}")
    print(f"Positions Closed:    {auto.get('positions_closed', 0)}")
    print(f"Still Open:          {auto.get('still_open', 0)}")
    print(f"Avg Hold Time:       {auto.get('avg_hold_time', 0):.2f} days")
    print(f"Success Rate:        {auto.get('discovery_success_rate', 0):.2f}%")
    print()
    
    # Recent trades
    trades = results.get('trade_history', [])
    print("📈 RECENT TRADES (Last 10)")
    print("=" * 80)
    print(f"{'Symbol':<12} {'Action':<4} {'Price':<12} {'PnL':<12} {'Date':<12}")
    print("-" * 80)
    
    for trade in trades[-10:]:
        symbol = trade.get('symbol', 'N/A')[:10]
        action = trade.get('action', 'N/A')
        price = f"${trade.get('price', 0):.2f}"
        pnl = trade.get('pnl', 0)
        pnl_str = f"${pnl:.2f}" if pnl else "OPEN"
        date = trade.get('date', 'N/A')
        
        print(f"{symbol:<12} {action:<4} {price:<12} {pnl_str:<12} {date:<12}")
    
    print("\n🎯 STRATEGY INSIGHTS:")
    print("=" * 40)
    print("✓ High-frequency trading (5-hour holds)")
    print("✓ Multi-asset diversification")
    print("✓ ML-powered entry signals")
    print("✓ Risk management exits")
    print("\n💡 Access full dashboard at: http://localhost:8080/backtest")

if __name__ == "__main__":
    display_auto_discovery_results() 