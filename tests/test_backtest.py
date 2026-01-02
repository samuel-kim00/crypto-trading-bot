#!/usr/bin/env python3
"""
Quick test script for VLM Strategy Backtesting
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append('src/analysis')

from backtester import VLMBacktester

async def run_demo_backtest():
    """Run a demonstration backtest"""
    print("🚀 VLM Strategy Backtesting Demo")
    print("=" * 50)
    
    # Initialize backtester
    backtester = VLMBacktester(initial_balance=10000)
    
    # Test parameters
    symbols = ['BTCUSDT', 'ETHUSDT']  # Start with 2 major coins
    start_date = '2024-01-01'
    end_date = '2024-03-01'  # 2 months of data
    
    print(f"📅 Testing Period: {start_date} to {end_date}")
    print(f"💰 Initial Balance: ${backtester.initial_balance:,.2f}")
    print(f"🪙 Trading Pairs: {', '.join(symbols)}")
    print(f"⚙️  Strategy: VLM (Volatility-Liquidity-Momentum)")
    print()
    
    # Run backtest
    print("⏳ Running backtest...")
    results = await backtester.run_backtest(symbols, start_date, end_date)
    
    # Display results
    summary = results['summary']
    print("\n📈 BACKTEST RESULTS")
    print("=" * 50)
    print(f"🏁 Final Balance:        ${summary['final_balance']:,.2f}")
    print(f"📊 Total Return:         {summary['total_return_pct']:+.2f}%")
    print(f"🎯 Total Trades:         {summary['total_trades']}")
    print(f"🏆 Win Rate:            {summary['win_rate_pct']:.1f}%")
    print(f"📈 Average Win:         {summary['avg_win_pct']:+.2f}%")
    print(f"📉 Average Loss:        {summary['avg_loss_pct']:+.2f}%")
    print(f"⚡ Sharpe Ratio:        {summary['sharpe_ratio']:.2f}")
    print(f"🔻 Max Drawdown:        {summary['max_drawdown_pct']:.2f}%")
    print(f"📊 Volatility:          {summary['volatility_pct']:.2f}%")
    
    # Performance metrics
    metrics = results['performance_metrics']
    print(f"💎 Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"💸 Total Fees:          ${metrics['total_fees_paid']:.2f}")
    print(f"⏱️  Avg Hold Time:       {metrics['avg_holding_time_hours']:.1f} hours")
    
    # Trade examples
    trade_history = results['trade_history']
    if trade_history:
        print(f"\n📋 Sample Trades (last 5):")
        print("-" * 50)
        for trade in trade_history[-5:]:
            action = trade['action'].upper()
            symbol = trade['symbol']
            price = trade['price']
            pnl = trade.get('pnl', 0)
            reason = trade['reason'][:30] + "..." if len(trade['reason']) > 30 else trade['reason']
            
            if action == 'SELL' and pnl != 0:
                pnl_str = f"{pnl:+.2f} ({trade.get('pnl_pct', 0):+.1f}%)"
                print(f"{action:4} {symbol:8} ${price:8.4f} | P&L: ${pnl_str:15} | {reason}")
            else:
                print(f"{action:4} {symbol:8} ${price:8.4f} | {reason}")
    
    # Save results
    filepath = backtester.save_backtest_results(results)
    print(f"\n💾 Results saved to: {filepath}")
    
    # Summary interpretation
    print(f"\n🧠 Strategy Analysis:")
    if summary['total_return_pct'] > 0:
        print("✅ The VLM strategy showed positive returns over this period")
    else:
        print("❌ The VLM strategy showed negative returns over this period")
    
    if summary['win_rate_pct'] > 50:
        print(f"✅ Good win rate of {summary['win_rate_pct']:.1f}%")
    else:
        print(f"⚠️  Win rate of {summary['win_rate_pct']:.1f}% could be improved")
    
    if summary['sharpe_ratio'] > 1:
        print(f"✅ Excellent risk-adjusted returns (Sharpe: {summary['sharpe_ratio']:.2f})")
    elif summary['sharpe_ratio'] > 0.5:
        print(f"✅ Good risk-adjusted returns (Sharpe: {summary['sharpe_ratio']:.2f})")
    else:
        print(f"⚠️  Risk-adjusted returns could be better (Sharpe: {summary['sharpe_ratio']:.2f})")
    
    print(f"\n🎯 Access full results via dashboard: http://localhost:8080/backtest")

if __name__ == "__main__":
    asyncio.run(run_demo_backtest()) 