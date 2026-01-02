#!/usr/bin/env python3
"""
Test script for Adaptive Backtesting with Live Strategy + ML Integration
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append('src/analysis')

from adaptive_backtester import AdaptiveBacktester

async def run_adaptive_demo():
    """Run a demonstration of adaptive backtesting"""
    print("🤖 ADAPTIVE BACKTESTING DEMO")
    print("=" * 60)
    print("🧠 Uses your LIVE trading strategy + ML predictions")
    print("🔄 Automatically adapts when strategy changes")
    print("📊 Confidence-based position sizing")
    print("🎯 Day trading vs long-term categorization")
    print()
    
    # Initialize adaptive backtester
    backtester = AdaptiveBacktester(initial_balance=10000)
    
    # Test parameters
    symbols = ['BTCUSDT', 'ETHUSDT']  # Start with major coins
    start_date = '2024-01-01'
    end_date = '2024-02-15'  # 1.5 months of data
    
    print(f"📅 Testing Period: {start_date} to {end_date}")
    print(f"💰 Initial Balance: ${backtester.initial_balance:,.2f}")
    print(f"🪙 Trading Pairs: {', '.join(symbols)}")
    print(f"⚙️  Strategy: Adaptive VLM + Enhanced ML Predictor")
    print()
    
    # Show live strategy config
    config = backtester.strategy_config
    print("📋 Live Strategy Configuration:")
    print(f"   • Risk per trade: {config['risk_per_trade']*100:.1f}%")
    print(f"   • Max position size: {config['max_position_size']*100:.1f}%")
    print(f"   • Stop loss: {config['stop_loss_pct']*100:.1f}%")
    print(f"   • Take profit levels: {[f'{tp*100:.0f}%' for tp in config['take_profit_levels']]}")
    print(f"   • Volume spike threshold: {config['volume_spike_threshold']}x")
    print(f"   • RSI range: {config['rsi_long_range']}")
    print(f"   • Time-based stop: {config['time_based_stop']}s")
    print()
    
    # Run adaptive backtest
    print("⏳ Running adaptive backtest...")
    results = await backtester.run_adaptive_backtest(symbols, start_date, end_date)
    
    # Display results
    summary = results['summary']
    strategy_info = results['strategy_info']
    ml_performance = results['ml_performance']
    
    print("\n🚀 ADAPTIVE BACKTEST RESULTS")
    print("=" * 60)
    print(f"🎯 Strategy Type:        {strategy_info['strategy_type']}")
    print(f"🤖 ML Integration:       {strategy_info['ml_integration']}")
    print(f"🔄 Live Bot Sync:        {'Yes' if strategy_info['live_bot_sync'] else 'No'}")
    print()
    print("📊 PERFORMANCE METRICS")
    print("-" * 30)
    print(f"🏁 Final Balance:        ${summary['final_balance']:,.2f}")
    print(f"📈 Total Return:         {summary['total_return_pct']:+.2f}%")
    print(f"🎯 Total Trades:         {summary['total_trades']}")
    print(f"🏆 Overall Win Rate:     {summary['win_rate_pct']:.1f}%")
    print(f"🧠 ML High-Conf Win Rate: {ml_performance['high_confidence_win_rate']:.1f}%")
    print(f"📈 Average Win:         {summary['avg_win_pct']:+.2f}%")
    print(f"📉 Average Loss:        {summary['avg_loss_pct']:+.2f}%")
    print(f"⚡ Sharpe Ratio:        {summary['sharpe_ratio']:.2f}")
    print(f"🔻 Max Drawdown:        {summary['max_drawdown_pct']:.2f}%")
    print(f"📊 Volatility:          {summary['volatility_pct']:.2f}%")
    
    # ML performance analysis
    print(f"\n🤖 ML INTEGRATION ANALYSIS")
    print("-" * 30)
    print(f"ML Predictions Used:     {ml_performance['predictions_used']}")
    print(f"Average Confidence:      {ml_performance['confidence_avg']:.1f}%")
    print(f"High-Confidence Trades:  Better win rate than overall")
    
    # Performance metrics
    metrics = results['performance_metrics']
    print(f"\n💎 ADVANCED METRICS")
    print("-" * 30)
    print(f"Profit Factor:          {metrics['profit_factor']:.2f}")
    print(f"Total Fees Paid:        ${metrics['total_fees_paid']:.2f}")
    print(f"Average Hold Time:      {metrics['avg_holding_time_hours']:.1f} hours")
    
    # Adaptive features
    print(f"\n🔄 ADAPTIVE FEATURES")
    print("-" * 30)
    for feature in strategy_info['adaptive_features']:
        print(f"✅ {feature}")
    
    # Trade examples
    trade_history = results['trade_history']
    if trade_history:
        print(f"\n📋 SAMPLE TRADES (Last 10):")
        print("-" * 80)
        print(f"{'ACTION':<6} {'SYMBOL':<8} {'PRICE':<10} {'P&L':<12} {'CONF':<5} {'REASON':<30}")
        print("-" * 80)
        
        for trade in trade_history[-10:]:
            action = trade['action'].upper()
            symbol = trade['symbol']
            price = trade['price']
            pnl = trade.get('pnl', 0)
            confidence = trade.get('confidence', 0)
            reason = trade['reason'][:28] + "..." if len(trade['reason']) > 28 else trade['reason']
            
            if action in ['SELL', 'PARTIAL_TP'] and pnl != 0:
                pnl_str = f"${pnl:+.2f} ({trade.get('pnl_pct', 0):+.1f}%)"
                print(f"{action:<6} {symbol:<8} ${price:<9.4f} {pnl_str:<12} {confidence:<4.0f}% {reason}")
            else:
                print(f"{action:<6} {symbol:<8} ${price:<9.4f} {'N/A':<12} {confidence:<4.0f}% {reason}")
    
    # Save results
    filepath = backtester.save_adaptive_results(results)
    print(f"\n💾 Results saved to: {filepath}")
    
    # Strategy analysis and recommendations
    print(f"\n🧠 ADAPTIVE STRATEGY ANALYSIS:")
    
    if summary['total_return_pct'] > 0:
        print("✅ The adaptive strategy showed positive returns")
    else:
        print("❌ The adaptive strategy showed negative returns")
    
    if summary['win_rate_pct'] > 50:
        print(f"✅ Good overall win rate of {summary['win_rate_pct']:.1f}%")
    else:
        print(f"⚠️  Overall win rate of {summary['win_rate_pct']:.1f}% could be improved")
    
    if ml_performance['high_confidence_win_rate'] > summary['win_rate_pct']:
        print(f"✅ ML high-confidence trades outperformed overall strategy")
        print(f"   ({ml_performance['high_confidence_win_rate']:.1f}% vs {summary['win_rate_pct']:.1f}%)")
    else:
        print(f"⚠️  ML predictions need calibration")
    
    if summary['sharpe_ratio'] > 1:
        print(f"✅ Excellent risk-adjusted returns (Sharpe: {summary['sharpe_ratio']:.2f})")
    elif summary['sharpe_ratio'] > 0.5:
        print(f"✅ Good risk-adjusted returns (Sharpe: {summary['sharpe_ratio']:.2f})")
    else:
        print(f"⚠️  Risk-adjusted returns could be better (Sharpe: {summary['sharpe_ratio']:.2f})")
    
    # Live strategy adaptation benefits
    print(f"\n🎯 ADAPTIVE BENEFITS:")
    print("• Strategy automatically syncs with your live bot configuration")
    print("• ML predictions enhance entry signals with confidence scoring")
    print("• Position sizing adapts based on prediction confidence")
    print("• Strategy parameters update if you modify your live bot")
    print("• Day trading vs long-term categorization improves precision")
    
    print(f"\n🌐 Access full interactive results: http://localhost:8080/backtest")
    print(f"📊 Select 'Adaptive Mode' and run with same parameters")

if __name__ == "__main__":
    asyncio.run(run_adaptive_demo()) 