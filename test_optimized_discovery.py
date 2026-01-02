#!/usr/bin/env python3
"""Test optimized Auto Discovery settings for higher returns"""

import requests
import json

def test_optimized_auto_discovery():
    """Test auto discovery with optimized settings for higher returns"""
    
    print("🚀 Testing OPTIMIZED Auto Discovery Settings")
    print("=" * 50)
    
    # OPTIMIZED SETTINGS FOR HIGHER RETURNS
    optimized_settings = {
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",      # 5 months for more opportunities
        "initial_balance": 10000,
        "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"],  # More symbols
        "max_positions": 5             # Allow more concurrent positions
    }
    
    print("📊 OPTIMIZED SETTINGS:")
    print(f"Period: {optimized_settings['start_date']} to {optimized_settings['end_date']} (5 months)")
    print(f"Symbols: {len(optimized_settings['symbols'])} cryptocurrencies")
    print(f"Max Positions: {optimized_settings['max_positions']}")
    print(f"Initial Balance: ${optimized_settings['initial_balance']:,}")
    print()
    
    print("🎯 EXPECTED IMPROVEMENTS:")
    print("• 5-month period = more market cycles")
    print("• 4 symbols = 4x more opportunities") 
    print("• 5 positions = higher portfolio utilization")
    print("• Should target 50-200% returns")
    print()
    
    try:
        print("🔄 Running optimized backtest...")
        response = requests.post(
            "http://localhost:8081/api/run_auto_discovery",
            json=optimized_settings,
            timeout=300  # 5 minutes for longer period
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                results = result['results']
                
                print("✅ OPTIMIZED RESULTS:")
                print("=" * 30)
                print(f"Initial Balance:  ${results['initial_balance']:,.2f}")
                print(f"Final Balance:    ${results['final_balance']:,.2f}")
                print(f"Total Return:     {results['total_return']:.2f}%")
                print(f"Total Trades:     {results['total_trades']}")
                print(f"Win Rate:         {results['win_rate']:.1f}%")
                print()
                
                # Compare to your current results
                current_return = 1902.6  # From your screenshot
                new_return = results['total_return']
                
                if new_return > current_return:
                    improvement = new_return - current_return
                    print(f"🎉 IMPROVEMENT: +{improvement:.1f}% better returns!")
                else:
                    print(f"📊 Result: {new_return:.1f}% (vs {current_return:.1f}% before)")
                
                # Profit calculation
                profit = results['final_balance'] - results['initial_balance']
                print(f"💰 Total Profit: ${profit:,.2f}")
                
                if results['total_trades'] > 0:
                    avg_profit_per_trade = profit / results['total_trades']
                    print(f"📈 Avg Profit/Trade: ${avg_profit_per_trade:,.2f}")
                
            else:
                print(f"❌ Error: {result.get('error')}")
        
        else:
            print(f"❌ HTTP Error: {response.status_code}")
    
    except requests.exceptions.Timeout:
        print("⏱️ Timeout - longer periods need more time")
        print("💡 Try shorter periods or fewer symbols")
    except Exception as e:
        print(f"❌ Error: {e}")

def suggest_more_optimizations():
    """Suggest additional optimizations for even higher returns"""
    
    print("\n🔧 MORE OPTIMIZATION IDEAS:")
    print("=" * 40)
    
    print("1. 📅 TIMING OPTIMIZATION:")
    print("   • Try bull market periods (Mar-Nov 2024)")
    print("   • Avoid bear markets or sideways periods")
    print()
    
    print("2. 🎯 PARAMETER TUNING:")
    print("   • Lower confidence threshold (30 instead of 40)")
    print("   • Increase position size (20% instead of 10%)")
    print("   • Adjust take profit (25% instead of 15%)")
    print()
    
    print("3. 🚀 ADVANCED STRATEGIES:")
    print("   • Add momentum indicators")
    print("   • Implement trailing stops")
    print("   • Use compound position sizing")
    print()
    
    print("4. 📊 SYMBOL SELECTION:")
    print("   • Add high-volatility coins (SOLUSDT, DOGEUSDT)")
    print("   • Focus on trending cryptocurrencies")
    print("   • Remove low-performing symbols")

if __name__ == "__main__":
    test_optimized_auto_discovery()
    suggest_more_optimizations() 