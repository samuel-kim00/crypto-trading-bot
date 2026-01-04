#!/usr/bin/env python3
"""Test the optimized auto discovery API for higher returns"""

import requests
import json

def test_optimized_auto_discovery():
    """Test the optimized auto discovery backtester via API"""
    
    print("🚀 Testing OPTIMIZED Auto Discovery API for Higher Returns")
    print("=" * 60)
    
    # Test with March 2024 (strong bull market period)
    payload = {
        "start_date": "2024-03-01",
        "end_date": "2024-04-01",
        "initial_balance": 10000,
        "symbols": ["BTCUSDT", "ETHUSDT"]
    }
    
    print("📊 OPTIMIZED SETTINGS:")
    print(f"Period: {payload['start_date']} to {payload['end_date']} (March 2024 - Bull Market)")
    print(f"Initial Balance: ${payload['initial_balance']:,}")
    print("🎯 AGGRESSIVE PARAMETERS:")
    print("• Position Size: 20% (vs 10% conservative)")
    print("• Take Profit: 25% (vs 15% conservative)")
    print("• Stop Loss: 3% (vs 5% conservative)")
    print("• Confidence Threshold: 30 (vs 40 conservative)")
    print("• Max Positions: 4 (vs 3 conservative)")
    print()
    
    try:
        print("🔄 Running OPTIMIZED backtest...")
        response = requests.post(
            "http://localhost:8081/api/run_optimized_auto_discovery",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("✅ OPTIMIZED Auto Discovery API working!")
                print()
                
                # Display results
                results = result['results']
                print("📊 OPTIMIZED RESULTS:")
                print("=" * 30)
                print(f"Initial Balance:  ${results['initial_balance']:,.2f}")
                print(f"Final Balance:    ${results['final_balance']:,.2f}")
                print(f"Total Return:     {results['total_return']:.2f}%")
                print(f"Total Trades:     {results['total_trades']}")
                print(f"Win Rate:         {results['win_rate']:.1f}%")
                print()
                
                # Show optimization details
                if 'optimization_details' in result:
                    print("🔧 OPTIMIZATION DETAILS:")
                    for key, value in result['optimization_details'].items():
                        print(f"• {key.replace('_', ' ').title()}: {value}")
                    print()
                
                # Compare to conservative version
                conservative_return = 2.62  # Your previous result
                optimized_return = results['total_return']
                
                print("📈 COMPARISON:")
                print(f"Conservative:     {conservative_return:.2f}%")
                print(f"Optimized:        {optimized_return:.2f}%")
                
                if optimized_return > conservative_return:
                    improvement = optimized_return - conservative_return
                    multiplier = optimized_return / conservative_return if conservative_return > 0 else float('inf')
                    print(f"🎉 IMPROVEMENT:   +{improvement:.2f}% ({multiplier:.1f}x better!)")
                else:
                    print(f"📊 Result:        {optimized_return:.2f}% vs {conservative_return:.2f}%")
                
                # Profit calculation
                profit = results['final_balance'] - results['initial_balance']
                print(f"💰 Total Profit:  ${profit:,.2f}")
                
                if results['total_trades'] > 0:
                    avg_profit_per_trade = profit / results['total_trades']
                    print(f"📈 Avg Profit/Trade: ${avg_profit_per_trade:,.2f}")
                
                # Show autonomous trading stats
                if 'autonomous_trading' in results:
                    auto_stats = results['autonomous_trading']
                    print()
                    print("🤖 AUTONOMOUS TRADING PERFORMANCE:")
                    print(f"• Scans Performed: {auto_stats['scans_performed']}")
                    print(f"• Opportunities Found: {auto_stats['opportunities_found']}")
                    print(f"• Success Rate: {auto_stats['discovery_success_rate']:.1f}%")
                    print(f"• Max Concurrent Positions: {auto_stats['max_concurrent_positions']}")
                
                print()
                print("🎯 CONCLUSION:")
                if optimized_return > 10:
                    print("✅ EXCELLENT: Optimized parameters delivering strong returns!")
                elif optimized_return > 5:
                    print("✅ GOOD: Solid improvement over conservative approach!")
                elif optimized_return > 0:
                    print("✅ POSITIVE: Making money with optimized risk management!")
                else:
                    print("⚠️ LEARNING: Market conditions may not favor aggressive approach")
                
            else:
                print(f"❌ API Error: {result.get('error', 'Unknown error')}")
        
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out")
        print("💡 The optimized backtest may be taking longer due to more opportunities")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def suggest_next_steps():
    """Suggest next steps for even higher returns"""
    
    print("\n🚀 NEXT STEPS FOR EVEN HIGHER RETURNS:")
    print("=" * 50)
    
    print("1. 📅 TEST DIFFERENT PERIODS:")
    print("   • Try April 2024 (Bitcoin halving)")
    print("   • Try November 2024 (Election pump)")
    print("   • Test 2-3 month periods")
    print()
    
    print("2. 💰 SCALE UP CAPITAL:")
    print("   • Test with $50,000 initial balance")
    print("   • Use $100,000 for institutional-level testing")
    print("   • Calculate annual returns at scale")
    print()
    
    print("3. 🔧 FURTHER OPTIMIZATION:")
    print("   • Add trailing stops for bigger winners")
    print("   • Implement dynamic position sizing")
    print("   • Use compound returns reinvestment")
    print()
    
    print("4. 🎯 LIVE IMPLEMENTATION:")
    print("   • Paper trade with real APIs")
    print("   • Add real-time data feeds")
    print("   • Implement risk monitoring")

if __name__ == "__main__":
    test_optimized_auto_discovery()
    suggest_next_steps() 