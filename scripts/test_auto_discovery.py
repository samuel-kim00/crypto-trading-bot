#!/usr/bin/env python3

import sys
import os
import asyncio
import traceback

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_auto_discovery():
    try:
        print("Testing Auto-Discovery Backtester...")
        
        from auto_discovery_backtester import AutoDiscoveryBacktester
        
        backtester = AutoDiscoveryBacktester(10000)
        print("✅ Backtester initialized successfully")
        
        # Test with a very short period
        print("🔄 Running 1-week test...")
        results = await backtester.run_auto_discovery_backtest('2024-01-01', '2024-01-07')
        
        print("✅ Test completed successfully!")
        print(f"📊 Results keys: {list(results.keys())}")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"📈 Final Balance: ${summary.get('final_balance', 0):,.2f}")
            print(f"🎯 Total Trades: {summary.get('total_trades', 0)}")
            print(f"📊 Win Rate: {summary.get('win_rate', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_auto_discovery())
    if success:
        print("🎉 Auto-discovery test PASSED")
    else:
        print("💥 Auto-discovery test FAILED") 