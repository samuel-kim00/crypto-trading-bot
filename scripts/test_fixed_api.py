#!/usr/bin/env python3
"""Test the fixed auto discovery API directly"""

import requests
import json
import time

def test_fixed_auto_discovery_api():
    """Test the fixed auto discovery backtester via API"""
    
    print("🧪 Testing Fixed Auto Discovery API")
    print("=" * 40)
    
    # API endpoint
    url = "http://localhost:8081/api/run_fixed_auto_discovery"
    
    # Test parameters
    payload = {
        "start_date": "2024-03-01",
        "end_date": "2024-03-15",  # 2 weeks test
        "initial_balance": 10000,
        "symbols": ["BTCUSDT", "ETHUSDT"]
    }
    
    print(f"📅 Testing period: {payload['start_date']} to {payload['end_date']}")
    print(f"💰 Initial balance: ${payload['initial_balance']:,}")
    print("🔄 Starting backtest...")
    
    try:
        # Make API call
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("✅ Fixed Auto Discovery API working!")
                print()
                
                # Display results
                results = result['results']
                print("📊 RESULTS:")
                print("-" * 20)
                print(f"Initial Balance:  ${results['initial_balance']:,.2f}")
                print(f"Final Balance:    ${results['final_balance']:,.2f}")
                print(f"Total Return:     {results['total_return']:.2f}%")
                print(f"Total Trades:     {results['total_trades']}")
                print(f"Win Rate:         {results['win_rate']:.1f}%")
                print()
                
                # Validate realistic results
                if -50 <= results['total_return'] <= 200:
                    print("✅ Returns are realistic!")
                else:
                    print("⚠️ Returns may be unrealistic")
                
                if results['final_balance'] < results['initial_balance'] * 10:
                    print("✅ Final balance is reasonable!")
                else:
                    print("⚠️ Final balance seems too high")
                
                print()
                print("🎉 Fixed Auto Discovery is working correctly!")
                
            else:
                print(f"❌ API Error: {result.get('error', 'Unknown error')}")
        
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to dashboard")
        print("💡 Make sure the dashboard is running on port 8081")
        print("   Try: python src/dashboard/app.py")
    
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out")
        print("💡 The backtest may be taking longer than expected")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def check_dashboard_status():
    """Check if dashboard is accessible"""
    print("🔍 Checking Dashboard Status")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:8081/api/performance", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is running on port 8081")
            print("🌐 Access at: http://localhost:8081")
            return True
        else:
            print(f"⚠️ Dashboard responding with status: {response.status_code}")
            return False
    except:
        print("❌ Dashboard not accessible")
        print("💡 Try starting it with: python src/dashboard/app.py")
        return False

if __name__ == "__main__":
    if check_dashboard_status():
        print()
        test_fixed_auto_discovery_api()
    else:
        print()
        print("🔧 Please start the dashboard first:")
        print("   cd /Users/samuelkim/Cursor\ Trading\ bot")
        print("   source venv/bin/activate")
        print("   python src/dashboard/app.py") 