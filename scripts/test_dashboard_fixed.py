#!/usr/bin/env python3
"""Test that the dashboard is now using the FIXED auto discovery backtester"""

import requests
import json
import time

def test_dashboard_with_fixed_backtester():
    """Test the dashboard auto discovery endpoint to verify it's using the fixed version"""
    
    print("🧪 Testing Dashboard with FIXED Auto Discovery")
    print("=" * 50)
    
    # Wait for dashboard to start
    print("⏳ Waiting for dashboard to start...")
    time.sleep(3)
    
    # Check if dashboard is running
    try:
        response = requests.get("http://localhost:8081/api/performance", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is running on port 8081")
        else:
            print(f"⚠️ Dashboard responding with status: {response.status_code}")
            return False
    except:
        print("❌ Dashboard not accessible")
        return False
    
    # Test the auto discovery endpoint with a short period
    print("\n🔬 Testing Auto Discovery with SHORT period (should be realistic)...")
    
    payload = {
        "start_date": "2024-03-01",
        "end_date": "2024-03-05",  # Just 4 days
        "initial_balance": 10000,
        "symbols": ["BTCUSDT"]  # Just one symbol
    }
    
    try:
        print(f"📤 Sending request: {payload}")
        response = requests.post(
            "http://localhost:8081/api/run_auto_discovery", 
            json=payload, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print("✅ Auto Discovery API call successful!")
                
                # Check the results
                results = result['results']
                total_return = results.get('total_return', 0)
                final_balance = results.get('final_balance', 0)
                total_trades = results.get('total_trades', 0)
                
                print("\n📊 RESULTS:")
                print(f"Total Return:     {total_return:.2f}%")
                print(f"Final Balance:    ${final_balance:,.2f}")
                print(f"Total Trades:     {total_trades}")
                
                # Validate if results are realistic
                if -50 <= total_return <= 100:  # Reasonable for 4 days
                    print("✅ Returns are REALISTIC!")
                    if "realistic" in result.get('message', '').lower() or "fixed" in result.get('message', '').lower():
                        print("✅ Message indicates FIXED version!")
                        print("\n🎉 SUCCESS: Dashboard is now using the FIXED Auto Discovery!")
                        return True
                    else:
                        print("⚠️ Message doesn't indicate fixed version")
                else:
                    print(f"❌ Returns are UNREALISTIC: {total_return:.2f}%")
                    print("❌ Still using the BUGGY version!")
                
            else:
                print(f"❌ API Error: {result.get('error', 'Unknown error')}")
        
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out (may still be processing)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    success = test_dashboard_with_fixed_backtester()
    if success:
        print("\n🚀 Now refresh your browser and try the Auto Discovery again!")
        print("   You should see realistic results instead of impossible returns!")
    else:
        print("\n🔧 May need to restart the dashboard or check the code.") 