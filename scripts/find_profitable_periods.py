#!/usr/bin/env python3
"""Find the most profitable time periods for Auto Discovery"""

import requests
import json
from datetime import datetime, timedelta

def test_period(start_date, end_date, description):
    """Test a specific time period"""
    
    settings = {
        "start_date": start_date,
        "end_date": end_date,
        "initial_balance": 10000,
        "symbols": ["BTCUSDT", "ETHUSDT"]  # Keep it simple
    }
    
    try:
        print(f"🧪 Testing {description} ({start_date} to {end_date})...")
        response = requests.post(
            "http://localhost:8081/api/run_auto_discovery",
            json=settings,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                results = result['results']
                return {
                    'period': description,
                    'start': start_date,
                    'end': end_date,
                    'return': results['total_return'],
                    'trades': results['total_trades'],
                    'win_rate': results['win_rate'],
                    'final_balance': results['final_balance']
                }
        
        return {'period': description, 'return': 'ERROR', 'trades': 0, 'win_rate': 0}
    
    except Exception as e:
        return {'period': description, 'return': 'TIMEOUT', 'trades': 0, 'win_rate': 0}

def find_best_periods():
    """Test multiple periods to find the most profitable"""
    
    print("🔍 Finding Most Profitable Periods for Auto Discovery")
    print("=" * 60)
    
    # Test different periods
    test_periods = [
        ("2024-03-01", "2024-04-01", "March 2024 (1 month)"),
        ("2024-04-01", "2024-05-01", "April 2024 (1 month)"),
        ("2024-05-01", "2024-06-01", "May 2024 (1 month)"),
        ("2024-03-01", "2024-05-01", "Mar-Apr 2024 (2 months)"),
        ("2024-04-01", "2024-06-01", "Apr-May 2024 (2 months)"),
        ("2024-02-01", "2024-04-01", "Feb-Mar 2024 (2 months)"),
    ]
    
    results = []
    
    for start, end, desc in test_periods:
        result = test_period(start, end, desc)
        results.append(result)
        
        if isinstance(result['return'], (int, float)):
            print(f"✅ {desc}: {result['return']:.1f}% return, {result['trades']} trades")
        else:
            print(f"❌ {desc}: {result['return']}")
    
    print("\n📊 RESULTS SUMMARY:")
    print("=" * 50)
    
    # Sort by return
    valid_results = [r for r in results if isinstance(r['return'], (int, float))]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: x['return'], reverse=True)
        
        print("🏆 TOP PERFORMING PERIODS:")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"{i}. {result['period']}")
            print(f"   Return: {result['return']:.2f}%")
            print(f"   Trades: {result['trades']}")
            print(f"   Win Rate: {result['win_rate']:.1f}%")
            print(f"   Final: ${result['final_balance']:,.2f}")
            print()
        
        best_period = sorted_results[0]
        print(f"🎯 RECOMMENDATION:")
        print(f"Use period: {best_period['start']} to {best_period['end']}")
        print(f"Expected return: {best_period['return']:.1f}%")
        
        # Calculate what this means for scaling up
        if best_period['return'] > 0:
            scaled_return = best_period['return'] * 6  # If we did this 6 times per year
            print(f"🚀 Potential annual return: ~{scaled_return:.0f}% (if repeated 6x)")
    
    else:
        print("❌ No successful periods found")

if __name__ == "__main__":
    find_best_periods() 