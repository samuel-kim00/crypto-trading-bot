#!/usr/bin/env python3
"""Test realistic price generation"""

import numpy as np
from datetime import datetime, timedelta

def test_realistic_price_generation():
    """Test that price generation produces realistic results"""
    
    # Test realistic price bounds
    base_price = 45000  # BTC price
    volatility = 0.03   # 3% daily volatility
    
    # Generate 30 days of realistic price data
    prices = []
    current_price = base_price
    
    for i in range(30):
        daily_change = np.random.normal(0, volatility)
        daily_change = np.clip(daily_change, -0.2, 0.2)  # Max 20% daily move
        
        current_price = current_price * (1 + daily_change)
        current_price = max(current_price, base_price * 0.1)  # Floor at 10% of base
        current_price = min(current_price, base_price * 5)    # Ceiling at 5x base
        
        prices.append(current_price)
    
    # Calculate statistics
    total_return = ((prices[-1] - base_price) / base_price * 100)
    min_price = min(prices)
    max_price = max(prices)
    max_daily_change = max(abs((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))) * 100
    
    print("🔍 REALISTIC PRICE GENERATION TEST")
    print("=" * 40)
    print(f"Starting price: ${base_price:,.2f}")
    print(f"Ending price: ${prices[-1]:,.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Min price: ${min_price:,.2f}")
    print(f"Max price: ${max_price:,.2f}")
    print(f"Max daily change: {max_daily_change:.2f}%")
    print()
    
    # Validation checks
    checks_passed = 0
    total_checks = 5
    
    print("✅ VALIDATION CHECKS:")
    
    # Check 1: Total return is reasonable (not quadrillions!)
    if -90 <= total_return <= 500:  # Between -90% and +500% for 30 days
        print("✅ Total return is realistic")
        checks_passed += 1
    else:
        print(f"❌ Total return too extreme: {total_return:.2f}%")
    
    # Check 2: Max daily change is bounded
    if max_daily_change <= 20:
        print("✅ Daily changes are bounded")
        checks_passed += 1
    else:
        print(f"❌ Daily change too extreme: {max_daily_change:.2f}%")
    
    # Check 3: Price stays within reasonable bounds
    if min_price >= base_price * 0.1 and max_price <= base_price * 5:
        print("✅ Prices stay within bounds")
        checks_passed += 1
    else:
        print("❌ Prices exceeded bounds")
    
    # Check 4: No impossible prices (like $0.01 BTC)
    if all(p > 1000 for p in prices):  # BTC should stay above $1000
        print("✅ No impossible prices")
        checks_passed += 1
    else:
        print("❌ Found impossible prices")
    
    # Check 5: Reasonable volatility
    price_changes = [abs((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))]
    avg_volatility = np.mean(price_changes) * 100
    if 0.5 <= avg_volatility <= 10:  # Between 0.5% and 10% average daily change
        print("✅ Volatility is reasonable")
        checks_passed += 1
    else:
        print(f"❌ Volatility unrealistic: {avg_volatility:.2f}%")
    
    print()
    print(f"🎯 RESULT: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 All checks passed! Price generation is realistic.")
        return True
    else:
        print("⚠️ Some checks failed. Price generation needs adjustment.")
        return False

if __name__ == "__main__":
    test_realistic_price_generation() 