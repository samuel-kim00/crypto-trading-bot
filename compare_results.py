#!/usr/bin/env python3
"""Compare buggy vs fixed auto discovery results"""

import json
import glob

def compare_auto_discovery_results():
    """Compare the buggy vs fixed auto discovery results"""
    
    print("🔍 COMPARING AUTO DISCOVERY RESULTS")
    print("=" * 60)
    
    # Find buggy results
    buggy_files = glob.glob("src/analysis/reports/auto_discovery_backtest_*.json")
    fixed_files = glob.glob("src/analysis/reports/fixed_auto_discovery_*.json")
    
    if not buggy_files:
        print("❌ No buggy auto discovery results found")
        return
    
    if not fixed_files:
        print("❌ No fixed auto discovery results found")
        return
    
    # Get latest of each
    latest_buggy = max(buggy_files)
    latest_fixed = max(fixed_files)
    
    # Load results
    with open(latest_buggy, 'r') as f:
        buggy_results = json.load(f)
    
    with open(latest_fixed, 'r') as f:
        fixed_results = json.load(f)
    
    print("🐛 BUGGY VERSION RESULTS:")
    print("-" * 30)
    print(f"Initial Balance:  ${buggy_results['initial_balance']:,.2f}")
    print(f"Final Balance:    ${buggy_results['final_balance']:,.2f}")
    print(f"Total Return:     {buggy_results['total_return']:,.2f}%")
    print(f"Total Trades:     {buggy_results['total_trades']}")
    print(f"Win Rate:         {buggy_results['win_rate']:.1f}%")
    print()
    
    print("✅ FIXED VERSION RESULTS:")
    print("-" * 30)
    print(f"Initial Balance:  ${fixed_results['initial_balance']:,.2f}")
    print(f"Final Balance:    ${fixed_results['final_balance']:,.2f}")
    print(f"Total Return:     {fixed_results['total_return']:,.2f}%")
    print(f"Total Trades:     {fixed_results['total_trades']}")
    print(f"Win Rate:         {fixed_results['win_rate']:.1f}%")
    print()
    
    print("📋 REALITY CHECK:")
    print("-" * 30)
    
    # Check if buggy results are realistic
    buggy_realistic = True
    if buggy_results['total_return'] > 1000:  # More than 10x in short period
        print("❌ Buggy version: Unrealistic returns (>1000%)")
        buggy_realistic = False
    
    if buggy_results['final_balance'] > 1000000:  # More than $1M from $10K
        print("❌ Buggy version: Impossible balance growth")
        buggy_realistic = False
    
    if buggy_realistic:
        print("✅ Buggy version: Results appear realistic")
    
    # Check if fixed results are realistic
    fixed_realistic = True
    if -50 <= fixed_results['total_return'] <= 200:  # Between -50% and +200%
        print("✅ Fixed version: Realistic returns")
    else:
        print("❌ Fixed version: Returns still unrealistic")
        fixed_realistic = False
    
    if 5000 <= fixed_results['final_balance'] <= 50000:  # Reasonable range
        print("✅ Fixed version: Realistic final balance")
    else:
        print("❌ Fixed version: Final balance unrealistic")
        fixed_realistic = False
    
    print()
    print("🎯 CONCLUSION:")
    print("-" * 30)
    if not buggy_realistic and fixed_realistic:
        print("🎉 SUCCESS: Fixed version produces realistic results!")
        print("   The Auto Discovery backtester has been successfully debugged.")
    elif buggy_realistic and fixed_realistic:
        print("✅ Both versions produce realistic results")
    elif not buggy_realistic and not fixed_realistic:
        print("⚠️ Both versions have issues")
    else:
        print("🤔 Mixed results - needs further investigation")

if __name__ == "__main__":
    compare_auto_discovery_results() 