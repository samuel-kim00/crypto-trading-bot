#!/usr/bin/env python3
import json

with open('src/analysis/reports/auto_discovery_20250619_191741.json', 'r') as f:
    data = json.load(f)

print('🔍 LATEST AUTO DISCOVERY RUN RESULTS:')
print('=' * 40)
print(f'Total Return: {data.get("total_return", "N/A")}%')
print(f'Final Balance: ${data.get("final_balance", "N/A"):,.2f}')
print(f'Total Trades: {data.get("total_trades", "N/A")}')
print(f'Win Rate: {data.get("win_rate", "N/A")}%')

if data.get("total_return", 0) > 1000:
    print('\n❌ THIS IS FROM THE BUGGY VERSION!')
    print('The dashboard is still using the broken backtester.')
else:
    print('\n✅ This looks like fixed results.') 