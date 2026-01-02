#!/usr/bin/env python3
"""
System Test Script
Tests all major components of the trading bot system
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestSystemComponents(unittest.TestCase):
    
    def test_performance_tracker_import(self):
        """Test that PerformanceTracker can be imported and initialized"""
        from analysis.performance_tracker import PerformanceTracker
        tracker = PerformanceTracker()
        self.assertIsNotNone(tracker)
        
    def test_media_analyzer_import(self):
        """Test that MediaAnalyzer can be imported and initialized"""
        from analysis.media_analyzer_v2 import MediaAnalyzer
        analyzer = MediaAnalyzer()
        self.assertIsNotNone(analyzer)
        
    @patch('ccxt.binance')
    def test_trading_bot_import(self, mock_binance):
        """Test that TradingBot can be imported (with mocked exchange)"""
        mock_exchange = MagicMock()
        mock_binance.return_value = mock_exchange
        
        from core.trading_bot import TradingBot
        # This should not fail during import
        self.assertTrue(True)
        
    def test_config_files_exist(self):
        """Test that all required config files exist"""
        config_files = [
            'config/strategy_config.json',
            'config/active_positions.json',
            'config/strategy_status.json',
            'config/schedule_config.json'
        ]
        
        for config_file in config_files:
            full_path = os.path.join(os.path.dirname(__file__), '..', config_file)
            self.assertTrue(os.path.exists(full_path), f"Config file {config_file} not found")
            
    def test_data_directories_exist(self):
        """Test that all required data directories exist"""
        data_dirs = [
            'data',
            'logs',
            'reports'
        ]
        
        for data_dir in data_dirs:
            full_path = os.path.join(os.path.dirname(__file__), '..', data_dir)
            self.assertTrue(os.path.exists(full_path), f"Data directory {data_dir} not found")
            
    def test_python_packages_structure(self):
        """Test that Python packages have __init__.py files"""
        package_dirs = [
            'src',
            'src/core',
            'src/analysis', 
            'src/dashboard',
            'src/utils'
        ]
        
        for package_dir in package_dirs:
            init_file = os.path.join(os.path.dirname(__file__), '..', package_dir, '__init__.py')
            self.assertTrue(os.path.exists(init_file), f"Missing __init__.py in {package_dir}")

if __name__ == '__main__':
    print("Running system component tests...")
    unittest.main(verbosity=2) 