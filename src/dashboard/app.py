import os
import sys
import json
import time
import asyncio
import tempfile
import traceback
import logging
import glob
from datetime import datetime
from threading import Lock

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np

# Add utils directory to path for PDF generator
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))

# Configure Flask app with correct template directory
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
socketio = SocketIO(app)

# Global variables
data_lock = Lock()
initial_budget = 1000.0
performance_data = {
    'trades': [],
    'active_trades': [],
    'daily_stats': {'trades': 0, 'win_rate': 0, 'pnl': 0},
    'weekly_stats': {'trades': 0, 'win_rate': 0, 'pnl': 0},
    'monthly_stats': {'trades': 0, 'win_rate': 0, 'pnl': 0}
}

# Configure logging
logging.basicConfig(level=logging.INFO)

def _make_json_serializable(obj):
    """Convert pandas/numpy objects to JSON serializable format"""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle infinity and NaN values
        if np.isinf(obj):
            return None if np.isneginf(obj) else 999999999  # Large number instead of infinity
        elif np.isnan(obj):
            return None
        else:
            return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj == float('inf'):
        return 999999999  # Large number instead of infinity
    elif obj == float('-inf'):
        return -999999999  # Large negative number instead of negative infinity
    elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return None
    else:
        return obj

def load_initial_budget():
    global initial_budget
    try:
        # First try to get REAL balance from lightweight bot heartbeat
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        heartbeat_path = os.path.join(project_root, 'logs', 'trading_bot_heartbeat.json')
        
        if os.path.exists(heartbeat_path):
            with open(heartbeat_path, 'r') as f:
                heartbeat_data = json.load(f)
                if heartbeat_data.get('mode') == 'lightweight_real_balance':
                    real_balance = heartbeat_data.get('total_balance_usd', 0)
                    if real_balance > 0:
                        initial_budget = float(real_balance)
                        print(f"✅ Loaded REAL balance from lightweight bot: ${initial_budget:.2f}")
                        return
        
        # Fallback to config file
        config_path = os.path.join(project_root, 'config', 'strategy_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            initial_budget = float(config.get('initial_budget', 9.77))
        print(f"✅ Loaded initial budget from config: ${initial_budget:.2f}")
    except Exception as e:
        print(f"Error loading initial budget: {str(e)}")
        initial_budget = 9.77  # Set to actual starting amount

def load_performance_data():
    global performance_data
    try:
        # Get project root directory (two levels up from src/dashboard/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(project_root, 'data', 'performance_data.json')
        
        with open(data_path, 'r') as f:
            performance_data = json.load(f)
    except Exception as e:
        print(f"Error loading performance data: {str(e)}")

def update_dashboard_data():
    with data_lock:
        # Try to get REAL balance from lightweight bot heartbeat
        real_balance = None
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            heartbeat_path = os.path.join(project_root, 'logs', 'trading_bot_heartbeat.json')
            
            if os.path.exists(heartbeat_path):
                with open(heartbeat_path, 'r') as f:
                    heartbeat_data = json.load(f)
                    if heartbeat_data.get('mode') == 'lightweight_real_balance':
                        real_balance = heartbeat_data.get('total_balance_usd', 0)
                        if real_balance > 0:
                            print(f"📊 Using REAL balance from lightweight bot: ${real_balance:.2f}")
        except Exception as e:
            print(f"Error reading real balance: {str(e)}")
        
        # Calculate current balance - use real balance if available
        if real_balance and real_balance > 0:
            current_balance = real_balance
            total_pnl = current_balance - initial_budget
        else:
            # Fallback to calculated balance from trades
            total_pnl = sum(trade.get('pnl', 0) for trade in performance_data.get('trades', []))
            current_balance = initial_budget + total_pnl
        
        # Calculate win rate
        total_trades = len(performance_data.get('trades', []))
        winning_trades = len([t for t in performance_data.get('trades', []) if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'initial_budget': initial_budget,
            'current_balance': current_balance,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'active_trades': performance_data.get('active_trades', []),
            'trade_history': performance_data.get('trades', []),
            'daily_stats': performance_data.get('daily_stats', {'trades': 0, 'win_rate': 0, 'pnl': 0}),
            'weekly_stats': performance_data.get('weekly_stats', {'trades': 0, 'win_rate': 0, 'pnl': 0}),
            'monthly_stats': performance_data.get('monthly_stats', {'trades': 0, 'win_rate': 0, 'pnl': 0}),
            'real_balance_active': real_balance is not None and real_balance > 0
        }

def background_update():
    """Background task to update dashboard data"""
    while True:
        try:
            dashboard_data = update_dashboard_data()
            socketio.emit('dashboard_update', dashboard_data)
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Background update error: {str(e)}")
            time.sleep(5)  # Update every 5 seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/performance')
def get_performance():
    """Get current performance data"""
    try:
        dashboard_data = update_dashboard_data()
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@socketio.on('connect')
def handle_connect():
    try:
        dashboard_data = update_dashboard_data()
        emit('dashboard_update', dashboard_data)
    except Exception as e:
        print(f"Connect error: {str(e)}")

@app.route('/api/enhanced_report')
def get_enhanced_report():
    """Get the latest enhanced weekly trading report"""
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Check both possible locations for reports
        reports_path1 = os.path.join(project_root, 'reports', 'enhanced_report_*.json')
        reports_path2 = os.path.join(project_root, 'src', 'analysis', 'reports', 'enhanced_report_*.json')
        
        report_files = glob.glob(reports_path1) + glob.glob(reports_path2)
        if not report_files:
            return jsonify({'error': 'No enhanced reports found'})
        
        latest_report = max(report_files)
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/generate_report')
def generate_new_report():
    """Generate a new enhanced weekly report on demand"""
    try:
        # Add proper path for analysis module
        analysis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
            
        from enhanced_predictor import EnhancedPredictor
        
        async def generate():
            predictor = EnhancedPredictor()
            return await predictor.generate_enhanced_report()
        
        report = asyncio.run(generate())
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/weekly_report')
def get_weekly_report():
    """Get the latest weekly trading report (fallback)"""
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Check both possible locations for reports
        reports_path1 = os.path.join(project_root, 'reports', 'weekly_report_*.json')
        reports_path2 = os.path.join(project_root, 'src', 'analysis', 'reports', 'weekly_report_*.json')
        
        report_files = glob.glob(reports_path1) + glob.glob(reports_path2)
        if not report_files:
            # Try enhanced reports
            return get_enhanced_report()
        
        latest_report = max(report_files)
        with open(latest_report, 'r') as f:
            report = json.load(f)
        
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/download_report_pdf')
def download_report_pdf():
    """Generate and download the latest report as PDF"""
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Get the latest enhanced report from both possible locations
        reports_path1 = os.path.join(project_root, 'reports', 'enhanced_report_*.json')
        reports_path2 = os.path.join(project_root, 'src', 'analysis', 'reports', 'enhanced_report_*.json')
        
        report_files = glob.glob(reports_path1) + glob.glob(reports_path2)
        if not report_files:
            return jsonify({'error': 'No reports found to convert to PDF'})
        
        latest_report = max(report_files)
        with open(latest_report, 'r') as f:
            report_data = json.load(f)
        
        # Generate PDF using direct path
        from pdf_generator import WeeklyReportPDFGenerator
        
        # Create PDF in temp directory first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"weekly_report_{timestamp}.pdf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, pdf_filename)
            
            # Generate PDF directly to temp path
            generator = WeeklyReportPDFGenerator()
            generator.generate_pdf(report_data, temp_pdf_path)
            
            return send_file(
                temp_pdf_path,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype='application/pdf'
            )
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/generate_and_download_pdf')
def generate_and_download_pdf():
    """Generate new report and immediately download as PDF"""
    try:
        # Add proper path for analysis module
        analysis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
            
        from enhanced_predictor import EnhancedPredictor
        
        async def generate():
            predictor = EnhancedPredictor()
            return await predictor.generate_enhanced_report()
        
        # Generate new report
        report_data = asyncio.run(generate())
        
        # Generate PDF using direct path
        from pdf_generator import WeeklyReportPDFGenerator
        
        # Create PDF in temp directory first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"weekly_report_{timestamp}.pdf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, pdf_filename)
            
            # Generate PDF directly to temp path
            generator = WeeklyReportPDFGenerator()
            generator.generate_pdf(report_data, temp_pdf_path)
            
            return send_file(
                temp_pdf_path,
                as_attachment=True,
                download_name=pdf_filename,
                mimetype='application/pdf'
            )
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/backtest')
def backtest_page():
    """Serve the backtest dashboard page"""
    return render_template('backtest.html')

@app.route('/ai-training')
def ai_training_page():
    """Serve the AI training dashboard page"""
    return render_template('ai_training.html')

@app.route('/api/run_backtest', methods=['POST'])
def run_backtest_api():
    """Run backtest with given parameters - supports both basic and adaptive modes"""
    try:
        print(f"[BACKTEST] Starting backtest API call...")
        sys.path.append('src/analysis')
        
        data = request.get_json()
        print(f"[BACKTEST] Request data: {data}")
        
        backtest_mode = data.get('mode', 'adaptive')  # Default to adaptive
        print(f"[BACKTEST] Mode: {backtest_mode}")
        
        if backtest_mode == 'adaptive':
            print(f"[BACKTEST] Running adaptive mode...")
            # Use the new adaptive backtester with live strategy + ML
            # Add the project root and analysis directory to the path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            analysis_path = os.path.join(project_root, 'src', 'analysis')
            if analysis_path not in sys.path:
                sys.path.insert(0, analysis_path)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            print(f"[BACKTEST] Importing AdaptiveBacktester...")
            from adaptive_backtester import AdaptiveBacktester
            
            print(f"[BACKTEST] Creating backtester instance...")
            backtester = AdaptiveBacktester(
                initial_balance=data.get('initial_balance', 10000)
            )
            
            async def run_adaptive_backtest():
                return await backtester.run_adaptive_backtest(
                    symbols=data.get('symbols', ['BTCUSDT', 'ETHUSDT']),
                    start_date=data.get('start_date', '2024-01-01'),
                    end_date=data.get('end_date', '2024-06-01')
                )
            
            print(f"[BACKTEST] Running adaptive backtest...")
            results = asyncio.run(run_adaptive_backtest())
            print(f"[BACKTEST] Adaptive backtest completed")
            
            # Save results and return the saved (serialized) version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = backtester.save_adaptive_results(results, f"adaptive_backtest_{timestamp}.json")
            
            # Load the saved results to ensure proper serialization
            with open(filepath, 'r') as f:
                saved_results = json.load(f)
            
            # Ensure results are JSON serializable (handle Infinity/NaN values)
            serializable_results = _make_json_serializable(saved_results)
            print(f"[BACKTEST] Returning adaptive results")
            
            return jsonify(serializable_results)
        
        else:
            print(f"[BACKTEST] Running basic mode...")
            # Use basic backtester
            # Add the project root and analysis directory to the path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            analysis_path = os.path.join(project_root, 'src', 'analysis')
            if analysis_path not in sys.path:
                sys.path.insert(0, analysis_path)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            print(f"[BACKTEST] Importing VLMBacktester...")
            from backtester import VLMBacktester
            
            print(f"[BACKTEST] Creating backtester instance...")
            # Initialize backtester
            backtester = VLMBacktester(initial_balance=data.get('initial_balance', 10000))
            
            async def run_basic_backtest():
                return await backtester.run_backtest(
                    symbols=data.get('symbols', ['BTCUSDT']),
                    start_date=data.get('start_date', '2024-01-01'),
                    end_date=data.get('end_date', '2024-06-01')
                )
            
            print(f"[BACKTEST] Running basic backtest...")
            results = asyncio.run(run_basic_backtest())
            print(f"[BACKTEST] Basic backtest completed, type: {type(results)}")
            
            # Ensure results are JSON serializable (handle Infinity/NaN values)
            print(f"[BACKTEST] Making results JSON serializable...")
            serializable_results = _make_json_serializable(results)
            print(f"[BACKTEST] Serialization completed")
            
            print(f"[BACKTEST] Returning basic results")
            return jsonify(serializable_results)
            
    except Exception as e:
        print(f"[BACKTEST] ERROR: {str(e)}")
        print(f"[BACKTEST] ERROR TYPE: {type(e)}")
        import traceback
        print(f"[BACKTEST] TRACEBACK: {traceback.format_exc()}")
        return jsonify({'error': str(e)})

@app.route('/api/latest_backtest')
def get_latest_backtest():
    """Get the latest backtest results"""
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Check both possible locations for backtest results
        backtest_path1 = os.path.join(project_root, 'reports', 'backtest_*.json')
        backtest_path2 = os.path.join(project_root, 'src', 'analysis', 'reports', 'backtest_*.json')
        
        backtest_files = glob.glob(backtest_path1) + glob.glob(backtest_path2)
        if not backtest_files:
            return jsonify({'error': 'No backtest results found'})
        
        latest_backtest = max(backtest_files)
        with open(latest_backtest, 'r') as f:
            results = json.load(f)
        
        # Ensure results are JSON serializable
        serializable_results = _make_json_serializable(results)
        return jsonify(serializable_results)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/download_backtest_results')
def download_backtest_results():
    """Download latest backtest results as JSON"""
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Get the latest backtest results
        backtest_path1 = os.path.join(project_root, 'reports', 'backtest_*.json')
        backtest_path2 = os.path.join(project_root, 'src', 'analysis', 'reports', 'backtest_*.json')
        
        backtest_files = glob.glob(backtest_path1) + glob.glob(backtest_path2)
        if not backtest_files:
            return jsonify({'error': 'No backtest results found'})
        
        latest_backtest = max(backtest_files)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return send_file(
            latest_backtest,
            as_attachment=True,
            download_name=f"backtest_results_{timestamp}.json",
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/run_auto_discovery', methods=['POST'])
def run_auto_discovery_api():
    """Run auto-discovery backtest to find optimal strategies"""
    try:
        logging.info("Starting auto-discovery backtest")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        data = request.get_json()
        
        # Calculate expected duration and set timeout
        start_date = datetime.strptime(data.get('start_date', '2024-01-01'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date', '2024-06-01'), '%Y-%m-%d')
        duration_days = (end_date - start_date).days
        
        # Set timeout based on duration (more generous timeouts after optimization)
        if duration_days > 150:  # 5+ months
            timeout_minutes = 25
        elif duration_days > 90:  # 3+ months
            timeout_minutes = 20
        elif duration_days > 30:   # 1+ months
            timeout_minutes = 12  # Increased for 1-month periods
        else:
            timeout_minutes = 8   # Even short periods need more time
        
        logging.info(f"Auto-discovery parameters - Period: {data.get('start_date')} to {data.get('end_date')}")
        logging.info(f"Duration: {duration_days} days, Timeout: {timeout_minutes} minutes")
        logging.info(f"Balance: ${data.get('initial_balance', 10000):,.2f}, Max positions: {data.get('max_positions', 3)}")
        
        # Provide runtime estimation
        if duration_days > 150:
            estimated_time = "15-20 minutes"
        elif duration_days > 90:
            estimated_time = "10-15 minutes"
        elif duration_days > 30:
            estimated_time = "6-10 minutes"  # Updated estimate
        elif duration_days > 7:
            estimated_time = "3-5 minutes"
        else:
            estimated_time = "1-3 minutes"
        
        logging.info(f"Estimated completion time: {estimated_time}")
        
        # Import and initialize the FIXED auto-discovery backtester
        try:
            # Use proper import instead of exec to avoid issues
            sys.path.insert(0, 'src/analysis')
            from auto_discovery_backtester_fixed import FixedAutoDiscoveryBacktester
            
            backtester = FixedAutoDiscoveryBacktester(
                initial_balance=data.get('initial_balance', 10000)
            )
            
            symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT'])
            start_date_str = data.get('start_date', '2024-01-01')
            end_date_str = data.get('end_date', '2024-06-01')
            
        except Exception as e:
            logging.error(f"Import error: {e}")
            return jsonify({'success': False, 'error': f'Could not import FIXED auto-discovery backtester: {str(e)}'})
        
        # Run the FIXED auto-discovery backtest with timeout
        try:
            logging.info(f"Running FIXED auto-discovery backtest with {timeout_minutes}-minute timeout...")
            
            async def run_fixed_auto_discovery_with_timeout():
                return await asyncio.wait_for(
                    backtester.run_fixed_auto_discovery_backtest(
                        start_date=start_date_str,
                        end_date=end_date_str
                    ),
                    timeout=timeout_minutes * 60  # Convert to seconds
                )
            
            results = asyncio.run(run_fixed_auto_discovery_with_timeout())
            logging.info("FIXED Auto-discovery completed successfully")
            
            # Save results
            logging.debug("Saving FIXED auto-discovery results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = backtester.save_fixed_results(results, f"fixed_auto_discovery_{timestamp}.json")
            logging.info(f"Saved to {filepath}")
            
            # Ensure results are JSON serializable (handle Infinity/NaN values)
            serializable_results = _make_json_serializable(results)
            
            logging.info("Returning FIXED results...")
            # Return in the format expected by the frontend
            return jsonify({
                'success': True,
                'message': 'FIXED Auto-discovery backtest completed successfully with realistic results!',
                'results': serializable_results,
                'note': 'Now using realistic price generation and proper risk management'
            })
            
        except asyncio.TimeoutError:
            logging.error(f"Auto-discovery timed out after {timeout_minutes} minutes")
            return jsonify({
                'success': False,
                'error': f'Auto-discovery backtest timed out ({timeout_minutes} minutes). Try a shorter date range or reduce the number of symbols.',
                'timeout_minutes': timeout_minutes,
                'duration_days': duration_days,
                'suggestion': 'For 6-month backtests, try breaking into 2-3 month periods or use fewer symbols.'
            })
            
        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            logging.error(f"Exception type: {type(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'success': False, 'error': str(e)})
            
    except Exception as e:
        logging.error(f"Failed to run auto-discovery: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run_fixed_auto_discovery', methods=['POST'])
def run_fixed_auto_discovery_api():
    """Run the FIXED auto-discovery backtest with realistic results"""
    try:
        logging.info("Starting FIXED auto-discovery backtest")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        data = request.get_json()
        
        # Calculate expected duration and set timeout
        start_date = datetime.strptime(data.get('start_date', '2024-01-01'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date', '2024-03-01'), '%Y-%m-%d')
        duration_days = (end_date - start_date).days
        
        logging.info(f"Fixed auto-discovery parameters - Period: {data.get('start_date')} to {data.get('end_date')}")
        logging.info(f"Duration: {duration_days} days, Balance: ${data.get('initial_balance', 10000):,.2f}")
        
        # Import the FIXED auto-discovery backtester
        try:
            # Create the fixed backtester inline to avoid import issues
            exec(open('src/analysis/auto_discovery_backtester_fixed.py').read(), globals())
            
            backtester = FixedAutoDiscoveryBacktester(
                initial_balance=data.get('initial_balance', 10000)
            )
            
            symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT'])
            start_date_str = data.get('start_date', '2024-01-01')
            end_date_str = data.get('end_date', '2024-03-01')
            
        except Exception as e:
            logging.error(f"Import error: {e}")
            return jsonify({'success': False, 'error': f'Could not import fixed auto-discovery backtester: {str(e)}'})
        
        # Run the FIXED auto-discovery backtest
        try:
            logging.info(f"Running FIXED auto-discovery backtest...")
            
            async def run_fixed_auto_discovery():
                return await backtester.run_fixed_auto_discovery_backtest(
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            
            results = asyncio.run(run_fixed_auto_discovery())
            logging.info("FIXED auto-discovery completed successfully")
            
            # Save results
            logging.debug("Saving fixed auto-discovery results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = backtester.save_fixed_results(results, f"fixed_auto_discovery_{timestamp}.json")
            logging.info(f"Saved to {filepath}")
            
            # Ensure results are JSON serializable
            serializable_results = _make_json_serializable(results)
            
            logging.info("Returning FIXED results...")
            # Return in the format expected by the frontend
            return jsonify({
                'success': True,
                'message': 'FIXED Auto-discovery backtest completed successfully with realistic results!',
                'results': serializable_results,
                'note': 'This version uses realistic price generation and proper risk management'
            })
            
        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            logging.error(f"Exception type: {type(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'success': False, 'error': str(e)})
            
    except Exception as e:
        logging.error(f"Failed to run FIXED auto-discovery: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run_optimized_auto_discovery', methods=['POST'])
def run_optimized_auto_discovery_api():
    """Run the OPTIMIZED auto-discovery backtest for higher returns"""
    try:
        logging.info("Starting OPTIMIZED auto-discovery backtest for higher returns")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        data = request.get_json()
        
        # Calculate expected duration and set timeout
        start_date = datetime.strptime(data.get('start_date', '2024-03-01'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date', '2024-04-01'), '%Y-%m-%d')
        duration_days = (end_date - start_date).days
        
        logging.info(f"Optimized auto-discovery parameters - Period: {data.get('start_date')} to {data.get('end_date')}")
        logging.info(f"Duration: {duration_days} days, Balance: ${data.get('initial_balance', 10000):,.2f}")
        logging.info("🚀 Using AGGRESSIVE parameters for higher returns!")
        
        # Import the OPTIMIZED auto-discovery backtester
        try:
            # Load the optimized backtester
            exec(open('src/analysis/auto_discovery_backtester_optimized.py').read(), globals())
            
            backtester = OptimizedAutoDiscoveryBacktester(
                initial_balance=data.get('initial_balance', 10000)
            )
            
            start_date_str = data.get('start_date', '2024-03-01')
            end_date_str = data.get('end_date', '2024-04-01')
            
        except Exception as e:
            logging.error(f"Import error: {e}")
            return jsonify({'success': False, 'error': f'Could not import optimized auto-discovery backtester: {str(e)}'})
        
        # Run the OPTIMIZED auto-discovery backtest
        try:
            logging.info(f"Running OPTIMIZED auto-discovery backtest...")
            
            async def run_optimized_auto_discovery():
                return await backtester.run_optimized_auto_discovery_backtest(
                    start_date=start_date_str,
                    end_date=end_date_str
                )
            
            results = asyncio.run(run_optimized_auto_discovery())
            logging.info("OPTIMIZED auto-discovery completed successfully")
            
            # Save results
            logging.debug("Saving optimized auto-discovery results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = backtester.save_optimized_results(results, f"optimized_auto_discovery_{timestamp}.json")
            logging.info(f"Saved to {filepath}")
            
            # Ensure results are JSON serializable
            serializable_results = _make_json_serializable(results)
            
            logging.info("Returning OPTIMIZED results...")
            # Return in the format expected by the frontend
            return jsonify({
                'success': True,
                'message': 'OPTIMIZED Auto-discovery backtest completed with AGGRESSIVE parameters for higher returns!',
                'results': serializable_results,
                'note': 'This version uses 20% position sizing, 25% take profits, and 3% stop losses for maximum returns',
                'optimization_details': {
                    'position_size': '20% (vs 10% conservative)',
                    'take_profit': '25% (vs 15% conservative)', 
                    'stop_loss': '3% (vs 5% conservative)',
                    'confidence_threshold': '30 (vs 40 conservative)',
                    'max_positions': '4 (vs 3 conservative)'
                }
            })
            
        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            logging.error(f"Exception type: {type(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'success': False, 'error': str(e)})
            
    except Exception as e:
        logging.error(f"Failed to run OPTIMIZED auto-discovery: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_self_learning_bot', methods=['POST'])
def train_self_learning_bot_api():
    """Train the self-learning AI trading bot"""
    try:
        logging.info("🧠 Starting Self-Learning Bot Training...")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        data = request.get_json()
        symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT'])
        
        logging.info(f"Training self-learning bot on symbols: {symbols}")
        
        # Import self-learning integration
        from self_learning_integration import train_self_learning_bot
        
        # Run training
        async def run_training():
            return await train_self_learning_bot(symbols)
        
        results = asyncio.run(run_training())
        
        if results['success']:
            logging.info("✅ Self-learning bot training completed successfully!")
            return jsonify({
                'success': True,
                'message': 'Self-Learning AI Bot trained successfully with multiple AI techniques!',
                'results': results,
                'ai_methods': [
                    'Deep Q-Network (Reinforcement Learning)',
                    'Genetic Algorithm (Strategy Evolution)', 
                    'Pattern Recognition Neural Network',
                    'Ensemble Decision Making'
                ],
                'capabilities': [
                    'Autonomous strategy development',
                    'Continuous learning from market data',
                    'Multi-model ensemble predictions',
                    'Risk-adjusted position sizing',
                    'Real-time adaptation'
                ]
            })
        else:
            return jsonify({
                'success': False,
                'error': results.get('error', 'Training failed'),
                'training_duration': results.get('training_duration', 0)
            })
            
    except Exception as e:
        logging.error(f"Failed to train self-learning bot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/run_self_learning_backtest', methods=['POST'])
def run_self_learning_backtest_api():
    """Run backtest using self-learning AI models"""
    try:
        logging.info("🤖 Starting Self-Learning AI Backtest...")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        data = request.get_json()
        symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        start_date = data.get('start_date', '2024-01-01')
        end_date = data.get('end_date', '2024-03-01')
        
        logging.info(f"Self-learning backtest: {start_date} to {end_date} on {symbols}")
        
        # Import self-learning integration
        from self_learning_integration import run_self_learning_backtest
        
        # Run backtest
        async def run_ai_backtest():
            return await run_self_learning_backtest(symbols, start_date, end_date)
        
        results = asyncio.run(run_ai_backtest())
        
        if results['success']:
            logging.info("✅ Self-learning backtest completed successfully!")
            
            # Ensure results are JSON serializable
            serializable_results = _make_json_serializable(results)
            
            return jsonify({
                'success': True,
                'message': 'Self-Learning AI Backtest completed with autonomous decision making!',
                'results': serializable_results,
                'ai_insights': {
                    'strategy_type': 'Autonomous AI Ensemble',
                    'decision_making': 'Multi-model consensus voting',
                    'learning_methods': results.get('ai_methods_used', []),
                    'confidence_based_trading': True,
                    'real_time_adaptation': True
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': results.get('error', 'Backtest failed'),
                'suggestion': results.get('suggestion', 'Check if models are trained')
            })
            
    except Exception as e:
        logging.error(f"Failed to run self-learning backtest: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_ai_prediction/<symbol>')
def get_ai_prediction_api(symbol):
    """Get live AI prediction for a symbol"""
    try:
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import self-learning integration
        from self_learning_integration import get_ai_prediction
        
        # Get prediction
        async def get_prediction():
            return await get_ai_prediction(symbol)
        
        prediction = asyncio.run(get_prediction())
        
        if 'error' not in prediction:
            return jsonify({
                'success': True,
                'prediction': prediction,
                'ai_analysis': {
                    'multi_model_ensemble': True,
                    'confidence_threshold': 0.7,
                    'real_time_processing': True
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': prediction['error'],
                'symbol': symbol
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'symbol': symbol})

@app.route('/api/self_learning_status')
def get_self_learning_status_api():
    """Get self-learning bot training status"""
    try:
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import self-learning integration
        from self_learning_integration import get_self_learning_status, get_ai_model_info
        
        status = get_self_learning_status()
        model_info = get_ai_model_info()
        
        # Check for auto-training status
        auto_training_status = {}
        try:
            from auto_training_scheduler import AutoTrainingScheduler
            scheduler = AutoTrainingScheduler()
            auto_training_status = scheduler.get_training_status()
        except:
            auto_training_status = {
                'is_training': False,
                'training_count': 0,
                'training_log': []
            }
        
        return jsonify({
            'success': True,
            'training_status': status,
            'model_info': model_info,
            'auto_training': auto_training_status,
            'capabilities': {
                'reinforcement_learning': 'Deep Q-Network for autonomous trading decisions',
                'genetic_evolution': 'Strategy parameter optimization through evolution',
                'pattern_recognition': 'Neural network for market pattern identification',
                'ensemble_voting': 'Multi-model consensus for final decisions',
                'continuous_learning': 'Real-time adaptation to market changes'
            },
            'learning_indicators': {
                'models_trained': model_info.get('training_summary', {}).get('total_models', 0) > 0,
                'genetic_evolved': model_info.get('genetic_algorithm', {}).get('evolved', False),
                'patterns_learned': model_info.get('pattern_recognition', {}).get('trained', False),
                'ensemble_ready': all([
                    model_info.get('dqn_model', {}).get('trained', False),
                    model_info.get('genetic_algorithm', {}).get('evolved', False),
                    model_info.get('pattern_recognition', {}).get('trained', False)
                ])
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_auto_training', methods=['POST'])
def start_auto_training_api():
    """Start automatic AI training in background"""
    try:
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from auto_training_scheduler import AutoTrainingScheduler
        import threading
        
        # Start training in background
        scheduler = AutoTrainingScheduler()
        thread = threading.Thread(target=scheduler.start_training_now, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Auto-training started in background! Check status for progress.',
            'note': 'Training will run automatically and update models continuously'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ai_training_log')
def get_ai_training_log_api():
    """Get AI training progress log"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        
        from auto_training_scheduler import AutoTrainingScheduler
        scheduler = AutoTrainingScheduler()
        status = scheduler.get_training_status()
        
        return jsonify({
            'success': True,
            'training_log': status.get('training_log', []),
            'is_training': status.get('is_training', False),
            'current_progress': status.get('current_progress', 0),
            'training_count': status.get('training_count', 0),
            'last_training': status.get('last_training', None)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_advanced_ai', methods=['POST'])
def train_advanced_ai_api():
    """Train advanced market intelligence AI with multiple data sources"""
    try:
        logging.info("🌐 Starting Advanced Market Intelligence AI Training...")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from advanced_market_ai import train_advanced_market_ai
        import threading
        
        # Start advanced training in background
        thread = threading.Thread(target=lambda: asyncio.run(train_advanced_market_ai()), daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Advanced Market Intelligence AI training started!',
            'features': [
                '📰 News sentiment analysis from multiple sources',
                '📊 Macro economic correlation analysis', 
                '📱 Social media sentiment tracking',
                '📈 Advanced chart pattern recognition',
                '🔗 Multi-asset correlation analysis',
                '🏛️ Market regime classification',
                '🧠 Situation-aware model training'
            ],
            'training_scenarios': [
                'Bull market momentum patterns',
                'Bear market reversal signals',
                'Sideways consolidation strategies',
                'High volatility event responses', 
                'Macro correlation events'
            ],
            'note': 'Training adapts to various market situations and external factors'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/advanced_ai_status')
def get_advanced_ai_status_api():
    """Get advanced AI training status and intelligence data"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_path = os.path.join(project_root, 'data', 'advanced_ai_training_results.json')
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                
            return jsonify({
                'success': True,
                'training_completed': True,
                'training_summary': results.get('training_summary', {}),
                'market_intelligence': {
                    'news_sentiment': results.get('intelligence_data', {}).get('news_sentiment', {}),
                    'market_regime': results.get('intelligence_data', {}).get('market_regimes', {}),
                    'macro_indicators': results.get('intelligence_data', {}).get('macro_events', {}),
                    'pattern_analysis': results.get('intelligence_data', {}).get('technical_patterns', {}),
                    'correlations': results.get('intelligence_data', {}).get('correlation_matrix', {})
                },
                'trained_scenarios': list(results.get('trained_models', {}).keys()),
                'last_update': results.get('timestamp', 'Unknown')
            })
        else:
            return jsonify({
                'success': True,
                'training_completed': False,
                'message': 'Advanced AI training not yet completed',
                'status': 'Training in progress or not started'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/gather_market_intelligence', methods=['POST'])
def gather_market_intelligence_api():
    """Gather market intelligence from YouTube and web sources"""
    try:
        logging.info("🌐 Starting market intelligence gathering...")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from youtube_market_intelligence import gather_market_intelligence
        import threading
        
        # Start intelligence gathering in background
        thread = threading.Thread(target=lambda: asyncio.run(gather_market_intelligence()), daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Market intelligence gathering started!',
            'sources': [
                '📹 YouTube crypto channels analysis',
                '📰 Crypto news sentiment analysis',
                '📊 Social media trends tracking',
                '🔍 Market-moving news detection',
                '📈 Price prediction extraction',
                '🎯 Market consensus determination'
            ],
            'note': 'Gathering intelligence from multiple sources to understand market sentiment'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market_intelligence_status')
def get_market_intelligence_status_api():
    """Get market intelligence status and data"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        intelligence_path = os.path.join(project_root, 'data', 'market_intelligence.json')
        
        if os.path.exists(intelligence_path):
            with open(intelligence_path, 'r') as f:
                intelligence = json.load(f)
                
            return jsonify({
                'success': True,
                'intelligence_available': True,
                'combined_sentiment': intelligence.get('combined_sentiment', {}),
                'market_consensus': intelligence.get('market_consensus', {}),
                'youtube_intelligence': intelligence.get('youtube_intelligence', {}),
                'web_intelligence': intelligence.get('web_intelligence', {}),
                'last_update': intelligence.get('timestamp', 'Unknown')
            })
        else:
            return jsonify({
                'success': True,
                'intelligence_available': False,
                'message': 'Market intelligence not yet gathered',
                'status': 'Gathering in progress or not started'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_comprehensive_ai_training', methods=['POST'])
def start_comprehensive_ai_training_api():
    """Start comprehensive AI training with all intelligence sources"""
    try:
        logging.info("🚀 Starting comprehensive AI training with all intelligence sources...")
        
        # Add the project root and analysis directory to the path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        analysis_path = os.path.join(project_root, 'src', 'analysis')
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Start comprehensive training process
        async def comprehensive_training():
            # Step 1: Gather market intelligence
            from youtube_market_intelligence import gather_market_intelligence
            intelligence_data = await gather_market_intelligence()
            
            # Step 2: Train advanced AI
            from advanced_market_ai import train_advanced_market_ai
            ai_results = await train_advanced_market_ai()
            
            # Step 3: Train self-learning bot with new intelligence
            from self_learning_integration import train_self_learning_bot
            bot_results = await train_self_learning_bot(['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
            
            return {
                'intelligence_data': intelligence_data,
                'ai_results': ai_results,
                'bot_results': bot_results
            }
        
        import threading
        thread = threading.Thread(target=lambda: asyncio.run(comprehensive_training()), daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Comprehensive AI training started with all intelligence sources!',
            'training_pipeline': [
                '1. 📹 YouTube & Web Intelligence Gathering',
                '2. 🧠 Advanced Market AI Training',
                '3. 🤖 Self-Learning Bot Enhancement',
                '4. 📊 Pattern Recognition Training',
                '5. 🔗 Multi-Asset Correlation Analysis',
                '6. 🎯 Situation-Aware Model Development'
            ],
            'expected_duration': '20-30 minutes',
            'note': 'Training will adapt to various market situations using external intelligence'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/real_balance_status')
def get_real_balance_status():
    """Get real balance status from lightweight bot"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        heartbeat_path = os.path.join(project_root, 'logs', 'trading_bot_heartbeat.json')
        
        if os.path.exists(heartbeat_path):
            with open(heartbeat_path, 'r') as f:
                heartbeat_data = json.load(f)
                
            return jsonify({
                'success': True,
                'real_balance_active': heartbeat_data.get('mode') == 'lightweight_real_balance',
                'total_balance_usd': heartbeat_data.get('total_balance_usd', 0),
                'usdt_balance': heartbeat_data.get('usdt_balance', 0),
                'last_update': heartbeat_data.get('timestamp', 0),
                'status': heartbeat_data.get('status', 'unknown'),
                'memory_usage': heartbeat_data.get('memory_usage', 0),
                'mode': heartbeat_data.get('mode', 'unknown')
            })
        else:
            return jsonify({
                'success': False,
                'real_balance_active': False,
                'error': 'No heartbeat file found - lightweight bot may not be running'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'real_balance_active': False,
            'error': str(e)
        })

def run_dashboard():
    """Run the dashboard server"""
    # Try multiple ports to avoid conflicts
    ports = [8081, 8082, 8083, 8084, 8085]
    
    for port in ports:
        try:
            # Check if port is available
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result != 0:  # Port is available
                logging.info(f"Starting dashboard server on http://localhost:{port}")
                socketio.run(app, host='0.0.0.0', port=port, debug=False, use_reloader=False)
                return
            else:
                logging.warning(f"Port {port} is in use, trying next port...")
                
        except Exception as e:
            logging.warning(f"Failed to start on port {port}: {str(e)}")
            continue
    
    logging.error("Could not find an available port for dashboard")
    raise Exception("No available ports for dashboard")

if __name__ == '__main__':
    # Load the correct initial budget from config
    load_initial_budget()
    run_dashboard()
