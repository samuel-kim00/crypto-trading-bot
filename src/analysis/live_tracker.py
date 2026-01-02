import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LiveTracker:
    def __init__(self):
        # Initialize Binance client
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        
        # Trading parameters
        self.trading_pairs = {
            'BTCUSDT': 0.001,  # symbol: quantity
            'ETHUSDT': 0.01,
            'BNBUSDT': 0.1,
            'ADAUSDT': 100,
            'DOGEUSDT': 1000
        }
        self.timeframes = ['4h', '1h']
        
    def get_historical_data(self, symbol, timeframe):
        """Get historical klines/candlestick data"""
        klines = self.client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=100
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        return df
        
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # EMA
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        
        return df
        
    def create_chart(self, df, timeframe, symbol):
        """Create interactive chart with price and indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} Price ({timeframe})', 'RSI', 'MACD')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # EMAs
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['ema_9'],
                name='EMA 9',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['ema_21'],
                name='EMA 21',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['bb_high'],
                name='BB High',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['bb_low'],
                name='BB Low',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd_signal'],
                name='Signal',
                line=dict(color='orange')
            ),
            row=3, col=1
        )
        
        # MACD Histogram
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['macd_hist'],
                name='Histogram',
                marker_color=np.where(df['macd_hist'] >= 0, 'green', 'red')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        return fig
        
    def get_current_prices(self):
        """Get current prices for all trading pairs"""
        prices = {}
        for symbol in self.trading_pairs.keys():
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            prices[symbol] = float(ticker['price'])
        return prices
        
    def get_account_balances(self):
        """Get account balances for all assets"""
        balances = {}
        account = self.client.get_account()
        for balance in account['balances']:
            if float(balance['free']) > 0 or float(balance['locked']) > 0:
                balances[balance['asset']] = {
                    'free': float(balance['free']),
                    'locked': float(balance['locked'])
                }
        return balances

def main():
    st.set_page_config(page_title="Live Trading Tracker", layout="wide")
    
    st.title("Live Trading Tracker")
    
    tracker = LiveTracker()
    
    # Sidebar
    st.sidebar.title("Account Info")
    
    # Get current prices and balances
    prices = tracker.get_current_prices()
    balances = tracker.get_account_balances()
    
    # Display USDT balance
    usdt_balance = balances.get('USDT', {'free': 0, 'locked': 0})
    st.sidebar.metric("USDT Balance", f"${usdt_balance['free']:,.2f}")
    
    # Display other asset balances
    st.sidebar.subheader("Asset Balances")
    for asset, balance in balances.items():
        if asset != 'USDT':
            st.sidebar.write(f"{asset}: {balance['free']:,.8f}")
    
    # Trading pair selector
    selected_pair = st.sidebar.selectbox(
        "Select Trading Pair",
        list(tracker.trading_pairs.keys()),
        index=0
    )
    
    # Timeframe selector
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ['1h', '4h'],
        index=0
    )
    
    # Auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    
    # Main content
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            # Get and process data
            df = tracker.get_historical_data(selected_pair, timeframe)
            df = tracker.calculate_indicators(df)
            
            # Create and display chart
            fig = tracker.create_chart(df, timeframe, selected_pair)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display latest indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RSI", f"{df['rsi'].iloc[-1]:.2f}")
            with col2:
                st.metric("MACD", f"{df['macd'].iloc[-1]:.2f}")
            with col3:
                st.metric("MACD Signal", f"{df['macd_signal'].iloc[-1]:.2f}")
            
            # Display latest price action
            st.subheader("Latest Price Action")
            latest_data = df.iloc[-1]
            st.write(f"Time: {latest_data['timestamp']}")
            st.write(f"Open: ${float(latest_data['open']):,.2f}")
            st.write(f"High: ${float(latest_data['high']):,.2f}")
            st.write(f"Low: ${float(latest_data['low']):,.2f}")
            st.write(f"Close: ${float(latest_data['close']):,.2f}")
            st.write(f"Volume: {float(latest_data['volume']):,.2f}")
            
            # Display current prices for all pairs
            st.subheader("Current Prices")
            prices = tracker.get_current_prices()
            cols = st.columns(len(prices))
            for i, (symbol, price) in enumerate(prices.items()):
                with cols[i]:
                    st.metric(symbol, f"${price:,.2f}")
        
        if not auto_refresh:
            break
            
        time.sleep(10)  # Update every 10 seconds

if __name__ == "__main__":
    main() 