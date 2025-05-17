import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from data_service import get_stock_data, get_crypto_data, get_economic_data
from utils import format_large_number, create_kpi_metric

# Page configuration
st.set_page_config(
    page_title="Financial Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard title
st.title("📊 Financial Performance Dashboard")
st.subheader("Real-time financial market insights")

# Add theme customization options in the sidebar
st.sidebar.title("Dashboard Settings")

# User role selection
user_role = st.sidebar.selectbox(
    "Select User Role",
    ["Investor", "Analyst", "Executive"]
)

# Display mode options
display_mode = st.sidebar.radio(
    "Display Mode",
    ["Standard", "Dark Mode", "High Contrast"]
)

# Data refresh rate
refresh_rate = st.sidebar.slider(
    "Data Refresh Interval (minutes)",
    min_value=1,
    max_value=60,
    value=15
)

# Apply theme based on display mode selection
if display_mode == "Dark Mode":
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
elif display_mode == "High Contrast":
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stMetric {
        background-color: #222222;
        border-radius: 5px;
        padding: 15px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Show last updated time and next refresh
last_update = datetime.now()
st.sidebar.write(f"Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
next_update = last_update + timedelta(minutes=refresh_rate)
st.sidebar.write(f"Next refresh: {next_update.strftime('%Y-%m-%d %H:%M:%S')}")

# Information about user role view
if user_role == "Investor":
    st.sidebar.info("Investor View: Focus on stock performance and investment opportunities")
elif user_role == "Analyst":
    st.sidebar.info("Analyst View: Detailed technical indicators and market analysis")
elif user_role == "Executive":
    st.sidebar.info("Executive View: High-level overview of market and economic performance")

# Add a refresh button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🔄 Refresh Data"):
        st.rerun()

# Add a time indicator to show when data was last updated
st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs for different dashboard sections
tab1, tab2, tab3 = st.tabs(["Market Overview", "Stock Analysis", "Economic Indicators"])

with tab1:
    st.header("Market Overview")
    
    # Loading state for data
    with st.spinner("Loading market data..."):
        # Fetch data for dashboard
        try:
            stock_data = get_stock_data()
            crypto_data = get_crypto_data()
            economic_data = get_economic_data()
            
            # Check if data is available
            if stock_data.empty or crypto_data.empty or economic_data.empty:
                st.error("Unable to retrieve complete market data. Please try again later.")
            
            # Create summary KPIs
            st.subheader("Key Performance Indicators")
            
            # Display KPIs in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # S&P 500 Index value
                sp500_current = stock_data[stock_data['symbol'] == 'SPY']['close'].iloc[-1]
                sp500_prev = stock_data[stock_data['symbol'] == 'SPY']['close'].iloc[-2]
                sp500_change = (sp500_current - sp500_prev) / sp500_prev * 100
                create_kpi_metric("S&P 500", f"${sp500_current:.2f}", f"{sp500_change:.2f}%")
            
            with col2:
                # Bitcoin value
                btc_current = crypto_data[crypto_data['symbol'] == 'BTC']['price'].iloc[-1]
                btc_prev = crypto_data[crypto_data['symbol'] == 'BTC']['price'].iloc[-2]
                btc_change = (btc_current - btc_prev) / btc_prev * 100
                create_kpi_metric("Bitcoin", f"${btc_current:.2f}", f"{btc_change:.2f}%")
            
            with col3:
                # GDP growth rate
                gdp_growth = economic_data[economic_data['indicator'] == 'GDP_growth']['value'].iloc[-1]
                gdp_prev = economic_data[economic_data['indicator'] == 'GDP_growth']['value'].iloc[-2]
                gdp_change = gdp_growth - gdp_prev
                create_kpi_metric("GDP Growth", f"{gdp_growth:.1f}%", f"{gdp_change:+.1f} pts")
            
            with col4:
                # Unemployment rate
                unemployment = economic_data[economic_data['indicator'] == 'Unemployment']['value'].iloc[-1]
                unemployment_prev = economic_data[economic_data['indicator'] == 'Unemployment']['value'].iloc[-2]
                unemployment_change = unemployment - unemployment_prev
                create_kpi_metric("Unemployment", f"{unemployment:.1f}%", f"{unemployment_change:+.1f} pts")
            
            # Market trends section
            st.subheader("Market Trends")
            
            # Filter for date range
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                # Determine date range from data
                min_date = stock_data['date'].min()
                max_date = stock_data['date'].max()
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            
            with date_col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
            # Filter data based on selected dates
            filtered_stock_data = stock_data[(stock_data['date'] >= pd.Timestamp(start_date)) & 
                                            (stock_data['date'] <= pd.Timestamp(end_date))]
            
            # Stock market trends chart
            st.subheader("Stock Market Performance")
            
            # Select stocks to display
            selected_stocks = st.multiselect(
                "Select stocks to display",
                options=filtered_stock_data['symbol'].unique().tolist(),
                default=['AAPL', 'MSFT', 'AMZN', 'GOOGL']
            )
            
            if selected_stocks:
                # Filter data for selected stocks
                stock_trend_data = filtered_stock_data[filtered_stock_data['symbol'].isin(selected_stocks)]
                
                # Create line chart for stock prices
                fig = px.line(
                    stock_trend_data, 
                    x='date', 
                    y='close', 
                    color='symbol',
                    title="Stock Price Trends",
                    labels={'close': 'Closing Price ($)', 'date': 'Date', 'symbol': 'Stock Symbol'}
                )
                
                fig.update_layout(
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one stock to display")
            
            # Cryptocurrency trends
            st.subheader("Cryptocurrency Performance")
            
            # Select cryptocurrencies to display
            selected_cryptos = st.multiselect(
                "Select cryptocurrencies to display",
                options=crypto_data['symbol'].unique().tolist(),
                default=['BTC', 'ETH']
            )
            
            if selected_cryptos:
                # Filter data for selected cryptocurrencies
                crypto_trend_data = crypto_data[crypto_data['symbol'].isin(selected_cryptos)]
                
                # Create line chart for crypto prices
                fig = px.line(
                    crypto_trend_data, 
                    x='date', 
                    y='price', 
                    color='symbol',
                    title="Cryptocurrency Price Trends",
                    labels={'price': 'Price ($)', 'date': 'Date', 'symbol': 'Crypto Symbol'}
                )
                
                fig.update_layout(
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one cryptocurrency to display")
        except Exception as e:
            st.error(f"Error loading market overview data: {e}")
            st.info("Please check your network connection and try refreshing the page.")

with tab2:
    st.header("Stock Analysis")
    
    try:
        # Stock selector
        stock_symbols = stock_data['symbol'].unique().tolist()
        selected_stock = st.selectbox("Select a stock for detailed analysis", stock_symbols)
        
        if selected_stock:
            # Get data for selected stock
            stock_detail = stock_data[stock_data['symbol'] == selected_stock]
            
            # Display current stock information
            current_price = stock_detail['close'].iloc[-1]
            prev_price = stock_detail['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price * 100
            
            # Stock details in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                create_kpi_metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
            
            with col2:
                volume = stock_detail['volume'].iloc[-1]
                vol_prev = stock_detail['volume'].iloc[-2]
                vol_change = (volume - vol_prev) / vol_prev * 100
                create_kpi_metric("Volume", format_large_number(volume), f"{vol_change:.2f}%")
            
            with col3:
                high = stock_detail['high'].iloc[-1]
                create_kpi_metric("Day High", f"${high:.2f}", "")
            
            with col4:
                low = stock_detail['low'].iloc[-1]
                create_kpi_metric("Day Low", f"${low:.2f}", "")
            
            # Display candlestick chart
            st.subheader(f"{selected_stock} Price Movement")
            
            # Date filter for stock details
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                # Determine date range from data
                min_date = stock_detail['date'].min()
                max_date = stock_detail['date'].max()
                detail_start_date = st.date_input("Analysis Start Date", min_date, key="detail_start", min_value=min_date, max_value=max_date)
            
            with date_col2:
                detail_end_date = st.date_input("Analysis End Date", max_date, key="detail_end", min_value=min_date, max_value=max_date)
            
            # Filter data for date range
            date_filtered_stock = stock_detail[(stock_detail['date'] >= pd.Timestamp(detail_start_date)) & 
                                             (stock_detail['date'] <= pd.Timestamp(detail_end_date))]
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=date_filtered_stock['date'],
                open=date_filtered_stock['open'],
                high=date_filtered_stock['high'],
                low=date_filtered_stock['low'],
                close=date_filtered_stock['close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            
            fig.update_layout(
                title=f"{selected_stock} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            st.subheader(f"{selected_stock} Trading Volume")
            
            # Create volume bar chart
            fig = px.bar(
                date_filtered_stock,
                x='date',
                y='volume',
                title=f"{selected_stock} Trading Volume",
                labels={'volume': 'Volume', 'date': 'Date'}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators section
            st.subheader("Technical Indicators")
            
            # Select technical indicators to display
            tech_indicators = st.multiselect(
                "Select technical indicators",
                ["Simple Moving Averages", "Bollinger Bands", "RSI", "MACD"],
                default=["Simple Moving Averages"]
            )
            
            # Technical analysis container
            tech_col1, tech_col2 = st.columns([3, 1])
            
            with tech_col2:
                # Technical analysis settings
                st.subheader("Settings")
                
                # SMA periods settings
                sma_short = st.number_input("Short SMA Period", min_value=5, max_value=50, value=20)
                sma_long = st.number_input("Long SMA Period", min_value=20, max_value=200, value=50)
                
                # Bollinger Bands settings
                bb_period = st.number_input("Bollinger Bands Period", min_value=5, max_value=50, value=20)
                bb_std = st.number_input("Bollinger Bands Std Dev", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
                
                # RSI settings
                rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=14)
                
                # MACD settings
                macd_fast = st.number_input("MACD Fast Period", min_value=5, max_value=30, value=12)
                macd_slow = st.number_input("MACD Slow Period", min_value=10, max_value=50, value=26)
                macd_signal = st.number_input("MACD Signal Period", min_value=5, max_value=20, value=9)
                
                # Trading signals alert threshold
                alert_threshold = st.slider("Alert Threshold (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
            
            with tech_col1:
                # Calculate technical indicators based on selection
                if "Simple Moving Averages" in tech_indicators:
                    # Calculate simple moving averages
                    date_filtered_stock[f'SMA_{sma_short}'] = date_filtered_stock['close'].rolling(window=sma_short).mean()
                    date_filtered_stock[f'SMA_{sma_long}'] = date_filtered_stock['close'].rolling(window=sma_long).mean()
                    
                    # Create SMA chart
                    fig = go.Figure()
                    
                    # Add close price
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add SMA short
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock[f'SMA_{sma_short}'],
                        mode='lines',
                        name=f'SMA {sma_short}',
                        line=dict(color='orange')
                    ))
                    
                    # Add SMA long
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock[f'SMA_{sma_long}'],
                        mode='lines',
                        name=f'SMA {sma_long}',
                        line=dict(color='green')
                    ))
                    
                    # Check for SMA crossover
                    date_filtered_stock['SMA_crossover'] = np.where(
                        date_filtered_stock[f'SMA_{sma_short}'] > date_filtered_stock[f'SMA_{sma_long}'],
                        1, -1
                    )
                    
                    # Find crossover points
                    crossover_points = date_filtered_stock[date_filtered_stock['SMA_crossover'].diff() != 0].copy()
                    
                    if not crossover_points.empty:
                        # Add crossover markers
                        for idx, row in crossover_points.iterrows():
                            if row['SMA_crossover'] == 1:  # Bullish crossover
                                fig.add_trace(go.Scatter(
                                    x=[row['date']],
                                    y=[row['close']],
                                    mode='markers',
                                    name='Buy Signal',
                                    marker=dict(color='green', size=10, symbol='triangle-up'),
                                    showlegend=True
                                ))
                            else:  # Bearish crossover
                                fig.add_trace(go.Scatter(
                                    x=[row['date']],
                                    y=[row['close']],
                                    mode='markers',
                                    name='Sell Signal',
                                    marker=dict(color='red', size=10, symbol='triangle-down'),
                                    showlegend=True
                                ))
                    
                    fig.update_layout(
                        title="Moving Averages with Crossover Signals",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show alerts for recent crossovers
                    recent_crossovers = crossover_points[crossover_points['date'] >= (date_filtered_stock['date'].max() - pd.Timedelta(days=7))]
                    
                    if not recent_crossovers.empty:
                        st.subheader("Recent Signals")
                        for idx, row in recent_crossovers.iterrows():
                            signal_type = "Buy Signal (Bullish Crossover)" if row['SMA_crossover'] == 1 else "Sell Signal (Bearish Crossover)"
                            signal_date = row['date'].strftime('%Y-%m-%d')
                            if row['SMA_crossover'] == 1:
                                st.success(f"🔔 {signal_type} detected on {signal_date} at ${row['close']:.2f}")
                            else:
                                st.warning(f"🔔 {signal_type} detected on {signal_date} at ${row['close']:.2f}")
                
                # Bollinger Bands
                if "Bollinger Bands" in tech_indicators:
                    # Calculate Bollinger Bands
                    date_filtered_stock['BB_middle'] = date_filtered_stock['close'].rolling(window=bb_period).mean()
                    date_filtered_stock['BB_std'] = date_filtered_stock['close'].rolling(window=bb_period).std()
                    date_filtered_stock['BB_upper'] = date_filtered_stock['BB_middle'] + (date_filtered_stock['BB_std'] * bb_std)
                    date_filtered_stock['BB_lower'] = date_filtered_stock['BB_middle'] - (date_filtered_stock['BB_std'] * bb_std)
                    
                    # Create Bollinger Bands chart
                    fig = go.Figure()
                    
                    # Add close price
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue')
                    ))
                    
                    # Add Bollinger Bands
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock['BB_upper'],
                        mode='lines',
                        name='Upper Band',
                        line=dict(color='rgba(250, 120, 120, 0.7)')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock['BB_middle'],
                        mode='lines',
                        name='Middle Band',
                        line=dict(color='rgba(150, 150, 150, 0.7)')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock['BB_lower'],
                        mode='lines',
                        name='Lower Band',
                        line=dict(color='rgba(120, 250, 120, 0.7)')
                    ))
                    
                    # Find price touching or crossing bands
                    date_filtered_stock['BB_touch_upper'] = date_filtered_stock['close'] >= date_filtered_stock['BB_upper']
                    date_filtered_stock['BB_touch_lower'] = date_filtered_stock['close'] <= date_filtered_stock['BB_lower']
                    
                    # Add markers for price crossing bands
                    upper_touches = date_filtered_stock[date_filtered_stock['BB_touch_upper']].copy()
                    lower_touches = date_filtered_stock[date_filtered_stock['BB_touch_lower']].copy()
                    
                    if not upper_touches.empty:
                        fig.add_trace(go.Scatter(
                            x=upper_touches['date'],
                            y=upper_touches['close'],
                            mode='markers',
                            name='Overbought',
                            marker=dict(color='red', size=8, symbol='circle'),
                            showlegend=True
                        ))
                    
                    if not lower_touches.empty:
                        fig.add_trace(go.Scatter(
                            x=lower_touches['date'],
                            y=lower_touches['close'],
                            mode='markers',
                            name='Oversold',
                            marker=dict(color='green', size=8, symbol='circle'),
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title=f"Bollinger Bands ({bb_period} periods, {bb_std} std dev)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show BB alerts
                    recent_upper_touches = upper_touches[upper_touches['date'] >= (date_filtered_stock['date'].max() - pd.Timedelta(days=7))]
                    recent_lower_touches = lower_touches[lower_touches['date'] >= (date_filtered_stock['date'].max() - pd.Timedelta(days=7))]
                    
                    if not recent_upper_touches.empty or not recent_lower_touches.empty:
                        st.subheader("Bollinger Bands Alerts")
                    
                    if not recent_upper_touches.empty:
                        for idx, row in recent_upper_touches.iterrows():
                            st.warning(f"🔔 Overbought signal detected on {row['date'].strftime('%Y-%m-%d')} at ${row['close']:.2f}")
                    
                    if not recent_lower_touches.empty:
                        for idx, row in recent_lower_touches.iterrows():
                            st.success(f"🔔 Oversold signal detected on {row['date'].strftime('%Y-%m-%d')} at ${row['close']:.2f}")
                
                # RSI
                if "RSI" in tech_indicators:
                    # Calculate RSI
                    delta = date_filtered_stock['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(window=rsi_period).mean()
                    avg_loss = loss.rolling(window=rsi_period).mean()
                    
                    rs = avg_gain / avg_loss
                    date_filtered_stock['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Create RSI chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=date_filtered_stock['date'],
                        y=date_filtered_stock['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ))
                    
                    # Add overbought/oversold lines
                    fig.add_shape(
                        type="line",
                        x0=date_filtered_stock['date'].min(),
                        y0=70,
                        x1=date_filtered_stock['date'].max(),
                        y1=70,
                        line=dict(color="red", width=2, dash="dash")
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=date_filtered_stock['date'].min(),
                        y0=30,
                        x1=date_filtered_stock['date'].max(),
                        y1=30,
                        line=dict(color="green", width=2, dash="dash")
                    )
                    
                    # Detect overbought/oversold conditions
                    date_filtered_stock['RSI_overbought'] = date_filtered_stock['RSI'] > 70
                    date_filtered_stock['RSI_oversold'] = date_filtered_stock['RSI'] < 30
                    
                    # Add markers for overbought/oversold
                    rsi_overbought = date_filtered_stock[date_filtered_stock['RSI_overbought']].copy()
                    rsi_oversold = date_filtered_stock[date_filtered_stock['RSI_oversold']].copy()
                    
                    if not rsi_overbought.empty:
                        fig.add_trace(go.Scatter(
                            x=rsi_overbought['date'],
                            y=rsi_overbought['RSI'],
                            mode='markers',
                            name='Overbought',
                            marker=dict(color='red', size=8, symbol='triangle-down'),
                            showlegend=True
                        ))
                    
                    if not rsi_oversold.empty:
                        fig.add_trace(go.Scatter(
                            x=rsi_oversold['date'],
                            y=rsi_oversold['RSI'],
                            mode='markers',
                            name='Oversold',
                            marker=dict(color='green', size=8, symbol='triangle-up'),
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title=f"Relative Strength Index (RSI {rsi_period})",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        height=300,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                        yaxis=dict(range=[0, 100])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show RSI alerts
                    recent_overbought = rsi_overbought[rsi_overbought['date'] >= (date_filtered_stock['date'].max() - pd.Timedelta(days=7))]
                    recent_oversold = rsi_oversold[rsi_oversold['date'] >= (date_filtered_stock['date'].max() - pd.Timedelta(days=7))]
                    
                    if not recent_overbought.empty or not recent_oversold.empty:
                        st.subheader("RSI Alerts")
                    
                    if not recent_overbought.empty:
                        for idx, row in recent_overbought.iterrows():
                            st.warning(f"🔔 RSI Overbought signal detected on {row['date'].strftime('%Y-%m-%d')} - RSI: {row['RSI']:.2f}")
                    
                    if not recent_oversold.empty:
                        for idx, row in recent_oversold.iterrows():
                            st.success(f"🔔 RSI Oversold signal detected on {row['date'].strftime('%Y-%m-%d')} - RSI: {row['RSI']:.2f}")
                
                # MACD
                if "MACD" in tech_indicators:
                    # Calculate MACD
                    date_filtered_stock['EMA_fast'] = date_filtered_stock['close'].ewm(span=macd_fast, adjust=False).mean()
                    date_filtered_stock['EMA_slow'] = date_filtered_stock['close'].ewm(span=macd_slow, adjust=False).mean()
                    date_filtered_stock['MACD'] = date_filtered_stock['EMA_fast'] - date_filtered_stock['EMA_slow']
                    date_filtered_stock['MACD_signal'] = date_filtered_stock['MACD'].ewm(span=macd_signal, adjust=False).mean()
                    date_filtered_stock['MACD_histogram'] = date_filtered_stock['MACD'] - date_filtered_stock['MACD_signal']
                    
                    # Create MACD chart
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                       vertical_spacing=0.1, 
                                       row_heights=[0.7, 0.3])
                    
                    # Add price chart
                    fig.add_trace(
                        go.Scatter(
                            x=date_filtered_stock['date'],
                            y=date_filtered_stock['close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    
                    # Add MACD line
                    fig.add_trace(
                        go.Scatter(
                            x=date_filtered_stock['date'],
                            y=date_filtered_stock['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue')
                        ),
                        row=2, col=1
                    )
                    
                    # Add MACD signal line
                    fig.add_trace(
                        go.Scatter(
                            x=date_filtered_stock['date'],
                            y=date_filtered_stock['MACD_signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red')
                        ),
                        row=2, col=1
                    )
                    
                    # Add MACD histogram
                    colors = ['green' if val >= 0 else 'red' for val in date_filtered_stock['MACD_histogram']]
                    
                    fig.add_trace(
                        go.Bar(
                            x=date_filtered_stock['date'],
                            y=date_filtered_stock['MACD_histogram'],
                            name='Histogram',
                            marker_color=colors
                        ),
                        row=2, col=1
                    )
                    
                    # Detect MACD crossovers
                    date_filtered_stock['MACD_crossover'] = np.where(
                        date_filtered_stock['MACD'] > date_filtered_stock['MACD_signal'],
                        1, -1
                    )
                    
                    # Find crossover points
                    macd_crossover_points = date_filtered_stock[date_filtered_stock['MACD_crossover'].diff() != 0].copy()
                    
                    if not macd_crossover_points.empty:
                        # Add crossover markers to MACD
                        for idx, row in macd_crossover_points.iterrows():
                            if row['MACD_crossover'] == 1:  # Bullish crossover
                                fig.add_trace(go.Scatter(
                                    x=[row['date']],
                                    y=[row['MACD']],
                                    mode='markers',
                                    name='MACD Buy',
                                    marker=dict(color='green', size=10, symbol='triangle-up'),
                                    showlegend=True
                                ), row=2, col=1)
                                
                                # Add to price chart too
                                fig.add_trace(go.Scatter(
                                    x=[row['date']],
                                    y=[row['close']],
                                    mode='markers',
                                    name='MACD Buy Signal',
                                    marker=dict(color='green', size=10, symbol='triangle-up'),
                                    showlegend=False
                                ), row=1, col=1)
                            else:  # Bearish crossover
                                fig.add_trace(go.Scatter(
                                    x=[row['date']],
                                    y=[row['MACD']],
                                    mode='markers',
                                    name='MACD Sell',
                                    marker=dict(color='red', size=10, symbol='triangle-down'),
                                    showlegend=True
                                ), row=2, col=1)
                                
                                # Add to price chart too
                                fig.add_trace(go.Scatter(
                                    x=[row['date']],
                                    y=[row['close']],
                                    mode='markers',
                                    name='MACD Sell Signal',
                                    marker=dict(color='red', size=10, symbol='triangle-down'),
                                    showlegend=False
                                ), row=1, col=1)
                    
                    fig.update_layout(
                        title=f"MACD ({macd_fast}, {macd_slow}, {macd_signal})",
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="MACD", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show MACD alerts
                    recent_macd_crossovers = macd_crossover_points[macd_crossover_points['date'] >= (date_filtered_stock['date'].max() - pd.Timedelta(days=7))]
                    
                    if not recent_macd_crossovers.empty:
                        st.subheader("MACD Alerts")
                        for idx, row in recent_macd_crossovers.iterrows():
                            signal_type = "Buy Signal (Bullish Crossover)" if row['MACD_crossover'] == 1 else "Sell Signal (Bearish Crossover)"
                            signal_date = row['date'].strftime('%Y-%m-%d')
                            if row['MACD_crossover'] == 1:
                                st.success(f"🔔 MACD {signal_type} detected on {signal_date}")
                            else:
                                st.warning(f"🔔 MACD {signal_type} detected on {signal_date}")
            
            # Technical Analysis Summary
            if user_role == "Analyst" or user_role == "Investor":
                st.subheader("Technical Analysis Summary")
                
                # Get latest data point
                latest_data = date_filtered_stock.iloc[-1]
                latest_date = latest_data['date'].strftime('%Y-%m-%d')
                
                # Create summary metrics
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    # Trend analysis based on SMAs
                    if 'SMA_20' in date_filtered_stock.columns and 'SMA_50' in date_filtered_stock.columns:
                        sma_trend = "Bullish" if latest_data['SMA_20'] > latest_data['SMA_50'] else "Bearish"
                        sma_color = "green" if sma_trend == "Bullish" else "red"
                        st.markdown(f"**SMA Trend:** <span style='color:{sma_color}'>{sma_trend}</span>", unsafe_allow_html=True)
                    
                    # RSI analysis if available
                    if 'RSI' in date_filtered_stock.columns:
                        rsi_value = latest_data['RSI']
                        if rsi_value > 70:
                            rsi_status = "Overbought"
                            rsi_color = "red"
                        elif rsi_value < 30:
                            rsi_status = "Oversold"
                            rsi_color = "green"
                        else:
                            rsi_status = "Neutral"
                            rsi_color = "gray"
                        st.markdown(f"**RSI ({rsi_period}):** <span style='color:{rsi_color}'>{rsi_value:.2f} - {rsi_status}</span>", unsafe_allow_html=True)
                
                with summary_col2:
                    # MACD analysis if available
                    if 'MACD' in date_filtered_stock.columns and 'MACD_signal' in date_filtered_stock.columns:
                        macd_value = latest_data['MACD']
                        macd_signal_value = latest_data['MACD_signal']
                        macd_status = "Bullish" if macd_value > macd_signal_value else "Bearish"
                        macd_color = "green" if macd_status == "Bullish" else "red"
                        st.markdown(f"**MACD:** <span style='color:{macd_color}'>{macd_status}</span>", unsafe_allow_html=True)
                    
                    # Bollinger Bands analysis if available
                    if 'BB_middle' in date_filtered_stock.columns and 'BB_upper' in date_filtered_stock.columns and 'BB_lower' in date_filtered_stock.columns:
                        current_price = latest_data['close']
                        bb_upper = latest_data['BB_upper']
                        bb_lower = latest_data['BB_lower']
                        
                        if current_price > bb_upper:
                            bb_status = "Overbought"
                            bb_color = "red"
                        elif current_price < bb_lower:
                            bb_status = "Oversold"
                            bb_color = "green"
                        else:
                            bb_status = "Neutral"
                            bb_color = "gray"
                        
                        st.markdown(f"**Bollinger Bands:** <span style='color:{bb_color}'>{bb_status}</span>", unsafe_allow_html=True)
                
                with summary_col3:
                    # Volume analysis
                    recent_volume = date_filtered_stock.iloc[-5:]['volume'].mean()
                    prev_volume = date_filtered_stock.iloc[-10:-5]['volume'].mean()
                    volume_change = (recent_volume - prev_volume) / prev_volume * 100
                    volume_status = "Increasing" if volume_change > 0 else "Decreasing"
                    volume_color = "green" if volume_change > 10 else ("red" if volume_change < -10 else "gray")
                    
                    st.markdown(f"**Volume Trend:** <span style='color:{volume_color}'>{volume_status} ({volume_change:.2f}%)</span>", unsafe_allow_html=True)
                    
                    # Price momentum
                    price_momentum = ((latest_data['close'] / date_filtered_stock.iloc[-5]['close']) - 1) * 100
                    momentum_status = "Strong" if abs(price_momentum) > 5 else "Moderate"
                    momentum_direction = "Up" if price_momentum > 0 else "Down"
                    momentum_color = "green" if price_momentum > 0 else "red"
                    
                    st.markdown(f"**Momentum:** <span style='color:{momentum_color}'>{momentum_status} {momentum_direction} ({price_momentum:.2f}%)</span>", unsafe_allow_html=True)
                
                # Overall recommendation based on technical indicators
                st.subheader("Technical Recommendation")
                
                # Initialize scores
                buy_signals = 0
                sell_signals = 0
                
                # SMA trend
                if 'SMA_20' in date_filtered_stock.columns and 'SMA_50' in date_filtered_stock.columns:
                    if latest_data['SMA_20'] > latest_data['SMA_50']:
                        buy_signals += 1
                    else:
                        sell_signals += 1
                
                # RSI
                if 'RSI' in date_filtered_stock.columns:
                    if latest_data['RSI'] < 30:
                        buy_signals += 1
                    elif latest_data['RSI'] > 70:
                        sell_signals += 1
                
                # MACD
                if 'MACD' in date_filtered_stock.columns and 'MACD_signal' in date_filtered_stock.columns:
                    if latest_data['MACD'] > latest_data['MACD_signal']:
                        buy_signals += 1
                    else:
                        sell_signals += 1
                
                # Bollinger Bands
                if 'BB_middle' in date_filtered_stock.columns and 'BB_upper' in date_filtered_stock.columns and 'BB_lower' in date_filtered_stock.columns:
                    if latest_data['close'] < latest_data['BB_lower']:
                        buy_signals += 1
                    elif latest_data['close'] > latest_data['BB_upper']:
                        sell_signals += 1
                
                # Determine overall signal
                total_signals = buy_signals + sell_signals
                if total_signals > 0:
                    buy_percentage = (buy_signals / total_signals) * 100
                    recommendation = ""
                    
                    if buy_percentage >= 75:
                        recommendation = "Strong Buy"
                        rec_color = "darkgreen"
                    elif buy_percentage >= 60:
                        recommendation = "Buy"
                        rec_color = "green"
                    elif buy_percentage >= 40:
                        recommendation = "Neutral"
                        rec_color = "gray"
                    elif buy_percentage >= 25:
                        recommendation = "Sell"
                        rec_color = "red"
                    else:
                        recommendation = "Strong Sell"
                        rec_color = "darkred"
                    
                    st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>{recommendation}</h3>", unsafe_allow_html=True)
                    st.progress(int(buy_percentage))
                    st.caption(f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}")
                else:
                    st.info("Not enough technical indicators selected to generate a recommendation.")
                
                # Alerts for significant price movements
                recent_price_change = ((latest_data['close'] / date_filtered_stock.iloc[-2]['close']) - 1) * 100
                if abs(recent_price_change) > alert_threshold:
                    if recent_price_change > 0:
                        st.success(f"🚨 **Alert:** {selected_stock} price increased by {recent_price_change:.2f}% on {latest_date}, exceeding the alert threshold of {alert_threshold:.1f}%")
                    else:
                        st.error(f"🚨 **Alert:** {selected_stock} price decreased by {abs(recent_price_change):.2f}% on {latest_date}, exceeding the alert threshold of {alert_threshold:.1f}%")
                
                # Volume alerts
                recent_volume_change = ((latest_data['volume'] / date_filtered_stock.iloc[-2]['volume']) - 1) * 100
                if abs(recent_volume_change) > alert_threshold * 3:  # Higher threshold for volume
                    if recent_volume_change > 0:
                        st.info(f"📈 **Volume Alert:** {selected_stock} trading volume increased by {recent_volume_change:.2f}% on {latest_date}")
                    else:
                        st.info(f"📉 **Volume Alert:** {selected_stock} trading volume decreased by {abs(recent_volume_change):.2f}% on {latest_date}")
    except Exception as e:
        st.error(f"Error loading stock analysis data: {e}")
        st.info("Please check your network connection and try refreshing the page.")


with tab3:
    st.header("Economic Indicators")
    
    try:
        # Fetch data for dashboard if not already fetched
        if 'economic_data' not in locals():
            economic_data = get_economic_data()
            
        # Enhanced Economic Dashboard for different user roles
        if user_role == "Executive":
            # Executive view - focused on high-level overview with alerts
            
            # Economic health score calculation
            st.subheader("Economic Health Dashboard")
            
            # Economic indicators overview with gauges
            indicators = economic_data['indicator'].unique().tolist()
            
            # Create health score metrics
            health_metrics = {
                'GDP_growth': {'weight': 0.3, 'optimal': 3.0, 'warning': 1.0, 'critical': 0.0},
                'Inflation': {'weight': 0.25, 'optimal': 2.0, 'warning': 3.5, 'critical': 5.0, 'inverse': True},
                'Unemployment': {'weight': 0.25, 'optimal': 3.5, 'warning': 5.0, 'critical': 7.0, 'inverse': True},
                'Interest_Rate': {'weight': 0.1, 'optimal': 2.5, 'warning': 4.0, 'critical': 6.0, 'inverse': True},
                'Consumer_Confidence': {'weight': 0.1, 'optimal': 100.0, 'warning': 80.0, 'critical': 70.0}
            }
            
            # Calculate scores
            eco_scores = {}
            overall_score = 0
            max_score = 0
            
            for indicator, metrics in health_metrics.items():
                if indicator in indicators:
                    current_value = economic_data[economic_data['indicator'] == indicator]['value'].iloc[-1]
                    
                    # Calculate indicator score (0-100)
                    if metrics.get('inverse', False):
                        # Lower is better for inverse metrics
                        if current_value <= metrics['optimal']:
                            score = 100
                        elif current_value >= metrics['critical']:
                            score = 0
                        else:
                            score = 100 * (metrics['critical'] - current_value) / (metrics['critical'] - metrics['optimal'])
                    else:
                        # Higher is better for normal metrics
                        if current_value >= metrics['optimal']:
                            score = 100
                        elif current_value <= metrics['critical']:
                            score = 0
                        else:
                            score = 100 * (current_value - metrics['critical']) / (metrics['optimal'] - metrics['critical'])
                    
                    eco_scores[indicator] = {'value': current_value, 'score': score}
                    overall_score += score * metrics['weight']
                    max_score += metrics['weight']
            
            # Normalize overall score
            if max_score > 0:
                overall_score = (overall_score / max_score) * 100
            
            # Display overall score
            score_col1, score_col2 = st.columns([1, 3])
            
            with score_col1:
                # Overall health gauge
                if overall_score >= 80:
                    gauge_color = "green"
                    health_status = "Excellent"
                elif overall_score >= 60:
                    gauge_color = "gold"
                    health_status = "Good"
                elif overall_score >= 40:
                    gauge_color = "orange"
                    health_status = "Moderate"
                else:
                    gauge_color = "red"
                    health_status = "Concerning"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={'text': "Economic Health Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 60], 'color': "gray"},
                            {'range': [60, 80], 'color': "lightgreen"},
                            {'range': [80, 100], 'color': "green"},
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': overall_score
                        }
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"**Health Status: {health_status}**")
            
            with score_col2:
                # Individual indicator scores
                st.subheader("Key Economic Indicators")
                
                # Create metrics for each indicator
                for i, (indicator, data) in enumerate(eco_scores.items()):
                    if i % 3 == 0:
                        metric_cols = st.columns(3)
                    
                    col_idx = i % 3
                    with metric_cols[col_idx]:
                        # Format the indicator name for display
                        display_name = indicator.replace('_', ' ')
                        
                        # Determine color based on score
                        if data['score'] >= 80:
                            score_color = "green"
                        elif data['score'] >= 60:
                            score_color = "orange"
                        else:
                            score_color = "red"
                        
                        # Create a gauge for each indicator
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=data['score'],
                            title={'text': display_name},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': score_color},
                                'steps': [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 80], 'color': "gray"},
                                    {'range': [80, 100], 'color': "lightgreen"},
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': data['score']
                                }
                            },
                            domain={'x': [0, 1], 'y': [0, 1]}
                        ))
                        
                        fig.update_layout(
                            height=150,
                            margin=dict(l=10, r=10, t=30, b=10),
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"Current: **{data['value']:.2f}%**")
            
            # Economic alerts section
            st.subheader("Economic Risk Alerts")
            
            # Check for economic warning conditions
            alerts = []
            for indicator, metrics in health_metrics.items():
                if indicator in indicators:
                    current_value = economic_data[economic_data['indicator'] == indicator]['value'].iloc[-1]
                    prev_value = economic_data[economic_data['indicator'] == indicator]['value'].iloc[-2]
                    change = current_value - prev_value
                    
                    # Alert for concerning trends
                    if metrics.get('inverse', False):
                        # For inverse metrics (lower is better)
                        if current_value >= metrics['critical']:
                            alerts.append({
                                'indicator': indicator,
                                'severity': 'critical',
                                'message': f"{indicator.replace('_', ' ')} is at {current_value:.2f}%, which is above the critical threshold of {metrics['critical']}%"
                            })
                        elif current_value >= metrics['warning']:
                            alerts.append({
                                'indicator': indicator,
                                'severity': 'warning',
                                'message': f"{indicator.replace('_', ' ')} is at {current_value:.2f}%, approaching concerning levels"
                            })
                        
                        # Alert for rapid increases
                        if change > 0.5:
                            alerts.append({
                                'indicator': indicator,
                                'severity': 'trend',
                                'message': f"{indicator.replace('_', ' ')} increased by {change:.2f} points in the last period"
                            })
                    else:
                        # For normal metrics (higher is better)
                        if current_value <= metrics['critical']:
                            alerts.append({
                                'indicator': indicator,
                                'severity': 'critical',
                                'message': f"{indicator.replace('_', ' ')} is at {current_value:.2f}%, which is below the critical threshold of {metrics['critical']}%"
                            })
                        elif current_value <= metrics['warning']:
                            alerts.append({
                                'indicator': indicator,
                                'severity': 'warning',
                                'message': f"{indicator.replace('_', ' ')} is at {current_value:.2f}%, approaching concerning levels"
                            })
                        
                        # Alert for rapid decreases
                        if change < -0.5:
                            alerts.append({
                                'indicator': indicator,
                                'severity': 'trend',
                                'message': f"{indicator.replace('_', ' ')} decreased by {abs(change):.2f} points in the last period"
                            })
            
            # Display alerts
            if alerts:
                for alert in alerts:
                    if alert['severity'] == 'critical':
                        st.error(f"🚨 **Critical Alert:** {alert['message']}")
                    elif alert['severity'] == 'warning':
                        st.warning(f"⚠️ **Warning:** {alert['message']}")
                    else:
                        st.info(f"📊 **Trend Alert:** {alert['message']}")
            else:
                st.success("No economic risk alerts at this time. All indicators are within expected ranges.")
            
            # Economic Forecast
            st.subheader("Economic Growth Forecast")
            
            # Create simplified forecast based on current trends
            if 'GDP_growth' in indicators:
                gdp_data = economic_data[economic_data['indicator'] == 'GDP_growth'].copy()
                
                # Sort by date to ensure proper time sequence
                gdp_data = gdp_data.sort_values('date')
                
                # Get recent GDP trend
                recent_gdp = gdp_data.iloc[-4:]['value'].values
                
                # Calculate simple trend for forecasting
                gdp_trend = np.polyfit(range(len(recent_gdp)), recent_gdp, 1)[0]
                
                # Create forecast for next 4 quarters
                forecast_periods = 4
                last_date = gdp_data['date'].iloc[-1]
                last_value = gdp_data['value'].iloc[-1]
                
                # Generate forecasted dates and values
                forecast_dates = [last_date + pd.DateOffset(months=3*i) for i in range(1, forecast_periods+1)]
                forecast_values = [last_value + gdp_trend*i for i in range(1, forecast_periods+1)]
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'value': forecast_values,
                    'type': ['Forecast'] * forecast_periods
                })
                
                # Add type column to original data
                gdp_data['type'] = 'Historical'
                
                # Combine historical and forecast data
                combined_df = pd.concat([gdp_data[['date', 'value', 'type']], forecast_df])
                
                # Create forecast chart
                fig = px.line(
                    combined_df, 
                    x='date', 
                    y='value',
                    color='type',
                    title="GDP Growth Forecast",
                    labels={'value': 'GDP Growth (%)', 'date': 'Date', 'type': 'Data Type'},
                    color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
                )
                
                # Add confidence interval for forecast
                for i, date in enumerate(forecast_dates):
                    fig.add_trace(go.Scatter(
                        x=[date, date],
                        y=[forecast_values[i] - 0.5, forecast_values[i] + 0.5],
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.2)', width=10),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast summary
                forecast_end = forecast_values[-1]
                if forecast_end > last_value:
                    st.success(f"📈 **Forecast:** GDP growth is projected to increase to {forecast_end:.2f}% in the next year")
                elif forecast_end < last_value:
                    st.error(f"📉 **Forecast:** GDP growth is projected to decrease to {forecast_end:.2f}% in the next year")
                else:
                    st.info(f"📊 **Forecast:** GDP growth is projected to remain stable around {forecast_end:.2f}% in the next year")
            
        else:
            # Standard view for Investor and Analyst roles
            # Display economic indicators
            st.subheader("Key Economic Metrics")
            
            # Economic indicators filter
            indicators = economic_data['indicator'].unique().tolist()
            selected_indicators = st.multiselect(
                "Select economic indicators to display",
                options=indicators,
                default=['GDP_growth', 'Inflation', 'Unemployment']
            )
            
            if selected_indicators:
                # Filter data for selected indicators
                filtered_eco_data = economic_data[economic_data['indicator'].isin(selected_indicators)]
                
                # Create line chart for economic indicators
                fig = px.line(
                    filtered_eco_data, 
                    x='date', 
                    y='value', 
                    color='indicator',
                    title="Economic Indicators Trends",
                    labels={'value': 'Value (%)', 'date': 'Date', 'indicator': 'Indicator'}
                )
                
                fig.update_layout(
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one economic indicator to display")
            
            # Create detailed economic metrics section
            st.subheader("Economic Metrics Breakdown")
            
            # Select specific indicator for breakdown
            selected_breakdown = st.selectbox(
                "Select an indicator for detailed breakdown",
                options=indicators
            )
            
            if selected_breakdown:
                # Get data for selected indicator
                breakdown_data = economic_data[economic_data['indicator'] == selected_breakdown]
                
                # Display current value and change
                current_value = breakdown_data['value'].iloc[-1]
                prev_value = breakdown_data['value'].iloc[-2]
                value_change = current_value - prev_value
                
                # Determine if the change is positive or negative for this indicator
                if selected_breakdown in ['Inflation', 'Unemployment', 'Interest_Rate']:
                    # For these indicators, lower is generally better
                    is_positive_change = value_change < 0
                else:
                    # For other indicators like GDP growth and Consumer Confidence, higher is better
                    is_positive_change = value_change > 0
                
                # Create metrics columns
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    # Current value
                    create_kpi_metric(
                        f"Current {selected_breakdown.replace('_', ' ')}", 
                        f"{current_value:.2f}%", 
                        f"{value_change:+.2f} pts"
                    )
                
                with metrics_col2:
                    # Year-over-year change
                    if len(breakdown_data) >= 5:
                        yoy_value = breakdown_data['value'].iloc[-5]
                        yoy_change = ((current_value - yoy_value) / yoy_value) * 100
                        create_kpi_metric(
                            "Year-over-Year Change", 
                            f"{yoy_change:+.2f}%", 
                            ""
                        )
                
                with metrics_col3:
                    # Trend direction
                    recent_values = breakdown_data.iloc[-6:]['value'].values
                    trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    
                    trend_direction = "Rising" if trend_slope > 0 else "Falling"
                    trend_strength = "Strongly" if abs(trend_slope) > 0.5 else "Moderately" if abs(trend_slope) > 0.1 else "Slightly"
                    
                    create_kpi_metric(
                        "Trend", 
                        f"{trend_strength} {trend_direction}", 
                        f"{trend_slope:+.3f} pts/period"
                    )
                
                # Create detailed visualization with annotations
                st.subheader(f"{selected_breakdown.replace('_', ' ')} Analysis")
                
                # Create more informative chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add bar chart for values
                fig.add_trace(
                    go.Bar(
                        x=breakdown_data['date'],
                        y=breakdown_data['value'],
                        name=selected_breakdown,
                        marker_color='lightblue'
                    ),
                    secondary_y=False,
                )
                
                # Add line for trend
                x_values = list(range(len(breakdown_data)))
                trend_line = np.poly1d(np.polyfit(x_values, breakdown_data['value'], 1))
                trend_values = trend_line(x_values)
                
                fig.add_trace(
                    go.Scatter(
                        x=breakdown_data['date'],
                        y=trend_values,
                        name="Trend Line",
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    secondary_y=False,
                )
                
                # Calculate and add moving average
                breakdown_data['MA_3'] = breakdown_data['value'].rolling(window=3).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=breakdown_data['date'],
                        y=breakdown_data['MA_3'],
                        name="3-Period Moving Average",
                        line=dict(color='green', width=2)
                    ),
                    secondary_y=False,
                )
                
                # Add period-over-period change
                breakdown_data['change'] = breakdown_data['value'].diff()
                
                fig.add_trace(
                    go.Scatter(
                        x=breakdown_data['date'],
                        y=breakdown_data['change'],
                        name="Period Change",
                        line=dict(color='purple')
                    ),
                    secondary_y=True,
                )
                
                # Update layout and axis titles
                fig.update_layout(
                    title=f"{selected_breakdown.replace('_', ' ')} Historical Analysis",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=40, b=40),
                    hovermode="x unified"
                )
                
                fig.update_yaxes(title_text=f"{selected_breakdown.replace('_', ' ')} (%)", secondary_y=False)
                fig.update_yaxes(title_text="Period-over-Period Change", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insights section
                st.subheader("Indicator Insights")
                
                # Generate insights based on the data
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("**Statistical Summary**")
                    
                    # Statistical summary
                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Current', 'Trend'],
                        'Value': [
                            f"{breakdown_data['value'].mean():.2f}%",
                            f"{breakdown_data['value'].median():.2f}%",
                            f"{breakdown_data['value'].min():.2f}%",
                            f"{breakdown_data['value'].max():.2f}%",
                            f"{breakdown_data['value'].std():.2f}%",
                            f"{current_value:.2f}%",
                            f"{trend_slope:+.3f} pts/period"
                        ]
                    })
                    
                    st.table(stats_df)
                
                with insights_col2:
                    st.markdown("**Indicator Context**")
                    
                    # Context information based on indicator
                    if selected_breakdown == 'GDP_growth':
                        if current_value < 0:
                            status = "Recession territory"
                            color = "red"
                        elif current_value < 2:
                            status = "Slow growth"
                            color = "orange"
                        elif current_value < 4:
                            status = "Moderate growth"
                            color = "green"
                        else:
                            status = "Strong growth"
                            color = "darkgreen"
                    elif selected_breakdown == 'Inflation':
                        if current_value < 1:
                            status = "Very low (deflation risk)"
                            color = "orange"
                        elif current_value < 2:
                            status = "Below target"
                            color = "lightgreen"
                        elif current_value <= 3:
                            status = "Target range"
                            color = "green"
                        else:
                            status = "Above target"
                            color = "red"
                    elif selected_breakdown == 'Unemployment':
                        if current_value < 4:
                            status = "Full employment"
                            color = "green"
                        elif current_value < 5:
                            status = "Strong labor market"
                            color = "lightgreen"
                        elif current_value < 7:
                            status = "Moderate unemployment"
                            color = "orange"
                        else:
                            status = "High unemployment"
                            color = "red"
                    elif selected_breakdown == 'Interest_Rate':
                        if current_value < 1:
                            status = "Very accommodative"
                            color = "blue"
                        elif current_value < 3:
                            status = "Accommodative"
                            color = "green"
                        elif current_value < 5:
                            status = "Neutral to restrictive"
                            color = "orange"
                        else:
                            status = "Highly restrictive"
                            color = "red"
                    else:
                        status = "Neutral"
                        color = "gray"
                    
                    st.markdown(f"**Current Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
                    
                    # Generate contextual interpretation
                    if trend_slope > 0.5:
                        trend_text = "rapidly increasing"
                    elif trend_slope > 0.1:
                        trend_text = "steadily increasing"
                    elif trend_slope > -0.1:
                        trend_text = "relatively stable"
                    elif trend_slope > -0.5:
                        trend_text = "steadily decreasing"
                    else:
                        trend_text = "rapidly decreasing"
                    
                    st.markdown(f"This indicator is currently **{trend_text}**. The current value of {current_value:.2f}% is {(current_value - breakdown_data['value'].mean()):.2f} points from the historical average.")
                    
                    # Provide economic context
                    if selected_breakdown == 'GDP_growth':
                        st.markdown("""**GDP Growth** is the primary measure of economic expansion. Growth rates:
                        - Below 0%: Recession
                        - 0-2%: Slow growth
                        - 2-4%: Healthy growth
                        - Above 4%: Rapid expansion""")
                    elif selected_breakdown == 'Inflation':
                        st.markdown("""**Inflation** measures price increases in the economy. Central banks typically target:
                        - Around 2%: Price stability
                        - Below 1%: Risk of deflation
                        - Above 3-4%: Rising inflation concern""")
                    elif selected_breakdown == 'Unemployment':
                        st.markdown("""**Unemployment Rate** indicates labor market health:
                        - Below 4%: Full employment
                        - 4-5%: Strong job market
                        - 5-6%: Moderate unemployment
                        - Above 6%: Elevated unemployment""")
                    elif selected_breakdown == 'Interest_Rate':
                        st.markdown("""**Interest Rates** reflect monetary policy stance:
                        - Below 1%: Very accommodative
                        - 1-3%: Accommodative
                        - 3-5%: Neutral to restrictive
                        - Above 5%: Highly restrictive""")
                    elif selected_breakdown == 'Consumer_Confidence':
                        st.markdown("""**Consumer Confidence** reflects economic outlook from consumers:
                        - Above 100: Optimistic outlook
                        - 80-100: Moderate confidence
                        - Below 80: Low confidence""")
                
                # Add economic indicator correlations if in Analyst role
                if user_role == "Analyst":
                    st.subheader("Indicator Correlations")
                    
                    # Create pivot table for correlation analysis
                    eco_pivot = economic_data.pivot(index='date', columns='indicator', values='value')
                    
                    # Calculate correlation matrix
                    corr_matrix = eco_pivot.corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title="Economic Indicator Correlation Matrix"
                    )
                    
                    fig.update_layout(
                        height=500,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display correlation with selected indicator
                    st.subheader(f"Correlations with {selected_breakdown.replace('_', ' ')}")
                    
                    # Get correlations with selected indicator
                    correlations = corr_matrix[selected_breakdown].drop(selected_breakdown).sort_values(ascending=False)
                    
                    # Create bar chart for correlations
                    fig = px.bar(
                        x=correlations.index,
                        y=correlations.values,
                        labels={'x': 'Indicator', 'y': 'Correlation Coefficient'},
                        title=f"Correlation with {selected_breakdown.replace('_', ' ')}",
                        color=correlations.values,
                        color_continuous_scale='RdBu_r',
                        text=correlations.values.round(2)
                    )
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explain strongest correlation
                    strongest_corr = correlations.index[0]
                    corr_value = correlations.iloc[0]
                    
                    if abs(corr_value) > 0.7:
                        strength = "strong"
                    elif abs(corr_value) > 0.4:
                        strength = "moderate"
                    else:
                        strength = "weak"
                    
                    direction = "positive" if corr_value > 0 else "negative"
                    
                    st.markdown(f"There is a **{strength} {direction} correlation** ({corr_value:.2f}) between {selected_breakdown.replace('_', ' ')} and {strongest_corr.replace('_', ' ')}.")
                    
                    if corr_value > 0:
                        st.markdown(f"This means that when {selected_breakdown.replace('_', ' ')} increases, {strongest_corr.replace('_', ' ')} tends to increase as well.")
                    else:
                        st.markdown(f"This means that when {selected_breakdown.replace('_', ' ')} increases, {strongest_corr.replace('_', ' ')} tends to decrease.")
        
        # Economic data export section (available to all roles)
        with st.expander("Export Economic Data"):
            st.markdown("Download the economic data for further analysis:")
            
            # Create download button for CSV
            csv = economic_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="economic_indicators.csv",
                mime="text/csv"
            )
            
            # Show data preview
            st.dataframe(economic_data.head())
    
    except Exception as e:
        st.error(f"Error loading economic indicators data: {e}")
        st.info("Please check your network connection and try refreshing the page.")# Add sidebar for dashboard settings
st.sidebar.title("Dashboard Settings")

# Theme selector
theme = st.sidebar.radio(
    "Display Theme",
    options=["Light", "Dark"]
)

# Update interval selector
update_interval = st.sidebar.slider(
    "Data refresh interval (minutes)",
    min_value=1,
    max_value=60,
    value=15
)

st.sidebar.info(f"Data will refresh every {update_interval} minutes")

# Add dashboard information
st.sidebar.markdown("---")
st.sidebar.subheader("About This Dashboard")
st.sidebar.markdown("""
This financial dashboard provides real-time insights into market performance, 
stock analysis, and economic indicators. 

**Features:**
- Market overview with key indices
- Detailed stock analysis with technical indicators
- Economic indicators tracking

**Data Sources:**
- Stock market data
- Cryptocurrency market data
- Economic indicators from public sources
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2023 Financial Dashboard | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
