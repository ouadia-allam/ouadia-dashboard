import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Function to generate stock market data (since we don't have a real API)
def get_stock_data():
    """
    Get stock market data for various symbols.
    In a real application, this would fetch from an API.
    """
    # Define stock symbols
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NFLX', 'SPY']
    
    # Create date range for the past 365 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Initialize empty dataframe
    stock_data = pd.DataFrame()
    
    # Generate data for each symbol
    for symbol in symbols:
        # Set base price and volatility based on symbol
        if symbol == 'AAPL':
            base_price = 150.0
            volatility = 0.02
        elif symbol == 'MSFT':
            base_price = 300.0
            volatility = 0.015
        elif symbol == 'AMZN':
            base_price = 130.0
            volatility = 0.025
        elif symbol == 'GOOGL':
            base_price = 120.0
            volatility = 0.02
        elif symbol == 'TSLA':
            base_price = 250.0
            volatility = 0.04
        elif symbol == 'META':
            base_price = 300.0
            volatility = 0.03
        elif symbol == 'NFLX':
            base_price = 400.0
            volatility = 0.025
        elif symbol == 'SPY':
            base_price = 450.0
            volatility = 0.01
        
        # Generate price series with random walk
        np.random.seed(int(hash(symbol) % 2**32))
        price_changes = np.random.normal(0.0005, volatility, len(dates))
        prices = [base_price]
        
        for change in price_changes:
            prices.append(prices[-1] * (1 + change))
        
        prices = prices[1:]  # Remove the initial base price
        
        # Create temporary dataframe for this symbol
        temp_df = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'open': [price * (1 - np.random.uniform(0, 0.005)) for price in prices],
            'high': [price * (1 + np.random.uniform(0, 0.01)) for price in prices],
            'low': [price * (1 - np.random.uniform(0, 0.01)) for price in prices],
            'close': prices,
            'volume': [int(np.random.uniform(1000000, 10000000)) for _ in prices]
        })
        
        # Append to main dataframe
        stock_data = pd.concat([stock_data, temp_df])
    
    # Reset index
    stock_data = stock_data.reset_index(drop=True)
    
    return stock_data

# Function to generate cryptocurrency market data
def get_crypto_data():
    """
    Get cryptocurrency market data.
    In a real application, this would fetch from an API.
    """
    # Define cryptocurrency symbols
    symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'ADA']
    
    # Create date range for the past 365 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize empty dataframe
    crypto_data = pd.DataFrame()
    
    # Generate data for each symbol
    for symbol in symbols:
        # Set base price and volatility based on symbol
        if symbol == 'BTC':
            base_price = 30000.0
            volatility = 0.03
        elif symbol == 'ETH':
            base_price = 2000.0
            volatility = 0.035
        elif symbol == 'XRP':
            base_price = 0.5
            volatility = 0.04
        elif symbol == 'LTC':
            base_price = 80.0
            volatility = 0.03
        elif symbol == 'ADA':
            base_price = 0.3
            volatility = 0.045
        
        # Generate price series with random walk
        np.random.seed(int(hash(symbol) % 2**32))
        price_changes = np.random.normal(0.001, volatility, len(dates))
        prices = [base_price]
        
        for change in price_changes:
            prices.append(prices[-1] * (1 + change))
        
        prices = prices[1:]  # Remove the initial base price
        
        # Create temporary dataframe for this symbol
        temp_df = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'price': prices,
            'volume': [int(np.random.uniform(10000000, 100000000)) for _ in prices],
            'market_cap': [price * np.random.uniform(10000000, 100000000) for price in prices]
        })
        
        # Append to main dataframe
        crypto_data = pd.concat([crypto_data, temp_df])
    
    # Reset index
    crypto_data = crypto_data.reset_index(drop=True)
    
    return crypto_data

# Function to generate economic indicators data
def get_economic_data():
    """
    Get economic indicators data.
    In a real application, this would fetch from an API.
    """
    # Define economic indicators
    indicators = ['GDP_growth', 'Inflation', 'Unemployment', 'Interest_Rate', 'Consumer_Confidence']
    
    # Create date range for quarterly data for past 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    
    # Initialize empty dataframe
    economic_data = pd.DataFrame()
    
    # Generate data for each indicator
    for indicator in indicators:
        # Set base value and volatility based on indicator
        if indicator == 'GDP_growth':
            base_value = 2.5
            volatility = 0.5
            trend = -0.1  # Slight downward trend
        elif indicator == 'Inflation':
            base_value = 2.0
            volatility = 0.3
            trend = 0.2  # Slight upward trend
        elif indicator == 'Unemployment':
            base_value = 4.0
            volatility = 0.2
            trend = 0.0  # No trend
        elif indicator == 'Interest_Rate':
            base_value = 3.0
            volatility = 0.25
            trend = 0.1  # Slight upward trend
        elif indicator == 'Consumer_Confidence':
            base_value = 100.0
            volatility = 3.0
            trend = -0.5  # Downward trend
        
        # Generate value series with random walk and trend
        np.random.seed(int(hash(indicator) % 2**32))
        value_changes = np.random.normal(trend/len(dates), volatility, len(dates))
        values = [base_value]
        
        for change in value_changes:
            values.append(values[-1] + change)
        
        values = values[1:]  # Remove the initial base value
        
        # Create temporary dataframe for this indicator
        temp_df = pd.DataFrame({
            'date': dates,
            'indicator': indicator,
            'value': values
        })
        
        # Append to main dataframe
        economic_data = pd.concat([economic_data, temp_df])
    
    # Reset index
    economic_data = economic_data.reset_index(drop=True)
    
    return economic_data
