# Financial Dashboard Application

A comprehensive financial performance dashboard that provides real-time market insights with interactive visualizations and role-based customization.

## Features

- **Market Overview**: Key financial metrics, trends, and market summaries
- **Stock Analysis**: Detailed stock information with technical indicators and alerts
- **Economic Indicators**: Enhanced economic data with health scoring and forecasts
- **Role-Based Views**: Customized dashboards for Executives, Investors, and Analysts
- **Standalone Application**: Can be packaged as a Windows executable

## Running the Dashboard

### Option 1: Run directly with Streamlit

Run the Streamlit application with the streamlit command.

### Option 2: Use the batch file (Windows)

Double-click the `FinancialDashboard.bat` file to start the application.

## Creating the Executable (Windows)

To build a standalone Windows executable:

1. Make sure all required packages are installed
2. Run the build script: `python build_exe.py`
3. Once the build completes, find the executable in the `dist` folder
4. To create a desktop shortcut, run: `python create_shortcut.py`

## Dashboard Sections

1. **Market Overview**
   - Key Performance Indicators
   - Market trends and summaries
   - Interactive filters

2. **Stock Analysis**
   - Stock performance details
   - Technical indicators (MACD, RSI, Bollinger Bands)
   - Price movement alerts

3. **Economic Indicators**
   - Economic health scoring
   - Role-based analytics views
   - Correlation analysis
   - Economic forecasts

## Role-Based Views

- **Executive**: High-level metrics, alerts, and forecasts
- **Investor**: Detailed stock information and investment opportunities
- **Analyst**: Advanced analytics, correlation analysis, and detailed breakdowns
