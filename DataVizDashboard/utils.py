import streamlit as st

def format_large_number(num):
    """
    Format large numbers to K, M, B notation.
    
    Args:
        num (float or int): The number to format
        
    Returns:
        str: Formatted number string
    """
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.2f}"

def create_kpi_metric(title, value, change):
    """
    Create a KPI metric with title, value and change indicator.
    
    Args:
        title (str): The KPI title
        value (str): The current value
        change (str): The change value (with % or other indicator)
    """
    # Determine if change is positive, negative or neutral for color coding
    if change:
        if change.startswith('+') or (not change.startswith('-') and float(change.strip('%+').strip()) > 0):
            delta_color = "normal"  # Green in Streamlit
        elif change.startswith('-') or float(change.strip('%+').strip()) < 0:
            delta_color = "inverse"  # Red in Streamlit
        else:
            delta_color = "off"  # Gray in Streamlit
    else:
        delta_color = "off"
        change = "0%"
    
    # Display metric with delta
    st.metric(
        label=title,
        value=value,
        delta=change,
        delta_color=delta_color
    )
