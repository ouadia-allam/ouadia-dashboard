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
        st.info("Please check your network connection and try refreshing the page.")