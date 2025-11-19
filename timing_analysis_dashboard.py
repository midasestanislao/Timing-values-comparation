"""
Interactive Time-Series Performance Comparison Dashboard
========================================================
Compare performance timing data between DEMO and PROD environments
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Performance Comparison Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file_path):
    """Load and preprocess the CSV data."""
    try:
        df = pd.read_csv(file_path)
        df['Call Started'] = pd.to_datetime(df['Call Started'], format='%b %d, %I:%M %p', errors='coerce')
        df['Call Ended'] = pd.to_datetime(df['Call Ended'], format='%b %d, %I:%M %p', errors='coerce')
        df = df.dropna(subset=['Call Started', 'Call Ended'])
        
        current_year = datetime.now().year
        df['Call Started'] = df['Call Started'].apply(lambda x: x.replace(year=current_year))
        df['Call Ended'] = df['Call Ended'].apply(lambda x: x.replace(year=current_year))
        
        df['Date'] = pd.to_datetime(df['Call Started'].dt.date)
        df['Hour'] = df['Call Started'].dt.hour
        df['Day_of_Week'] = df['Call Started'].dt.day_name()
        df['Week'] = df['Call Started'].dt.isocalendar().week
        df['Month'] = df['Call Started'].dt.month
        df['Month_Name'] = df['Call Started'].dt.strftime('%B')
        df['Datetime'] = df['Call Started']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_statistics(data, metric_col='Total'):
    """Calculate key statistics including percentiles."""
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'p50': data.quantile(0.50),
        'p75': data.quantile(0.75),
        'p90': data.quantile(0.90),
        'p95': data.quantile(0.95),
        'p99': data.quantile(0.99),
        'count': len(data)
    }

def plot_comparison_time_series(df_demo, df_prod, date_col, value_col, title, percentile=90, aggregation='mean'):
    """Create comparative time series plot for DEMO vs PROD."""
    
    # Aggregate data
    if aggregation == 'mean':
        demo_agg = df_demo.groupby(date_col)[value_col].mean().reset_index()
        prod_agg = df_prod.groupby(date_col)[value_col].mean().reset_index()
        agg_label = 'Mean'
    elif aggregation == 'median':
        demo_agg = df_demo.groupby(date_col)[value_col].median().reset_index()
        prod_agg = df_prod.groupby(date_col)[value_col].median().reset_index()
        agg_label = 'Median'
    else:
        demo_agg = df_demo.groupby(date_col)[value_col].max().reset_index()
        prod_agg = df_prod.groupby(date_col)[value_col].max().reset_index()
        agg_label = 'Max'
    
    # Calculate percentiles
    demo_p = df_demo[value_col].quantile(percentile / 100)
    prod_p = df_prod[value_col].quantile(percentile / 100)
    
    fig = go.Figure()
    
    # DEMO line
    fig.add_trace(go.Scatter(
        x=demo_agg[date_col],
        y=demo_agg[value_col],
        mode='lines+markers',
        name=f'DEMO {agg_label}',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=6),
        hovertemplate=f'<b>DEMO</b><br>Date: %{{x}}<br>{agg_label}: %{{y:.3f}}s<extra></extra>'
    ))
    
    # PROD line
    fig.add_trace(go.Scatter(
        x=prod_agg[date_col],
        y=prod_agg[value_col],
        mode='lines+markers',
        name=f'PROD {agg_label}',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(size=6),
        hovertemplate=f'<b>PROD</b><br>Date: %{{x}}<br>{agg_label}: %{{y:.3f}}s<extra></extra>'
    ))
    
    # DEMO percentile
    fig.add_hline(
        y=demo_p,
        line_dash="dash",
        line_color="#FF6B6B",
        opacity=0.5,
        annotation_text=f"DEMO P{percentile}: {demo_p:.3f}s",
        annotation_position="left"
    )
    
    # PROD percentile
    fig.add_hline(
        y=prod_p,
        line_dash="dash",
        line_color="#4ECDC4",
        opacity=0.5,
        annotation_text=f"PROD P{percentile}: {prod_p:.3f}s",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=f'{value_col} (seconds)',
        hovermode='x unified',
        template=None,
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_comparison_components(df_demo, df_prod, date_col, title):
    """Create stacked area chart comparison for components."""
    demo_agg = df_demo.groupby(date_col)[['EOU', 'LLM', 'TTS']].mean().reset_index()
    prod_agg = df_prod.groupby(date_col)[['EOU', 'LLM', 'TTS']].mean().reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('DEMO Environment', 'PROD Environment'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # DEMO stacked area
    fig.add_trace(go.Scatter(
        x=demo_agg[date_col], y=demo_agg['EOU'],
        mode='lines', name='DEMO EOU', stackgroup='demo',
        fillcolor='rgba(255, 107, 107, 0.6)',
        line=dict(width=0.5, color='rgb(255, 107, 107)'),
        legendgroup='demo',
        hovertemplate='<b>EOU:</b> %{y:.3f}s<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=demo_agg[date_col], y=demo_agg['LLM'],
        mode='lines', name='DEMO LLM', stackgroup='demo',
        fillcolor='rgba(255, 159, 64, 0.6)',
        line=dict(width=0.5, color='rgb(255, 159, 64)'),
        legendgroup='demo',
        hovertemplate='<b>LLM:</b> %{y:.3f}s<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=demo_agg[date_col], y=demo_agg['TTS'],
        mode='lines', name='DEMO TTS', stackgroup='demo',
        fillcolor='rgba(255, 205, 86, 0.6)',
        line=dict(width=0.5, color='rgb(255, 205, 86)'),
        legendgroup='demo',
        hovertemplate='<b>TTS:</b> %{y:.3f}s<extra></extra>'
    ), row=1, col=1)
    
    # PROD stacked area
    fig.add_trace(go.Scatter(
        x=prod_agg[date_col], y=prod_agg['EOU'],
        mode='lines', name='PROD EOU', stackgroup='prod',
        fillcolor='rgba(78, 205, 196, 0.6)',
        line=dict(width=0.5, color='rgb(78, 205, 196)'),
        legendgroup='prod',
        hovertemplate='<b>EOU:</b> %{y:.3f}s<extra></extra>'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=prod_agg[date_col], y=prod_agg['LLM'],
        mode='lines', name='PROD LLM', stackgroup='prod',
        fillcolor='rgba(54, 162, 235, 0.6)',
        line=dict(width=0.5, color='rgb(54, 162, 235)'),
        legendgroup='prod',
        hovertemplate='<b>LLM:</b> %{y:.3f}s<extra></extra>'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=prod_agg[date_col], y=prod_agg['TTS'],
        mode='lines', name='PROD TTS', stackgroup='prod',
        fillcolor='rgba(75, 192, 192, 0.6)',
        line=dict(width=0.5, color='rgb(75, 192, 192)'),
        legendgroup='prod',
        hovertemplate='<b>TTS:</b> %{y:.3f}s<extra></extra>'
    ), row=1, col=2)
    
    fig.update_layout(
        title=title,
        height=500,
        hovermode='x unified',
        template=None
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
    
    return fig

def plot_comparison_bars(stats_demo, stats_prod, metric_name):
    """Create side-by-side bar comparison."""
    metrics = ['mean', 'median', 'p90', 'p95', 'max']
    metric_labels = ['Mean', 'Median', 'P90', 'P95', 'Max']
    
    demo_values = [stats_demo[m] for m in metrics]
    prod_values = [stats_prod[m] for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='DEMO',
        x=metric_labels,
        y=demo_values,
        marker_color='#FF6B6B',
        text=[f'{v:.3f}s' for v in demo_values],
        textposition='outside',
        hovertemplate='<b>DEMO %{x}</b><br>Value: %{y:.3f}s<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='PROD',
        x=metric_labels,
        y=prod_values,
        marker_color='#4ECDC4',
        text=[f'{v:.3f}s' for v in prod_values],
        textposition='outside',
        hovertemplate='<b>PROD %{x}</b><br>Value: %{y:.3f}s<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{metric_name} Statistics Comparison',
        xaxis_title='Metric',
        yaxis_title='Time (seconds)',
        barmode='group',
        template=None,
        height=400,
        showlegend=True
    )
    
    return fig

def plot_difference_analysis(df_demo, df_prod, date_col, value_col, title, aggregation='mean'):
    """Show the difference between DEMO and PROD over time."""
    
    if aggregation == 'mean':
        demo_agg = df_demo.groupby(date_col)[value_col].mean().reset_index()
        prod_agg = df_prod.groupby(date_col)[value_col].mean().reset_index()
    elif aggregation == 'median':
        demo_agg = df_demo.groupby(date_col)[value_col].median().reset_index()
        prod_agg = df_prod.groupby(date_col)[value_col].median().reset_index()
    else:
        demo_agg = df_demo.groupby(date_col)[value_col].max().reset_index()
        prod_agg = df_prod.groupby(date_col)[value_col].max().reset_index()
    
    # Merge on date
    merged = demo_agg.merge(prod_agg, on=date_col, suffixes=('_demo', '_prod'))
    merged['difference'] = merged[f'{value_col}_demo'] - merged[f'{value_col}_prod']
    merged['percent_diff'] = (merged['difference'] / merged[f'{value_col}_prod']) * 100
    
    fig = go.Figure()
    
    colors = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in merged['difference']]
    
    fig.add_trace(go.Bar(
        x=merged[date_col],
        y=merged['difference'],
        marker_color=colors,
        name='Difference',
        hovertemplate='<b>Date:</b> %{x}<br><b>Difference:</b> %{y:.3f}s<br><b>DEMO-PROD</b><extra></extra>',
        text=[f'{x:+.2f}s' for x in merged['difference']],
        textposition='outside'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Difference (DEMO - PROD) in seconds',
        template=None,
        height=400,
        showlegend=False
    )
    
    return fig

def plot_percentile_comparison(df_demo, df_prod, metrics=['Total', 'EOU', 'LLM', 'TTS']):
    """Create percentile comparison across metrics."""
    
    percentiles = [50, 75, 90, 95, 99]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, (metric, pos) in enumerate(zip(metrics, positions)):
        demo_percentiles = [df_demo[metric].quantile(p/100) for p in percentiles]
        prod_percentiles = [df_prod[metric].quantile(p/100) for p in percentiles]
        percentile_labels = [f'P{p}' for p in percentiles]
        
        fig.add_trace(
            go.Scatter(
                x=percentile_labels,
                y=demo_percentiles,
                mode='lines+markers',
                name='DEMO' if idx == 0 else None,
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=8),
                showlegend=(idx == 0),
                legendgroup='demo',
                hovertemplate='<b>DEMO</b><br>%{x}: %{y:.3f}s<extra></extra>'
            ),
            row=pos[0], col=pos[1]
        )
        
        fig.add_trace(
            go.Scatter(
                x=percentile_labels,
                y=prod_percentiles,
                mode='lines+markers',
                name='PROD' if idx == 0 else None,
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=8),
                showlegend=(idx == 0),
                legendgroup='prod',
                hovertemplate='<b>PROD</b><br>%{x}: %{y:.3f}s<extra></extra>'
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        title_text="Percentile Comparison Across All Metrics",
        height=700,
        template=None
    )
    
    fig.update_yaxes(title_text="Time (seconds)")
    
    return fig

def create_summary_table(stats_demo, stats_prod, metrics=['Total', 'EOU', 'LLM', 'TTS']):
    """Create comprehensive comparison table."""
    
    summary_data = []
    
    for metric in metrics:
        demo_stats = calculate_statistics(stats_demo[metric])
        prod_stats = calculate_statistics(stats_prod[metric])
        
        for stat_name in ['mean', 'median', 'p90', 'p95', 'max']:
            demo_val = demo_stats[stat_name]
            prod_val = prod_stats[stat_name]
            diff = demo_val - prod_val
            pct_diff = (diff / prod_val * 100) if prod_val != 0 else 0
            
            summary_data.append({
                'Metric': metric,
                'Statistic': stat_name.upper(),
                'DEMO (s)': round(demo_val, 3),
                'PROD (s)': round(prod_val, 3),
                'Difference (s)': round(diff, 3),
                'Difference (%)': round(pct_diff, 1)
            })
    
    return pd.DataFrame(summary_data)

def main():
    """Main application function."""
    
    st.title("📊 Performance Comparison Dashboard")
    st.subheader("DEMO vs PROD Environment Analysis")
    st.divider()
    
    # Sidebar controls
    st.sidebar.title("📋 Analysis Controls")
    st.sidebar.divider()
    
    # File path inputs
    with st.sidebar.expander("📁 File Paths", expanded=False):
        demo_path = st.text_input(
            "DEMO file path",
            value="./ticket_timing_data_DEMO.csv",
            help="Path to DEMO CSV file"
        )
        prod_path = st.text_input(
            "PROD file path",
            value="./ticket_timing_data_PROD.csv",
            help="Path to PROD CSV file"
        )
    
    # Load data
    df_demo = load_data(demo_path)
    df_prod = load_data(prod_path)
    
    if df_demo is None or df_prod is None or df_demo.empty or df_prod.empty:
        st.error("Unable to load data. Please check the file paths.")
        return
    
    view_type = st.sidebar.selectbox(
        "Time Aggregation",
        ["Daily", "Weekly", "Monthly", "Full Period"]
    )
    
    metric_to_analyze = st.sidebar.selectbox(
        "Primary Metric",
        ["Total", "EOU", "LLM", "TTS"]
    )
    
    percentile_value = st.sidebar.slider(
        "Percentile to Highlight",
        min_value=50,
        max_value=99,
        value=90,
        step=5
    )
    
    agg_method = st.sidebar.radio(
        "Aggregation Method",
        ["mean", "median", "max"]
    )
    
    st.sidebar.divider()
    st.sidebar.info("💡 Hover over charts for details. Use zoom and pan tools.")
    
    # Overview Statistics
    st.header("📈 Overview Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "DEMO Measurements",
            f"{len(df_demo):,}",
            help="Total measurements in DEMO"
        )
    
    with col2:
        st.metric(
            "PROD Measurements",
            f"{len(df_prod):,}",
            help="Total measurements in PROD"
        )
    
    with col3:
        st.metric(
            "DEMO Tickets",
            f"{df_demo['Ticket ID'].nunique()}",
            help="Unique tickets in DEMO"
        )
    
    with col4:
        st.metric(
            "PROD Tickets",
            f"{df_prod['Ticket ID'].nunique()}",
            help="Unique tickets in PROD"
        )
    
    st.divider()
    
    # Key Performance Indicators Comparison
    st.header(f"🎯 Key Performance Indicators - {metric_to_analyze}")
    
    stats_demo = calculate_statistics(df_demo[metric_to_analyze])
    stats_prod = calculate_statistics(df_prod[metric_to_analyze])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_to_show = [
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('90th %ile', 'p90'),
        ('95th %ile', 'p95'),
        ('Max', 'max')
    ]
    
    for col, (label, key) in zip([col1, col2, col3, col4, col5], metrics_to_show):
        demo_val = stats_demo[key]
        prod_val = stats_prod[key]
        diff = demo_val - prod_val
        pct_diff = (diff / prod_val * 100) if prod_val != 0 else 0
        
        with col:
            st.metric(
                label,
                f"D: {demo_val:.3f}s",
                delta=None
            )
            st.metric(
                "",
                f"P: {prod_val:.3f}s",
                delta=f"{diff:+.3f}s ({pct_diff:+.1f}%)"
            )
    
    st.divider()
    
    # Statistics Bar Comparison
    st.header("📊 Statistics Comparison")
    fig_bars = plot_comparison_bars(stats_demo, stats_prod, metric_to_analyze)
    st.plotly_chart(fig_bars, use_container_width=True)
    
    st.divider()
    
    # Main visualizations based on selected view
    st.header(f"📈 {view_type} Comparative Analysis")
    
    date_col_map = {
        "Daily": "Date",
        "Weekly": "Week_Year",
        "Monthly": "Month_Year",
        "Full Period": "Datetime"
    }
    
    # Add week/month identifiers if needed
    if view_type == "Weekly":
        df_demo['Week_Year'] = df_demo['Call Started'].dt.strftime('%Y-W%W')
        df_prod['Week_Year'] = df_prod['Call Started'].dt.strftime('%Y-W%W')
    elif view_type == "Monthly":
        df_demo['Month_Year'] = df_demo['Call Started'].dt.strftime('%Y-%m')
        df_prod['Month_Year'] = df_prod['Call Started'].dt.strftime('%Y-%m')
    
    date_col = date_col_map[view_type]
    
    # Time series comparison
    st.subheader(f"{view_type} Performance Trends")
    fig1 = plot_comparison_time_series(
        df_demo, df_prod, date_col, metric_to_analyze,
        f"{view_type} {agg_method.capitalize()} {metric_to_analyze} - DEMO vs PROD",
        percentile=percentile_value,
        aggregation=agg_method
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Difference analysis
    st.subheader(f"Performance Difference (DEMO - PROD)")
    fig_diff = plot_difference_analysis(
        df_demo, df_prod, date_col, metric_to_analyze,
        f"{view_type} Difference in {metric_to_analyze}",
        aggregation=agg_method
    )
    st.plotly_chart(fig_diff, use_container_width=True)
    
    # Component breakdown comparison
    st.subheader(f"{view_type} Component Breakdown")
    fig2 = plot_comparison_components(
        df_demo, df_prod, date_col,
        f"{view_type} Response Time Components Comparison"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    
    # Percentile comparison across all metrics
    st.header("📉 Percentile Analysis Across All Metrics")
    fig_percentiles = plot_percentile_comparison(df_demo, df_prod)
    st.plotly_chart(fig_percentiles, use_container_width=True)
    
    st.divider()
    
    # Comprehensive comparison table
    st.header("📋 Comprehensive Comparison Table")
    summary_df = create_summary_table(df_demo, df_prod)
    
    # Style the dataframe
    def color_difference(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'background-color: rgba(255, 107, 107, 0.3)'
            elif val < 0:
                return 'background-color: rgba(78, 205, 196, 0.3)'
        return ''
    
    styled_df = summary_df.style.applymap(
        color_difference,
        subset=['Difference (s)', 'Difference (%)']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    st.divider()
    
    # Additional insights
    st.header("🔍 Additional Insights")
    
    tab1, tab2 = st.tabs(["Environment Comparison Summary", "Metric Breakdown"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 DEMO Environment Summary")
            for metric in ['Total', 'EOU', 'LLM', 'TTS']:
                stats = calculate_statistics(df_demo[metric])
                st.write(f"**{metric}**: Mean={stats['mean']:.3f}s, P90={stats['p90']:.3f}s")
        
        with col2:
            st.subheader("🟢 PROD Environment Summary")
            for metric in ['Total', 'EOU', 'LLM', 'TTS']:
                stats = calculate_statistics(df_prod[metric])
                st.write(f"**{metric}**: Mean={stats['mean']:.3f}s, P90={stats['p90']:.3f}s")
    
    with tab2:
        st.subheader("Component Performance Comparison")
        
        components = ['EOU', 'LLM', 'TTS']
        comp_data = []
        
        for comp in components:
            demo_stats = calculate_statistics(df_demo[comp])
            prod_stats = calculate_statistics(df_prod[comp])
            
            comp_data.append({
                'Component': comp,
                'DEMO Mean': f"{demo_stats['mean']:.3f}s",
                'PROD Mean': f"{prod_stats['mean']:.3f}s",
                'DEMO P90': f"{demo_stats['p90']:.3f}s",
                'PROD P90': f"{prod_stats['p90']:.3f}s",
                'Improvement': f"{((demo_stats['mean'] - prod_stats['mean']) / demo_stats['mean'] * 100):+.1f}%"
            })
        
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)
    
    # Export section
    st.divider()
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_summary = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Summary",
            data=csv_summary,
            file_name="demo_vs_prod_comparison.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create combined dataset
        df_demo_export = df_demo.copy()
        df_demo_export['Environment'] = 'DEMO'
        df_prod_export = df_prod.copy()
        df_prod_export['Environment'] = 'PROD'
        combined_df = pd.concat([df_demo_export, df_prod_export], ignore_index=True)
        
        csv_combined = combined_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Dataset",
            data=csv_combined,
            file_name="demo_prod_combined.csv",
            mime="text/csv"
        )
    
    # Footer
    st.divider()
    st.caption("Performance Comparison Dashboard | DEMO vs PROD Analysis")
    st.caption("💡 Use the sidebar to customize your comparison analysis")

if __name__ == "__main__":
    main()