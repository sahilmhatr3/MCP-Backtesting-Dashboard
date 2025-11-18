import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def render_metrics(stats: Dict) -> None:
    """Render metrics cards."""
    if not stats:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", stats.get('total_return_pct', '0.00%'))
    
    with col2:
        st.metric("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}")
    
    with col3:
        st.metric("Max Drawdown", stats.get('max_drawdown_pct', '0.00%'))
    
    with col4:
        st.metric("Win Rate", stats.get('win_rate_pct', '0.00%'))


def render_price_chart(data: pd.DataFrame, strategy_name: str, strategy_params: Dict) -> None:
    """Render price chart with indicators."""
    if data is None or data.empty:
        return
    
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Chart", "Volume")
    )
    
    data_sorted = data.sort_values('Date').copy()
    
    fig.add_trace(
        go.Scatter(
            x=data_sorted['Date'],
            y=data_sorted['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    if strategy_name == "MovingAverageCrossover":
        fast_period = strategy_params.get('fast_period', 10)
        slow_period = strategy_params.get('slow_period', 30)
        
        data_sorted['Fast_MA'] = data_sorted['Close'].rolling(window=fast_period).mean()
        data_sorted['Slow_MA'] = data_sorted['Close'].rolling(window=slow_period).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['Fast_MA'],
                name=f'MA {fast_period}',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['Slow_MA'],
                name=f'MA {slow_period}',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Bar(
            x=data_sorted['Date'],
            y=data_sorted['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def render_trade_log(trade_log: Optional[pd.DataFrame]) -> None:
    """Render trade log table."""
    if trade_log is None or trade_log.empty:
        st.info("No trades generated in this backtest.")
        return
    
    st.subheader("Trade Log")
    st.dataframe(trade_log, use_container_width=True)


def render_ai_insights(stats: Dict, trade_log: Optional[pd.DataFrame], sentiment_data: Optional[list]) -> None:
    """Render AI insights section."""
    st.subheader("AI Insights")
    
    if st.button("Generate Analysis", type="primary"):
        try:
            from llm.openai_client import OpenAIClient
            with st.spinner("Generating AI insights..."):
                client = OpenAIClient()
                explanation = client.explain_results(stats, trade_log, sentiment_data)
                st.markdown(f"**Analysis:**\n\n{explanation}")
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")


def render_export_options(stats: Dict, trade_log: Optional[pd.DataFrame]) -> None:
    """Render export options."""
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if trade_log is not None and not trade_log.empty:
            csv = trade_log.to_csv(index=False)
            st.download_button(
                label="Download Trade Log (CSV)",
                data=csv,
                file_name="trade_log.csv",
                mime="text/csv"
            )
    
    with col2:
        if stats:
            report = f"""# Backtest Report

## Statistics
- Total Return: {stats.get('total_return_pct', '0.00%')}
- Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {stats.get('max_drawdown_pct', '0.00%')}
- Win Rate: {stats.get('win_rate_pct', '0.00%')}
- Total Trades: {stats.get('total_trades', 0)}

## Trade Log
{trade_log.to_string() if trade_log is not None and not trade_log.empty else 'No trades'}
"""
            st.download_button(
                label="Download Report (Markdown)",
                data=report,
                file_name="backtest_report.md",
                mime="text/markdown"
            )

