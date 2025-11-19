import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List

from llm.openai_client import OpenAIClient


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
    
    data_sorted = data.sort_values('Date').copy()
    
    if strategy_name == "MACDStrategy":
        fast_period = strategy_params.get('fast_period', 12)
        slow_period = strategy_params.get('slow_period', 26)
        signal_period = strategy_params.get('signal_period', 9)
        
        exp1 = data_sorted['Close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data_sorted['Close'].ewm(span=slow_period, adjust=False).mean()
        data_sorted['MACD'] = exp1 - exp2
        data_sorted['Signal'] = data_sorted['MACD'].ewm(span=signal_period, adjust=False).mean()
        data_sorted['Histogram'] = data_sorted['MACD'] - data_sorted['Signal']
        
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=("Price Chart", "MACD Indicator", "Volume")
        )
        
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
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        colors = ['green' if val >= 0 else 'red' for val in data_sorted['Histogram']]
        fig.add_trace(
            go.Bar(
                x=data_sorted['Date'],
                y=data_sorted['Histogram'],
                name='Histogram',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=data_sorted['Date'],
                y=data_sorted['Volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        return
    
    if strategy_name == "BollingerBandsStrategy":
        period = strategy_params.get('period', 20)
        devfactor = strategy_params.get('devfactor', 2.0)
        
        data_sorted['SMA'] = data_sorted['Close'].rolling(window=period).mean()
        data_sorted['STD'] = data_sorted['Close'].rolling(window=period).std()
        data_sorted['Upper'] = data_sorted['SMA'] + (data_sorted['STD'] * devfactor)
        data_sorted['Lower'] = data_sorted['SMA'] - (data_sorted['STD'] * devfactor)
        
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price Chart with Bollinger Bands", "Volume")
        )
        
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
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['Upper'],
                mode='lines',
                name=f'Upper Band ({devfactor}σ)',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['SMA'],
                mode='lines',
                name=f'SMA ({period})',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['Lower'],
                mode='lines',
                name=f'Lower Band ({devfactor}σ)',
                line=dict(color='green', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)'
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
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        return
    
    if strategy_name == "MomentumStrategy":
        period = strategy_params.get('period', 10)
        threshold = strategy_params.get('threshold', 0.02)
        
        data_sorted['Momentum'] = ((data_sorted['Close'] / data_sorted['Close'].shift(period)) - 1) * 100
        data_sorted['Momentum_Pct'] = data_sorted['Momentum'] / 100
        
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=("Price Chart", "Momentum Indicator", "Volume")
        )
        
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
        
        colors = ['green' if val >= threshold * 100 else 'red' if val <= -threshold * 100 else 'gray' for val in data_sorted['Momentum']]
        fig.add_trace(
            go.Bar(
                x=data_sorted['Date'],
                y=data_sorted['Momentum'],
                name='Momentum (%)',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=[threshold * 100] * len(data_sorted),
                mode='lines',
                name=f'Buy Threshold ({threshold*100:.1f}%)',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=[-threshold * 100] * len(data_sorted),
                mode='lines',
                name=f'Sell Threshold ({-threshold*100:.1f}%)',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=[0] * len(data_sorted),
                mode='lines',
                name='Zero Line',
                line=dict(color='gray', width=1, dash='dot')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=data_sorted['Date'],
                y=data_sorted['Volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Momentum (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        return
    
    if strategy_name == "RSIStrategy":
        rsi_period = strategy_params.get('rsi_period', 14)
        rsi_oversold = strategy_params.get('rsi_oversold', 30)
        rsi_overbought = strategy_params.get('rsi_overbought', 70)
        
        delta = data_sorted['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        data_sorted['RSI'] = 100 - (100 / (1 + rs))
        
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=("Price Chart", "RSI Indicator", "Volume")
        )
        
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
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=[rsi_oversold] * len(data_sorted),
                mode='lines',
                name=f'Oversold ({rsi_oversold})',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=[rsi_overbought] * len(data_sorted),
                mode='lines',
                name=f'Overbought ({rsi_overbought})',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=data_sorted['Date'],
                y=data_sorted['Volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        return
    
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price Chart", "Volume")
    )
    
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
                name=f'Fast MA ({fast_period})',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['Slow_MA'],
                name=f'Slow MA ({slow_period})',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
    
    elif strategy_name == "SentimentStrategy":
        data_sorted['SMA_20'] = data_sorted['Close'].rolling(window=20).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data_sorted['Date'],
                y=data_sorted['SMA_20'],
                name='SMA (20)',
                line=dict(color='orange', width=1, dash='dash')
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
    
    csv_data = trade_log.to_csv(index=False)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="trade_log.csv",
            mime="text/csv",
            key="trade_log_download"
        )
    with col2:
        show_copy = st.checkbox("Show Copyable Text", value=st.session_state.get('show_trade_copy', False), key="show_copy_checkbox")
        st.session_state.show_trade_copy = show_copy
    
    st.dataframe(trade_log, use_container_width=True)
    
    if st.session_state.get('show_trade_copy', False):
        st.markdown("**Copy to Clipboard (select all text below and copy):**")
        st.text_area(
            "Trade Log CSV",
            value=csv_data,
            height=200,
            key="trade_log_text_area",
            label_visibility="collapsed"
        )


def render_sentiment_details(sentiment_data: Optional[List[dict]], ticker: Optional[str] = None) -> None:
    """Render sentiment analysis details and news feed."""
    if not sentiment_data:
        return
    
    st.subheader("Sentiment Analysis & News Feed")
    
    filtered_data = sentiment_data
    if ticker:
        filtered_data = [item for item in sentiment_data if item.get('ticker') == ticker]
    
    if not filtered_data:
        st.info("No sentiment data available for this ticker.")
        return
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for item in filtered_data:
        sentiment = item.get('sentiment', 'neutral')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    total = len(filtered_data)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", total)
    with col2:
        pct = (sentiment_counts['positive'] / total * 100) if total > 0 else 0
        st.metric("Positive", f"{sentiment_counts['positive']} ({pct:.1f}%)")
    with col3:
        pct = (sentiment_counts['negative'] / total * 100) if total > 0 else 0
        st.metric("Negative", f"{sentiment_counts['negative']} ({pct:.1f}%)")
    with col4:
        pct = (sentiment_counts['neutral'] / total * 100) if total > 0 else 0
        st.metric("Neutral", f"{sentiment_counts['neutral']} ({pct:.1f}%)")
    
    st.markdown("---")
    
    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        ["All", "Positive", "Negative", "Neutral"],
        key="sentiment_filter"
    )
    
    filtered_articles = filtered_data
    if sentiment_filter != "All":
        filtered_articles = [item for item in filtered_data if item.get('sentiment', 'neutral').lower() == sentiment_filter.lower()]
    
    if filtered_articles:
        st.markdown(f"### News Articles ({len(filtered_articles)} found)")
        
        for idx, article in enumerate(filtered_articles[:50]):
            sentiment = article.get('sentiment', 'neutral')
            sentiment_label = {
                'positive': '[+]',
                'negative': '[-]',
                'neutral': '[~]'
            }.get(sentiment, '[?]')
            
            with st.expander(f"{sentiment_label} {article.get('headline', 'No headline')[:80]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Date:** {article.get('date', 'Unknown')}")
                    st.write(f"**Source:** {article.get('source', 'Unknown')}")
                    if article.get('url'):
                        st.markdown(f"[Read full article →]({article.get('url')})")
                with col2:
                    st.write(f"**Sentiment:** {sentiment.upper()}")
    else:
        st.info("No articles found with the selected filter.")


def _create_cumulative_pnl_chart(trade_log: pd.DataFrame, height: int = 200) -> Optional[go.Figure]:
    """Create cumulative PnL chart."""
    if 'exit_date' not in trade_log.columns or 'pnl' not in trade_log.columns:
        return None
    try:
        trade_log_copy = trade_log.copy()
        trade_log_copy['exit_date'] = pd.to_datetime(trade_log_copy['exit_date'], errors='coerce')
        trade_log_copy = trade_log_copy.sort_values('exit_date')
        trade_log_copy['cumulative_pnl'] = trade_log_copy['pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trade_log_copy['exit_date'],
            y=trade_log_copy['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative PnL',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=4)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="Cumulative PnL Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative PnL ($)",
            height=height,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig
    except Exception:
        return None


def _create_trade_distribution_chart(trade_log: pd.DataFrame, height: int = 200) -> Optional[go.Figure]:
    """Create trade distribution chart."""
    if 'pnl' not in trade_log.columns:
        return None
    try:
        winning = len(trade_log[trade_log['pnl'] > 0])
        losing = len(trade_log[trade_log['pnl'] < 0])
        neutral = len(trade_log[trade_log['pnl'] == 0])
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Winning', 'Losing', 'Neutral'],
                y=[winning, losing, neutral],
                marker_color=['#2ecc71', '#e74c3c', '#95a5a6'],
                text=[winning, losing, neutral],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Trade Distribution",
            xaxis_title="Trade Type",
            yaxis_title="Number of Trades",
            height=height,
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig
    except Exception:
        return None


def _create_duration_chart(trade_log: pd.DataFrame, height: int = 200) -> Optional[go.Figure]:
    """Create trade duration distribution chart."""
    if 'entry_date' not in trade_log.columns or 'exit_date' not in trade_log.columns:
        return None
    try:
        trade_log_copy = trade_log.copy()
        trade_log_copy['entry_date'] = pd.to_datetime(trade_log_copy['entry_date'], errors='coerce')
        trade_log_copy['exit_date'] = pd.to_datetime(trade_log_copy['exit_date'], errors='coerce')
        trade_log_copy['duration'] = (trade_log_copy['exit_date'] - trade_log_copy['entry_date']).dt.days
        trade_log_copy = trade_log_copy[trade_log_copy['duration'] >= 0]
        
        if trade_log_copy.empty:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trade_log_copy['duration'],
            nbinsx=20,
            marker_color='#3498db',
            opacity=0.7
        ))
        fig.update_layout(
            title="Trade Duration Distribution",
            xaxis_title="Duration (days)",
            yaxis_title="Frequency",
            height=height,
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig
    except Exception:
        return None


def _create_monthly_pnl_chart(trade_log: pd.DataFrame, height: int = 200) -> Optional[go.Figure]:
    """Create monthly PnL breakdown chart."""
    if 'exit_date' not in trade_log.columns or 'pnl' not in trade_log.columns:
        return None
    try:
        trade_log_copy = trade_log.copy()
        trade_log_copy['exit_date'] = pd.to_datetime(trade_log_copy['exit_date'], errors='coerce')
        trade_log_copy = trade_log_copy.dropna(subset=['exit_date'])
        trade_log_copy['month'] = trade_log_copy['exit_date'].dt.to_period('M').astype(str)
        monthly_pnl = trade_log_copy.groupby('month')['pnl'].sum().reset_index()
        
        if monthly_pnl.empty:
            return None
        
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in monthly_pnl['pnl']]
        fig = go.Figure(data=[
            go.Bar(
                x=monthly_pnl['month'],
                y=monthly_pnl['pnl'],
                marker_color=colors,
                text=[f"${x:,.0f}" for x in monthly_pnl['pnl']],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Monthly PnL Breakdown",
            xaxis_title="Month",
            yaxis_title="PnL ($)",
            height=height,
            showlegend=False,
            xaxis_tickangle=-45,
            margin=dict(l=40, r=40, t=50, b=60)
        )
        return fig
    except Exception:
        return None


def _create_pnl_distribution_chart(trade_log: pd.DataFrame, height: int = 200) -> Optional[go.Figure]:
    """Create PnL distribution histogram."""
    if 'pnl' not in trade_log.columns:
        return None
    try:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trade_log['pnl'],
            nbinsx=30,
            marker_color='#9b59b6',
            opacity=0.7
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="PnL Distribution",
            xaxis_title="PnL ($)",
            yaxis_title="Frequency",
            height=height,
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig
    except Exception:
        return None


def render_ai_insights(stats: Dict, trade_log: Optional[pd.DataFrame], sentiment_data: Optional[list], strategy_name: Optional[str] = None, strategy_params: Optional[Dict] = None) -> None:
    """Render strategy insights section with collapsible visualizations and AI analysis panel."""
    st.subheader("Strategy Insights")
    
    if trade_log is None or trade_log.empty:
        st.info("No trade data available for analysis.")
        return
    
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
        st.markdown("#### Performance Visualizations")
        st.caption("Click on any chart tile to expand and view full details.")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            fig_small = _create_cumulative_pnl_chart(trade_log, height=180)
            if fig_small:
                with st.expander("Cumulative PnL Over Time", expanded=False):
                    fig_large = _create_cumulative_pnl_chart(trade_log, height=450)
                    if fig_large:
                        st.plotly_chart(fig_large, use_container_width=True)
                st.plotly_chart(fig_small, use_container_width=True, key="cumulative_preview")
            
            fig_small = _create_duration_chart(trade_log, height=180)
            if fig_small:
                with st.expander("Trade Duration Distribution", expanded=False):
                    fig_large = _create_duration_chart(trade_log, height=450)
                    if fig_large:
                        st.plotly_chart(fig_large, use_container_width=True)
                st.plotly_chart(fig_small, use_container_width=True, key="duration_preview")
        
        with viz_col2:
            fig_small = _create_trade_distribution_chart(trade_log, height=180)
            if fig_small:
                with st.expander("Trade Distribution", expanded=False):
                    fig_large = _create_trade_distribution_chart(trade_log, height=450)
                    if fig_large:
                        st.plotly_chart(fig_large, use_container_width=True)
                st.plotly_chart(fig_small, use_container_width=True, key="distribution_preview")
            
            fig_small = _create_monthly_pnl_chart(trade_log, height=180)
            if fig_small:
                with st.expander("Monthly PnL Breakdown", expanded=False):
                    fig_large = _create_monthly_pnl_chart(trade_log, height=450)
                    if fig_large:
                        st.plotly_chart(fig_large, use_container_width=True)
                st.plotly_chart(fig_small, use_container_width=True, key="monthly_preview")
        
        fig_small = _create_pnl_distribution_chart(trade_log, height=180)
        if fig_small:
            with st.expander("PnL Distribution", expanded=False):
                fig_large = _create_pnl_distribution_chart(trade_log, height=450)
                if fig_large:
                    st.plotly_chart(fig_large, use_container_width=True)
            st.plotly_chart(fig_small, use_container_width=True, key="pnl_dist_preview")
    
    with main_col2:
        st.markdown("#### Intelligent Insights")
        if 'ai_insights_generated' not in st.session_state:
            st.session_state.ai_insights_generated = False
        if 'ai_insights_text' not in st.session_state:
            st.session_state.ai_insights_text = ""
        
        if st.button("Generate Analysis", type="primary", use_container_width=True):
            with st.spinner("Generating analysis..."):
                try:
                    client = OpenAIClient()
                    explanation = client.explain_results(
                        stats, 
                        trade_log, 
                        sentiment_data,
                        strategy_name,
                        strategy_params
                    )
                    st.session_state.ai_insights_text = explanation
                    st.session_state.ai_insights_generated = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.ai_insights_generated = False
        
        st.markdown("---")
        
        if st.session_state.ai_insights_generated and st.session_state.ai_insights_text:
            st.markdown("**Analysis Report:**")
            st.markdown(st.session_state.ai_insights_text)
        else:
            st.info("Click 'Generate Analysis' to get AI-powered insights on your strategy performance.")


def render_export_options(stats: Dict, trade_log: Optional[pd.DataFrame]) -> None:
    """Render export options."""
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if trade_log is not None and not trade_log.empty:
            csv_data = trade_log.to_csv(index=False)
            st.download_button(
                label="Download Trade Log CSV",
                data=csv_data,
                file_name="trade_log.csv",
                mime="text/csv"
            )
    
    with col2:
        if stats:
            report_text = f"""
Backtest Report
================

Statistics:
- Total Return: {stats.get('total_return_pct', 'N/A')}
- Sharpe Ratio: {stats.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {stats.get('max_drawdown_pct', 'N/A')}
- Win Rate: {stats.get('win_rate_pct', 'N/A')}
- Total Trades: {stats.get('total_trades', 'N/A')}
- Winning Trades: {stats.get('winning_trades', 'N/A')}
- Losing Trades: {stats.get('losing_trades', 'N/A')}
- Average Win: ${stats.get('avg_win', 'N/A')}
- Average Loss: ${stats.get('avg_loss', 'N/A')}
- Profit Factor: {stats.get('profit_factor', 'N/A')}
"""
            st.download_button(
                label="Download Report (TXT)",
                data=report_text,
                file_name="backtest_report.txt",
                mime="text/plain"
            )


def render_landing_page() -> None:
    """Render landing page when no backtest has been run."""
    
    st.markdown("""
    <style>
    .landing-container {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        padding: 2rem;
        border-radius: 8px;
        margin: -1rem -1rem 1rem -1rem;
    }
    .landing-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    .landing-subtitle {
        font-size: 1rem;
        color: #a0a8b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.25rem 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.65rem;
        color: #8b94a8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4fc3f7;
        line-height: 1;
        margin-bottom: 0.5rem;
        font-family: 'Courier New', monospace;
    }
    .metric-desc {
        font-size: 0.75rem;
        color: #a0a8b8;
        margin-top: 0.25rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid rgba(79, 195, 247, 0.3);
        padding-bottom: 0.5rem;
    }
    .feature-row {
        display: flex;
        align-items: flex-start;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s ease;
    }
    .feature-row:hover {
        background: rgba(255, 255, 255, 0.03);
        padding-left: 0.5rem;
        border-left: 2px solid #4fc3f7;
    }
    .feature-row:last-child {
        border-bottom: none;
    }
    .feature-icon {
        width: 24px;
        text-align: center;
        font-size: 0.9rem;
        color: #4fc3f7;
        margin-right: 1rem;
        margin-top: 0.2rem;
        flex-shrink: 0;
    }
    .feature-content {
        flex: 1;
    }
    .feature-title {
        font-weight: 600;
        color: #ffffff;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .feature-desc {
        font-size: 0.8rem;
        color: #a0a8b8;
        line-height: 1.4;
    }
    .strategy-item {
        padding: 0.85rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s ease;
    }
    .strategy-item:hover {
        background: rgba(255, 255, 255, 0.03);
        padding-left: 0.5rem;
    }
    .strategy-item:last-child {
        border-bottom: none;
    }
    .strategy-name {
        font-weight: 600;
        color: #ffffff;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .strategy-desc {
        font-size: 0.75rem;
        color: #a0a8b8;
    }
    .quick-start-box {
        background: rgba(79, 195, 247, 0.1);
        border: 1px solid rgba(79, 195, 247, 0.2);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .quick-start-step {
        font-size: 0.9rem;
        color: #ffffff;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
    }
    .quick-start-step:last-child {
        margin-bottom: 0;
    }
    .step-number {
        background: #4fc3f7;
        color: #0a0e27;
        font-weight: 700;
        width: 24px;
        height: 24px;
        border-radius: 4px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.75rem;
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    .footer-text {
        text-align: center;
        color: #8b94a8;
        font-size: 0.85rem;
        padding: 1.5rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 2rem;
    }
    .footer-text strong {
        color: #4fc3f7;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="landing-header">MCP Backtesting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">Equity Strategy Backtesting | Multi-Source Data Integration</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Strategies</div>
            <div class="metric-value">6</div>
            <div class="metric-desc">MA • RSI • MACD • BB • Momentum • Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Scenarios</div>
            <div class="metric-value">6</div>
            <div class="metric-desc">Pre-configured</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Data Sources</div>
            <div class="metric-value">2</div>
            <div class="metric-desc">Market • News</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Indicators</div>
            <div class="metric-value">3+</div>
            <div class="metric-desc">MA • RSI • Volume</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Analytics</div>
            <div class="metric-value">8+</div>
            <div class="metric-desc">Metrics tracked</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown('<div class="section-title">System Capabilities</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-row">
                <div class="feature-icon">•</div>
                <div class="feature-content">
                    <div class="feature-title">Market Data Integration</div>
                    <div class="feature-desc">Historical OHLCV via yfinance • Real-time data fetching • Multi-ticker support</div>
                </div>
            </div>
            <div class="feature-row">
                <div class="feature-icon">•</div>
                <div class="feature-content">
                    <div class="feature-title">Trading Strategies</div>
                    <div class="feature-desc">Moving Average Crossover • RSI • Sentiment-based signals • Customizable parameters</div>
                </div>
            </div>
            <div class="feature-row">
                <div class="feature-icon">•</div>
                <div class="feature-content">
                    <div class="feature-title">Sentiment Analysis</div>
                    <div class="feature-desc">News sentiment integration • Multi-source aggregation • Real-time processing</div>
                </div>
            </div>
            <div class="feature-row">
                <div class="feature-icon">•</div>
                <div class="feature-content">
                    <div class="feature-title">Performance Analytics</div>
                    <div class="feature-desc">Sharpe ratio • Max drawdown • Win rate • Trade logs • P&L tracking • Equity curves</div>
                </div>
            </div>
            <div class="feature-row">
                <div class="feature-icon">•</div>
                <div class="feature-content">
                    <div class="feature-title">Strategy Insights</div>
                    <div class="feature-desc">Natural language analysis • Strategy performance explanations • Pattern recognition</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="section-title">Quick Start</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="quick-start-box">
            <div class="quick-start-step"><span class="step-number">1</span>Select demo scenario</div>
            <div class="quick-start-step"><span class="step-number">2</span>Configure parameters</div>
            <div class="quick-start-step"><span class="step-number">3</span>Run backtest</div>
            <div class="quick-start-step"><span class="step-number">4</span>Analyze results</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Strategy Library</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="strategy-item">
            <div class="strategy-name">Moving Average Crossover</div>
            <div class="strategy-desc">Fast/Slow MA crossovers</div>
        </div>
        <div class="strategy-item">
            <div class="strategy-name">RSI Strategy</div>
            <div class="strategy-desc">Oversold/Overbought signals</div>
        </div>
        <div class="strategy-item">
            <div class="strategy-name">MACD Strategy</div>
            <div class="strategy-desc">MACD line crossovers</div>
        </div>
        <div class="strategy-item">
            <div class="strategy-name">Bollinger Bands</div>
            <div class="strategy-desc">Mean reversion signals</div>
        </div>
        <div class="strategy-item">
            <div class="strategy-name">Momentum Strategy</div>
            <div class="strategy-desc">Price momentum signals</div>
        </div>
        <div class="strategy-item">
            <div class="strategy-name">Sentiment Strategy</div>
            <div class="strategy-desc">News sentiment integration</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer-text">
        Configure backtest parameters in sidebar → Click <strong>Run Backtest</strong> to begin
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
