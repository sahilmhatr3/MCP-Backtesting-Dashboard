import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from strategies.prebuilt import get_strategy_class, get_strategy_params
from backtest.engine import BacktestEngine
from config import static_flows
from utils import ui_components, data_fetchers


st.set_page_config(
    page_title="MCP Backtesting Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM",
    "SPY", "QQQ", "DIA"
]


def initialize_session_state():
    """Initialize session state variables."""
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'trade_log' not in st.session_state:
        st.session_state.trade_log = None
    if 'market_data' not in st.session_state:
        st.session_state.market_data = None
    if 'sentiment_data' not in st.session_state:
        st.session_state.sentiment_data = None
    if 'strategy_params' not in st.session_state:
        st.session_state.strategy_params = {}
    if 'show_trade_copy' not in st.session_state:
        st.session_state.show_trade_copy = False


def render_header():
    """Render app header."""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("MCP Backtesting Dashboard")
        st.markdown("**Equity Strategy Backtesting with Multi-Source Data Integration**")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Home", type="secondary", use_container_width=True):
            st.session_state.backtest_results = None
            st.session_state.trade_log = None
            st.session_state.market_data = None
            st.session_state.sentiment_data = None
            st.rerun()
    st.markdown("---")


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.header("Control Panel")
    
    st.sidebar.markdown("### Demo Scenarios")
    scenario_options = ["Custom"] + static_flows.get_all_scenario_names()
    selected_scenario = st.sidebar.selectbox(
        "Load Scenario",
        scenario_options,
        help="Select a pre-configured scenario or choose Custom for manual setup"
    )
    
    scenario_config = None
    if selected_scenario != "Custom":
        scenario_config = static_flows.get_scenario(selected_scenario)
        st.sidebar.info(f"**{selected_scenario}**\n\n{scenario_config.get('description', '')}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Market Data")
    
    tickers = st.sidebar.multiselect(
        "Stock Tickers",
        POPULAR_TICKERS,
        default=scenario_config["tickers"] if scenario_config else ["AAPL"],
        help="Select one or more stock tickers to analyze"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(scenario_config["start_date"], "%Y-%m-%d").date() if scenario_config else datetime.now() - timedelta(days=365),
            help="Backtest start date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.strptime(scenario_config["end_date"], "%Y-%m-%d").date() if scenario_config else datetime.now(),
            help="Backtest end date"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Strategy")
    
    strategy_names = [
        "MovingAverageCrossover",
        "RSIStrategy",
        "MACDStrategy",
        "BollingerBandsStrategy",
        "MomentumStrategy",
        "SentimentStrategy"
    ]
    default_index = 0
    if scenario_config and scenario_config["strategy"] in strategy_names:
        default_index = strategy_names.index(scenario_config["strategy"])
    
    selected_strategy = st.sidebar.selectbox(
        "Trading Strategy",
        strategy_names,
        index=default_index,
        help="Select a trading strategy to backtest"
    )
    
    strategy_class = get_strategy_class(selected_strategy)
    default_params = get_strategy_params(selected_strategy)
    
    if scenario_config:
        params = scenario_config["parameters"]
    else:
        params = default_params.copy()
    
    st.sidebar.markdown("#### Strategy Parameters")
    strategy_params = {}
    
    if selected_strategy == "MovingAverageCrossover":
        strategy_params["fast_period"] = st.sidebar.slider(
            "Fast MA Period",
            min_value=5,
            max_value=50,
            value=params.get("fast_period", 10),
            help="Fast moving average period"
        )
        strategy_params["slow_period"] = st.sidebar.slider(
            "Slow MA Period",
            min_value=20,
            max_value=200,
            value=params.get("slow_period", 30),
            help="Slow moving average period"
        )
    elif selected_strategy == "RSIStrategy":
        strategy_params["rsi_period"] = st.sidebar.slider(
            "RSI Period",
            min_value=5,
            max_value=30,
            value=params.get("rsi_period", 14),
            help="RSI calculation period"
        )
        strategy_params["rsi_oversold"] = st.sidebar.slider(
            "Oversold Threshold",
            min_value=10,
            max_value=40,
            value=params.get("rsi_oversold", 30),
            help="RSI level to trigger buy signal"
        )
        strategy_params["rsi_overbought"] = st.sidebar.slider(
            "Overbought Threshold",
            min_value=60,
            max_value=90,
            value=params.get("rsi_overbought", 70),
            help="RSI level to trigger sell signal"
        )
    elif selected_strategy == "MACDStrategy":
        strategy_params["fast_period"] = st.sidebar.slider(
            "Fast EMA Period",
            min_value=5,
            max_value=20,
            value=params.get("fast_period", 12),
            help="Fast exponential moving average period"
        )
        strategy_params["slow_period"] = st.sidebar.slider(
            "Slow EMA Period",
            min_value=20,
            max_value=35,
            value=params.get("slow_period", 26),
            help="Slow exponential moving average period"
        )
        strategy_params["signal_period"] = st.sidebar.slider(
            "Signal Period",
            min_value=5,
            max_value=15,
            value=params.get("signal_period", 9),
            help="Signal line period"
        )
    elif selected_strategy == "BollingerBandsStrategy":
        strategy_params["period"] = st.sidebar.slider(
            "Period",
            min_value=10,
            max_value=50,
            value=params.get("period", 20),
            help="Moving average period for Bollinger Bands"
        )
        strategy_params["devfactor"] = st.sidebar.slider(
            "Deviation Factor",
            min_value=1.0,
            max_value=3.0,
            value=float(params.get("devfactor", 2.0)),
            step=0.1,
            help="Standard deviation multiplier for bands"
        )
    elif selected_strategy == "MomentumStrategy":
        strategy_params["period"] = st.sidebar.slider(
            "Momentum Period",
            min_value=5,
            max_value=30,
            value=params.get("period", 10),
            help="Period for momentum calculation"
        )
        strategy_params["threshold"] = st.sidebar.slider(
            "Momentum Threshold",
            min_value=0.01,
            max_value=0.10,
            value=float(params.get("threshold", 0.02)),
            step=0.01,
            help="Minimum momentum percentage to trigger signal"
        )
    elif selected_strategy == "SentimentStrategy":
        strategy_params["sentiment_threshold"] = st.sidebar.slider(
            "Sentiment Threshold",
            min_value=0.0,
            max_value=1.0,
            value=params.get("sentiment_threshold", 0.6),
            step=0.05,
            help="Minimum sentiment score (0-1) to trigger buy signal. Higher = more conservative."
        )
        strategy_params["lookback_period"] = st.sidebar.slider(
            "Lookback Period (days)",
            min_value=1,
            max_value=30,
            value=params.get("lookback_period", 5),
            help="Number of days to look back when calculating sentiment score"
        )
        strategy_params["min_articles"] = st.sidebar.slider(
            "Minimum Articles",
            min_value=1,
            max_value=10,
            value=params.get("min_articles", 1),
            help="Minimum number of articles with sentiment needed to make a decision"
        )
        strategy_params["use_sma_filter"] = st.sidebar.checkbox(
            "Use SMA Filter",
            value=params.get("use_sma_filter", True),
            help="Only buy when price is above SMA (trend filter)"
        )
        if strategy_params["use_sma_filter"]:
            strategy_params["sma_period"] = st.sidebar.slider(
                "SMA Period",
                min_value=5,
                max_value=50,
                value=params.get("sma_period", 20),
                help="Period for Simple Moving Average filter"
            )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Options")
    
    if selected_strategy == "SentimentStrategy":
        include_sentiment = st.sidebar.checkbox(
            "Enable Sentiment Analysis",
            value=scenario_config["include_sentiment"] if scenario_config else True,
            help="Required for SentimentStrategy - fetches and analyzes news sentiment"
        )
        if not include_sentiment:
            st.sidebar.warning("Sentiment analysis is required for SentimentStrategy")
    else:
        include_sentiment = False
        st.sidebar.info("Sentiment analysis is only available for SentimentStrategy")
    
    st.sidebar.markdown("---")
    
    run_backtest = st.sidebar.button(
        "Run Backtest",
        type="primary",
        use_container_width=True
    )
    
    return {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "strategy": selected_strategy,
        "strategy_class": strategy_class,
        "strategy_params": strategy_params,
        "include_sentiment": include_sentiment,
        "run_backtest": run_backtest
    }


def run_backtest_analysis(config):
    """Run backtest with given configuration."""
    if not config["tickers"]:
        st.error("Please select at least one ticker.")
        return None, None, None, None
    
    market_data = data_fetchers.fetch_market_data(
        config["tickers"],
        config["start_date"],
        config["end_date"]
    )
    
    if market_data is None:
        st.error("Failed to fetch market data. Please check ticker symbols and date range.")
        return None, None, None, None
    
    st.session_state.market_data = market_data
    
    sentiment_data = None
    if config["include_sentiment"]:
        sentiment_data = data_fetchers.fetch_sentiment_data(
            config["tickers"],
            config["start_date"],
            config["end_date"]
        )
        st.session_state.sentiment_data = sentiment_data
    
    engine = BacktestEngine(initial_cash=10000.0, commission=0.001)
    
    ticker_data = market_data[market_data['Ticker'] == config["tickers"][0]].copy()
    
    with st.spinner("Running backtest..."):
        trade_log, stats = engine.run_backtest(
            config["strategy_class"],
            ticker_data,
            config["strategy_params"],
            sentiment_data
        )
    
    return trade_log, stats, ticker_data, sentiment_data


def main():
    """Main application function."""
    initialize_session_state()
    
    config = render_sidebar()
    
    if config["run_backtest"]:
        trade_log, stats, data, sentiment_data = run_backtest_analysis(config)
        
        if stats:
            st.session_state.backtest_results = stats
            st.session_state.trade_log = trade_log
            st.session_state.strategy_params = config["strategy_params"]
            if sentiment_data:
                st.session_state.sentiment_data = sentiment_data
    
    if st.session_state.backtest_results:
        render_header()
        stats = st.session_state.backtest_results
        trade_log = st.session_state.trade_log
        data = st.session_state.market_data
        
        ui_components.render_metrics(stats)
        st.markdown("---")
        
        ui_components.render_price_chart(data, config["strategy"], st.session_state.strategy_params)
        st.markdown("---")
        
        ui_components.render_trade_log(trade_log)
        st.markdown("---")
        
        if config["strategy"] == "SentimentStrategy" and st.session_state.get("sentiment_data"):
            ui_components.render_sentiment_details(
                st.session_state.sentiment_data,
                config["tickers"][0] if config["tickers"] else None
            )
            st.markdown("---")
        
        ui_components.render_ai_insights(
            stats, 
            trade_log, 
            st.session_state.get("sentiment_data"),
            config["strategy"],
            st.session_state.strategy_params
        )
        st.markdown("---")
        
        ui_components.render_export_options(stats, trade_log)
    else:
        ui_components.render_landing_page()


if __name__ == "__main__":
    main()
