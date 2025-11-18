import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Optional

from adapters.yfinance_adapter import YFinanceAdapter
from adapters.news_adapter import NewsAdapter
from adapters.sentiment_adapter import SentimentAdapter


def fetch_market_data(tickers: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch market data for selected tickers."""
    adapter = YFinanceAdapter()
    all_data = []
    
    for ticker in tickers:
        with st.spinner(f"Fetching data for {ticker}..."):
            data = adapter.get_data(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            if data is not None and not data.empty:
                data['Ticker'] = ticker
                all_data.append(data)
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)


def fetch_sentiment_data(tickers: List[str], start_date: datetime, end_date: datetime) -> List[dict]:
    """Fetch and analyze sentiment data."""
    news_adapter = NewsAdapter()
    sentiment_adapter = SentimentAdapter()
    
    all_sentiment = []
    
    for ticker in tickers:
        with st.spinner(f"Fetching news for {ticker}..."):
            articles = news_adapter.get_news(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            if articles:
                headlines = [a.get('headline', '') for a in articles]
                sentiment_results = sentiment_adapter.get_sentiment(headlines)
                
                for i, article in enumerate(articles):
                    if i < len(sentiment_results):
                        all_sentiment.append({
                            'date': article.get('date', ''),
                            'headline': article.get('headline', ''),
                            'sentiment': sentiment_results[i].get('sentiment', 'neutral'),
                            'ticker': ticker
                        })
    
    return all_sentiment

