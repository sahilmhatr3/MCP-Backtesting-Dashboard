import yfinance as yf
import pandas as pd
from typing import Optional
from datetime import datetime


class YFinanceAdapter:
    """Adapter for fetching market price data using yfinance."""
    
    def __init__(self):
        """Initialize the yfinance adapter."""
        pass
    
    def get_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a given ticker and date range.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            Returns None if data fetch fails
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {ticker} in date range {start_date} to {end_date}")
                return None
            
            data.index.name = 'Date'
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

