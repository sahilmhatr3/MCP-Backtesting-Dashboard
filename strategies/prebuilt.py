import backtrader as bt
from typing import Dict, Optional


class MovingAverageCrossover(bt.Strategy):
    """Moving average crossover strategy."""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize indicators."""
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.order = None
    
    def log(self, txt, dt=None):
        """Logging function."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Execute strategy logic on each bar."""
        if self.order:
            return
        
        if not self.position:
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                self.order = self.buy()
        else:
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called at the end of backtest."""
        self.log(f'Fast Period: {self.params.fast_period}, Slow Period: {self.params.slow_period}, Ending Value: {self.broker.getvalue():.2f}')


class RSIStrategy(bt.Strategy):
    """RSI-based trading strategy."""
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize indicators."""
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None
    
    def log(self, txt, dt=None):
        """Logging function."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Execute strategy logic on each bar."""
        if self.order:
            return
        
        if not self.position:
            if self.rsi < self.params.rsi_oversold:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}')
                self.order = self.buy()
        else:
            if self.rsi > self.params.rsi_overbought:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}, RSI: {self.rsi[0]:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called at the end of backtest."""
        self.log(f'RSI Period: {self.params.rsi_period}, Oversold: {self.params.rsi_oversold}, Overbought: {self.params.rsi_overbought}, Ending Value: {self.broker.getvalue():.2f}')


class SentimentStrategy(bt.Strategy):
    """Strategy that incorporates sentiment signals."""
    
    params = (
        ('sentiment_threshold', 0.6),
        ('lookback_period', 5),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize indicators and sentiment data."""
        self.sma = bt.indicators.SMA(self.data.close, period=20)
        self.sentiment_signal = None
        self.sentiment_data = []
        self.order = None
    
    def set_sentiment_data(self, sentiment_data: Dict):
        """Set sentiment data from external source."""
        self.sentiment_data = sentiment_data
    
    def get_sentiment_for_date(self, date):
        """Get sentiment score for a given date."""
        if not self.sentiment_data:
            return 0.5
        
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        for item in self.sentiment_data:
            if item.get('date') == date_str:
                sentiment = item.get('sentiment', 'neutral')
                if sentiment == 'positive':
                    return 1.0
                elif sentiment == 'negative':
                    return 0.0
                else:
                    return 0.5
        
        return 0.5
    
    def calculate_sentiment_score(self):
        """Calculate average sentiment score over lookback period."""
        if not self.sentiment_data:
            return 0.5
        
        scores = []
        for i in range(min(self.params.lookback_period, len(self.datas[0]))):
            try:
                date = self.datas[0].datetime.date(-i)
                score = self.get_sentiment_for_date(date)
                scores.append(score)
            except:
                continue
        
        if not scores:
            return 0.5
        
        return sum(scores) / len(scores)
    
    def log(self, txt, dt=None):
        """Logging function."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade notifications."""
        if not trade.isclosed:
            return
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Execute strategy logic on each bar."""
        if self.order:
            return
        
        sentiment_score = self.calculate_sentiment_score()
        price_above_sma = self.data.close[0] > self.sma[0]
        
        if not self.position:
            if sentiment_score > self.params.sentiment_threshold and price_above_sma:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}, Sentiment: {sentiment_score:.2f}')
                self.order = self.buy()
        else:
            if sentiment_score < (1 - self.params.sentiment_threshold) or not price_above_sma:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}, Sentiment: {sentiment_score:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called at the end of backtest."""
        self.log(f'Sentiment Threshold: {self.params.sentiment_threshold}, Lookback: {self.params.lookback_period}, Ending Value: {self.broker.getvalue():.2f}')


def get_strategy_class(strategy_name: str) -> Optional[bt.Strategy]:
    """
    Get strategy class by name.
    
    Args:
        strategy_name: Name of the strategy ('MovingAverageCrossover', 'RSIStrategy', 'SentimentStrategy')
        
    Returns:
        Strategy class or None if not found
    """
    strategies = {
        'MovingAverageCrossover': MovingAverageCrossover,
        'RSIStrategy': RSIStrategy,
        'SentimentStrategy': SentimentStrategy,
    }
    return strategies.get(strategy_name)


def get_strategy_params(strategy_name: str) -> Dict:
    """
    Get default parameters for a strategy.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'MovingAverageCrossover': {
            'fast_period': 10,
            'slow_period': 30,
        },
        'RSIStrategy': {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
        },
        'SentimentStrategy': {
            'sentiment_threshold': 0.6,
            'lookback_period': 5,
        },
    }
    return defaults.get(strategy_name, {})


