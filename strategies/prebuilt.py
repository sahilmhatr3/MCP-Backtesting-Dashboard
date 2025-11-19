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
    """Strategy that uses news sentiment to make trading decisions."""
    
    params = (
        ('sentiment_threshold', 0.6),
        ('lookback_period', 5),
        ('min_articles', 1),
        ('use_sma_filter', True),
        ('sma_period', 20),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize indicators."""
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
        self.sentiment_data = []
        self.order = None
    
    def set_sentiment_data(self, sentiment_data):
        """Set sentiment data from external source."""
        self.sentiment_data = sentiment_data if sentiment_data else []
    
    def get_sentiment_for_date(self, date):
        """Get sentiment score for a specific date."""
        if not self.sentiment_data:
            return None
        
        try:
            if hasattr(date, 'strftime'):
                date_obj = date
            else:
                from datetime import datetime
                date_obj = datetime.strptime(str(date), '%Y-%m-%d').date()
        except Exception:
            return None
        
        best_match = None
        min_diff = float('inf')
        
        for item in self.sentiment_data:
            item_date_str = item.get('date', '')
            if not item_date_str:
                continue
            
            try:
                from datetime import datetime
                item_date = datetime.strptime(item_date_str, '%Y-%m-%d').date()
                diff = abs((date_obj - item_date).days)
                
                if diff < min_diff and diff <= 7:
                    min_diff = diff
                    best_match = item
            except Exception:
                continue
        
        if best_match:
            sentiment = best_match.get('sentiment', 'neutral')
            if sentiment == 'positive':
                return 1.0
            elif sentiment == 'negative':
                return 0.0
            else:
                return 0.5
        
        return None
    
    def calculate_sentiment_score(self):
        """Calculate weighted sentiment score over lookback period."""
        if not self.sentiment_data:
            return 0.5
        
        if len(self.datas[0]) < self.params.lookback_period:
            return 0.5
        
        sentiment_scores = []
        for i in range(self.params.lookback_period):
            try:
                if len(self.datas[0]) > i:
                    date = self.datas[0].datetime.date(-i)
                    score = self.get_sentiment_for_date(date)
                    if score is not None:
                        sentiment_scores.append(score)
            except Exception:
                continue
        
        if len(sentiment_scores) < self.params.min_articles:
            return 0.5
        
        non_neutral = [s for s in sentiment_scores if s != 0.5]
        if not non_neutral:
            return 0.5
        
        avg_score = sum(non_neutral) / len(non_neutral)
        return avg_score
    
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
        
        if len(self.datas[0]) < max(self.params.sma_period, self.params.lookback_period):
            return
        
        sentiment_score = self.calculate_sentiment_score()
        
        price_above_sma = True
        if self.params.use_sma_filter and len(self.sma) > 0:
            price_above_sma = self.data.close[0] > self.sma[0]
        
        if not self.position:
            if sentiment_score > self.params.sentiment_threshold and price_above_sma:
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}, Sentiment: {sentiment_score:.3f}, Threshold: {self.params.sentiment_threshold:.3f}')
                self.order = self.buy()
        else:
            sell_threshold = 1.0 - self.params.sentiment_threshold
            if sentiment_score < sell_threshold or not price_above_sma:
                self.log(f'SELL CREATE, Price: {self.data.close[0]:.2f}, Sentiment: {sentiment_score:.3f}, Sell Threshold: {sell_threshold:.3f}')
                self.order = self.sell()
    
    def stop(self):
        """Called at the end of backtest."""
        self.log(f'Ending Value: {self.broker.getvalue():.2f}')


class MACDStrategy(bt.Strategy):
    """MACD-based trading strategy."""
    
    params = (
        ('fast_period', 12),
        ('slow_period', 26),
        ('signal_period', 9),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize indicators."""
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.fast_period,
            period_me2=self.params.slow_period,
            period_signal=self.params.signal_period
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
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
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}, MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f}')
                self.order = self.buy()
        else:
            if self.crossover < 0:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}, MACD: {self.macd.macd[0]:.2f}, Signal: {self.macd.signal[0]:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called at the end of backtest."""
        self.log(f'MACD Fast: {self.params.fast_period}, Slow: {self.params.slow_period}, Signal: {self.params.signal_period}, Ending Value: {self.broker.getvalue():.2f}')


class BollingerBandsStrategy(bt.Strategy):
    """Bollinger Bands mean reversion strategy."""
    
    params = (
        ('period', 20),
        ('devfactor', 2.0),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize indicators."""
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.period,
            devfactor=self.params.devfactor
        )
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
            if self.data.close[0] < self.bb.lines.bot[0]:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}, Lower Band: {self.bb.lines.bot[0]:.2f}')
                self.order = self.buy()
        else:
            if self.data.close[0] > self.bb.lines.top[0]:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}, Upper Band: {self.bb.lines.top[0]:.2f}')
                self.order = self.sell()
    
    def stop(self):
        """Called at the end of backtest."""
        self.log(f'Period: {self.params.period}, Dev Factor: {self.params.devfactor}, Ending Value: {self.broker.getvalue():.2f}')


class MomentumStrategy(bt.Strategy):
    """Momentum-based trading strategy."""
    
    params = (
        ('period', 10),
        ('threshold', 0.02),
        ('printlog', False),
    )
    
    def __init__(self):
        """Initialize indicators."""
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.period)
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
        
        if len(self.datas[0]) < self.params.period:
            return
        
        if len(self.momentum) < 1:
            return
        
        price_periods_ago = self.data.close[-self.params.period]
        if price_periods_ago == 0:
            return
        
        momentum_pct = self.momentum[0] / price_periods_ago
        
        if not self.position:
            if momentum_pct > self.params.threshold:
                self.log(f'BUY CREATE, {self.data.close[0]:.2f}, Momentum: {momentum_pct:.2%}')
                self.order = self.buy()
        else:
            if momentum_pct < -self.params.threshold:
                self.log(f'SELL CREATE, {self.data.close[0]:.2f}, Momentum: {momentum_pct:.2%}')
                self.order = self.sell()
    
    def stop(self):
        """Called at the end of backtest."""
        self.log(f'Period: {self.params.period}, Threshold: {self.params.threshold}, Ending Value: {self.broker.getvalue():.2f}')


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
        'MACDStrategy': MACDStrategy,
        'BollingerBandsStrategy': BollingerBandsStrategy,
        'MomentumStrategy': MomentumStrategy,
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
            'min_articles': 1,
            'use_sma_filter': True,
            'sma_period': 20,
        },
        'MACDStrategy': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
        },
        'BollingerBandsStrategy': {
            'period': 20,
            'devfactor': 2.0,
        },
        'MomentumStrategy': {
            'period': 10,
            'threshold': 0.02,
        },
    }
    return defaults.get(strategy_name, {})


