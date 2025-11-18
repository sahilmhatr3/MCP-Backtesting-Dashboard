import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
from datetime import datetime


class BacktestEngine:
    """Engine for running backtests using Backtrader."""
    
    def __init__(self, initial_cash: float = 10000.0, commission: float = 0.001):
        """
        Initialize backtest engine.
        
        Args:
            initial_cash: Starting capital
            commission: Commission rate (0.001 = 0.1%)
        """
        self.initial_cash = initial_cash
        self.commission = commission
    
    def run_backtest(
        self,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        parameters: Dict,
        sentiment_data: Optional[List[Dict]] = None
    ) -> tuple[pd.DataFrame, Dict]:
        """
        Run a backtest with given strategy and data.
        
        Args:
            strategy_class: Strategy class from strategies/prebuilt.py
            data: DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume)
            parameters: Dictionary of strategy parameters
            sentiment_data: Optional sentiment data for SentimentStrategy
            
        Returns:
            Tuple of (trade_log DataFrame, statistics dictionary)
        """
        try:
            cerebro = bt.Cerebro()
            
            if 'Date' in data.columns:
                data = data.set_index('Date')
            
            data.index = pd.to_datetime(data.index)
            
            bt_data = bt.feeds.PandasData(
                dataname=data,
                datetime=None,
                open=0,
                high=1,
                low=2,
                close=3,
                volume=4,
                openinterest=-1
            )
            
            cerebro.adddata(bt_data)
            
            if sentiment_data and strategy_class.__name__ == 'SentimentStrategy':
                strategy_instance = strategy_class(**parameters)
                strategy_instance.set_sentiment_data(sentiment_data)
                cerebro.addstrategy(strategy_class, **parameters)
            else:
                cerebro.addstrategy(strategy_class, **parameters)
            
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)
            
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            results = cerebro.run()
            strategy = results[0]
            
            trade_log = self._extract_trades(strategy)
            statistics = self._calculate_statistics(strategy, trade_log)
            
            return trade_log, statistics
            
        except Exception as e:
            print(f"Error running backtest: {str(e)}")
            return pd.DataFrame(), {}
    
    def _extract_trades(self, strategy: bt.Strategy) -> pd.DataFrame:
        """Extract trade log from strategy."""
        trades = []
        
        if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'trades'):
            trade_analyzer = strategy.analyzers.trades
            if hasattr(trade_analyzer, 'rets'):
                rets = trade_analyzer.rets
                if rets:
                    total_trades = rets.get('total', {}).get('total', 0) if isinstance(rets.get('total'), dict) else 0
                    
                    if total_trades > 0:
                        won = rets.get('won', {})
                        lost = rets.get('lost', {})
                        
                        if isinstance(won, dict) and 'total' in won:
                            for i in range(won.get('total', 0)):
                                trades.append({
                                    'entry_date': '',
                                    'exit_date': '',
                                    'entry_price': 0,
                                    'exit_price': 0,
                                    'pnl': won.get('pnl', {}).get('average', 0) if isinstance(won.get('pnl'), dict) else 0,
                                    'pnl_pct': 0,
                                    'size': 0,
                                    'duration': 0
                                })
                        
                        if isinstance(lost, dict) and 'total' in lost:
                            for i in range(lost.get('total', 0)):
                                trades.append({
                                    'entry_date': '',
                                    'exit_date': '',
                                    'entry_price': 0,
                                    'exit_price': 0,
                                    'pnl': lost.get('pnl', {}).get('average', 0) if isinstance(lost.get('pnl'), dict) else 0,
                                    'pnl_pct': 0,
                                    'size': 0,
                                    'duration': 0
                                })
        
        return pd.DataFrame(trades)
    
    def _calculate_statistics(self, strategy: bt.Strategy, trade_log: pd.DataFrame) -> Dict:
        """Calculate backtest statistics."""
        final_value = strategy.broker.getvalue()
        total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100
        
        sharpe_ratio = 0.0
        if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'sharpe'):
            sharpe = strategy.analyzers.sharpe
            if hasattr(sharpe, 'ratio'):
                sharpe_ratio = sharpe.ratio if sharpe.ratio and not np.isnan(sharpe.ratio) else 0.0
        
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'drawdown'):
            dd = strategy.analyzers.drawdown
            if hasattr(dd, 'max'):
                max_drawdown = abs(dd.max.drawdown) if dd.max.drawdown else 0.0
                max_drawdown_pct = abs(dd.max.drawdown) if dd.max.drawdown else 0.0
        
        total_trades = len(trade_log) if not trade_log.empty else 0
        
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        
        if not trade_log.empty and 'pnl' in trade_log.columns:
            winning_trades = len(trade_log[trade_log['pnl'] > 0])
            losing_trades = len(trade_log[trade_log['pnl'] < 0])
            total_pnl = trade_log['pnl'].sum()
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_win = trade_log[trade_log['pnl'] > 0]['pnl'].mean() if not trade_log.empty and 'pnl' in trade_log.columns and winning_trades > 0 else 0.0
        avg_loss = abs(trade_log[trade_log['pnl'] < 0]['pnl'].mean()) if not trade_log.empty and 'pnl' in trade_log.columns and losing_trades > 0 else 0.0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if avg_loss > 0 and losing_trades > 0 else 0.0
        
        return {
            'initial_cash': self.initial_cash,
            'final_value': round(final_value, 2),
            'total_return': round(total_return, 2),
            'total_return_pct': f"{total_return:.2f}%",
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': f"{max_drawdown_pct:.2f}%",
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'win_rate_pct': f"{win_rate:.2f}%",
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2)
        }

