import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Type
from datetime import datetime


class TradeListAnalyzer(bt.Analyzer):
    """Analyzer to track individual trades."""
    
    def __init__(self):
        self.trades = []
    
    def notify_trade(self, trade):
        """Called when a trade is closed."""
        if trade.isclosed:
            entry_date = ''
            exit_date = ''
            entry_price = 0
            exit_price = 0
            size = 0
            
            try:
                if hasattr(trade, 'dtopen'):
                    dt_open = trade.dtopen
                    if hasattr(dt_open, 'date'):
                        entry_date = dt_open.date().isoformat()
                    elif hasattr(dt_open, 'num'):
                        entry_date = pd.Timestamp.fromordinal(int(dt_open.num)).date().isoformat()
                    elif isinstance(dt_open, (int, float)):
                        entry_date = pd.Timestamp.fromordinal(int(dt_open)).date().isoformat()
                    else:
                        entry_date = str(dt_open)
                
                if hasattr(trade, 'dtclose'):
                    dt_close = trade.dtclose
                    if hasattr(dt_close, 'date'):
                        exit_date = dt_close.date().isoformat()
                    elif hasattr(dt_close, 'num'):
                        exit_date = pd.Timestamp.fromordinal(int(dt_close.num)).date().isoformat()
                    elif isinstance(dt_close, (int, float)):
                        exit_date = pd.Timestamp.fromordinal(int(dt_close)).date().isoformat()
                    else:
                        exit_date = str(dt_close)
            except Exception:
                pass
            
            entry_price = 0
            exit_price = 0
            size = 0
            
            if hasattr(trade, 'size') and trade.size:
                size = abs(int(trade.size))
            
            if hasattr(trade, 'price') and trade.price:
                entry_price = round(float(trade.price), 2)
            elif hasattr(trade, 'value') and size > 0:
                try:
                    entry_price = round(float(trade.value) / size, 2)
                except:
                    entry_price = 0
            elif hasattr(trade, 'data') and hasattr(trade.data, 'close'):
                try:
                    entry_price = round(float(trade.data.close[0]), 2)
                except:
                    entry_price = 0
            
            pnl = round(float(trade.pnl), 2) if hasattr(trade, 'pnl') and trade.pnl else 0
            pnlcomm = round(float(trade.pnlcomm), 2) if hasattr(trade, 'pnlcomm') and trade.pnlcomm else pnl
            duration = int(trade.barlen) if hasattr(trade, 'barlen') and trade.barlen else 0
            
            if size == 0 and entry_price > 0:
                size = abs(int(10000 / entry_price)) if entry_price > 0 else 100
            
            if size > 0 and entry_price > 0:
                if pnlcomm != 0:
                    exit_price = round(entry_price + (pnlcomm / size), 2)
                else:
                    exit_price = entry_price
            elif hasattr(trade, 'data') and hasattr(trade.data, 'close'):
                try:
                    exit_price = round(float(trade.data.close[0]), 2)
                except:
                    exit_price = 0
            
            pnl_pct = 0
            if entry_price > 0 and size > 0:
                cost_basis = entry_price * size
                if cost_basis > 0:
                    pnl_pct = round((pnlcomm / cost_basis) * 100, 2)
            
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnlcomm,
                'pnl_pct': pnl_pct,
                'size': size,
                'duration': duration
            })


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
            
            ticker = None
            if 'Ticker' in data.columns:
                ticker = data['Ticker'].iloc[0] if not data.empty else None
            
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
                filtered_sentiment = sentiment_data
                if ticker:
                    filtered_sentiment = [item for item in sentiment_data if item.get('ticker') == ticker]
                
                class SentimentStrategyWrapper(strategy_class):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.set_sentiment_data(filtered_sentiment)
                
                cerebro.addstrategy(SentimentStrategyWrapper, **parameters)
            else:
                cerebro.addstrategy(strategy_class, **parameters)
            
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)
            
            cerebro.addsizer(bt.sizers.PercentSizer, percents=98)
            
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(TradeListAnalyzer, _name='tradelist')
            
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
        if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'tradelist'):
            tradelist = strategy.analyzers.tradelist
            if hasattr(tradelist, 'trades') and tradelist.trades:
                return pd.DataFrame(tradelist.trades)
        
        trades = []
        
        if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'trades'):
            trade_analyzer = strategy.analyzers.trades
            if hasattr(trade_analyzer, 'rets'):
                rets = trade_analyzer.rets
                if rets:
                    total_trades = rets.get('total', {}).get('total', 0) if isinstance(rets.get('total'), dict) else 0
                    
                    if total_trades > 0:
                        total_won = rets.get('won', {}).get('total', 0) if isinstance(rets.get('won'), dict) else 0
                        total_lost = rets.get('lost', {}).get('total', 0) if isinstance(rets.get('lost'), dict) else 0
                        
                        avg_win = rets.get('won', {}).get('pnl', {}).get('average', 0) if isinstance(rets.get('won', {}).get('pnl'), dict) else 0
                        avg_loss = rets.get('lost', {}).get('pnl', {}).get('average', 0) if isinstance(rets.get('lost', {}).get('pnl'), dict) else 0
                        
                        for i in range(total_won):
                            trades.append({
                                'entry_date': '',
                                'exit_date': '',
                                'entry_price': 0,
                                'exit_price': 0,
                                'pnl': round(avg_win, 2),
                                'pnl_pct': 0,
                                'size': 0,
                                'duration': 0
                            })
                        
                        for i in range(total_lost):
                            trades.append({
                                'entry_date': '',
                                'exit_date': '',
                                'entry_price': 0,
                                'exit_price': 0,
                                'pnl': round(avg_loss, 2),
                                'pnl_pct': 0,
                                'size': 0,
                                'duration': 0
                            })
        
        return pd.DataFrame(trades)
    
    def _calculate_statistics(self, strategy: bt.Strategy, trade_log: pd.DataFrame) -> Dict:
        """Calculate backtest statistics."""
        final_value = strategy.broker.getvalue()
        if self.initial_cash > 0:
            total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100
        else:
            total_return = 0.0
        
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
        
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(strategy, trade_log)
        sharpe_ratio = self._calculate_sharpe_ratio(strategy, trade_log)
        
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
    
    def _calculate_max_drawdown(self, strategy: bt.Strategy, trade_log: pd.DataFrame) -> tuple[float, float]:
        """Calculate max drawdown from Backtrader analyzer or trade log."""
        if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'drawdown'):
            dd = strategy.analyzers.drawdown
            try:
                if hasattr(dd, 'rets') and dd.rets:
                    rets = dd.rets
                    if 'max' in rets:
                        max_info = rets['max']
                        max_dd_pct = 0.0
                        max_dd = 0.0
                        
                        if 'drawdown' in max_info:
                            drawdown_val = float(max_info['drawdown'])
                            if abs(drawdown_val) <= 1.0:
                                max_dd_pct = abs(drawdown_val) * 100
                            elif abs(drawdown_val) <= 100.0:
                                max_dd_pct = abs(drawdown_val)
                            else:
                                max_dd_pct = min(abs(drawdown_val) / 100, 100.0)
                        
                        if 'moneydown' in max_info:
                            max_dd = abs(float(max_info['moneydown']))
                        
                        if max_dd_pct > 0 or max_dd > 0:
                            if max_dd_pct == 0 and max_dd > 0:
                                max_dd_pct = (max_dd / self.initial_cash) * 100
                            
                            max_dd_pct = min(max_dd_pct, 100.0)
                            return max_dd, max_dd_pct
            except Exception as e:
                pass
        
        if trade_log.empty or 'pnl' not in trade_log.columns:
            return 0.0, 0.0
        
        equity = self.initial_cash
        peak_equity = self.initial_cash
        max_dd = 0.0
        max_dd_pct = 0.0
        
        for _, trade in trade_log.iterrows():
            equity += trade['pnl']
            if equity < 0:
                equity = 0
            
            if equity > peak_equity:
                peak_equity = equity
            
            if peak_equity > 0:
                drawdown = peak_equity - equity
                drawdown_pct = (drawdown / peak_equity * 100)
                
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_pct = min(drawdown_pct, 100.0)
        
        return max_dd, max_dd_pct
    
    def _calculate_sharpe_ratio(self, strategy: bt.Strategy, trade_log: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from Backtrader analyzers."""
        if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'sharpe'):
            sharpe = strategy.analyzers.sharpe
            try:
                analysis = sharpe.get_analysis()
                if isinstance(analysis, dict):
                    if 'sharperatio' in analysis:
                        ratio = analysis['sharperatio']
                        if ratio is not None and not np.isnan(ratio) and not np.isinf(ratio):
                            return float(ratio)
            except Exception:
                pass
        
        return 0.0

