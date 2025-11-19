import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


class OpenAIClient:
    """Client for OpenAI API interactions."""
    
    def __init__(self):
        """Initialize OpenAI client with API key from environment."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
    
    def analyze_sentiment(self, headlines: List[str]) -> List[Dict[str, str]]:
        """
        Analyze sentiment for a list of headlines.
        
        Args:
            headlines: List of news headlines to analyze
            
        Returns:
            List of dictionaries with 'headline' and 'sentiment' keys
            Sentiment values: 'positive', 'negative', or 'neutral'
        """
        if not headlines:
            return []
        
        try:
            prompt = """Analyze the sentiment of each news headline. 
Return only one word per headline: 'positive', 'negative', or 'neutral'.
Format: one sentiment per line, in the same order as the headlines.

Headlines:
"""
            for i, headline in enumerate(headlines, 1):
                prompt += f"{i}. {headline}\n"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis tool. Return only sentiment labels."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            sentiments = response.choices[0].message.content.strip().split('\n')
            results = []
            
            for i, headline in enumerate(headlines):
                if i < len(sentiments):
                    sentiment = sentiments[i].strip().lower()
                    if 'positive' in sentiment:
                        sentiment = 'positive'
                    elif 'negative' in sentiment:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                else:
                    sentiment = 'neutral'
                
                results.append({
                    'headline': headline,
                    'sentiment': sentiment
                })
            
            return results
            
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return [{'headline': h, 'sentiment': 'neutral'} for h in headlines]
    
    def explain_results(
        self, 
        backtest_stats: Dict, 
        trades: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[List] = None,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive explanation of backtest results using LLM.
        
        Args:
            backtest_stats: Dictionary with metrics (Sharpe, return, drawdown, etc.)
            trades: Optional DataFrame with trade records
            sentiment_data: Optional sentiment analysis data
            strategy_name: Optional strategy name
            strategy_params: Optional strategy parameters
            
        Returns:
            String explanation of the results
        """
        try:
            stats_text = f"""
PERFORMANCE METRICS:
- Total Return: {backtest_stats.get('total_return_pct', backtest_stats.get('total_return', 'N/A'))}
- Initial Capital: ${backtest_stats.get('initial_cash', 'N/A'):,.2f}
- Final Value: ${backtest_stats.get('final_value', 'N/A'):,.2f}
- Total PnL: ${backtest_stats.get('total_pnl', 'N/A'):,.2f}
- Sharpe Ratio: {backtest_stats.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {backtest_stats.get('max_drawdown_pct', backtest_stats.get('max_drawdown', 'N/A'))}
- Win Rate: {backtest_stats.get('win_rate_pct', backtest_stats.get('win_rate', 'N/A'))}
- Total Trades: {backtest_stats.get('total_trades', 'N/A')}
- Winning Trades: {backtest_stats.get('winning_trades', 'N/A')}
- Losing Trades: {backtest_stats.get('losing_trades', 'N/A')}
- Average Win: ${backtest_stats.get('avg_win', 'N/A'):,.2f}
- Average Loss: ${backtest_stats.get('avg_loss', 'N/A'):,.2f}
- Profit Factor: {backtest_stats.get('profit_factor', 'N/A')}
"""
            
            if strategy_name:
                stats_text += f"\nSTRATEGY: {strategy_name}"
            
            if strategy_params:
                params_str = ", ".join([f"{k}={v}" for k, v in strategy_params.items()])
                stats_text += f"\nPARAMETERS: {params_str}"
            
            trade_analysis = ""
            if trades is not None and not trades.empty and len(trades) > 0:
                try:
                    if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
                        trades['entry_date'] = pd.to_datetime(trades['entry_date'], errors='coerce')
                        trades['exit_date'] = pd.to_datetime(trades['exit_date'], errors='coerce')
                        trades['duration'] = (trades['exit_date'] - trades['entry_date']).dt.days
                        avg_duration = trades['duration'].mean() if 'duration' in trades.columns else 0
                        trade_analysis += f"\n- Average Trade Duration: {avg_duration:.1f} days"
                    
                    if 'pnl' in trades.columns:
                        best_trade = trades['pnl'].max() if len(trades) > 0 else 0
                        worst_trade = trades['pnl'].min() if len(trades) > 0 else 0
                        trade_analysis += f"\n- Best Trade: ${best_trade:,.2f}"
                        trade_analysis += f"\n- Worst Trade: ${worst_trade:,.2f}"
                        
                        if 'pnl_pct' in trades.columns:
                            best_pct = trades['pnl_pct'].max() if len(trades) > 0 else 0
                            worst_pct = trades['pnl_pct'].min() if len(trades) > 0 else 0
                            trade_analysis += f"\n- Best Trade %: {best_pct:.2f}%"
                            trade_analysis += f"\n- Worst Trade %: {worst_pct:.2f}%"
                except Exception:
                    pass
                
                if trade_analysis:
                    stats_text += f"\nTRADE ANALYSIS:{trade_analysis}"
            
            if sentiment_data:
                try:
                    pos_count = sum(1 for item in sentiment_data if item.get('sentiment') == 'positive')
                    neg_count = sum(1 for item in sentiment_data if item.get('sentiment') == 'negative')
                    neu_count = sum(1 for item in sentiment_data if item.get('sentiment') == 'neutral')
                    total_sent = len(sentiment_data)
                    stats_text += f"\nSENTIMENT DATA:\n- Total Articles: {total_sent}\n- Positive: {pos_count} ({pos_count/total_sent*100:.1f}%)\n- Negative: {neg_count} ({neg_count/total_sent*100:.1f}%)\n- Neutral: {neu_count} ({neu_count/total_sent*100:.1f}%)"
                except Exception:
                    pass
            
            prompt = f"""You are a senior quantitative analyst providing a comprehensive retrospective analysis of a trading strategy backtest.

Analyze the following backtest results and provide a detailed, insightful analysis covering:

1. PERFORMANCE ASSESSMENT: Evaluate the overall performance. Is this strategy profitable? How does it compare to buy-and-hold or market benchmarks?

2. RISK ANALYSIS: Assess the risk-adjusted returns. Is the Sharpe ratio acceptable? How significant is the drawdown? What are the risk concerns?

3. TRADE QUALITY: Analyze the win rate, profit factor, and average win/loss ratios. Are trades well-executed? Is there room for improvement in trade selection?

4. STRATEGY EFFECTIVENESS: Based on the parameters and results, is the strategy working as intended? What are its strengths and weaknesses?

5. RECOMMENDATIONS: Provide 2-3 actionable recommendations for improving the strategy (parameter tuning, risk management, entry/exit rules, etc.).

{stats_text}

Provide a comprehensive, professional analysis in 4-5 well-structured paragraphs. Be specific, quantitative, and actionable. Use financial terminology appropriately."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior quantitative analyst with expertise in algorithmic trading, risk management, and strategy optimization. Provide detailed, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return "Unable to generate analysis at this time."

