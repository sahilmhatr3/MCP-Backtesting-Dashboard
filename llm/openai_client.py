import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

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
        trades: Optional[List] = None,
        sentiment_data: Optional[Dict] = None
    ) -> str:
        """
        Generate explanation of backtest results using LLM.
        
        Args:
            backtest_stats: Dictionary with metrics (Sharpe, return, drawdown, etc.)
            trades: Optional list of trade records
            sentiment_data: Optional sentiment analysis data
            
        Returns:
            String explanation of the results
        """
        try:
            stats_text = f"""
Backtest Results:
- Total Return: {backtest_stats.get('total_return', 'N/A')}
- Sharpe Ratio: {backtest_stats.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {backtest_stats.get('max_drawdown', 'N/A')}
- Win Rate: {backtest_stats.get('win_rate', 'N/A')}
- Total Trades: {backtest_stats.get('total_trades', 'N/A')}
"""
            
            if sentiment_data:
                stats_text += f"\nSentiment Analysis: {sentiment_data.get('summary', 'N/A')}"
            
            prompt = f"""Analyze these backtesting results and provide a concise explanation 
of the strategy's performance, key insights, and any notable patterns.

{stats_text}

Provide a clear, professional analysis in 2-3 paragraphs."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst explaining trading strategy results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return "Unable to generate analysis at this time."

