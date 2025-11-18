from typing import List, Dict, Optional
from llm.openai_client import OpenAIClient


class SentimentAdapter:
    """Adapter for sentiment analysis of news headlines."""
    
    def __init__(self):
        """Initialize sentiment adapter with OpenAI client."""
        try:
            self.llm_client = OpenAIClient()
        except ValueError as e:
            print(f"Warning: {str(e)}. Sentiment analysis will be unavailable.")
            self.llm_client = None
    
    def get_sentiment(self, headlines: List[str]) -> List[Dict[str, str]]:
        """
        Get sentiment scores for a list of headlines.
        
        Args:
            headlines: List of news headlines to analyze
            
        Returns:
            List of dictionaries with 'headline' and 'sentiment' keys
        """
        if not self.llm_client:
            return [{'headline': h, 'sentiment': 'neutral'} for h in headlines]
        
        if not headlines:
            return []
        
        try:
            return self.llm_client.analyze_sentiment(headlines)
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return [{'headline': h, 'sentiment': 'neutral'} for h in headlines]
    
    def aggregate_sentiment(self, sentiment_results: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Aggregate sentiment scores into summary statistics.
        
        Args:
            sentiment_results: List of sentiment analysis results
            
        Returns:
            Dictionary with sentiment distribution percentages
        """
        if not sentiment_results:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        total = len(sentiment_results)
        positive = sum(1 for r in sentiment_results if r.get('sentiment') == 'positive')
        negative = sum(1 for r in sentiment_results if r.get('sentiment') == 'negative')
        neutral = total - positive - negative
        
        return {
            'positive': round(positive / total * 100, 2),
            'negative': round(negative / total * 100, 2),
            'neutral': round(neutral / total * 100, 2)
        }

