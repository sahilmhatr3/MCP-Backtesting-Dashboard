from typing import Dict, List


DEMO_SCENARIOS: Dict[str, Dict] = {
    "AAPL MA Crossover 2018-2022": {
        "tickers": ["AAPL"],
        "start_date": "2018-01-01",
        "end_date": "2022-12-31",
        "strategy": "MovingAverageCrossover",
        "parameters": {
            "fast_period": 10,
            "slow_period": 30
        },
        "include_sentiment": False,
        "description": "Moving average crossover strategy on Apple stock from 2018-2022"
    },
    
    "Tech Stocks RSI 2020-2023": {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "strategy": "RSIStrategy",
        "parameters": {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70
        },
        "include_sentiment": False,
        "description": "RSI strategy on tech stocks during 2020-2023 period"
    },
    
    "S&P 500 with Sentiment 2016-2020": {
        "tickers": ["SPY"],
        "start_date": "2016-01-01",
        "end_date": "2020-12-31",
        "strategy": "SentimentStrategy",
        "parameters": {
            "sentiment_threshold": 0.6,
            "lookback_period": 5
        },
        "include_sentiment": True,
        "description": "Sentiment-based strategy on S&P 500 ETF with news sentiment analysis"
    },
    
    "AAPL Quick Test 2024": {
        "tickers": ["AAPL"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "strategy": "MovingAverageCrossover",
        "parameters": {
            "fast_period": 5,
            "slow_period": 20
        },
        "include_sentiment": False,
        "description": "Quick test with shorter moving averages on recent AAPL data"
    },
    
    "Multi-Stock MA Crossover 2019-2021": {
        "tickers": ["AAPL", "MSFT", "AMZN"],
        "start_date": "2019-01-01",
        "end_date": "2021-12-31",
        "strategy": "MovingAverageCrossover",
        "parameters": {
            "fast_period": 15,
            "slow_period": 50
        },
        "include_sentiment": False,
        "description": "Moving average crossover on multiple tech stocks"
    },
    
    "RSI Conservative 2021-2023": {
        "tickers": ["AAPL"],
        "start_date": "2021-01-01",
        "end_date": "2023-12-31",
        "strategy": "RSIStrategy",
        "parameters": {
            "rsi_period": 21,
            "rsi_oversold": 25,
            "rsi_overbought": 75
        },
        "include_sentiment": False,
        "description": "Conservative RSI strategy with wider thresholds"
    }
}


def get_scenario(scenario_name: str) -> Dict:
    """
    Get a demo scenario by name.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        Dictionary with scenario configuration
    """
    return DEMO_SCENARIOS.get(scenario_name, {})


def get_all_scenario_names() -> List[str]:
    """
    Get list of all available scenario names.
    
    Returns:
        List of scenario names
    """
    return list(DEMO_SCENARIOS.keys())


def get_scenario_description(scenario_name: str) -> str:
    """
    Get description for a scenario.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        Description string
    """
    scenario = get_scenario(scenario_name)
    return scenario.get("description", "")

