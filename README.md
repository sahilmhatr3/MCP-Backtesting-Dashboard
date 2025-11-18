# MCP Backtesting Dashboard

A modular dashboard for equity strategy backtesting, integrating market data, media-scraped sentiment, and LLM-generated reports using a multi-adapter MCP-style architecture.

## Features

- Multi-source data integration (market prices via yfinance, news scraping)
- Sentiment analysis using OpenAI API
- Backtesting engine with Backtrader
- Pre-built trading strategies
- LLM-powered result explanations
- Interactive Streamlit dashboard

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

On WSL/Ubuntu, install python3-venv if needed:
```bash
sudo apt install -y python3.12-venv python3-pip
```

### Installation

**Option 1: Automated Setup**
```bash
./setup.sh
```

**Option 2: Manual Setup**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file for OpenAI API integration:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

Activate the virtual environment:
```bash
source venv/bin/activate
```

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── adapters/              # Data source adapters
│   ├── yfinance_adapter.py
│   ├── news_adapter.py
│   └── sentiment_adapter.py
├── strategies/            # Trading strategies
│   └── prebuilt.py
├── backtest/              # Backtesting engine
│   └── engine.py
├── llm/                   # OpenAI integration
│   └── openai_client.py
└── requirements.txt      # Python dependencies
```

## Testing

Test the adapters:
```bash
python3 test_adapters.py
```

Test OpenAI integration:
```bash
python3 test_openai.py
```

## Credits

- Market data: yfinance
- Backtesting: Backtrader
- LLM: OpenAI API
- News scraping: BeautifulSoup, requests
