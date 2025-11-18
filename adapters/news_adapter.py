import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, date
import time


class NewsAdapter:
    """Adapter for fetching news articles related to stock tickers."""
    
    def __init__(self):
        """Initialize the news adapter."""
        self.base_url = "https://finance.yahoo.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
    
    def get_news(
        self, 
        ticker: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        max_articles: int = 50
    ) -> List[Dict[str, str]]:
        """
        Fetch news articles for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date filter (optional, format: 'YYYY-MM-DD')
            end_date: End date filter (optional, format: 'YYYY-MM-DD')
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of dictionaries with keys: 'date', 'headline', 'source', 'url'
        """
        try:
            url = f"{self.base_url}/quote/{ticker}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            news_section = soup.find('div', {'id': 'quoteNewsStream'}) or soup.find('section', {'data-module': 'NewsStream'})
            if not news_section:
                news_section = soup
            
            news_items = news_section.find_all(['li', 'div'], class_=lambda x: x and ('news' in x.lower() or 'stream' in x.lower()))
            
            if not news_items:
                links = soup.find_all('a', href=lambda x: x and '/news/' in x)
                for link in links[:max_articles]:
                    headline = link.get_text(strip=True)
                    href = link.get('href', '')
                    if headline and len(headline) > 10:
                        full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                        articles.append({
                            'date': date.today().isoformat(),
                            'headline': headline,
                            'source': 'Yahoo Finance',
                            'url': full_url
                        })
            else:
                for item in news_items[:max_articles]:
                    try:
                        headline_elem = item.find('h3') or item.find('h4') or item.find('a')
                        if not headline_elem:
                            continue
                        
                        headline = headline_elem.get_text(strip=True)
                        if not headline or len(headline) < 10:
                            continue
                        
                        link_elem = item.find('a', href=True)
                        link = link_elem.get('href', '') if link_elem else ''
                        if link and not link.startswith('http'):
                            link = f"{self.base_url}{link}" if link.startswith('/') else f"{self.base_url}/{link}"
                        
                        date_elem = item.find('time') or item.find('span', class_=lambda x: x and 'date' in x.lower())
                        article_date = None
                        if date_elem:
                            date_str = date_elem.get_text(strip=True) or date_elem.get('datetime', '')
                            if date_str:
                                article_date = self._parse_date(date_str)
                        
                        source_elem = item.find(['div', 'span'], class_=lambda x: x and 'source' in x.lower())
                        source = source_elem.get_text(strip=True) if source_elem else "Yahoo Finance"
                        
                        articles.append({
                            'date': article_date or date.today().isoformat(),
                            'headline': headline,
                            'source': source,
                            'url': link
                        })
                    except Exception:
                        continue
            
            if start_date or end_date:
                articles = self._filter_by_date(articles, start_date, end_date)
            
            return articles[:max_articles] if articles else []
            
        except requests.RequestException as e:
            print(f"Error fetching news for {ticker}: {str(e)}")
            return []
        except Exception as e:
            print(f"Error parsing news for {ticker}: {str(e)}")
            return []
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats to ISO format."""
        try:
            date_str = date_str.lower().strip()
            if 'min' in date_str or 'hour' in date_str:
                return date.today().isoformat()
            
            for fmt in ['%b %d, %Y', '%B %d, %Y', '%Y-%m-%d', '%m/%d/%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.date().isoformat()
                except ValueError:
                    continue
            return date.today().isoformat()
        except Exception:
            return date.today().isoformat()
    
    def _filter_by_date(
        self, 
        articles: List[Dict[str, str]], 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> List[Dict[str, str]]:
        """Filter articles by date range."""
        filtered = []
        for article in articles:
            article_date = article.get('date', '')
            if not article_date:
                continue
            
            try:
                art_date = datetime.fromisoformat(article_date).date()
                if start_date:
                    start = datetime.fromisoformat(start_date).date()
                    if art_date < start:
                        continue
                if end_date:
                    end = datetime.fromisoformat(end_date).date()
                    if art_date > end:
                        continue
                filtered.append(article)
            except Exception:
                continue
        
        return filtered

