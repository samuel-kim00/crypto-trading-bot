#!/usr/bin/env python3
"""
YouTube and Web Market Intelligence Scraper
Analyzes crypto content from YouTube and other sources for market intelligence
"""

import os
import sys
import json
import asyncio
import logging
import requests
from datetime import datetime, timedelta
import youtube_dl
from textblob import TextBlob
import feedparser
import re
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YouTubeMarketIntelligence:
    def __init__(self):
        self.intelligence_data = {}
        self.youtube_channels = [
            'UCRvqjQPSeaWn-uEx-w0XOIg',  # Coin Bureau
            'UCG7GXzBq4Y8sZhXEv0d0gLw',  # Altcoin Daily
            'UCxOoJkD-PGnNWCOlWzWQnuQ',  # The Moon
            'UCpTyYOGK8zk0HDCkLzjV-3A',  # InvestAnswers
            'UCdHUWaTmk8L7zlOPB7C5O0g'   # Benjamin Cowen
        ]
        
    async def gather_youtube_intelligence(self):
        """Gather market intelligence from YouTube crypto channels"""
        logging.info("📹 Gathering YouTube market intelligence...")
        
        try:
            # Search for recent crypto videos
            crypto_keywords = [
                'bitcoin price prediction',
                'ethereum analysis',
                'crypto market update',
                'altcoin season',
                'market crash',
                'bull market',
                'crypto news today'
            ]
            
            all_video_data = []
            
            for keyword in crypto_keywords:
                try:
                    video_data = await self.search_youtube_videos(keyword, max_results=5)
                    all_video_data.extend(video_data)
                except Exception as e:
                    logging.warning(f"Failed to search for '{keyword}': {e}")
            
            # Analyze video content for sentiment and insights
            intelligence = await self.analyze_video_content(all_video_data)
            
            return intelligence
            
        except Exception as e:
            logging.error(f"Error gathering YouTube intelligence: {e}")
            return {}
    
    async def search_youtube_videos(self, query, max_results=10):
        """Search YouTube for crypto-related videos"""
        try:
            # YouTube Data API v3 search (simplified version)
            # In production, you'd use the official YouTube API
            
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            
            # Simulate video data (in real implementation, parse HTML or use API)
            video_data = []
            
            for i in range(max_results):
                video_data.append({
                    'title': f"Sample video about {query} {i+1}",
                    'description': f"Analysis of {query} with market insights",
                    'upload_date': (datetime.now() - timedelta(days=i)).isoformat(),
                    'view_count': 10000 + i * 1000,
                    'like_count': 500 + i * 50,
                    'comment_count': 100 + i * 10,
                    'channel': f"Crypto Channel {i % 3 + 1}",
                    'duration': f"{5 + i}:30",
                    'tags': query.split()
                })
            
            return video_data
            
        except Exception as e:
            logging.error(f"Error searching YouTube: {e}")
            return []
    
    async def analyze_video_content(self, video_data):
        """Analyze video content for market sentiment and insights"""
        try:
            logging.info(f"🔍 Analyzing {len(video_data)} videos for market intelligence...")
            
            sentiment_scores = []
            market_themes = {}
            price_predictions = []
            trending_topics = {}
            
            for video in video_data:
                # Analyze title and description sentiment
                text = f"{video['title']} {video['description']}"
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                sentiment_scores.append(sentiment)
                
                # Extract market themes
                themes = self.extract_market_themes(text)
                for theme in themes:
                    market_themes[theme] = market_themes.get(theme, 0) + 1
                
                # Extract price predictions
                predictions = self.extract_price_predictions(text)
                price_predictions.extend(predictions)
                
                # Track trending topics
                for tag in video.get('tags', []):
                    trending_topics[tag] = trending_topics.get(tag, 0) + 1
            
            # Calculate overall sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Determine market mood
            if avg_sentiment > 0.3:
                market_mood = 'bullish'
            elif avg_sentiment < -0.3:
                market_mood = 'bearish'
            else:
                market_mood = 'neutral'
            
            # Get top themes and topics
            top_themes = sorted(market_themes.items(), key=lambda x: x[1], reverse=True)[:5]
            top_topics = sorted(trending_topics.items(), key=lambda x: x[1], reverse=True)[:10]
            
            intelligence = {
                'youtube_sentiment': {
                    'average_sentiment': avg_sentiment,
                    'sentiment_score': avg_sentiment * 100,
                    'market_mood': market_mood,
                    'total_videos_analyzed': len(video_data)
                },
                'market_themes': dict(top_themes),
                'price_predictions': price_predictions,
                'trending_topics': dict(top_topics),
                'content_metrics': {
                    'total_views': sum(v.get('view_count', 0) for v in video_data),
                    'total_likes': sum(v.get('like_count', 0) for v in video_data),
                    'engagement_rate': self.calculate_engagement_rate(video_data)
                }
            }
            
            return intelligence
            
        except Exception as e:
            logging.error(f"Error analyzing video content: {e}")
            return {}
    
    def extract_market_themes(self, text):
        """Extract market themes from video text"""
        themes = []
        text_lower = text.lower()
        
        theme_keywords = {
            'bull_market': ['bull', 'bullish', 'moon', 'pump', 'rally', 'breakout'],
            'bear_market': ['bear', 'bearish', 'crash', 'dump', 'correction', 'dip'],
            'altcoin_season': ['altcoin', 'alt season', 'alts', 'altcoins'],
            'bitcoin_dominance': ['btc dominance', 'bitcoin dominance', 'btc dom'],
            'defi': ['defi', 'decentralized finance', 'yield farming', 'liquidity'],
            'nft': ['nft', 'non-fungible', 'opensea', 'collectibles'],
            'regulation': ['regulation', 'sec', 'government', 'legal', 'compliance'],
            'institutional': ['institutional', 'banks', 'etf', 'corporate', 'adoption']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def extract_price_predictions(self, text):
        """Extract price predictions from video text"""
        predictions = []
        
        # Look for price patterns like "$50,000", "$2,500", "100k", etc.
        price_patterns = [
            r'\$([0-9,]+)',  # $50,000
            r'([0-9,]+)k',   # 100k
            r'([0-9,]+)\s*thousand',  # 50 thousand
            r'([0-9,]+)\s*million',   # 1 million
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and convert to number
                    clean_match = match.replace(',', '')
                    if 'k' in text[text.find(match):text.find(match)+10].lower():
                        price = float(clean_match) * 1000
                    elif 'million' in text[text.find(match):text.find(match)+20].lower():
                        price = float(clean_match) * 1000000
                    else:
                        price = float(clean_match)
                    
                    predictions.append({
                        'price': price,
                        'context': text[max(0, text.find(match)-50):text.find(match)+50]
                    })
                except:
                    continue
        
        return predictions
    
    def calculate_engagement_rate(self, video_data):
        """Calculate engagement rate from video metrics"""
        total_engagement = sum(
            v.get('like_count', 0) + v.get('comment_count', 0) 
            for v in video_data
        )
        total_views = sum(v.get('view_count', 0) for v in video_data)
        
        if total_views > 0:
            return (total_engagement / total_views) * 100
        return 0
    
    async def gather_web_intelligence(self):
        """Gather market intelligence from web sources"""
        logging.info("🌐 Gathering web market intelligence...")
        
        try:
            web_sources = [
                'https://cointelegraph.com/rss',
                'https://bitcoinist.com/feed/',
                'https://cryptonews.com/news/feed/',
                'https://decrypt.co/feed',
                'https://u.today/rss'
            ]
            
            all_articles = []
            
            for source in web_sources:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:10]:  # Last 10 articles per source
                        all_articles.append({
                            'title': entry.title,
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'source': source
                        })
                except Exception as e:
                    logging.warning(f"Failed to parse {source}: {e}")
            
            # Analyze web content
            web_intelligence = await self.analyze_web_content(all_articles)
            
            return web_intelligence
            
        except Exception as e:
            logging.error(f"Error gathering web intelligence: {e}")
            return {}
    
    async def analyze_web_content(self, articles):
        """Analyze web articles for market intelligence"""
        try:
            sentiment_scores = []
            key_events = []
            market_signals = {}
            
            for article in articles:
                text = f"{article['title']} {article['summary']}"
                
                # Sentiment analysis
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                sentiment_scores.append(sentiment)
                
                # Identify key market events
                if self.is_market_moving_news(text):
                    key_events.append({
                        'title': article['title'],
                        'sentiment': sentiment,
                        'published': article['published'],
                        'impact_score': abs(sentiment) * 100,
                        'source': article['source']
                    })
                
                # Extract market signals
                signals = self.extract_market_signals(text)
                for signal in signals:
                    market_signals[signal] = market_signals.get(signal, 0) + 1
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                'web_sentiment': {
                    'average_sentiment': avg_sentiment,
                    'sentiment_score': avg_sentiment * 100,
                    'articles_analyzed': len(articles)
                },
                'key_events': sorted(key_events, key=lambda x: x['impact_score'], reverse=True)[:5],
                'market_signals': dict(sorted(market_signals.items(), key=lambda x: x[1], reverse=True)[:10]),
                'news_volume': len(articles)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing web content: {e}")
            return {}
    
    def is_market_moving_news(self, text):
        """Identify if news is likely to move markets"""
        market_moving_keywords = [
            'sec', 'regulation', 'etf', 'approval', 'ban', 'legal',
            'hack', 'breach', 'exploit', 'partnership', 'adoption',
            'institutional', 'bank', 'government', 'fed', 'treasury',
            'halving', 'fork', 'upgrade', 'launch', 'listing'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in market_moving_keywords)
    
    def extract_market_signals(self, text):
        """Extract market signals from text"""
        signals = []
        text_lower = text.lower()
        
        signal_keywords = {
            'bullish': ['bullish', 'positive', 'up', 'rise', 'increase', 'pump'],
            'bearish': ['bearish', 'negative', 'down', 'fall', 'decrease', 'dump'],
            'volatile': ['volatile', 'volatility', 'unstable', 'wild', 'swing'],
            'stable': ['stable', 'steady', 'consolidation', 'sideways'],
            'breakthrough': ['breakthrough', 'breakout', 'all-time high', 'ath'],
            'correction': ['correction', 'pullback', 'retracement', 'dip']
        }
        
        for signal, keywords in signal_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                signals.append(signal)
        
        return signals
    
    async def run_complete_intelligence_gathering(self):
        """Run complete intelligence gathering from all sources"""
        try:
            logging.info("🚀 Starting complete market intelligence gathering...")
            
            # Gather from all sources
            youtube_data = await self.gather_youtube_intelligence()
            web_data = await self.gather_web_intelligence()
            
            # Combine intelligence
            combined_intelligence = {
                'timestamp': datetime.now().isoformat(),
                'youtube_intelligence': youtube_data,
                'web_intelligence': web_data,
                'combined_sentiment': {
                    'overall_score': self.calculate_combined_sentiment(youtube_data, web_data),
                    'source_breakdown': {
                        'youtube': youtube_data.get('youtube_sentiment', {}).get('sentiment_score', 0),
                        'web': web_data.get('web_sentiment', {}).get('sentiment_score', 0)
                    }
                },
                'market_consensus': self.determine_market_consensus(youtube_data, web_data)
            }
            
            # Save results
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            results_path = os.path.join(project_root, 'data', 'market_intelligence.json')
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(combined_intelligence, f, indent=2, default=str)
            
            logging.info("✅ Market intelligence gathering completed!")
            return combined_intelligence
            
        except Exception as e:
            logging.error(f"Error in complete intelligence gathering: {e}")
            return {'error': str(e)}
    
    def calculate_combined_sentiment(self, youtube_data, web_data):
        """Calculate combined sentiment score from all sources"""
        youtube_score = youtube_data.get('youtube_sentiment', {}).get('sentiment_score', 0)
        web_score = web_data.get('web_sentiment', {}).get('sentiment_score', 0)
        
        # Weighted average (YouTube gets slightly higher weight due to engagement)
        combined_score = (youtube_score * 0.6 + web_score * 0.4)
        
        return combined_score
    
    def determine_market_consensus(self, youtube_data, web_data):
        """Determine overall market consensus"""
        youtube_mood = youtube_data.get('youtube_sentiment', {}).get('market_mood', 'neutral')
        
        # Extract web sentiment mood
        web_sentiment = web_data.get('web_sentiment', {}).get('sentiment_score', 0)
        if web_sentiment > 30:
            web_mood = 'bullish'
        elif web_sentiment < -30:
            web_mood = 'bearish'
        else:
            web_mood = 'neutral'
        
        # Determine consensus
        if youtube_mood == web_mood:
            consensus = youtube_mood
            confidence = 'high'
        elif (youtube_mood in ['bullish', 'bearish'] and web_mood == 'neutral') or \
             (web_mood in ['bullish', 'bearish'] and youtube_mood == 'neutral'):
            consensus = youtube_mood if youtube_mood != 'neutral' else web_mood
            confidence = 'medium'
        else:
            consensus = 'mixed'
            confidence = 'low'
        
        return {
            'consensus': consensus,
            'confidence': confidence,
            'youtube_mood': youtube_mood,
            'web_mood': web_mood
        }

# Integration function
async def gather_market_intelligence():
    """Main function to gather market intelligence"""
    intelligence = YouTubeMarketIntelligence()
    return await intelligence.run_complete_intelligence_gathering()

if __name__ == '__main__':
    async def main():
        results = await gather_market_intelligence()
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main()) 