import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import logging
from typing import Dict, List, Optional
import json
from pathlib import Path
import asyncio
import aiohttp
from newsapi import NewsApiClient
import pytube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

class MediaAnalyzer:
    def __init__(self):
        # YouTube API setup
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        
        # Rate limiting settings
        self.youtube_quota_limit = 10000  # Daily quota limit
        self.youtube_quota_used = 0
        self.youtube_last_reset = datetime.now()
        self.youtube_request_delay = 1  # Delay between requests in seconds
        self.youtube_cache_duration = 3600  # Cache duration in seconds (1 hour)
        
        # News API setup
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.newsapi = NewsApiClient(api_key=self.news_api_key)
        
        # Popular crypto YouTube channels
        self.crypto_channels = [
            'UCqK_GSMbpiV8spgD3ZGloSw',  # Coin Bureau
            'UCdUSSt-IEUg2eq46rD7lu_g',  # Crypto Daily
            'UC7vVhkEfw4nOGp8TyDk7RcQ',  # Crypto Banter
            'UC6PlLNOI4K7D4f8FpBd0KZw',  # Crypto Jebb
            'UCqK_GSMbpiV8spgD3ZGloSw',  # BitBoy Crypto
        ]
        
        # Data storage paths
        self.data_dir = Path('data/media_analysis')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.youtube_data_file = self.data_dir / 'youtube_analysis.json'
        self.news_data_file = self.data_dir / 'news_analysis.json'
        self.correlation_data_file = self.data_dir / 'media_price_correlation.json'
        self.youtube_cache_file = self.data_dir / 'youtube_cache.json'
        
        # Load existing data
        self.load_data()
    
        self.sentiment_file = 'data/youtube_sentiment.json'
        self.ensure_files_exist()
    
    def ensure_files_exist(self):
        """Ensure necessary files exist"""
        if not os.path.exists(self.sentiment_file):
            self.save_sentiment_data({
                'timestamp': datetime.now().isoformat(),
                'videos': [],
                'analysis': {
                    'total_videos': 0,
                    'average_sentiment': 0,
                    'weighted_sentiment': 0,
                    'sentiment_std': 0,
                    'most_viewed': []
                }
            })

    def save_sentiment_data(self, data):
        """Save sentiment data to file"""
        with open(self.sentiment_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_sentiment_data(self):
        """Load sentiment data from file"""
        try:
            with open(self.sentiment_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'timestamp': datetime.now().isoformat(),
                'videos': [],
                'analysis': {
                    'total_videos': 0,
                    'average_sentiment': 0,
                    'weighted_sentiment': 0,
                    'sentiment_std': 0,
                    'most_viewed': []
                }
            }

    def analyze_text(self, text):
        """Analyze sentiment of text"""
        try:
            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity
            }
        except Exception as e:
            logging.error(f"Error analyzing text: {str(e)}")
            return {'polarity': 0, 'subjectivity': 0.5}

    def update_sentiment(self, video_data):
        """Update sentiment data with new video"""
        data = self.load_sentiment_data()
        
        # Add new video data
        data['videos'].append(video_data)
        
        # Keep only videos from last 7 days
        cutoff_date = datetime.now() - timedelta(days=7)
        data['videos'] = [
            v for v in data['videos']
            if datetime.fromisoformat(v['publish_date']) > cutoff_date
        ]
        
        # Update analysis
        if data['videos']:
            df = pd.DataFrame([{
                'polarity': v['sentiment']['combined']['polarity'],
                'views': v['views']
            } for v in data['videos']])
            
            data['analysis'] = {
                'total_videos': len(data['videos']),
                'average_sentiment': df['polarity'].mean(),
                'weighted_sentiment': (df['polarity'] * df['views']).sum() / df['views'].sum(),
                'sentiment_std': df['polarity'].std(),
                'most_viewed': sorted(data['videos'], key=lambda x: x['views'], reverse=True)[:5]
            }
        
        data['timestamp'] = datetime.now().isoformat()
        self.save_sentiment_data(data)

    def get_latest_sentiment(self):
        """Get latest sentiment analysis"""
        data = self.load_sentiment_data()
        
        # Check if data is stale (older than 24 hours)
        last_update = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - last_update > timedelta(hours=24):
            data['analysis']['is_stale'] = True
        
        return data['analysis']
    
    def load_data(self):
        """Load existing analysis data"""
        try:
            if self.youtube_data_file.exists():
                with open(self.youtube_data_file, 'r') as f:
                    self.youtube_data = json.load(f)
            else:
                self.youtube_data = {'videos': [], 'analysis': {}}
            
            if self.news_data_file.exists():
                with open(self.news_data_file, 'r') as f:
                    self.news_data = json.load(f)
            else:
                self.news_data = {'articles': [], 'analysis': {}}
            
            if self.correlation_data_file.exists():
                with open(self.correlation_data_file, 'r') as f:
                    self.correlation_data = json.load(f)
            else:
                self.correlation_data = {'correlations': []}
                
            # Load YouTube quota usage
            if self.youtube_cache_file.exists():
                with open(self.youtube_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.youtube_quota_used = cache_data.get('quota_used', 0)
                    self.youtube_last_reset = datetime.fromisoformat(cache_data.get('last_reset', datetime.now().isoformat()))
        except Exception as e:
            logging.error(f"Error loading media analysis data: {str(e)}")
            self.youtube_data = {'videos': [], 'analysis': {}}
            self.news_data = {'articles': [], 'analysis': {}}
            self.correlation_data = {'correlations': []}
    
    def save_data(self):
        """Save analysis data"""
        try:
            with open(self.youtube_data_file, 'w') as f:
                json.dump(self.youtube_data, f, indent=4)
            with open(self.news_data_file, 'w') as f:
                json.dump(self.news_data, f, indent=4)
            with open(self.correlation_data_file, 'w') as f:
                json.dump(self.correlation_data, f, indent=4)
                
            # Save YouTube quota usage
            cache_data = {
                'quota_used': self.youtube_quota_used,
                'last_reset': self.youtube_last_reset.isoformat()
            }
            with open(self.youtube_cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logging.error(f"Error saving media analysis data: {str(e)}")
    
    async def fetch_youtube_videos(self, days_back: int = 7) -> List[Dict]:
        """Fetch recent videos from crypto channels with rate limiting and caching"""
        videos = []
        try:
            # Check if we have cached data
            if self.youtube_cache_file.exists():
                with open(self.youtube_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if (datetime.now() - datetime.fromisoformat(cache_data['timestamp'])).total_seconds() < self.youtube_cache_duration:
                        return cache_data['videos']

            # Reset quota counter if it's a new day
            if (datetime.now() - self.youtube_last_reset).days >= 1:
                self.youtube_quota_used = 0
                self.youtube_last_reset = datetime.now()

            for channel_id in self.crypto_channels:
                # Check if we've exceeded quota
                if self.youtube_quota_used >= self.youtube_quota_limit:
                    logging.warning('YouTube quota limit reached, using cached data')
                    return self.youtube_data.get('videos', [])

                # Add delay between requests
                await asyncio.sleep(self.youtube_request_delay)

                request = self.youtube.search().list(
                    part="snippet",
                    channelId=channel_id,
                    order="date",
                    maxResults=10,  # Reduced from 50 to save quota
                    type="video"
                )
                try:
                    response = request.execute()
                    self.youtube_quota_used += 100  # Each search request costs 100 units
                except Exception as e:
                    if 'quotaExceeded' in str(e):
                        logging.warning('YouTube quota exceeded, using cached data')
                        return self.youtube_data.get('videos', [])
                    else:
                        raise

                for item in response['items']:
                    video_data = {
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'published_at': item['snippet']['publishedAt'],
                        'channel_title': item['snippet']['channelTitle'],
                        'view_count': 0,
                        'like_count': 0,
                        'comment_count': 0,
                        'transcript': '',
                        'sentiment': 0.0
                    }

                    # Get video statistics (costs 1 unit)
                    if self.youtube_quota_used < self.youtube_quota_limit:
                        stats_request = self.youtube.videos().list(
                            part="statistics",
                            id=video_data['video_id']
                        )
                        try:
                            stats_response = stats_request.execute()
                            self.youtube_quota_used += 1
                            
                            if stats_response['items']:
                                stats = stats_response['items'][0]['statistics']
                                video_data['view_count'] = int(stats.get('viewCount', 0))
                                video_data['like_count'] = int(stats.get('likeCount', 0))
                                video_data['comment_count'] = int(stats.get('commentCount', 0))
                        except Exception as e:
                            if 'quotaExceeded' in str(e):
                                logging.warning('YouTube quota exceeded while fetching stats')
                            else:
                                logging.error(f"Error fetching video stats: {str(e)}")

                    # Get video transcript (no quota cost)
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_data['video_id'])
                        formatter = TextFormatter()
                        video_data['transcript'] = formatter.format_transcript(transcript)
                    except:
                        video_data['transcript'] = ''

                    videos.append(video_data)

            # Cache the results
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'videos': videos,
                'quota_used': self.youtube_quota_used,
                'last_reset': self.youtube_last_reset.isoformat()
            }
            with open(self.youtube_cache_file, 'w') as f:
                json.dump(cache_data, f)

            return videos
        except Exception as e:
            logging.error(f"Error fetching YouTube videos: {str(e)}")
            return self.youtube_data.get('videos', [])  # Return cached data on error
    
    async def analyze_youtube_content(self, videos: List[Dict]) -> Dict:
        """Analyze YouTube content sentiment and impact"""
        analysis = {
            'overall_sentiment': 0.0,
            'channel_sentiments': {},
            'topic_sentiments': {},
            'engagement_metrics': {}
        }
        
        try:
            for video in videos:
                # Skip if title or description is None
                if not video.get('title') or not video.get('description'):
                    continue
                    
                # Analyze title and description
                title_sentiment = TextBlob(video['title']).sentiment.polarity
                desc_sentiment = TextBlob(video['description']).sentiment.polarity
                transcript_sentiment = TextBlob(video['transcript']).sentiment.polarity if video['transcript'] else 0
                
                # Calculate weighted sentiment
                video_sentiment = (
                    title_sentiment * 0.3 +
                    desc_sentiment * 0.2 +
                    transcript_sentiment * 0.5
                )
                
                # Update channel sentiments
                channel = video['channel_title']
                if channel not in analysis['channel_sentiments']:
                    analysis['channel_sentiments'][channel] = []
                analysis['channel_sentiments'][channel].append(video_sentiment)
                
                # Update engagement metrics
                analysis['engagement_metrics'][video['video_id']] = {
                    'views': video['view_count'],
                    'likes': video['like_count'],
                    'comments': video['comment_count']
                }
            
            # Calculate overall sentiment
            all_sentiments = []
            for channel_sentiments in analysis['channel_sentiments'].values():
                all_sentiments.extend(channel_sentiments)
            analysis['overall_sentiment'] = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
            
            return analysis
        except Exception as e:
            logging.error(f"Error analyzing YouTube content: {str(e)}")
            return analysis
    
    async def fetch_news_articles(self, days_back: int = 7) -> List[Dict]:
        """Fetch recent crypto news articles"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch news articles
            articles = self.newsapi.get_everything(
                q='cryptocurrency OR bitcoin OR ethereum',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            return articles['articles']
        except Exception as e:
            logging.error(f"Error fetching news articles: {str(e)}")
            return []
    
    async def analyze_news_content(self, articles: List[Dict]) -> Dict:
        """Analyze news content sentiment and impact"""
        analysis = {
            'overall_sentiment': 0.0,
            'source_sentiments': {},
            'topic_sentiments': {},
            'article_impacts': {}
        }
        
        try:
            for article in articles:
                # Skip if title or description is None
                if not article.get('title') or not article.get('description'):
                    continue
                    
                title_sentiment = TextBlob(article['title']).sentiment.polarity
                content_sentiment = TextBlob(article['description']).sentiment.polarity
                article_sentiment = (title_sentiment * 0.4 + content_sentiment * 0.6)
                
                source = article['source']['name']
                if source not in analysis['source_sentiments']:
                    analysis['source_sentiments'][source] = []
                analysis['source_sentiments'][source].append(article_sentiment)
                
                analysis['article_impacts'][article['url']] = {
                    'sentiment': article_sentiment,
                    'published_at': article['publishedAt'],
                    'relevance_score': article.get('relevancy_score', 0)
                }
            
            # Calculate overall sentiment
            all_sentiments = []
            for source_sentiments in analysis['source_sentiments'].values():
                all_sentiments.extend(source_sentiments)
            analysis['overall_sentiment'] = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
            
            return analysis
        except Exception as e:
            logging.error(f"Error analyzing news content: {str(e)}")
            return analysis
    
    async def analyze_media_impact(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Analyze impact of media on price movements"""
        try:
            # Get media data
            videos = await self.fetch_youtube_videos()
            youtube_analysis = await self.analyze_youtube_content(videos)
            
            articles = await self.fetch_news_articles()
            news_analysis = await self.analyze_news_content(articles)
            
            # Calculate price changes
            price_data['returns'] = price_data['close'].pct_change()
            
            # Combine media sentiment
            media_sentiment = (
                youtube_analysis['overall_sentiment'] * 0.6 +
                news_analysis['overall_sentiment'] * 0.4
            )
            
            # Calculate correlation
            sentiment_series = pd.Series([media_sentiment] * len(price_data))
            correlation = sentiment_series.corr(price_data['returns'])
            
            return {
                'media_sentiment': media_sentiment,
                'price_correlation': correlation,
                'youtube_analysis': youtube_analysis,
                'news_analysis': news_analysis
            }
        except Exception as e:
            logging.error(f"Error analyzing media impact: {str(e)}")
            return {
                'media_sentiment': 0.0,
                'price_correlation': 0.0,
                'youtube_analysis': {},
                'news_analysis': {}
            }
    
    async def update_analysis(self, symbol: str, price_data: pd.DataFrame):
        """Update media analysis for a symbol"""
        try:
            # Get media impact analysis
            impact_analysis = await self.analyze_media_impact(symbol, price_data)
            
            # Update correlation data
            self.correlation_data['correlations'].append({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'correlation': impact_analysis['price_correlation'],
                'media_sentiment': impact_analysis['media_sentiment']
            })
            
            # Keep only last 100 correlations
            if len(self.correlation_data['correlations']) > 100:
                self.correlation_data['correlations'] = self.correlation_data['correlations'][-100:]
            
            # Save updated data
            self.save_data()
            
            return {
                'impact_analysis': {
                    'media_sentiment': impact_analysis['media_sentiment'],
                    'price_impact': 'positive' if impact_analysis['price_correlation'] > 0.3 else
                                  'negative' if impact_analysis['price_correlation'] < -0.3 else 'neutral'
                },
                'youtube_analysis': impact_analysis['youtube_analysis'],
                'news_analysis': impact_analysis['news_analysis']
            }
        except Exception as e:
            logging.error(f"Error updating media analysis: {str(e)}")
            return None 