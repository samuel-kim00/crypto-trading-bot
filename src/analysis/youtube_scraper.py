import os
import json
import logging
from datetime import datetime, timedelta
from pytube import Search, YouTube
from textblob import TextBlob
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_scraper.log'),
        logging.StreamHandler()
    ]
)

class YouTubeScraper:
    def __init__(self):
        self.search_terms = [
            "crypto trading strategy",
            "bitcoin analysis",
            "ethereum price prediction",
            "cryptocurrency market analysis",
            "crypto trading signals",
            "blockchain technology news",
            "defi projects",
            "nft market analysis",
            "altcoin trading",
            "crypto market trends"
        ]
        self.data_dir = "data"
        self.sentiment_file = os.path.join(self.data_dir, "youtube_sentiment.json")
        self.ensure_data_dir()

    def ensure_data_dir(self):
        """Ensure the data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_video_details(self, video):
        """Extract relevant details from a video"""
        try:
            return {
                'title': video.title,
                'description': video.description,
                'views': video.views,
                'rating': video.rating,
                'length': video.length,
                'publish_date': video.publish_date.isoformat() if video.publish_date else None,
                'url': video.watch_url
            }
        except Exception as e:
            logging.error(f"Error getting video details: {str(e)}")
            return None

    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        try:
            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity
            }
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {str(e)}")
            return {'polarity': 0, 'subjectivity': 0.5}

    def scrape_videos(self):
        """Scrape and analyze YouTube videos"""
        try:
            all_videos = []
            one_week_ago = datetime.now() - timedelta(days=7)
            
            for term in self.search_terms:
                logging.info(f"Searching for: {term}")
                try:
                    search = Search(term)
                    videos = search.results[:10]  # Get top 10 results
                    
                    for video in videos:
                        try:
                            details = self.get_video_details(video)
                            if details:
                                # Only process recent videos
                                if details['publish_date']:
                                    publish_date = datetime.fromisoformat(details['publish_date'])
                                    if publish_date < one_week_ago:
                                        continue
                                
                                # Analyze sentiment
                                title_sentiment = self.analyze_sentiment(details['title'])
                                desc_sentiment = self.analyze_sentiment(details['description'])
                                
                                # Calculate combined sentiment
                                combined_polarity = (title_sentiment['polarity'] + desc_sentiment['polarity']) / 2
                                combined_subjectivity = (title_sentiment['subjectivity'] + desc_sentiment['subjectivity']) / 2
                                
                                video_data = {
                                    **details,
                                    'search_term': term,
                                    'sentiment': {
                                        'title': title_sentiment,
                                        'description': desc_sentiment,
                                        'combined': {
                                            'polarity': combined_polarity,
                                            'subjectivity': combined_subjectivity
                                        }
                                    }
                                }
                                all_videos.append(video_data)
                                
                        except Exception as e:
                            logging.error(f"Error processing video: {str(e)}")
                            continue
                            
                except Exception as e:
                    logging.error(f"Error searching for term '{term}': {str(e)}")
                    continue
            
            # Save results
            if all_videos:
                # Convert to DataFrame for analysis
                df = pd.DataFrame([{
                    'search_term': v['search_term'],
                    'polarity': v['sentiment']['combined']['polarity'],
                    'subjectivity': v['sentiment']['combined']['subjectivity'],
                    'views': v['views'],
                    'publish_date': v['publish_date']
                } for v in all_videos])
                
                # Calculate weighted sentiment scores
                df['weight'] = df['views'] / df['views'].sum()
                weighted_sentiment = (df['polarity'] * df['weight']).sum()
                
                # Save sentiment data
                sentiment_data = {
                    'timestamp': datetime.now().isoformat(),
                    'videos': all_videos,
                    'analysis': {
                        'total_videos': len(all_videos),
                        'average_sentiment': df['polarity'].mean(),
                        'weighted_sentiment': weighted_sentiment,
                        'sentiment_std': df['polarity'].std(),
                        'most_viewed': df.nlargest(5, 'views')[['search_term', 'polarity', 'views']].to_dict('records')
                    }
                }
                
                with open(self.sentiment_file, 'w') as f:
                    json.dump(sentiment_data, f, indent=2)
                logging.info(f"Saved sentiment data for {len(all_videos)} videos")
                
            else:
                logging.warning("No videos found")
            
        except Exception as e:
            logging.error(f"Error in scrape_videos: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        scraper = YouTubeScraper()
        scraper.scrape_videos()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        sys.exit(1) 