import tweepy
import praw
import requests
from typing import Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketSentimentAnalyzer:
    def __init__(self, config: Dict):
        """Initialize sentiment analyzer with API keys."""
        # Twitter/X API setup
        self.twitter_client = tweepy.Client(
            bearer_token=config.get('twitter_bearer_token'),
            consumer_key=config.get('twitter_api_key'),
            consumer_secret=config.get('twitter_api_secret'),
            access_token=config.get('twitter_access_token'),
            access_token_secret=config.get('twitter_access_secret')
        )

        # Reddit API setup
        self.reddit_client = praw.Reddit(
            client_id=config.get('reddit_client_id'),
            client_secret=config.get('reddit_client_secret'),
            user_agent='Crypto Sentiment Bot 1.0'
        )

        # Fear & Greed Index API
        self.fear_greed_url = "https://api.alternative.me/fng/"

    async def get_sentiment(self, token: str) -> Dict:
        """Get comprehensive market sentiment data."""
        try:
            # Collect data from all sources
            twitter_sentiment = await self.analyze_twitter_sentiment(token)
            reddit_sentiment = await self.analyze_reddit_sentiment(token)
            fear_greed = await self.get_fear_greed_index()

            # Combine sentiment scores
            overall_score = (
                twitter_sentiment['score'] * 0.4 +  # 40% weight to Twitter
                reddit_sentiment['score'] * 0.3 +   # 30% weight to Reddit
                fear_greed['value'] / 100 * 0.3     # 30% weight to Fear & Greed
            )

            return {
                'overall_score': overall_score,  # 0-1 scale, higher is more positive
                'twitter': twitter_sentiment,
                'reddit': reddit_sentiment,
                'fear_greed_index': fear_greed,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting sentiment for {token}: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def analyze_twitter_sentiment(self, token: str) -> Dict:
        """Analyze Twitter sentiment for token."""
        try:
            # Get recent tweets about the token
            query = f"#{token} OR #{token}USD OR #{token}USDT -is:retweet"
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=100
            )

            if not tweets.data:
                return {'score': 0.5, 'volume': 0, 'sources': []}

            # Here you would normally use a proper NLP model
            # This is a simplified version
            positive_keywords = ['bullish', 'buy', 'moon', 'pump', 'long']
            negative_keywords = ['bearish', 'sell', 'dump', 'short', 'crash']

            sentiment_score = 0
            sources = []

            for tweet in tweets.data:
                text = tweet.text.lower()
                pos_count = sum(1 for word in positive_keywords if word in text)
                neg_count = sum(1 for word in negative_keywords if word in text)

                if pos_count or neg_count:
                    score = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0.5
                    sentiment_score += score
                    sources.append({
                        'text': tweet.text,
                        'score': score
                    })

            return {
                'score': sentiment_score / len(tweets.data),
                'volume': len(tweets.data),
                'sources': sources[:5]  # Return top 5 relevant tweets
            }

        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
            return {'score': 0.5, 'volume': 0, 'sources': []}

    async def analyze_reddit_sentiment(self, token: str) -> Dict:
        """Analyze Reddit sentiment for token."""
        try:
            subreddits = ['CryptoCurrency', f'{token}', 'CryptoMarkets']
            posts = []

            for subreddit in subreddits:
                try:
                    for post in self.reddit_client.subreddit(subreddit).hot(limit=50):
                        if token.lower() in post.title.lower() or token.lower() in post.selftext.lower():
                            posts.append({
                                'title': post.title,
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments
                            })
                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit}: {e}")
                    continue

            if not posts:
                return {'score': 0.5, 'volume': 0, 'sources': []}

            # Calculate sentiment based on upvote ratios and comment activity
            sentiment_score = sum(post['upvote_ratio'] for post in posts) / len(posts)

            return {
                'score': sentiment_score,
                'volume': len(posts),
                'sources': sorted(posts, key=lambda x: x['score'], reverse=True)[:5]
            }

        except Exception as e:
            logger.error(f"Reddit sentiment error: {e}")
            return {'score': 0.5, 'volume': 0, 'sources': []}

    async def get_fear_greed_index(self) -> Dict:
        """Get Fear & Greed Index."""
        try:
            response = requests.get(self.fear_greed_url)
            data = response.json()

            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification'],
                'timestamp': data['data'][0]['timestamp']
            }

        except Exception as e:
            logger.error(f"Fear & Greed Index error: {e}")
            return {'value': 50, 'classification': 'Neutral', 'timestamp': None}