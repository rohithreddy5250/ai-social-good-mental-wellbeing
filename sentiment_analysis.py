"""
AI for Social Good - Mental Well-being Analysis
Sentiment Analysis and Text Processing Module
Mahindra University

This module provides functionality for analyzing social media content
to identify emotional and behavioral patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter

class SentimentAnalyzer:
    """
    A sentiment analysis system for social media content with privacy-preserving features.
    """
    
    def __init__(self):
        # Simple sentiment lexicon (expand with proper datasets)
        self.positive_words = {'happy', 'joy', 'love', 'excellent', 'good', 'wonderful', 
                               'positive', 'fortunate', 'correct', 'superior', 'great'}
        self.negative_words = {'sad', 'hate', 'bad', 'terrible', 'awful', 'negative',
                               'unfortunate', 'wrong', 'inferior', 'poor', 'depressed'}
        self.mental_health_keywords = {'anxious', 'depressed', 'lonely', 'stressed',
                                       'overwhelmed', 'hopeless', 'worried', 'isolated'}
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def anonymize_data(self, text: str) -> str:
        """Remove personally identifiable information."""
        # Replace names pattern (simplified)
        text = re.sub(r'@\w+', '[USER]', text)
        # Replace email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        # Replace phone numbers
        text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        return text
    
    def calculate_sentiment_score(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores for given text."""
        words = text.lower().split()
        
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        total = len(words)
        
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
        
        pos_score = pos_count / total
        neg_score = neg_count / total
        neutral_score = 1 - (pos_score + neg_score)
        
        # Compound score: -1 (most negative) to +1 (most positive)
        compound = (pos_count - neg_count) / (total + 1)
        
        return {
            'positive': round(pos_score, 3),
            'negative': round(neg_score, 3),
            'neutral': round(neutral_score, 3),
            'compound': round(compound, 3)
        }
    
    def detect_mental_health_indicators(self, text: str) -> Dict[str, any]:
        """Detect potential mental health indicators in text."""
        words = text.lower().split()
        
        indicators_found = [word for word in words if word in self.mental_health_keywords]
        
        return {
            'has_indicators': len(indicators_found) > 0,
            'indicator_count': len(indicators_found),
            'indicators': list(set(indicators_found)),
            'risk_level': 'high' if len(indicators_found) > 2 else 'medium' if len(indicators_found) > 0 else 'low'
        }
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """Analyze a batch of texts and return results as DataFrame."""
        results = []
        
        for idx, text in enumerate(texts):
            # Anonymize and preprocess
            anon_text = self.anonymize_data(text)
            clean_text = self.preprocess_text(anon_text)
            
            # Get sentiment scores
            sentiment = self.calculate_sentiment_score(clean_text)
            
            # Detect mental health indicators
            mh_indicators = self.detect_mental_health_indicators(clean_text)
            
            results.append({
                'text_id': idx,
                'original_length': len(text),
                'processed_length': len(clean_text),
                **sentiment,
                **mh_indicators
            })
        
        return pd.DataFrame(results)
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from analysis results."""
        return {
            'total_texts': len(df),
            'avg_sentiment': df['compound'].mean(),
            'positive_ratio': (df['compound'] > 0).sum() / len(df),
            'negative_ratio': (df['compound'] < 0).sum() / len(df),
            'texts_with_mh_indicators': df['has_indicators'].sum(),
            'high_risk_count': (df['risk_level'] == 'high').sum(),
            'most_common_indicators': df['indicators'].explode().value_counts().head(5).to_dict()
        }


def main():
    """Example usage of the SentimentAnalyzer."""
    # Sample data (use real data in production)
    sample_texts = [
        "I'm feeling really happy today! Life is wonderful.",
        "Feeling very anxious and overwhelmed with everything going on.",
        "Just another normal day at work.",
        "I hate how stressed and depressed I feel lately.",
        "Great news! Got accepted into the program I wanted!"
    ]
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze texts
    print("Analyzing social media content...\n")
    results_df = analyzer.analyze_batch(sample_texts)
    
    print("Analysis Results:")
    print(results_df.to_string())
    
    print("\n" + "="*50)
    print("Summary Statistics:")
    summary = analyzer.get_summary_statistics(results_df)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
