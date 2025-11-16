# AI for Social Good - Mental Well-being Analysis

**Mahindra University Research Project**

A machine learning system for analyzing social media content to identify emotional and behavioral patterns impacting mental well-being, with strong emphasis on ethical AI practices and privacy protection.

##  Project Overview

This project develops ML models to:
- Analyze social media content for sentiment and emotional patterns
- Identify behavioral indicators related to mental well-being
- Implement privacy-preserving data processing techniques
- Address ethical considerations in AI-driven mental health analysis

##  Key Features

- **Sentiment Analysis**: Multi-dimensional sentiment scoring (positive, negative, neutral, compound)
- **Mental Health Indicators**: Pattern detection for potential mental well-being concerns
- **Privacy Protection**: Automatic PII (Personally Identifiable Information) anonymization
- **Ethical AI**: Built-in safeguards and responsible AI practices
- **Batch Processing**: Efficient analysis of large text datasets
- **Statistical Reporting**: Comprehensive summary statistics and insights

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for version control)

##  Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ai-social-good-mental-wellbeing.git
cd ai-social-good-mental-wellbeing
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

##  Usage

### Quick Start

```bash
# Run the main analysis script
python sentiment_analysis.py

# Run the comprehensive demo
python demo.py
```

### Basic Example

```python
from sentiment_analysis import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Single text analysis
text = "Feeling anxious about upcoming exams but trying to stay positive."
result = analyzer.calculate_sentiment_score(text)
print(result)
# Output: {'positive': 0.067, 'negative': 0.067, 'neutral': 0.867, 'compound': 0.0}

# Detect mental health indicators
mh_result = analyzer.detect_mental_health_indicators(text)
print(mh_result)
# Output: {'has_indicators': True, 'indicator_count': 1, 'indicators': ['anxious'], 'risk_level': 'medium'}

# Batch analysis
texts = [
    "I'm so happy today!",
    "Feeling really stressed and overwhelmed.",
    "Just another normal day."
]
results_df = analyzer.analyze_batch(texts)
print(results_df)
```

### Advanced Usage

```python
# Privacy-protected analysis
text = "Contact me @john_doe at john@email.com"
anonymized = analyzer.anonymize_data(text)
print(anonymized)  # "Contact me [USER] at [EMAIL]"

# Get summary statistics
summary = analyzer.get_summary_statistics(results_df)
print(f"Average sentiment: {summary['avg_sentiment']}")
print(f"High risk texts: {summary['high_risk_count']}")
```

##  Output Format

The analyzer returns structured data including:
- Sentiment scores (positive, negative, neutral, compound)
- Mental health indicator flags
- Risk level assessment (low, medium, high)
- Anonymized text statistics
