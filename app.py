# Import libraries with error handling
try:
    import json
    import urllib.request
    import streamlit as st
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime, timedelta
    import time
    import os
    import re
    import threading
    from collections import deque
except ImportError as e:
    import streamlit as st
    st.error(f"""
    Missing required dependency: {str(e)}
    
    Please make sure you have a requirements.txt file with these packages:
    - streamlit
    - scikit-learn
    - pandas
    - matplotlib
    - numpy
    - urllib3
    
    If deploying on Streamlit Cloud, ensure requirements.txt is in your repository.
    """)
    st.stop()

# Set up the page
st.set_page_config(
    page_title="Automated Stock News Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: #f8f9fa;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .sentiment-score {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive-bg {
        background-color: #d4edda;
        color: #155724;
    }
    .negative-bg {
        background-color: #f8d7da;
        color: #721c24;
    }
    .neutral-bg {
        background-color: #e2e3e5;
        color: #383d41;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">📈 Automated Stock News Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("""
This application automatically fetches the latest stock news every hour, analyzes sentiment using machine learning,
and provides a comprehensive market sentiment report with trend analysis.
""")

# Initialize session state for storing news and analysis results
if 'news_data' not in st.session_state:
    st.session_state.news_data = pd.DataFrame(columns=['Title', 'Description', 'Sentiment', 'Confidence', 'Timestamp'])
if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = deque(maxlen=100)  # Store last 100 sentiment scores
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = ""

# Function to clean and preprocess text
def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove special characters and digits but keep spaces and letters
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to create a robust sample dataset
def create_sample_dataset():
    """Create a guaranteed sample dataset with enough data for training"""
    sample_data = pd.DataFrame({
        'Sentence': [
            # Positive samples (20)
            'Company shares surge after positive earnings report',
            'Stock market reaches all time high records',
            'Profits increase significantly this quarter',
            'New product launch exceeds expectations',
            'Company announces major expansion plans',
            'Revenue growth beats analyst predictions',
            'Dividend increase announced for shareholders',
            'Merger creates market leading company',
            'Innovation drives competitive advantage',
            'Strong demand boosts sales figures',
            'Market share increases substantially',
            'Positive outlook for future growth',
            'Investment in research pays off',
            'Cost reduction improves profitability',
            'Strategic partnership creates opportunities',
            'Customer satisfaction at record levels',
            'Export business expands to new markets',
            'Technology upgrade increases efficiency',
            'Brand recognition reaches new heights',
            'Successful quarter for all divisions',
            
            # Negative samples (20)
            'Stock prices plummet due to market uncertainty',
            'Company reports significant financial losses',
            'Sales decline amid economic slowdown',
            'Layoffs announced due to restructuring',
            'Competition intensifies affecting market share',
            'Regulatory challenges impact operations',
            'Supply chain disruptions cause delays',
            'Debt levels rise creating financial pressure',
            'Product recall affects company reputation',
            'Management changes create uncertainty',
            'Economic downturn impacts all sectors',
            'Cost overruns on major projects',
            'Market volatility creates investor concern',
            'Legal issues threaten company stability',
            'Natural disaster affects production',
            'Cybersecurity breach compromises data',
            'Consumer confidence declines sharply',
            'Trade wars impact international business',
            'Inflation concerns affect pricing strategy',
            'Quarterly results disappoint investors'
        ],
        'Sentiment': [
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative'
        ]
    })
    
    # Clean the sample data
    sample_data['Sentence'] = sample_data['Sentence'].apply(clean_text)
    return sample_data

# Load and train model (with caching to improve performance)
@st.cache_resource
def load_model():
    try:
        # Start with guaranteed sample data
        data = create_sample_dataset()
        
        # Try to load the original dataset if it exists and has data
        try:
            if os.path.exists("Sentiment_Stock_data.csv"):
                original_data = pd.read_csv("Sentiment_Stock_data.csv")
                if len(original_data) > 0:
                    # Clean the text data
                    original_data['Sentence'] = original_data['Sentence'].apply(clean_text)
                    # Filter only valid sentiments
                    original_data = original_data[original_data['Sentiment'].isin(['positive', 'negative'])]
                    # Remove empty sentences
                    original_data = original_data[original_data['Sentence'].str.len() > 0]
                    
                    if len(original_data) > 0:
                        # Combine with sample data
                        data = pd.concat([data, original_data], ignore_index=True)
        except Exception as e:
            st.warning(f"Could not load original dataset: {e}")
        
        # Try to load enhanced dataset if it exists and has data
        try:
            if os.path.exists("Sentiment_Stock_data_enhanced.csv"):
                enhanced_data = pd.read_csv("Sentiment_Stock_data_enhanced.csv")
                if len(enhanced_data) > 0:
                    # Clean the text data
                    enhanced_data['Sentence'] = enhanced_data['Sentence'].apply(clean_text)
                    # Filter only valid sentiments
                    enhanced_data = enhanced_data[enhanced_data['Sentiment'].isin(['positive', 'negative'])]
                    # Remove empty sentences
                    enhanced_data = enhanced_data[enhanced_data['Sentence'].str.len() > 0]
                    
                    if len(enhanced_data) > 0:
                        # Combine with existing data
                        data = pd.concat([data, enhanced_data], ignore_index=True)
        except Exception as e:
            st.warning(f"Could not load enhanced dataset: {e}")
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Apply TF-IDF vectorization with simpler parameters
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english', 
            lowercase=True,
            min_df=1,  # Lower minimum document frequency
            max_features=1000  # Limit features to avoid memory issues
        )

        # Fit and transform the training text
        X = vectorizer.fit_transform(data['Sentence'])
        
        # Check if vocabulary was created
        if len(vectorizer.vocabulary_) == 0:
            # Try without stopwords if vocabulary is empty
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True,
                min_df=1,
                max_features=1000,
                stop_words=None  # Remove stopwords filter
            )
            X = vectorizer.fit_transform(data['Sentence'])
            
            if len(vectorizer.vocabulary_) == 0:
                raise Exception("Empty vocabulary - all words were filtered out")
        
        # Train Naive Bayes classifier
        NB = MultinomialNB()
        NB.fit(X, data['Sentiment'])
        
        return vectorizer, NB, data
        
    except Exception as e:
        st.error(f"Error in model loading: {str(e)}")
        # Final fallback to guaranteed sample data
        data = create_sample_dataset()
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), 
            stop_words=None, 
            lowercase=True,
            min_df=1
        )
        X = vectorizer.fit_transform(data['Sentence'])
        
        NB = MultinomialNB()
        NB.fit(X, data['Sentiment'])
        
        return vectorizer, NB, data

# Load the model
vectorizer, model, data = load_model()

# Function to fetch news from API
def fetch_news():
    try:
        # TODO: replace API_KEY with your API key.
        apikey = "0235be61680e03536cfdc55b496c48dc"
        category = "business"
        url = f"https://gnews.io/api/v4/top-headlines?category=business&lang=en&country=in&max=10&apikey={apikey}"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data["articles"]
            print(articles)
            
            new_articles = []
            for article in articles:
                # Clean and combine title and description for analysis
                text_to_analyze = clean_text(f"{article['title']} {article.get('description', '')}")
                
                if len(text_to_analyze) > 0:
                    # Analyze sentiment
                    X_input = vectorizer.transform([text_to_analyze])
                    confidence_scores = model.predict_proba(X_input)[0]
                    
                    # Get confidence for positive and negative classes
                    negative_confidence = confidence_scores[0]  # Class 0 (negative)
                    positive_confidence = confidence_scores[1]  # Class 1 (positive)
                    
                    # Determine sentiment based on confidence levels
                    if positive_confidence > 0.5:
                        predicted_class = 'positive'
                        confidence = positive_confidence
                    else:
                        predicted_class = 'negative'
                        confidence = negative_confidence
                    
                    # Add to new articles list
                    new_articles.append({
                        'Title': article['title'],
                        'Description': article.get('description', ''),
                        'Sentiment': predicted_class,
                        'Confidence': confidence,
                        'Timestamp': datetime.now()
                    })
            
            return new_articles
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# Function to analyze sentiment of text
def analyze_sentiment(text):
    cleaned_text = clean_text(text)
    
    if len(cleaned_text) == 0:
        return None, None
    
    try:
        # Transform input text and make prediction
        X_input = vectorizer.transform([cleaned_text])
        confidence_scores = model.predict_proba(X_input)[0]
        
        # Get confidence for positive and negative classes
        negative_confidence = confidence_scores[0]  # Class 0 (negative)
        positive_confidence = confidence_scores[1]  # Class 1 (positive)
        
        # Determine sentiment based on confidence levels
        if positive_confidence > 0.5:
            predicted_class = 'positive'
            confidence = positive_confidence
        else:
            predicted_class = 'negative'
            confidence = negative_confidence
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Function to generate analysis report
def generate_analysis_report(news_df):
    if len(news_df) == 0:
        return "No news data available for analysis."
    
    # Calculate sentiment scores (1 for positive, 0 for negative)
    sentiment_scores = [1 if s == 'positive' else 0 for s in news_df['Sentiment']]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    # Count positive and negative news
    positive_count = sum(1 for s in sentiment_scores if s == 1)
    negative_count = len(sentiment_scores) - positive_count
    
    # Get most positive and negative news
    most_positive = news_df.loc[news_df['Confidence'].idxmax()] if positive_count > 0 else None
    most_negative = news_df.loc[news_df[news_df['Sentiment'] == 'negative']['Confidence'].idxmax()] if negative_count > 0 else None
    
    # Generate report
    report = f"## Market Sentiment Analysis Report\n\n"
    report += f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**Total News Analyzed:** {len(news_df)}\n\n"
    report += f"**Positive News:** {positive_count} ({positive_count/len(news_df)*100:.1f}%)\n\n"
    report += f"**Negative News:** {negative_count} ({negative_count/len(news_df)*100:.1f}%)\n\n"
    report += f"**Overall Sentiment Score:** {avg_sentiment:.2f}/1.00\n\n"
    
    # Add sentiment interpretation
    if avg_sentiment >= 0.7:
        report += "**Market Outlook:** 🟢 Strongly Positive\n\n"
    elif avg_sentiment >= 0.6:
        report += "**Market Outlook:** 🟡 Moderately Positive\n\n"
    elif avg_sentiment >= 0.4:
        report += "**Market Outlook:** ⚪ Neutral\n\n"
    elif avg_sentiment >= 0.3:
        report += "**Market Outlook:** 🟠 Moderately Negative\n\n"
    else:
        report += "**Market Outlook:** 🔴 Strongly Negative\n\n"
    
    # Add most positive news
    if most_positive is not None:
        report += "### Most Positive News\n\n"
        report += f"**Headline:** {most_positive['Title']}\n\n"
        report += f"**Description:** {most_positive['Description']}\n\n"
        report += f"**Confidence:** {most_positive['Confidence']:.2%}\n\n"
    
    # Add most negative news
    if most_negative is not None:
        report += "### Most Negative News\n\n"
        report += f"**Headline:** {most_negative['Title']}\n\n"
        report += f"**Description:** {most_negative['Description']}\n\n"
        report += f"**Confidence:** {most_negative['Confidence']:.2%}\n\n"
    
    # Add summary of positive news
    if positive_count > 0:
        report += "### Positive News Summary\n\n"
        positive_news = news_df[news_df['Sentiment'] == 'positive'].sort_values('Confidence', ascending=False)
        for i, (idx, row) in enumerate(positive_news.head(3).iterrows(), 1):
            report += f"{i}. {row['Title']} (Confidence: {row['Confidence']:.2%})\n\n"
    
    # Add summary of negative news
    if negative_count > 0:
        report += "### Negative News Summary\n\n"
        negative_news = news_df[news_df['Sentiment'] == 'negative'].sort_values('Confidence', ascending=False)
        for i, (idx, row) in enumerate(negative_news.head(3).iterrows(), 1):
            report += f"{i}. {row['Title']} (Confidence: {row['Confidence']:.2%})\n\n"
    
    return report

# Function to manually fetch and analyze news
def fetch_and_analyze_news():
    with st.spinner("Fetching and analyzing latest news..."):
        new_articles = fetch_news()
        
        if new_articles:
            # Convert to DataFrame
            new_df = pd.DataFrame(new_articles)
            
            # Update session state
            if len(st.session_state.news_data) == 0:
                st.session_state.news_data = new_df
            else:
                st.session_state.news_data = pd.concat([st.session_state.news_data, new_df], ignore_index=True)
            
            # Remove duplicates based on title
            st.session_state.news_data = st.session_state.news_data.drop_duplicates(subset=['Title'])
            
            # Calculate sentiment score (1 for positive, 0 for negative)
            sentiment_score = sum(1 if s == 'positive' else 0 for s in new_df['Sentiment']) / len(new_df)
            st.session_state.sentiment_history.append({
                'timestamp': datetime.now(),
                'score': sentiment_score,
                'positive_count': sum(1 for s in new_df['Sentiment'] if s == 'positive'),
                'negative_count': sum(1 for s in new_df['Sentiment'] if s == 'negative')
            })
            
            # Update last fetch time
            st.session_state.last_fetch_time = datetime.now()
            
            # Generate analysis report
            st.session_state.analysis_report = generate_analysis_report(st.session_state.news_data)
            
            st.success(f"Fetched and analyzed {len(new_df)} news articles!")
        else:
            st.error("No news articles were fetched. Please check your API key and connection.")

# Create sidebar
with st.sidebar:
    st.header("News Settings")
    
    # API key input
    api_key = st.text_input("Click Fetch Button:", type="password", value="API_KEY")
    
    # Manual fetch button
    if st.button("Fetch Latest News", use_container_width=True):
        fetch_and_analyze_news()
    
    # Auto-fetch toggle
    auto_fetch = st.toggle("Auto-fetch news every hour", value=False)
    
    st.header("About")
    st.info("""
    This sentiment analysis tool uses a Naive Bayes classifier trained on financial news data 
    to predict market sentiment from news headlines and articles.
    """)
    
    st.header("How to Use")
    st.write("""
    1. Enter your GNews API key
    2. Click 'Fetch Latest News' to get current news
    3. Enable auto-fetch to get news every hour
    4. View the sentiment analysis and trends
    """)
    
    # Show dataset info
    st.header("Dataset Info")
    st.write(f"Number of training samples: {len(data)}")
    st.write("Sentiment distribution:")
    sentiment_counts = data['Sentiment'].value_counts()
    st.write(sentiment_counts)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Latest News Analysis")
    
    # Display last fetch time
    if st.session_state.last_fetch_time:
        st.write(f"Last fetch: {st.session_state.last_fetch_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display news data if available
    if len(st.session_state.news_data) > 0:
        # Calculate current sentiment score
        sentiment_scores = [1 if s == 'positive' else 0 for s in st.session_state.news_data['Sentiment']]
        current_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Display sentiment score with appropriate styling
        sentiment_class = "positive-bg" if current_sentiment >= 0.5 else "negative-bg"
        if 0.4 <= current_sentiment < 0.6:
            sentiment_class = "neutral-bg"
            
        st.markdown(f'<div class="sentiment-score {sentiment_class}">Current Sentiment: {current_sentiment:.2f}/1.00</div>', unsafe_allow_html=True)
        
        # Display news in expandable sections
        positive_news = st.session_state.news_data[st.session_state.news_data['Sentiment'] == 'positive']
        negative_news = st.session_state.news_data[st.session_state.news_data['Sentiment'] == 'negative']
        
        with st.expander(f"Positive News ({len(positive_news)})", expanded=True):
            for _, news in positive_news.iterrows():
                st.markdown(f"**{news['Title']}**")
                st.markdown(f"*{news['Description']}*")
                st.markdown(f"Confidence: <span class='positive'>{news['Confidence']:.2%}</span>", unsafe_allow_html=True)
                st.markdown("---")
        
        with st.expander(f"Negative News ({len(negative_news)})"):
            for _, news in negative_news.iterrows():
                st.markdown(f"**{news['Title']}**")
                st.markdown(f"*{news['Description']}*")
                st.markdown(f"Confidence: <span class='negative'>{news['Confidence']:.2%}</span>", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("No news data available. Click 'Fetch Latest News' to get started.")

with col2:
    st.subheader("Sentiment Trends")
    
    # Display sentiment history chart if available
    if st.session_state.sentiment_history:
        # Prepare data for charting
        history_dates = [entry['timestamp'] for entry in st.session_state.sentiment_history]
        history_scores = [entry['score'] for entry in st.session_state.sentiment_history]
        
        # Create trend chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history_dates, history_scores, marker='o', linestyle='-', color='#1f77b4')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Neutral Line')
        ax.set_title('Sentiment Trend Over Time')
        ax.set_ylabel('Sentiment Score (0-1)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display summary statistics
        st.write("**Summary Statistics**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Average Sentiment", f"{np.mean(history_scores):.2f}")
            st.metric("Highest Sentiment", f"{max(history_scores):.2f}")
            
        with col_b:
            st.metric("Lowest Sentiment", f"{min(history_scores):.2f}")
            st.metric("Number of Analyses", len(history_scores))
    else:
        st.info("No trend data available yet. Fetch news to see trends.")

# Analysis report section
st.markdown("---")
st.subheader("Comprehensive Analysis Report")

if st.session_state.analysis_report:
    st.markdown(st.session_state.analysis_report)
else:
    st.info("No analysis report available yet. Fetch news to generate a report.")

# Manual news input section
st.markdown("---")
st.subheader("Manual News Analysis")

news_input = st.text_area(
    "Enter stock news headline or article to analyze:",
    height=100,
    placeholder="e.g., Ola Electric shares jump 15% in two sessions on PLI certification for its Gen 3"
)

if st.button("Analyze Manual Input"):
    if news_input:
        sentiment, confidence = analyze_sentiment(news_input)
        
        if sentiment:
            st.subheader("Analysis Result")
            
            if sentiment == 'positive':
                st.markdown(f'**Sentiment**: <span class="positive">Positive 📈</span>', unsafe_allow_html=True)
                st.markdown(f'**Confidence**: <span class="positive">{confidence:.2%}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'**Sentiment**: <span class="negative">Negative 📉</span>', unsafe_allow_html=True)
                st.markdown(f'**Confidence**: <span class="negative">{confidence:.2%}</span>', unsafe_allow_html=True)
        else:
            st.error("Could not analyze the text. Please try with different content.")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>This application automatically fetches and analyzes stock news using the GNews API.</p>
        <p>The model is trained on financial news data using a Naive Bayes classifier.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Auto-fetch functionality
if auto_fetch:
    # This is a simplified approach for Streamlit
    # In a production environment, you would use a proper scheduler
    current_time = datetime.now()
    if (st.session_state.last_fetch_time is None or 
        (current_time - st.session_state.last_fetch_time).seconds >= 3600):
        fetch_and_analyze_news()
        st.rerun()