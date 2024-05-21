import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import en_core_web_sm
# Specify the NLTK data directory
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# Load spaCy model
nlp = en_core_web_sm.load()

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()
# Function to extract stop words
def extract_stop_words(text):
    english_stopwords = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stop_words = [w for w in word_tokens if w.lower() in english_stopwords]
    return stop_words

# Function to perform Named Entity Recognition (NER)
def perform_ner(text):
    article = nlp(text);
    results = []
    for x in article.ents:
       results.append(f'Text : {x.text}')
       results.append(f'Label : {x.label_}')
    return results

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    scores = sid.polarity_scores(text)
    total = scores['pos'] + scores['neg'] + scores['neu']
    proportions = {
        'positive': round((scores['pos'] / total) * 100, 2),
        'negative': round((scores['neg'] / total) * 100, 2),
        'neutral': round((scores['neu'] / total) * 100, 2)
    }
    return proportions
def sentiment_run():
    st.title("Sentiment Analysis & NER Tool")

    # Input text area
    text = st.text_area("Enter text:")
    if st.button("Analyze"):
      # Extract stop words
      st.subheader("Stop words")
      stop_words = extract_stop_words(text)
      st.write(stop_words)

      # Perform NER
      st.subheader("Named Entity Recognition (NER)")
      ner_results = perform_ner(text)
      st.write(ner_results)

      # Perform sentiment analysis
      st.subheader("Sentiment Analysis")
      sentiment_results = perform_sentiment_analysis(text)
      st.write(sentiment_results)
sentiment_run()
