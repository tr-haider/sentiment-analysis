import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import en_core_web_sm
nlp = en_core_web_sm.load()
# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

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
    sentiment_classifier = pipeline(task="sentiment-analysis")
    preds = sentiment_classifier(text)
    preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
    return preds
def sentiment_run():
    st.title("Text Analysis Tool")

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
