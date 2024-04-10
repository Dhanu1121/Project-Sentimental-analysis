from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

modelpath = './model/model2.h5'

# Load ML model
model = load_model(modelpath)

# Load tokenizer
tokenizer = Tokenizer()
# Load the Twitter data
twitter_data = pd.read_csv('D:\Final Year Project\Project(latest)\SentimentalAnalysisMLModel\preprocessed_twitter_data.csv')['tweet'].tolist()

# Ensure all elements are strings
twitter_data = [str(tweet) for tweet in twitter_data]

# Fit tokenizer on the preprocessed Twitter data
tokenizer.fit_on_texts(twitter_data)

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove noisy text
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

def predict_sentiment_csv(text):
    # Preprocess the input text
    text = preprocess_text(text)
    sequence = text_to_word_sequence(text)
    sequence = tokenizer.texts_to_sequences([sequence])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Make prediction
    prediction = model.predict(padded_sequence)
    sentiment = ['Negative', 'Positive', 'Neutral'][prediction.argmax()]
    
    return sentiment

def predict_sentiment_text(text):
    # Preprocess the input text
    text = preprocess_text(text)
    sequence = text_to_word_sequence(text)
    sequence = tokenizer.texts_to_sequences([sequence])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Make prediction
    prediction = model.predict(padded_sequence)[0]  # Take the first prediction from the batch
    positive_prob = prediction[1]  # Probability of being positive
    negative_prob = prediction[0]  # Probability of being negative
    neutral_prob = prediction[2]   # Probability of being neutral

    # Calculate percentages
    total = positive_prob + negative_prob + neutral_prob
    positive_percentage = (positive_prob / total) * 100
    negative_percentage = (negative_prob / total) * 100
    neutral_percentage = (neutral_prob / total) * 100

    positive_percentage = "{:.2f}".format(positive_percentage)
    negative_percentage = "{:.2f}".format(negative_percentage)
    neutral_percentage = "{:.2f}".format(neutral_percentage)

    # Determine sentiment
    sentiment = ['Negative', 'Positive', 'Neutral'][prediction.argmax()]
    
    return sentiment, positive_percentage, negative_percentage, neutral_percentage

# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form['text']
    sentiment, positive_percentage, negative_percentage, neutral_percentage = predict_sentiment_text(text)
    return render_template('index.html', prediction=sentiment, positive_percentage=positive_percentage, negative_percentage=negative_percentage, neutral_percentage=neutral_percentage)

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        sentiments = [predict_sentiment_csv(text) for text in df['review']]
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        neutral_count = sentiments.count('Neutral')
        return render_template('csv_result.html', positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count)
    else:
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
