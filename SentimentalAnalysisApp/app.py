from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import pickle

app = Flask(__name__)

modelpath = './model/best_modelLSTM1.h5'

# Load ML model
model = load_model(modelpath)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Tokenization using TweetTokenizer
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

    return ' '.join(tokens)

def predict_sentiment_csv(text):
    # Preprocess the input text
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200)

    # Make prediction
    prediction = model.predict(padded_sequence)
    sentiment = ['Negative', 'Positive', 'Neutral'][prediction.argmax()]
    
    return sentiment

def predict_sentiment_text(text):
    # Preprocess the input text
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200)

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
