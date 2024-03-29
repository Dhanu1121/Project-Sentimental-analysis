from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

app = Flask(__name__)

modelpath = './model/sentiment_model.h5'

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


def predict_sentiment(text):
    # Preprocess the input text
    sequence = text_to_word_sequence(text)
    sequence = tokenizer.texts_to_sequences([sequence])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Make prediction
    prediction = model.predict(padded_sequence)
    sentiment = ['Negative', 'Positive', 'Neutral'][prediction.argmax()]
    
    return sentiment


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return render_template('index.html', prediction=sentiment)


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    file = request.files['file']
    if file:
        df = pd.read_csv(file)
        sentiments = [predict_sentiment(text) for text in df['review']]
        positive_count = sentiments.count('Positive')
        negative_count = sentiments.count('Negative')
        neutral_count = sentiments.count('Neutral')
        return render_template('csv_result.html', positive_count=positive_count, negative_count=negative_count, neutral_count=neutral_count)
    else:
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
