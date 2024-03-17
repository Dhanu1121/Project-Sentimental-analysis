from flask import Flask, render_template, request
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
twitter_data = pd.read_csv('./preprocessed_twitter_data.csv')['tweet'].tolist()

# Ensure all elements are strings
twitter_data = [str(tweet) for tweet in twitter_data]

# Fit tokenizer on the preprocessed Twitter data
tokenizer.fit_on_texts(twitter_data)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Convert text to string if it's not already a string
    if not isinstance(text, str):
        text = str(text)
    
    # Preprocess the input text
    sequence = text_to_word_sequence(text)
    sequence = tokenizer.texts_to_sequences([sequence])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Make prediction
    prediction = model.predict(padded_sequence)
    print(prediction.argmax())
    sentiment = ['Negative', 'Positive','Neutral'][prediction.argmax()]
    
    print(sentiment)
    
    return render_template('index.html', prediction=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
