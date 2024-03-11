from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

#Load ML model
model = load_model('sentiment_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)