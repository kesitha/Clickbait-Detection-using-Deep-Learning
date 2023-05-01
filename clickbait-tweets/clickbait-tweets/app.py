from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
# from model import model


# Load the saved model
import tensorflow as tf
model = tf.keras.models.load_model('model')

# Load the saved weights
model.load_weights('weights.h5')

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

maxlen = 500

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    token_text = pad_sequences(tokenizer.texts_to_sequences([tweet]), maxlen=maxlen)
    
    pred = model.predict(token_text)
    print(pred[0][0])
    pred = np.round(pred[0][0])    
    # result = 'This tweet is clickbait' if pred == 1.0 else 'This tweet is not clickbait'
    result = 'True' if pred == 1.0 else 'False'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
