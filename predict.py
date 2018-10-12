'''
Created on Oct 12, 2018

@author: earass
'''
from flask import Flask, render_template, request
from keras.models import load_model
from numpy import argmax
import pickle
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model('cls_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tkzr = pickle.load(handle)

categories = ['Environment', 'Sports', 'Technology']

def get_prediction(text):
    encoded = tkzr.texts_to_sequences([text])
    padded_docs = pad_sequences(encoded, maxlen=60, padding='post')
    prediction = model.predict(padded_docs, verbose=1)
    prediction = categories[argmax(prediction)]
    return prediction

@app.route("/")
def index():
    prediction = None
    text = request.args.get('text')
    if text:
        prediction = get_prediction(text)
    return render_template("index.html", data = prediction)

if __name__ == '__main__':
    app.run()
