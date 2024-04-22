import json
import numpy as np
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def app_go():
    return "Twitter Sentiment Analysis"

MODEL_PATH = "serving_model_dir/tweet-detection-model/1710973572"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json
    tweet = request_json.get("data")
    label = model.predict([tweet])
    prediction = tf.argmax(label[0]).numpy()
    class_labels = ['Negative', 'Positive']

    response_json = {
        "tweet": tweet,
        "label": class_labels[prediction],
    }

    return json.dumps(response_json)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)