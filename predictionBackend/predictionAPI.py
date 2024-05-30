import os
import sys

CURRENT_DIR = os.getcwd()
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(SRC_DIR)

from flask import Flask, request
import src.lstm_dataprep as lstm_dataprep
import src.lstm as lstm
import src.smoothing_pipeline as smoothing_pipeline

app = Flask(__name__)


@app.route("/predict_lstm", methods=['POST'])
def predict_lstm():
    data = request.json
    if data["trigger"]:
        test_preds = lstm.pipeline()

        # burada kayıtların database'e yazılması gerekecek.

    return {"prediction": "Prediction Triggered"}

@app.route("/predict_smoothing", methods=['POST'])
def predict_smoothing():
    data = request.json
    if data["trigger"]:
        test_preds = smoothing_pipeline.pipeline()

        # burada kayıtların database'e yazılması gerekecek.

    return {"prediction": "Prediction Triggered"}


if __name__ == '__main__':
    app.run(port=5001, debug=False)
