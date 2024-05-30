import os
import sys

CURRENT_DIR = os.getcwd()
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(SRC_DIR)
print(CURRENT_DIR)
print(sys.path)

from flask import Flask, request
import src.lstm_dataprep as lstm_dataprep
import src.lstm as lstm

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    if data["trigger"]:
        test_preds = lstm.pipeline()

        # burada kayıtların database'e yazılması gerekecek.

    return {"prediction": "Prediction Triggered"}


if __name__ == '__main__':
    app.run(port=5001, debug=True)
