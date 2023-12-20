import pickle

from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd

model_file = 'model_RF-ADASYN.bin'


with open(model_file, 'rb') as f_in_model:
    dv, model = pickle.load(f_in_model)


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    X = dv.transform([patient])
    y_pred = model.predict(X)
    prediction = y_pred[0]

    result = {
        "Status": prediction
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
