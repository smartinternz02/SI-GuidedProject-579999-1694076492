import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    output = prediction[0]
    if output == 1:
        launch="Successful"
    else:
        launch="Unsuccessful"

    return render_template("index.html", prediction_text = "Launch {}".format(launch))

if __name__ == "__main__":
    flask_app.run(debug=True)