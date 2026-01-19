from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        temp = float(request.form["temperature"])
        rain = float(request.form["rainfall"])
        hum = float(request.form["humidity"])
        air = float(request.form["air_quality"])

        result = model.predict([[temp, rain, hum, air]])[0]

        if result == 0:
            prediction = "Low Climate Risk"
        elif result == 1:
            prediction = "Moderate Climate Risk"
        else:
            prediction = "High Climate Risk"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
