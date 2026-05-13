from flask import Flask, jsonify, request
from flask_cors import CORS

import json
import joblib
import os
import random

app = Flask(__name__)
CORS(app)

# ✅ BASE DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ LOAD MODEL
model_path = os.path.join(
    BASE_DIR,
    "flood_model.pkl"
)

model = joblib.load(model_path)

# ✅ LOAD GEOJSON
geojson_path = os.path.join(
    BASE_DIR,
    "Wards.geojson"
)

with open(geojson_path) as f:

    geojson_data = json.load(f)

    wards = geojson_data["features"]


@app.route("/get-risk")
def get_risk():

    rainfall = float(
        request.args.get("rainfall", 250)
    )

    results = []

    for ward in wards:

        properties = ward["properties"]

        ward_no = properties["Ward_No"]

        # ✅ RANDOM DEMO VALUES
        elevation = round(
            random.uniform(1, 10),
            2
        )

        water_distance = round(
            random.uniform(0.1, 3),
            2
        )

        # ✅ MODEL INPUT
        features = [[
            rainfall,
            elevation,
            water_distance
        ]]

        score = model.predict(features)[0]

        # ✅ RISK LEVEL
        if score >= 150:

            level = "HIGH"

            reason = (
                "Low elevation and close to water"
            )

        elif score >= 100:

            level = "MEDIUM"

            reason = (
                "Moderate flood conditions"
            )

        else:

            level = "LOW"

            reason = (
                "Safer flood conditions"
            )

        results.append({

            "ward_no": ward_no,

            "ward_name":
            f"Ward {ward_no}",

            "rainfall": rainfall,

            "elevation": elevation,

            "water_distance":
            water_distance,

            "score":
            round(float(score), 2),

            "level": level,

            "reason": reason

        })

    return jsonify(results)


@app.route("/")
def home():

    return jsonify({
        "message":
        "FloodSense Backend Running"
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )