from flask import Flask, render_template, request, jsonify
import os
import re
import joblib

app = Flask(__name__)

MODEL_PATH = "models/phishshift_model.pkl"
scan_history = []

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


def analyze_message(message):
    warnings = []
    patterns = [
        (r"\burgent\b|\bimmediately\b|\bact now\b", "Urgent language"),
        (r"\bpassword\b|\botp\b|\bpin\b", "Sensitive information request"),
        (r"https?://\S+|www\.\S+", "Web link detected"),
        (r"\bwon\b|\bprize\b|\bclaim now\b", "Prize or reward"),
        (r"\bblocked\b|\bsuspended\b", "Threatening language"),
    ]

    for pattern, warning in patterns:
        if re.search(pattern, message, re.IGNORECASE):
            warnings.append(warning)

    return warnings


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    prediction = str(model.predict([message])[0]).lower()
    warnings = analyze_message(message)
    risk_score = min(100, len(warnings) * 20)

    if prediction == "phishing":
        risk_level = "High Risk"
    elif warnings:
        risk_level = "Caution"
    else:
        risk_level = "Low Risk"

    result = {
        "prediction": prediction,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "warnings": warnings,
    }

    scan_history.insert(0, result)
    return jsonify(result)


@app.route("/api/stats")
def stats():
    total = len(scan_history)
    phishing_count = sum(
        1 for item in scan_history
        if item["prediction"] == "phishing"
    )

    return jsonify({
        "total_scans": total,
        "phishing_count": phishing_count,
        "legitimate_count": total - phishing_count,
        "recent_scans": scan_history[:10],
    })


if __name__ == "__main__":
    app.run(debug=True)
