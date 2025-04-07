
from flask import Flask, render_template, request
from predict import predict_trust_score

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        url = request.form.get("url")
        if url:
            # ✅ Add "http://" if the URL doesn't start with it
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "http://" + url

            # ✅ Get prediction from your ML model
            prediction = predict_trust_score(url)

            if "score" in prediction:
                score = prediction["score"]
                feedback = "Legitimate ✅" if score > 0.5 else "Phishing 🚨"

                # ✅ Include features in the result
                result = {
                    "url": url,
                    "score": round(score, 3),
                    "feedback": feedback,
                    "color": "green" if score > 0.5 else "red",
                    "features": prediction["features"]  # ✅ add this line
                }
            else:
                error = prediction.get("error", "Unknown error occurred.")

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
