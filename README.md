# 🔐 Phishing Trust Score Checker

This is a machine learning-based web app that checks the **trustworthiness of URLs** and predicts a **Phishing Trust Score** between `0` (highly suspicious) and `1` (very safe).

## 🚀 Features

- 🧠 Machine learning model trained on phishing detection data
- 🌐 Enter any URL and get a Trust Score
- ✅ Real-time feature extraction
- 📊 Visual feedback and explanation of risky features

## 💡 How It Works

1. You enter a URL.
2. The app extracts up to 48 features (e.g., IP address usage, domain age, redirection).
3. A trained model (e.g., XGBoost) predicts the trust score.
4. The result is shown with explanations and a color-coded risk indicator.

## 🧪 Technologies Used

- Python + Flask
- Scikit-learn / XGBoost
- HTML/CSS/JS (Vanilla)


## 🛠️ Setup Instructions

```bash
git clone https://github.com/homeshwari524/phishing-trust-score.git
cd phishing-trust-score
pip install -r requirements.txt
python app.py
