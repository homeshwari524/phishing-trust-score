import numpy as np
from joblib import load
from preprocess import extract_features_from_url

# Load model and scaler
model = load("models/model_xgb.pkl")
scaler = load("models/scaler_xgb.pkl")

# Feature names — match with your 12 extracted features
feature_names = [
    "having_IP_Address",
    "URL_Length",
    "Shortening_Service",
    "Having_At_Symbol",
    "Double_slash_redirecting",
    "Prefix_Suffix",
    "SubDomains",
    "HTTPS_Token",
    "Domain_Age",
    "Web_Traffic",
    "Iframe",
    "Mouse_Over"
]

def predict_trust_score(url):
    try:
        features = extract_features_from_url(url)
        
        if features is None:
            return {"error": "Failed to extract features."}

        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        score = model.predict(scaled_features)[0]

        # ✅ Convert to dictionary
        feature_dict = dict(zip(feature_names, features))

        return {
            "score": float(score),
            "features": feature_dict
        }

    except Exception as e:
        return {"error": str(e)}
