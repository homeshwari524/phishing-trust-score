import joblib
import numpy as np
from preprocess import load_and_preprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_train, X_test, y_train, y_test, _ = load_and_preprocess("dataset/phishing.csv")

models = {
    "Random Forest": joblib.load("models/model_rf.pkl"),
    "Gradient Boosting": joblib.load("models/model_gb.pkl"),
    "XGBoost": joblib.load("models/model_xgb.pkl")
}

print("\nModel Comparison Table:")
print(f"{'Model':<20}{'MSE':<10}{'RMSE':<10}{'MAE':<10}{'R2 Score':<10}")
print("-"*60)

for name, model in models.items():
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<20}{mse:<10.4f}{rmse:<10.4f}{mae:<10.4f}{r2:<10.4f}")
    
