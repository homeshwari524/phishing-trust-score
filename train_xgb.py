from xgboost import XGBRegressor
from joblib import dump
from preprocess import load_and_preprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

X_train, X_test, y_train, y_test, scaler = load_and_preprocess("dataset/phishing.csv")

model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)


print("XGBoost Results:")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

os.makedirs("models", exist_ok=True)
dump(model, "models/model_xgb.pkl")
dump(scaler, "models/scaler_xgb.pkl")
print("✅ Model and Scaler saved to models/")