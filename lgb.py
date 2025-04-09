from lightgbm import LGBMRegressor
from joblib import dump
from preprocess import load_and_preprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# Load data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess("dataset/phishing.csv")

# Initialize and train the LightGBM model
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Print evaluation results
print("LightGBM Results:")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

# Save model and scaler
os.makedirs("models", exist_ok=True)
dump(model, "models/model_lgb.pkl")
dump(scaler, "models/scaler_lgb.pkl")
print("✅ LightGBM Model and Scaler saved to models/")
