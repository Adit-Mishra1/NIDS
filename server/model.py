# model.py — FINAL, STABLE VERSION (LightGBM + Scaler + Mask)

import os
import joblib
import numpy as np

MODEL_FILE_NAME = "lgbm_full_and_final_model.pkl"
SCALER_FILE_NAME = "scaler_final.pkl"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILE_NAME)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"LightGBM model file '{MODEL_FILE_NAME}' not found at: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file '{SCALER_FILE_NAME}' not found at: {SCALER_PATH}")

print(f"✅ Found Model at: {MODEL_PATH}")
print(f"✅ Found Scaler at: {SCALER_PATH}")

# Feature constants
ORIGINAL_DIMENSION = 2381
IRRELEVANT_FEATURE_INDICES = list(range(10, 245))
EXPECTED_DROPS = 235


def get_feature_mask():
    mask = np.ones(ORIGINAL_DIMENSION, dtype=bool)
    mask[IRRELEVANT_FEATURE_INDICES] = False
    final_dim = np.sum(mask)
    if final_dim != ORIGINAL_DIMENSION - EXPECTED_DROPS:
        raise ValueError("Feature mask dimension mismatch")
    return mask


FEATURE_MASK = get_feature_mask()
FINAL_FEATURE_DIMENSION = np.sum(FEATURE_MASK)  # Should be 2146


class NIDSModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def preprocess(self, full_vector):
        arr = np.asarray(full_vector, dtype=np.float32).ravel()

        # Pad or truncate
        if len(arr) < ORIGINAL_DIMENSION:
            padded = np.zeros(ORIGINAL_DIMENSION, dtype=np.float32)
            padded[:len(arr)] = arr
            arr = padded
        elif len(arr) > ORIGINAL_DIMENSION:
            arr = arr[:ORIGINAL_DIMENSION]

        # Apply mask → 2146 dims
        reduced_unscaled = arr[FEATURE_MASK]

        # Apply scaler
        reduced_scaled = self.scaler.transform(reduced_unscaled.reshape(1, -1)).ravel()

        return reduced_scaled

    def predict(self, feature_vector):
        if len(feature_vector) != FINAL_FEATURE_DIMENSION:
            feature_vector = self.preprocess(feature_vector)

        prob = float(self.model.predict(feature_vector.reshape(1, -1))[0])
        label = int(prob >= 0.5)

        return label, prob