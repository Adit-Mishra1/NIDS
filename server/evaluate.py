import lightgbm as lgb
import numpy as np
import scipy.sparse as sparse
import os
import joblib
from pathlib import Path
from google.colab import drive
from datasets import load_dataset
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import time

# --- Configuration Constants (Must match training script) ---

# Model and Data Paths
PROJECT_DIR = Path("/content/drive/MyDrive/malware_project")
MODEL_PATH = PROJECT_DIR / "lgbm_final_model.pkl"

# EMBER Data and Feature Configuration
HF_DATASET = "cw1521/ember2018-malware"
SPLIT = "test"  # Use 'test' split for independent evaluation

# NOTE: Since the test set is large, we load all *labeled* samples for evaluation.
# We do not need a MAX_SAMPLES limit here unless memory is still an issue.
# We will filter out unlabeled samples (label -1).
# We assume the entire labeled test set can be loaded into memory for evaluation.

# Feature Filtering (Match training script exactly)
IRRELEVANT_FEATURE_INDICES = list(range(10, 245))
EXPECTED_DROPS = 235
ORIGINAL_DIMENSION = 2381

# --- Utility Functions (Copied for self-containment) ---

def get_feature_mask(original_dimension=ORIGINAL_DIMENSION, irrelevant_indices=IRRELEVANT_FEATURE_INDICES):
    """Creates a boolean mask for feature selection."""
    mask = np.ones(original_dimension, dtype=bool)
    if irrelevant_indices:
        mask[irrelevant_indices] = False
    return mask

FEATURE_MASK = get_feature_mask()
FINAL_FEATURE_DIMENSION = np.sum(FEATURE_MASK)


def preprocess_sample(sample, feature_mask):
    """
    Applies feature masking and extracts label, only keeping labeled samples (0 or 1).
    """
    label = sample.get('y', -1)

    if label in [0, 1]:
        full_feature_vector = sample.get('x', None)
        if full_feature_vector is None:
            return None, None

        reduced_features = np.array(full_feature_vector, dtype=np.float32)[feature_mask]
        return reduced_features, label

    return None, None


def load_test_data(split_name, feature_mask):
    """
    Streams and preprocesses the EMBER test dataset, filtering for labeled samples.
    """
    print(f"\n🚀 Streaming EMBER2018 test dataset from Hugging Face ({HF_DATASET}, split: {split_name})...")

    dataset = load_dataset(HF_DATASET, split=split_name, streaming=True)
    dataset_iterator = iter(dataset)

    sparse_features_list = []
    labels_list = []
    count = 0

    # We iterate until StopIteration (end of split)
    while True:
        try:
            sample = next(dataset_iterator)
        except StopIteration:
            break

        features, label = preprocess_sample(sample, feature_mask)

        if features is not None and label is not None:
            # Convert to sparse immediately to save memory
            sparse_features_list.append(sparse.csr_matrix(features))
            labels_list.append(label)
            count += 1

            if count % 100000 == 0:
                 print(f"Loaded {count} labeled test samples...")

    print(f"✅ Finished loading. Total labeled test samples loaded: {count}. Final feature dimension: {FINAL_FEATURE_DIMENSION}")

    if not sparse_features_list:
        raise RuntimeError("No labeled samples were loaded from the test split. Check dataset split name or labels.")

    print("📦 Combining sparse feature rows...")
    sparse_features = sparse.vstack(sparse_features_list, format='csr')
    labels_array = np.array(labels_list, dtype=np.float32)

    print(f"Final test data shape: {sparse_features.shape}")

    return sparse_features, labels_array


def evaluate_model():
    """Main function to load the model, load test data, and compute metrics."""

    # --- 1. Drive Mounting and Model Loading ---
    print("--- 1. Setup and Model Loading ---")
    try:
        drive.mount('/content/drive', force_remount=True)
        print("✅ Drive mounted successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not mount Google Drive. Evaluation aborted.")
        print(f"Error details: {e}")
        return

    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Final model not found at {MODEL_PATH}. Run training first.")
        return

    try:
        bst = joblib.load(MODEL_PATH)
        print(f"✅ LightGBM model loaded successfully from: {MODEL_PATH.name}")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load model. Error: {e}")
        return

    # --- 2. Data Loading and Prediction ---
    print("\n--- 2. Test Data Loading and Prediction ---")

    # Load the raw test data
    test_features, test_labels = load_test_data(SPLIT, FEATURE_MASK)

    print("Performing predictions...")
    start_time = time.time()

    # Predict raw probabilities (needed for AUC)
    y_pred_proba = bst.predict(test_features)

    # Convert probabilities to binary class labels (using 0.5 threshold)
    y_pred = (y_pred_proba > 0.5).astype(int)

    end_time = time.time()
    print(f"Predictions completed in {end_time - start_time:.2f} seconds.")


    # --- 3. Metric Calculation ---
    print("\n--- 3. Evaluation Metrics ---")

    # AUC (Area Under the ROC Curve)
    auc = roc_auc_score(test_labels, y_pred_proba)

    # Standard Classification Metrics (require binary predictions)
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred)

    print(f"Total Test Samples Evaluated: {len(test_labels)}")
    print("-" * 40)
    print(f"1. AUC Score:       {auc:.4f}")
    print(f"2. F1 Score:        {f1:.4f}")
    print(f"3. Accuracy:        {accuracy:.4f}")
    print(f"4. Precision:       {precision:.4f}")
    print(f"5. Recall (TPR):    {recall:.4f}")
    print("-" * 40)

    # Print Confusion Matrix in an easy-to-read format
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix (True Labels vs. Predicted Labels):")
    print(f"[[{tn}, {fp}]")
    print(f" [{fn}, {tp}]]")
    print("\nInterpretation:")
    print(f"  True Positives (TP): {tp} (Correctly predicted malware)")
    print(f"  True Negatives (TN): {tn} (Correctly predicted benign)")
    print(f"  False Positives (FP): {fp} (Benign classified as malware)")
    print(f"  False Negatives (FN): {fn} (Malware classified as benign)")


if __name__ == "__main__":
    evaluate_model()