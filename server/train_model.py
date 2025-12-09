import lightgbm as lgb
import numpy as np
import scipy.sparse as sparse
import os
import joblib
from pathlib import Path
from google.colab import drive
import time
from datasets import load_dataset

# --- Configuration Constants ---

PROJECT_DIR = Path("/content/drive/MyDrive/malware_project")
MODEL_PATH = PROJECT_DIR / "lgbm_final_model.pkl"
CHECKPOINT_DIR = PROJECT_DIR / "lgbm_checkpoints"

HF_DATASET = "cw1521/ember2018-malware"
CONFIG_NAME = None
SPLIT = "train"
MAX_SAMPLES = 500000
VALIDATION_SPLIT_RATIO = 0.05

N_ESTIMATORS = 300
EARLY_STOPPING_ROUNDS = 10
CHECKPOINT_INTERVAL = 50

IRRELEVANT_FEATURE_INDICES = list(range(10, 245))
EXPECTED_DROPS = 235
ORIGINAL_DIMENSION = 2381


# --- Utility Functions ---

def get_feature_mask(original_dimension=ORIGINAL_DIMENSION, irrelevant_indices=IRRELEVANT_FEATURE_INDICES):
    mask = np.ones(original_dimension, dtype=bool)
    max_idx = max(irrelevant_indices)
    if max_idx >= original_dimension:
        raise ValueError(f"Irrelevant index {max_idx} out of bounds.")
    mask[irrelevant_indices] = False

    final_dimension = np.sum(mask)
    expected_final = original_dimension - EXPECTED_DROPS
    if final_dimension != expected_final:
        raise ValueError(f"Expected {expected_final} features, got {final_dimension}.")
    return mask


FEATURE_MASK = get_feature_mask()
FINAL_FEATURE_DIMENSION = np.sum(FEATURE_MASK)


def preprocess_sample(sample, feature_mask):
    label = sample.get('y', -1)
    if label in [0, 1]:
        full_feature_vector = sample.get('x', None)
        if full_feature_vector is None:
            return None, None
        reduced_features = np.array(full_feature_vector, dtype=np.float32)[feature_mask]
        return reduced_features, label
    return None, None


def load_and_preprocess_data(split_name, max_samples, feature_mask):
    print(f"\n🚀 Streaming EMBER2018 dataset from Hugging Face ({HF_DATASET}, split: {split_name})...")
    dataset = load_dataset(HF_DATASET, split=split_name, streaming=True)
    dataset = dataset.shuffle(seed=42)

    dataset_iterator = iter(dataset)
    sparse_features_list, labels_list = [], []
    count = 0

    while count < max_samples:
        try:
            sample = next(dataset_iterator)
        except StopIteration:
            print("End of dataset reached.")
            break

        features, label = preprocess_sample(sample, feature_mask)
        if features is not None and label is not None:
            sparse_features_list.append(sparse.csr_matrix(features))
            labels_list.append(label)
            count += 1
            if count % 10000 == 0:
                print(f"Loaded {count}/{max_samples} samples...")

    print(f"✅ Loaded {count} samples, feature dim {FINAL_FEATURE_DIMENSION}.")
    sparse_features = sparse.vstack(sparse_features_list, format='csr')
    labels_array = np.array(labels_list, dtype=np.int32)
    return sparse_features, labels_array, count


def prepare_data_splits(sparse_features, labels_array, total_count):
    val_size = int(total_count * VALIDATION_SPLIT_RATIO)
    np.random.seed(42)
    indices = np.random.permutation(total_count)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_features = sparse_features[train_indices, :]
    val_features = sparse_features[val_indices, :]
    train_labels = labels_array[train_indices]
    val_labels = labels_array[val_indices]

    train_data = lgb.Dataset(train_features, train_labels)
    val_data = lgb.Dataset(val_features, val_labels, reference=train_data)
    return train_data, val_data, total_count - val_size, val_size


def train_lightgbm_model():
    # --- 1. Drive Setup ---
    print("--- 1. Mounting Google Drive ---")
    if not os.path.ismount("/content/drive"):
        drive.mount('/content/drive', force_remount=True)
        print("✅ Drive mounted successfully.")
    else:
        print("Drive already mounted.")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. Checkpoint Resume Setup ---
    print("\n--- 2. Checking for Checkpoints ---")
    checkpoints = sorted(CHECKPOINT_DIR.glob('lgbm_checkpoint_*.txt'))
    initial_model_path = None
    start_iteration = 0

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        initial_model_path = str(latest_checkpoint)
        print(f"✅ Found checkpoint {latest_checkpoint.name}. Resuming training.")
    else:
        print("No checkpoint found. Starting from scratch.")

    # --- 3. Data Loading ---
    print("\n--- 3. Loading Data ---")
    sparse_features, labels_array, total_samples = load_and_preprocess_data(SPLIT, MAX_SAMPLES, FEATURE_MASK)
    train_data, val_data, train_size, val_size = prepare_data_splits(sparse_features, labels_array, total_samples)

    # --- 4. Training Parameters ---
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 512,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'is_sparse': True,
    }

    # --- 5. Training with Checkpointing ---
    print("\n--- 4. Starting LightGBM Training ---")

    def checkpoint_callback(env):
        if env.iteration > 0 and env.iteration % CHECKPOINT_INTERVAL == 0:
            ckpt_path = CHECKPOINT_DIR / f"lgbm_checkpoint_{env.iteration:04d}.txt"
            try:
                env.model.save_model(str(ckpt_path))
                print(f"💾 Saved checkpoint at iteration {env.iteration}: {ckpt_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")

    start_time = time.time()

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        checkpoint_callback
    ]

    bst = lgb.train(
        params,
        train_data,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[val_data],
        callbacks=callbacks,
        init_model=initial_model_path,
        keep_training_booster=True
    )

    total_duration = time.time() - start_time
    best_score = bst.best_score['valid_0']['auc']
    best_iter = bst.best_iteration

    print("\n--- 5. Training Complete ---")
    print(f"✅ Completed in {total_duration:.2f}s, best AUC = {best_score:.4f} at iteration {best_iter}")

    # Save final model in both formats
    try:
        joblib.dump(bst, MODEL_PATH)
        bst.save_model(str(MODEL_PATH.with_suffix('.txt')))
        print(f"✅ Final model saved to {MODEL_PATH} and {MODEL_PATH.with_suffix('.txt')}")
    except Exception as e:
        print(f"❌ Could not save model: {e}")


if __name__ == "__main__":
    train_lightgbm_model()