"""
Standalone evaluation script for PositionNet.
Loads the best saved checkpoint and runs the validation metrics.
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import build_position_net

DATASET_PATH = "dataset_tf.npz"
CHECKPOINT_DIR = "checkpoints_tf"
BATCH_SIZE = 64
TRAIN_SPLIT = 0.7
SEED = 42

def load_dataset(path):
    data = np.load(path)
    X, Y = data["X"], data["Y"]
    return X, Y

def split_data(X, Y):
    # Using the same seed guarantees we get the exact same validation set
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    split = int(len(X) * TRAIN_SPLIT)
    return X[idx[:split]], Y[idx[:split]], X[idx[split:]], Y[idx[split:]]

def topk_accuracy_np(y_true, y_pred):
    true_flat = y_true.reshape(len(y_true), -1)
    pred_flat = y_pred.reshape(len(y_pred), -1)
    
    accs = []
    for t, p in zip(true_flat, pred_flat):
        K_true = int(np.sum(t))
        if K_true == 0:
            continue
        top_indices = np.argpartition(-p, K_true)[:K_true]
        correct = np.sum(t[top_indices])
        accs.append(correct / K_true)
        
    return np.mean(accs) if accs else 0.0

def evaluate(model, X_val, Y_val):
    n = len(X_val)

    print("Running inference...")
    preds = []
    for i in tqdm(range(0, n, BATCH_SIZE), desc="Predict", ncols=80):
        preds.append(model(X_val[i:i + BATCH_SIZE], training=False).numpy())
    Y_pred = np.concatenate(preds, axis=0)

    pred_bin = (Y_pred >= 0.5).astype(np.float32)
    val_flat = Y_val.reshape(n, -1)
    pred_flat = Y_pred.reshape(n, -1)
    bin_flat = pred_bin.reshape(n, -1)

    tp = np.sum(val_flat * bin_flat, axis=1)
    sup_acc = np.mean(tp / (np.sum(val_flat, axis=1) + 1e-8))

    # Dynamic Top-K accuracy matched to actual non-zeros
    topk_acc = topk_accuracy_np(Y_val, Y_pred)

    cos_sim = np.mean(
        np.sum(val_flat * pred_flat, axis=1) /
        (np.linalg.norm(val_flat, axis=1) * np.linalg.norm(pred_flat, axis=1) + 1e-8)
    )

    bin_acc = np.mean(bin_flat == val_flat)
    tp_all = np.sum((bin_flat == 1) & (val_flat == 1))
    fp_all = np.sum((bin_flat == 1) & (val_flat == 0))
    fn_all = np.sum((bin_flat == 0) & (val_flat == 1))
    precision = tp_all / (tp_all + fp_all + 1e-8)
    recall = tp_all / (tp_all + fn_all + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n  Support accuracy (threshold=0.5): {sup_acc:.4f}")
    print(f"  Target Top-K accuracy:            {topk_acc:.4f}")
    print(f"  Cosine similarity:                {cos_sim:.4f}")
    print(f"  Per-bin accuracy:                 {bin_acc:.4f}")
    print(f"  Precision:                        {precision:.4f}")
    print(f"  Recall:                           {recall:.4f}")
    print(f"  F1 score:                         {f1:.4f}")
    print()

def main():
    tf.random.set_seed(SEED)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "position_net_best.weights.h5")

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    print("Loading dataset and splitting...")
    X, Y = load_dataset(DATASET_PATH)
    _, _, X_val, Y_val = split_data(X, Y)
    print(f"  Evaluating on {len(X_val)} validation samples.\n")

    print("Building model and loading weights...")
    # Build the model architecture
    model = build_position_net()
    
    # Load the best weights saved during Phase 1
    model.load_weights(ckpt_path)
    print("  ✓ Weights loaded successfully.")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    evaluate(model, X_val, Y_val)

if __name__ == "__main__":
    main()