"""
Training + evaluation pipeline for PositionNet.

Paper settings (Section IV):
    - Optimizer: AdamW, lr=1e-3
    - Loss: Cosine Similarity
    - Split: 70% train, 30% validation
    - Samples: 10^5
    - Fine-tuning lr: 1e-5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from model import build_position_net, cosine_similarity_loss

DATASET_PATH = "dataset_tf.npz"
CHECKPOINT_DIR = "checkpoints_tf_1"
BATCH_SIZE = 64
EPOCHS = 10
FINETUNE_EPOCHS = 10
LR = 1e-3
FINETUNE_LR = 1e-5
TRAIN_SPLIT = 0.7
SEED = 42
NQ = 6
NT = 16


def load_dataset(path):
    data = np.load(path)
    X, Y = data["X"], data["Y"]
    print(f"Loaded {path}: X={X.shape}, Y={Y.shape}")
    print(f"  Non-zero fraction: {Y.mean():.4f}")
    return X, Y


def split_data(X, Y):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    split = int(len(X) * TRAIN_SPLIT)
    return X[idx[:split]], Y[idx[:split]], X[idx[split:]], Y[idx[split:]]


def build_tf_dataset(X, Y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 10000), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def topk_accuracy_np(y_true, y_pred):
    """Evaluates percentage of true non-zero locations correctly ranked at the top.
    For each sample, dynamically finds the actual number of non-zeros (K_true),
    picks the top K_true predicted indices, and measures overlap.
    """
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


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = cosine_similarity_loss(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, pred


@tf.function
def val_step(model, x, y):
    pred = model(x, training=False)
    loss = cosine_similarity_loss(y, pred)
    return loss, pred


def run_epoch(model, optimizer, dataset, n_batches, training, desc):
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    pbar = tqdm(dataset, total=n_batches, desc=desc, ncols=100)
    for x, y in pbar:
        if training:
            loss, pred = train_step(model, optimizer, x, y)
        else:
            loss, pred = val_step(model, x, y)

        bs = x.shape[0]
        total_loss += loss.numpy() * bs
        total_acc += topk_accuracy_np(y.numpy(), pred.numpy()) * bs
        count += bs

        pbar.set_postfix(loss=f"{total_loss/count:.4f}", topk=f"{total_acc/count:.4f}")

    avg_loss = total_loss / count
    avg_acc = total_acc / count
    return avg_loss, avg_acc


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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    X, Y = load_dataset(DATASET_PATH)
    X_train, Y_train, X_val, Y_val = split_data(X, Y)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}\n")

    train_ds = build_tf_dataset(X_train, Y_train, shuffle=True)
    val_ds = build_tf_dataset(X_val, Y_val)
    n_train = int(np.ceil(len(X_train) / BATCH_SIZE))
    n_val = int(np.ceil(len(X_val) / BATCH_SIZE))

    model = build_position_net()
    model.summary()

    optimizer = keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "position_net_best.weights.h5")
    best_val_loss = float("inf")

    # Phase 1: Main training (lr=1e-3)
    print(f"\n── Phase 1: Training (lr={LR}) ──\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, optimizer, train_ds, n_train, True,
                                    f"Train {epoch:02d}/{EPOCHS}")
        va_loss, va_acc = run_epoch(model, optimizer, val_ds, n_val, False,
                                    f"Val   {epoch:02d}/{EPOCHS}")
        saved = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            model.save_weights(ckpt_path)
            saved = " ✓ saved"
        print(f"Epoch {epoch:02d} | train loss={tr_loss:.4f} topk={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} topk={va_acc:.4f}{saved}\n")

    # Phase 2: Fine-tuning (lr=1e-5)
    print(f"\n── Phase 2: Fine-tuning (lr={FINETUNE_LR}) ──\n")
    model.load_weights(ckpt_path)
    optimizer.learning_rate.assign(FINETUNE_LR)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, optimizer, train_ds, n_train, True,
                                    f"FT Train {epoch:02d}/{FINETUNE_EPOCHS}")
        va_loss, va_acc = run_epoch(model, optimizer, val_ds, n_val, False,
                                    f"FT Val   {epoch:02d}/{FINETUNE_EPOCHS}")
        saved = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            model.save_weights(ckpt_path)
            saved = " ✓ saved"
        print(f"FT Epoch {epoch:02d} | train loss={tr_loss:.4f} topk={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} topk={va_acc:.4f}{saved}\n")

    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    model.load_weights(ckpt_path)
    evaluate(model, X_val, Y_val)


if __name__ == "__main__":
    main()
