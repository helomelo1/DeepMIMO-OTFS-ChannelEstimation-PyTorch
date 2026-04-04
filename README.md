# DeepMIMO-OTFS Channel Estimation (Paper Implementation)

TensorFlow implementation of PositionNet from:

**"Deep Learning-based Channel Estimation for Massive MIMO-OTFS Communication Systems"**  
Payami & Blostein (WTS 2024)

This project includes:
- Synthetic data generation (EVA channel model)
- PositionNet model definition
- Training + fine-tuning
- Evaluation with support/top-k/cosine/F1 metrics

## Project Files

- `generate_data.py` - creates synthetic dataset
- `model.py` - PositionNet architecture + cosine similarity loss
- `train.py` - training and fine-tuning pipeline
- `eval.py` - standalone evaluation

## Requirements

- Python 3.10+
- TensorFlow 2.15+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Generate dataset:

```bash
python generate_data.py
```

2. Train model:

```bash
python train.py
```

3. Run evaluation:

```bash
python eval.py
```

## Important Note

Please make sure dataset and checkpoint paths are consistent across scripts before running.

Default values currently differ in some files (for example, `dataset_tf_3.npz` vs `dataset_tf.npz`, and different checkpoint folders). If needed, set the same values for:
- `DATASET_PATH`
- `CHECKPOINT_DIR`

## Output

After training/evaluation, you will get metrics such as:
- Support accuracy
- Top-k accuracy
- Cosine similarity
- Precision / Recall / F1

## Citation

If you use this implementation, please cite the original paper.
