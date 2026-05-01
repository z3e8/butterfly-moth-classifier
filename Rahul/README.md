# CNN Pipeline — Rahul Kishore (Final Submission, 96.8% Public LB)

This folder contains the full convolutional neural network (CNN) pipeline used
for our final Kaggle submission. The single script `cnn_butterfly.py` trains a
from-scratch ResNet-18-width network with SE attention, MixUp/CutMix, EMA
weights, and TTA + checkpoint ensembling at inference.

## Files

| File | Purpose |
|---|---|
| `cnn_butterfly.py` | Single-file training + inference script |
| `submission.csv` | Final 96.8% submission (columns `ID,TARGET`) |
| `training_curves.png` | Loss + accuracy per epoch over the 80-epoch run |
| `confusion_matrix.png` | Top-20 most-confused classes from the validation set |
| `README.md` | This file |

## Results

| Metric | Value |
|---|---|
| Train accuracy (final) | ~0.80 (on MixUp-augmented stream; un-mixed train acc is higher) |
| Validation accuracy (live weights, best) | **0.9444** (epoch 74) |
| Validation accuracy (EMA weights, best) | **0.9450** (epoch 71) |
| Kaggle public leaderboard | **0.968** |
| Parameters | 11.31 M |
| Train time | ~3 hours on a single T4 GPU |

The validation set is a 15% stratified hold-out of the provided training data
(N = 1,890 images). Public LB landing above val is consistent with the
regularization (MixUp, label smoothing, EMA, RandomErasing) doing real work
rather than overfitting the val split.

## Method (high level)

- **Architecture**: ResNet-18-width with Squeeze-and-Excite blocks. Four stages
  of 2 BasicBlocks each (channels 64-128-256-512). 3×3 stem (no 7×7 since
  inputs are only 160 px). AdaptiveAvgPool → Dropout(0.3) → Linear(512, 100).
- **Augmentation**: Random resized crop (scale 0.7–1.0), horizontal flip,
  rotation ±15°, color jitter (brightness/contrast/saturation 0.25, hue 0.05),
  RandomErasing (p = 0.25). MixUp (α = 0.2) and CutMix (α = 1.0) applied to
  50% of batches.
- **Training**: AdamW (peak LR 3e-3, WD 5e-4), cosine annealing with 5-epoch
  linear warmup, cross-entropy with label smoothing 0.1, gradient clipping
  at 5.0, mixed-precision (AMP) on CUDA, EMA of weights with decay 0.999,
  80 epochs, batch size 64, image size 160 px.
- **Inference**: TTA (logits averaged over original + horizontal flip), then
  ensemble of best-val checkpoint and EMA checkpoint by averaging softmax
  probabilities, then argmax.

## How to run

### Requirements

- Python ≥ 3.9
- CUDA GPU strongly recommended (T4 used; AMP enabled)
- Python packages (all standard, all on Colab by default):
  - `torch >= 2.0`
  - `numpy`
  - `pandas`
  - `pillow`
  - `scikit-learn`
  - `matplotlib`

CPU-only will work but takes ~10× longer.

### Data layout

Place the script next to the competition data:

```
project_root/
├── cnn_butterfly.py
├── train.csv
├── sample_submission.csv
├── train_images/
│   └── train_images/        ← yes, the nested folder is correct
│       ├── train_000001.jpg
│       └── ...
└── test_images/
    └── test_images/
        ├── test_000001.jpg
        └── ...
```

`train.csv` must have columns `ID,file_name,TARGET`. `sample_submission.csv`
must have columns `ID,TARGET`. Output `submission.csv` will match those
column names.

### Running on Google Colab (recommended — produced the 96.8% result)

1. Open a new Colab notebook. Set **Runtime → Change runtime type → GPU (T4)**.
2. Zip the project directory above (with images included) and upload
   `project.zip` to Google Drive at `MyDrive/ELEC 378 Final Project/`.
3. Run these three cells:

   ```python
   # Cell 1 — mount Drive
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   ```python
   # Cell 2 — copy + unzip
   !cp "/content/drive/MyDrive/ELEC 378 Final Project/Rahul-CNN/project.zip" /content/
   !unzip -qo /content/project.zip -d /content/
   ```

   ```python
   # Cell 3 — train + write submission.csv
   %cd /content/elec-378-sp-26-final-project
   !python cnn_butterfly.py
   ```

4. After ~3 hours, save outputs back to Drive so they survive the runtime
   shutting down:

   ```python
   !cp /content/elec-378-sp-26-final-project/submission.csv "/content/drive/MyDrive/ELEC 378 Final Project/Rahul-CNN/"
   !cp /content/elec-378-sp-26-final-project/best_cnn_model.pth "/content/drive/MyDrive/ELEC 378 Final Project/Rahul-CNN/"
   !cp /content/elec-378-sp-26-final-project/ema_cnn_model.pth "/content/drive/MyDrive/ELEC 378 Final Project/Rahul-CNN/"
   !cp /content/elec-378-sp-26-final-project/training_curves.png "/content/drive/MyDrive/ELEC 378 Final Project/Rahul-CNN/"
   !cp /content/elec-378-sp-26-final-project/confusion_matrix.png "/content/drive/MyDrive/ELEC 378 Final Project/Rahul-CNN/"
   ```

### Running locally

From the project root with the data layout shown above:

```bash
pip install torch numpy pandas pillow scikit-learn matplotlib
python cnn_butterfly.py
```

### CLI options

```
python cnn_butterfly.py                  # full training + submission
python cnn_butterfly.py --eval           # skip training, run inference using
                                         #   existing best_cnn_model.pth and
                                         #   ema_cnn_model.pth, write submission.csv
python cnn_butterfly.py --seed 1337      # different stratified split + init
python cnn_butterfly.py --epochs 50      # shorter run
python cnn_butterfly.py --img-size 128   # smaller inputs (faster, less accurate)
```

### Outputs

Written next to the script:

- `submission.csv` — Kaggle-format predictions (columns `ID`, `TARGET`)
- `best_cnn_model.pth` — checkpoint with highest live validation accuracy
- `ema_cnn_model.pth` — checkpoint with highest EMA validation accuracy
- `training_curves.png` — train/val loss + accuracy per epoch
- `confusion_matrix.png` — top-20 most-confused classes from the validation set

## Reproducing the 96.8% submission

Run with default settings on a single T4. Default seed is 42. Expected
per-epoch wall-clock time ≈ 132 seconds × 80 epochs ≈ 2h 56m, plus a few
seconds at the end for inference.

If `best_cnn_model.pth` and `ema_cnn_model.pth` are already present (e.g.
restored from Drive), `python cnn_butterfly.py --eval` will skip training and
regenerate `submission.csv` in about a minute.

## Hyperparameters

All hyperparameters are at the top of `cnn_butterfly.py` under "1.
HYPERPARAMETERS" and grouped by purpose. The defaults below produced 96.8%:

| Group | Setting | Value |
|---|---|---|
| Image | `IMG_SIZE` | 160 |
| Optim | `LEARNING_RATE` (peak) | 3e-3 |
| Optim | `WEIGHT_DECAY` | 5e-4 |
| Optim | `BATCH_SIZE` | 64 |
| Optim | `NUM_EPOCHS` | 80 |
| Optim | `WARMUP_EPOCHS` | 5 |
| Loss | `LABEL_SMOOTHING` | 0.1 |
| Reg | `DROPOUT_RATE` | 0.3 |
| Reg | `MIX_PROB` (MixUp/CutMix prob) | 0.5 |
| Reg | `MIXUP_ALPHA` / `CUTMIX_ALPHA` | 0.2 / 1.0 |
| Reg | `RANDOM_ERASE_PROB` | 0.25 |
| EMA | `EMA_DECAY` | 0.999 |
| Model | `BASE_FILTERS` | 64 |
| Model | `USE_SE` | True |
| Infra | `USE_AMP` | True |
| Infra | `GRAD_CLIP` | 5.0 |
| Eval | `TTA_HFLIP` | True |
| Eval | `ENSEMBLE_BEST_AND_EMA` | True |

## References (for the report)

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018.
- Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018.
- Yun et al., *CutMix: Regularization Strategy to Train Strong Classifiers
  with Localizable Features*, ICCV 2019.
- Szegedy et al., *Rethinking the Inception Architecture for Computer
  Vision* (label smoothing), CVPR 2016.
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm
  Restarts*, ICLR 2017 (cosine schedule).
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019
  (AdamW).
- Polyak & Juditsky, *Acceleration of Stochastic Approximation by
  Averaging*, SIAM J. Control Optim. 1992 (weight averaging / EMA).
- DeVries & Taylor, *Improved Regularization of Convolutional Neural
  Networks with Cutout*, arXiv 2017 (RandomErasing).
