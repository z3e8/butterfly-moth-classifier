"""
ELEC 378 Final Project -- CNN for Butterfly & Moth Classification
=================================================================
From-scratch CNN (no pre-trained weights, no outside data) for 100-class
butterfly/moth classification.

This is an upgraded pipeline over the original VGG-style baseline.  The main
changes, with short justifications:

  Architecture
    * Replaced stacked VGG-like ConvBlocks with a ResNet-18-style backbone
      (BasicBlocks with residual connections) and optional Squeeze-and-Excite
      channel attention.  Residuals help train deeper nets from scratch; SE
      gives cheap channel re-weighting, helpful for fine-grained classes that
      differ mostly in colour/texture.

  Data / augmentation
    * Stratified 85/15 split (stratification already present, kept).
    * Stronger training augmentation: random resized crop, horizontal flip,
      small rotation + affine, colour jitter, and Random Erasing (cutout).
    * Optional MixUp / CutMix with a clean switch; both regularise fine-grained
      classifiers.

  Training
    * AdamW instead of Adam (decoupled weight decay).
    * Cosine annealing schedule with linear warmup.
    * Label smoothing (0.1) in CrossEntropy.
    * Exponential Moving Average (EMA) of weights -- usually gives a free
      accuracy bump on the validation set.
    * Mixed precision (torch.cuda.amp) for 2-3x speed on T4 GPUs.
    * Gradient clipping at 5.0 to stabilise mixup-augmented loss spikes.

  Validation / inference
    * Best-val checkpoint + EMA checkpoint are both saved.
    * Test-time augmentation: average logits over the original and its
      horizontal flip.
    * Checkpoint ensemble: logits from best-val and EMA are averaged.

The CLI is preserved:
    python cnn_butterfly.py           # train + write submission.csv
    python cnn_butterfly.py --eval    # load checkpoints + write submission.csv
Extra flags:
    --seed <int>     override RANDOM_SEED
    --epochs <int>   override NUM_EPOCHS
    --img-size <int> override IMG_SIZE
"""

import os
import sys
import time
import copy
import math
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================================
# 0.  PATHS
# ============================================================================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV       = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV  = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR   = os.path.join(BASE_DIR, "train_images", "train_images")
TEST_IMG_DIR    = os.path.join(BASE_DIR, "test_images", "test_images")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_cnn_model.pth")
EMA_SAVE_PATH   = os.path.join(BASE_DIR, "ema_cnn_model.pth")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

# ============================================================================
# 1.  HYPERPARAMETERS
# ============================================================================
IMG_SIZE        = 160        # 128 -> 160 gives noticeable accuracy gain on fine-grained features
BATCH_SIZE      = 64
NUM_EPOCHS      = 80
WARMUP_EPOCHS   = 5
LEARNING_RATE   = 3e-3       # AdamW with cosine schedule tolerates a higher peak LR
WEIGHT_DECAY    = 5e-4
LABEL_SMOOTHING = 0.1
DROPOUT_RATE    = 0.3
VAL_SPLIT       = 0.15

# MixUp / CutMix -- with prob MIX_PROB, each batch samples one of them
USE_MIX         = True
MIXUP_ALPHA     = 0.2
CUTMIX_ALPHA    = 1.0
MIX_PROB        = 0.5        # 0.5 = mix half of batches
CUTMIX_RATIO    = 0.5        # of the mixed batches, half are CutMix, half MixUp

# EMA
USE_EMA         = True
EMA_DECAY       = 0.999

# Training infra
USE_AMP         = True       # mixed precision; auto-disabled on CPU
GRAD_CLIP       = 5.0
NUM_WORKERS     = 2          # T4 Colab has 2 CPU cores
RANDOM_SEED     = 42

# Augmentation
USE_HFLIP       = True
ROTATION_DEGREES = 15
COLOR_JITTER_BRIGHTNESS = 0.25
COLOR_JITTER_CONTRAST   = 0.25
COLOR_JITTER_SATURATION = 0.25
COLOR_JITTER_HUE        = 0.05
RANDOM_CROP_SCALE_MIN = 0.7
RANDOM_CROP_SCALE_MAX = 1.0
RANDOM_ERASE_PROB       = 0.25   # cutout; applied post-normalisation

# Model
BASE_FILTERS    = 64             # stem channels; resnet-18 uses 64
USE_SE          = True
SE_REDUCTION    = 16

# Inference
TTA_HFLIP       = True
ENSEMBLE_BEST_AND_EMA = True

# ============================================================================
# 2.  REPRODUCIBILITY
# ============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 3.  LABEL ENCODING
# ============================================================================
def build_label_maps(train_csv_path):
    df = pd.read_csv(train_csv_path)
    classes = sorted(df["TARGET"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    return df, classes, label_to_idx, idx_to_label

# ============================================================================
# 4.  DATASET + AUGMENTATION
# ============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class ButterflyDataset(Dataset):
    """Loads images from disk.  All augmentation is PIL-based for transparency
    (no torchvision transforms).  Augmentations are gated by the `augment`
    flag; validation and test use only resize + normalise.
    """

    def __init__(self, file_names, labels, img_dir, img_size=160, augment=False):
        self.file_names = file_names
        self.labels     = labels
        self.img_dir    = img_dir
        self.img_size   = img_size
        self.augment    = augment

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.file_names[idx])
        img  = Image.open(path).convert("RGB")

        if self.augment:
            img = self._augment_train(img)
        else:
            img = self._resize_center(img)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())

        if self.augment and RANDOM_ERASE_PROB > 0 and np.random.rand() < RANDOM_ERASE_PROB:
            tensor = self._random_erase(tensor)

        if self.labels is not None:
            return tensor, int(self.labels[idx])
        return tensor

    # ---- augmentation pieces ----
    def _resize_center(self, img):
        return img.resize((self.img_size, self.img_size), Image.BILINEAR)

    def _augment_train(self, img):
        # Random resized crop: picks a random sub-area with scale in [min, max]
        # and resizes back to img_size -- main source of scale/translation
        # invariance for fine-grained crops.
        w, h = img.size
        scale = np.random.uniform(RANDOM_CROP_SCALE_MIN, RANDOM_CROP_SCALE_MAX)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        left = np.random.randint(0, max(1, w - new_w + 1))
        top  = np.random.randint(0, max(1, h - new_h + 1))
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # H-flip (butterflies are roughly bilaterally symmetric)
        if USE_HFLIP and np.random.rand() < 0.5:
            img = ImageOps.mirror(img)

        # Small rotation (around image center; PIL handles this correctly).
        if ROTATION_DEGREES > 0:
            angle = np.random.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        # Colour jitter
        img = self._color_jitter(img)
        return img

    @staticmethod
    def _color_jitter(img):
        # Apply in random order for more variety
        ops = []
        if COLOR_JITTER_BRIGHTNESS > 0:
            ops.append(("b", np.random.uniform(1 - COLOR_JITTER_BRIGHTNESS, 1 + COLOR_JITTER_BRIGHTNESS)))
        if COLOR_JITTER_CONTRAST > 0:
            ops.append(("c", np.random.uniform(1 - COLOR_JITTER_CONTRAST, 1 + COLOR_JITTER_CONTRAST)))
        if COLOR_JITTER_SATURATION > 0:
            ops.append(("s", np.random.uniform(1 - COLOR_JITTER_SATURATION, 1 + COLOR_JITTER_SATURATION)))
        np.random.shuffle(ops)
        for kind, factor in ops:
            if kind == "b":
                img = ImageEnhance.Brightness(img).enhance(factor)
            elif kind == "c":
                img = ImageEnhance.Contrast(img).enhance(factor)
            elif kind == "s":
                img = ImageEnhance.Color(img).enhance(factor)
        # Hue shift via HSV
        if COLOR_JITTER_HUE > 0 and np.random.rand() < 0.5:
            hsv = img.convert("HSV")
            arr = np.asarray(hsv, dtype=np.int16).copy()
            shift = int(np.random.uniform(-COLOR_JITTER_HUE, COLOR_JITTER_HUE) * 255)
            arr[..., 0] = (arr[..., 0] + shift) % 256
            img = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")
        return img

    @staticmethod
    def _random_erase(tensor):
        """Cutout/RandomErasing on a (C,H,W) normalised tensor."""
        c, h, w = tensor.shape
        area = h * w
        for _ in range(10):
            target_area = np.random.uniform(0.02, 0.25) * area
            aspect = np.random.uniform(0.3, 3.3)
            eh = int(round(math.sqrt(target_area * aspect)))
            ew = int(round(math.sqrt(target_area / aspect)))
            if eh < h and ew < w:
                top  = np.random.randint(0, h - eh)
                left = np.random.randint(0, w - ew)
                tensor[:, top:top+eh, left:left+ew] = torch.randn(c, eh, ew)
                return tensor
        return tensor


# ============================================================================
# 5.  MODEL:  ResNet-18-style with optional SE
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excite: global pool -> MLP -> per-channel sigmoid gate."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        s = self.avg(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BasicBlock(nn.Module):
    """ResNet basic block (2 x 3x3 conv + residual + optional SE)."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch, reduction=SE_REDUCTION) if use_se else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ButterflyNet(nn.Module):
    """ResNet-18 width, from-scratch init, with SE blocks.

    Stem is 3x3/s1 (ImageNet's 7x7 stem throws away too much detail for
    160-pixel inputs).  Four stages of 2 blocks each, channels 64-128-256-512.
    """
    def __init__(self, num_classes=100, dropout=0.3, use_se=True):
        super().__init__()
        c1 = BASE_FILTERS
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(c1,     c1,     n_blocks=2, stride=1, use_se=use_se)
        self.stage2 = self._make_stage(c1,     c1 * 2, n_blocks=2, stride=2, use_se=use_se)
        self.stage3 = self._make_stage(c1 * 2, c1 * 4, n_blocks=2, stride=2, use_se=use_se)
        self.stage4 = self._make_stage(c1 * 4, c1 * 8, n_blocks=2, stride=2, use_se=use_se)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(c1 * 8, num_classes)
        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n_blocks, stride, use_se):
        layers = [BasicBlock(in_ch, out_ch, stride=stride, use_se=use_se)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================================
# 6.  MIXUP / CUTMIX
# ============================================================================
def mixup_batch(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def cutmix_batch(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, C, H, W = x.shape
    idx = torch.randperm(B, device=x.device)
    # random bbox
    cut_ratio = math.sqrt(1.0 - lam)
    cw, ch = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cw // 2, 0); x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0); y2 = min(cy + ch // 2, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # re-compute lam to reflect actual cut area
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
    return x, y, y[idx], lam


def mix_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ============================================================================
# 7.  EMA
# ============================================================================
class ModelEMA:
    """Simple exponential-moving-average copy of the model's parameters.
    The shadow model is updated every step.  At eval time we call
    self.ema.eval() and forward through it like any other module.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


# ============================================================================
# 8.  TRAINING
# ============================================================================
def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    ema=None, epoch_idx=0, epochs=1, steps_per_epoch=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    total_steps = epochs * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Schedule LR
        step = epoch_idx * steps_per_epoch + i
        lr = cosine_lr(step, total_steps, LEARNING_RATE, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # Decide mixing strategy for this batch
        mix_mode = "none"
        if USE_MIX and np.random.rand() < MIX_PROB:
            if np.random.rand() < CUTMIX_RATIO:
                images, y_a, y_b, lam = cutmix_batch(images, labels, CUTMIX_ALPHA)
                mix_mode = "cutmix"
            else:
                images, y_a, y_b, lam = mixup_batch(images, labels, MIXUP_ALPHA)
                mix_mode = "mixup"

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            if mix_mode == "none":
                loss = criterion(logits, labels)
            else:
                loss = mix_criterion(criterion, logits, y_a, y_b, lam)

        if scaler is not None:
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        # Approximate train acc using the (possibly mixed) batch -- imperfect
        # but fine for a sanity signal.
        _, pred = logits.max(1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = 0.0; correct = 0; total = 0
    preds, gts = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        running += loss.item() * images.size(0)
        _, p = logits.max(1)
        correct += (p == labels).sum().item()
        total   += labels.size(0)
        preds.extend(p.cpu().numpy().tolist())
        gts.extend(labels.cpu().numpy().tolist())
    return running / total, correct / total, preds, gts


# ============================================================================
# 9.  INFERENCE (TTA + ensemble)
# ============================================================================
@torch.no_grad()
def predict_logits(model, loader, device, tta_hflip=True):
    """Return stacked probability predictions of shape (N, num_classes)."""
    model.eval()
    all_logits = []
    for batch in loader:
        images = batch if not isinstance(batch, (tuple, list)) else batch[0]
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        if tta_hflip:
            flipped = torch.flip(images, dims=[3])
            probs_flip = F.softmax(model(flipped), dim=1)
            probs = 0.5 * (probs + probs_flip)
        all_logits.append(probs.cpu())
    return torch.cat(all_logits, dim=0)


def generate_submission(pred_indices, idx_to_label, submission_path):
    # Column names match sample_submission.csv: ID, TARGET
    sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
    image_ids = sample_sub["ID"].tolist()
    pred_labels = [idx_to_label[i] for i in pred_indices]
    df = pd.DataFrame({"ID": image_ids, "TARGET": pred_labels})
    df.to_csv(submission_path, index=False)
    print(f"[INFO] Submission saved to {submission_path}  ({len(df)} rows)")
    return df


# ============================================================================
# 10.  PLOTS
# ============================================================================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.grid(True); ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs,   label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.grid(True); ax2.legend()
    ax2.set_title("Accuracy")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[INFO] Training curves -> {save_path}")


def plot_confusion_matrix(cm, classes, save_path, top_n=20):
    misclass = cm.sum(axis=1) - np.diag(cm)
    top_idx = np.argsort(misclass)[-top_n:]
    cm_sub = cm[np.ix_(top_idx, top_idx)]
    labels = [classes[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_sub, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (top-{top_n} most confused)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"[INFO] Confusion matrix -> {save_path}")


# ============================================================================
# 11.  MAIN
# ============================================================================
def main():
    global IMG_SIZE, NUM_EPOCHS

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Skip training; load checkpoints and write submission.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    # Apply CLI overrides
    IMG_SIZE = args.img_size
    NUM_EPOCHS = args.epochs
    set_seed(args.seed)

    print("=" * 60)
    print("ELEC 378 -- Butterfly & Moth classifier (ResNet-SE + MixUp + EMA)")
    print("=" * 60)
    print(f"[INFO] Device: {DEVICE}  |  AMP: {USE_AMP and DEVICE.type == 'cuda'}")
    print(f"[INFO] IMG_SIZE={IMG_SIZE}  BATCH_SIZE={BATCH_SIZE}  EPOCHS={NUM_EPOCHS}  seed={args.seed}")

    # ---- data ----
    df, classes, label_to_idx, idx_to_label = build_label_maps(TRAIN_CSV)
    num_classes = len(classes)
    print(f"[INFO] {len(df)} training images, {num_classes} classes")

    train_files, val_files, train_labels, val_labels = train_test_split(
        df["file_name"].values,
        df["TARGET"].map(label_to_idx).values,
        test_size=VAL_SPLIT,
        stratify=df["TARGET"].values,
        random_state=args.seed,
    )
    print(f"[INFO] Train: {len(train_files)}  Val: {len(val_files)}")

    train_ds = ButterflyDataset(train_files, train_labels, TRAIN_IMG_DIR,
                                img_size=IMG_SIZE, augment=True)
    val_ds   = ButterflyDataset(val_files,   val_labels,   TRAIN_IMG_DIR,
                                img_size=IMG_SIZE, augment=False)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=NUM_WORKERS > 0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=NUM_WORKERS > 0)

    # ---- model ----
    model = ButterflyNet(num_classes=num_classes, dropout=DROPOUT_RATE,
                         use_se=USE_SE).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Params: {total/1e6:.2f}M")

    # ---- loss / optim / amp / ema ----
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.type == "cuda") else None
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    best_val_acc = 0.0
    best_ema_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    if not args.eval:
        steps_per_epoch = len(train_loader)
        print(f"[INFO] Steps/epoch: {steps_per_epoch}")

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, DEVICE,
                ema=ema, epoch_idx=epoch, epochs=NUM_EPOCHS,
                steps_per_epoch=steps_per_epoch,
            )
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
            ema_acc = None
            if ema is not None:
                _, ema_acc, _, _ = validate(ema.ema, val_loader, criterion, DEVICE)

            train_losses.append(tr_loss); val_losses.append(val_loss)
            train_accs.append(tr_acc);    val_accs.append(val_acc)

            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            msg = (f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                   f"train {tr_loss:.3f}/{tr_acc:.3f} | "
                   f"val {val_loss:.3f}/{val_acc:.3f}")
            if ema_acc is not None:
                msg += f" | ema {ema_acc:.3f}"
            msg += f" | lr {cur_lr:.2e} | {elapsed:.1f}s"
            print(msg)

            # Save best-val and best-ema checkpoints
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"    + new best val acc {best_val_acc:.4f}  -> {os.path.basename(MODEL_SAVE_PATH)}")
            if ema is not None and ema_acc is not None and ema_acc > best_ema_acc:
                best_ema_acc = ema_acc
                torch.save(ema.ema.state_dict(), EMA_SAVE_PATH)
                print(f"    + new best ema acc {best_ema_acc:.4f}  -> {os.path.basename(EMA_SAVE_PATH)}")

        print(f"\n[INFO] Best val acc: {best_val_acc:.4f}")
        if ema is not None:
            print(f"[INFO] Best ema acc: {best_ema_acc:.4f}")

        plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                             os.path.join(BASE_DIR, "training_curves.png"))

        # Final confusion matrix on the better of the two checkpoints
        final_state = EMA_SAVE_PATH if (ema is not None and best_ema_acc >= best_val_acc
                                        and os.path.exists(EMA_SAVE_PATH)) else MODEL_SAVE_PATH
        final_model = ButterflyNet(num_classes=num_classes, dropout=DROPOUT_RATE,
                                    use_se=USE_SE).to(DEVICE)
        final_model.load_state_dict(torch.load(final_state, map_location=DEVICE))
        _, final_acc, val_preds, val_gts = validate(final_model, val_loader, criterion, DEVICE)
        print(f"[INFO] Final chosen ckpt {os.path.basename(final_state)} val acc = {final_acc:.4f}")
        cm = confusion_matrix(val_gts, val_preds)
        plot_confusion_matrix(cm, classes, os.path.join(BASE_DIR, "confusion_matrix.png"))

    # ---- build test loader ----
    sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
    test_files = [f"{tid}.jpg" for tid in sample_sub["ID"]]
    test_ds = ButterflyDataset(test_files, labels=None, img_dir=TEST_IMG_DIR,
                               img_size=IMG_SIZE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin,
                             persistent_workers=NUM_WORKERS > 0)

    # ---- load best checkpoints for inference ----
    print("\n[INFO] Running inference with TTA" + (" + EMA ensemble" if ENSEMBLE_BEST_AND_EMA else ""))
    models_to_ensemble = []
    if os.path.exists(MODEL_SAVE_PATH):
        m = ButterflyNet(num_classes=num_classes, dropout=DROPOUT_RATE,
                         use_se=USE_SE).to(DEVICE)
        m.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        models_to_ensemble.append(("best_val", m))
    if ENSEMBLE_BEST_AND_EMA and os.path.exists(EMA_SAVE_PATH):
        m = ButterflyNet(num_classes=num_classes, dropout=DROPOUT_RATE,
                         use_se=USE_SE).to(DEVICE)
        m.load_state_dict(torch.load(EMA_SAVE_PATH, map_location=DEVICE))
        models_to_ensemble.append(("ema", m))

    if not models_to_ensemble:
        print("[ERROR] No saved checkpoints to run inference with.")
        sys.exit(1)

    probs_sum = None
    for name, m in models_to_ensemble:
        probs = predict_logits(m, test_loader, DEVICE, tta_hflip=TTA_HFLIP)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        print(f"    - {name}: probs shape {tuple(probs.shape)}")
    probs_avg = probs_sum / len(models_to_ensemble)
    pred_indices = probs_avg.argmax(dim=1).numpy().tolist()

    sub_df = generate_submission(pred_indices, idx_to_label, SUBMISSION_PATH)
    print("\n--- Submission preview ---")
    print(sub_df.head(10))
    print("\n[DONE]")


if __name__ == "__main__":
    main()
