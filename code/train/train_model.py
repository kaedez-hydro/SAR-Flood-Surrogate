"""
Flood mapping training — final recipe.

Combines the validated recipe from train_03261.py (F1 ≈ 0.8 under random split)
with spatial-split-first data loading to eliminate pixel leakage between
train and validation sets.

Key design choices (each validated by ablation):
  - Per-sample Dice + BCE — every patch contributes equally to Dice loss
  - No Focal Loss — avoids fitting boundary mixed pixels whose labels are uncertain
  - Adam (no weight decay) — preserves the capacity of a from-scratch UNet
  - CosineAnnealingLR (no restarts) — smooth convergence, no disruptive LR resets
  - Large batch + long patience — stable gradients, gives model time to converge
  - Spatial split by default — realistic validation metrics

Usage:
  python make_spatial_split.py --metadata-csv <path> --strategy zone_block_random
  export TRAIN_SPLIT_FILE="./splits/<name>.csv"
  python train_final.py
"""

import json
import logging
import multiprocessing
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from dataset import FloodDataset
from model import build_model


# =========================================================================
# Reproducibility
# =========================================================================

def seed_everything(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark = True is set later (after seed, before training)


# =========================================================================
# Interactive setup
# =========================================================================

def choose_option(prompt, options):
    print(prompt)
    for idx, (label, _) in enumerate(options, start=1):
        print(f"  [{idx}] {label}")
    while True:
        choice = input("Select: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][1]
        print("Invalid input, try again.")


def choose_yes_no(prompt, default=True):
    tip = "Y/n" if default else "y/N"
    while True:
        v = input(f"{prompt} ({tip}): ").strip().lower()
        if v == "":
            return default
        if v in {"y", "yes"}:
            return True
        if v in {"n", "no"}:
            return False
        print("Please input Y or N.")


def choose_int(prompt, default, lo=None, hi=None):
    while True:
        v = input(f"{prompt} [default={default}]: ").strip()
        if v == "":
            return default
        if not v.isdigit():
            print("Please input an integer.")
            continue
        n = int(v)
        if lo is not None and n < lo:
            print(f"Value must be >= {lo}")
            continue
        if hi is not None and n > hi:
            print(f"Value must be <= {hi}")
            continue
        return n


def commander_setup():
    print("=" * 72)
    print("Flood Mapping — Training Control Panel")
    print("=" * 72)

    data_domain = choose_option(
        "1) Data domain:",
        [("full_basin", "full_basin"), ("floodplain_domain", "floodplain_domain")],
    )

    model_name = choose_option(
        "2) Model:",
        [
            ("ResNet34-U-Net (pretrained encoder)", "resnet34_unet"),
            ("Vanilla U-Net (from scratch)", "vanilla_unet"),
            ("DeepLabV3+ (pretrained encoder)", "deeplabv3plus"),
        ],
    )

    in_channels = choose_int("3) Input channels (4-8)", default=6, lo=4, hi=8)

    raw_name = input("4) Experiment name (Enter for auto): ").strip()
    exp_name = raw_name or (
        f"{data_domain}_{model_name}_{in_channels}ch_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    use_aug = choose_yes_no("5) Data augmentation", default=True)
    cache_mem = choose_yes_no("6) Cache dataset in RAM", default=True)

    bs_override = choose_yes_no("7) Override batch size", default=False)
    default_bs = (
        180
        if torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory > 16 * 1024**3
        else 32
    )
    batch_size = choose_int("Batch size", default=default_bs, lo=1) if bs_override else default_bs

    seed = choose_int("8) Random seed", default=2026, lo=1)

    print("\n" + "-" * 72)
    print("Summary")
    print(f"  domain      : {data_domain}")
    print(f"  model       : {model_name}")
    print(f"  channels    : {in_channels}")
    print(f"  experiment  : {exp_name}")
    print(f"  augmentation: {use_aug}")
    print(f"  cache_memory: {cache_mem}")
    print(f"  batch_size  : {batch_size}")
    print(f"  seed        : {seed}")
    print("-" * 72)

    if not choose_yes_no("Start training?", default=True):
        raise SystemExit("Aborted.")

    return {
        "EXP_NAME": exp_name,
        "DATA_DOMAIN": data_domain,
        "MODEL_NAME": model_name,
        "IN_CHANNELS": in_channels,
        "AUGMENTATION": use_aug,
        "CACHE_MEMORY": cache_mem,
        "BATCH_SIZE": batch_size,
        "SEED": seed,
    }


# =========================================================================
# Loss  —  per-sample Dice + BCE  (validated recipe)
# =========================================================================

class CombinedLoss(nn.Module):
    """BCE + per-sample Dice (water patches only).

    - Dice computed per-sample for patches that contain water pixels.
      Dry patches are excluded from Dice because intersection ≡ 0 makes
      dice_loss ≈ 1.0 regardless of prediction quality, flooding the loss.
    - BCE applies to all patches (pos_weight handles class imbalance).
    - smooth = 1e-5 keeps Dice sensitive to small overlaps.
    - pos_weight via register_buffer → auto device placement.
    - label_smoothing softens hard targets to reduce overconfidence.
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: float = 4.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "pos_weight_tensor",
            torch.tensor([float(pos_weight)]),
            persistent=True,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Label smoothing: hard {0,1} → soft {eps, 1-eps}
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + \
                     0.5 * self.label_smoothing

        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=self.pos_weight_tensor,
            reduction="mean",
        )

        # Per-sample Dice — only for water-containing patches.
        # Dry patches have target.sum == 0, making Dice meaningless.
        has_water = target.sum(dim=(2, 3)) > 0  # [B, 1]

        if has_water.any():
            pred_sig = torch.sigmoid(pred)
            smooth = 1e-5
            inter = (pred_sig * target).sum(dim=(2, 3))  # [B, 1]
            union = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # [B, 1]
            dice = (2.0 * inter + smooth) / (union + smooth)  # [B, 1]
            dice_loss = 1.0 - dice[has_water].mean()
        else:
            dice_loss = torch.tensor(0.0, device=pred.device)

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


# =========================================================================
# Metrics
# =========================================================================

def compute_metrics(pred_logits, targets, threshold=0.5):
    preds = (torch.sigmoid(pred_logits) > threshold).float().view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum().item()
    fp = ((1.0 - targets) * preds).sum().item()
    fn = (targets * (1.0 - preds)).sum().item()
    eps = 1e-6
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)
    return {"iou": iou, "precision": precision, "recall": recall,
            "f1": f1, "dice": dice}


# =========================================================================
# Hardware monitor
# =========================================================================

class Monitor:
    @staticmethod
    def snapshot():
        s = {}
        try:
            import psutil
            s["cpu_pct"] = psutil.cpu_percent(interval=None)
            s["ram_gb"]  = round(psutil.virtual_memory().used / 1024**3, 2)
            s["ram_pct"] = psutil.virtual_memory().percent
        except Exception:
            s["cpu_pct"] = -1
            s["ram_gb"]  = -1
            s["ram_pct"] = -1
        if torch.cuda.is_available():
            s["gpu_alloc_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            s["gpu_resv_gb"]  = round(torch.cuda.memory_reserved() / 1024**3, 2)
        else:
            s["gpu_alloc_gb"] = 0
            s["gpu_resv_gb"]  = 0
        return s


# =========================================================================
# Logging & book-keeping
# =========================================================================

def setup_logger(output_dir):
    p = os.path.join(output_dir, "train_full_log.txt")
    logger = logging.getLogger("FloodTrain")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(p, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_snapshot(output_dir, config):
    info = {
        "System": platform.system(),
        "Python": sys.version,
        "PyTorch": torch.__version__,
        "CUDA": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    with open(os.path.join(output_dir, "env_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    shutil.copy2(os.path.abspath(__file__),
                 os.path.join(output_dir, "train_code_backup.py"))


def write_summary(history, config, total_time, stop_reason, output_dir):
    if not history:
        return
    first = history[0]
    last  = history[-1]
    best  = min(history, key=lambda x: x["val_loss"])
    avg_s = sum(x["time_sec"] for x in history) / len(history)
    txt = f"""# Training Summary

- Experiment: {config['EXP_NAME']}
- Domain: {config['DATA_DOMAIN']}
- Model: {config['MODEL_NAME']}
- Channels: {config['IN_CHANNELS']}
- Seed: {config['SEED']}
- Stop reason: {stop_reason}
- Epochs: {len(history)} / {config['EPOCHS']}
- Total time (h): {total_time / 3600:.2f}
- Avg epoch time (s): {avg_s:.1f}

## Best Epoch ({best['epoch']})
- Val Loss: {best['val_loss']:.4f}
- IoU: {best['val_iou']:.4f}
- F1: {best['val_f1']:.4f}
- Dice: {best['val_dice']:.4f}

## Trajectory
- Epoch 1: val_loss={first['val_loss']:.4f}, iou={first['val_iou']:.4f}
- Last epoch: val_loss={last['val_loss']:.4f}, iou={last['val_iou']:.4f}
"""
    p = os.path.join(output_dir, "LLM_Summary.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)


# =========================================================================
# Data loading  —  spatial-split-first
# =========================================================================

def build_dataloaders(config, logger):
    domain_dir = f"./train_data_{config['DATA_DOMAIN']}"
    if not os.path.exists(domain_dir):
        legacy = "./train_data"
        if os.path.exists(legacy):
            logger.info(f"[Data] fallback to legacy: {legacy}")
            domain_dir = legacy
        else:
            raise FileNotFoundError(
                f"Dataset not found: {domain_dir}. Run preprocess.py first."
            )

    train_ds = FloodDataset(domain_dir, transform=config["AUGMENTATION"],
                            cache_memory=config["CACHE_MEMORY"])
    val_ds   = FloodDataset(domain_dir, transform=False,
                            cache_memory=config["CACHE_MEMORY"])

    split_mode = config.get("SPLIT_MODE", "predefined")
    split_file = config.get("SPLIT_FILE", "")

    if split_mode == "predefined" and split_file and os.path.exists(split_file):
        # ---- spatial split ----
        sdf = pd.read_csv(split_file)
        if not {"patch_name", "split"}.issubset(sdf.columns):
            raise ValueError("Split file needs columns: patch_name, split")
        smap = {str(r["patch_name"]): str(r["split"]).strip().lower()
                for _, r in sdf.iterrows()}
        t_idx, v_idx = [], []
        for i, fn in enumerate(train_ds.files):
            f = smap.get(fn, "")
            if f == "train":
                t_idx.append(i)
            elif f == "val":
                v_idx.append(i)
        if not t_idx or not v_idx:
            raise ValueError(f"Empty split from {split_file}: "
                             f"train={len(t_idx)} val={len(v_idx)}")
        train_set = Subset(train_ds, t_idx)
        val_set   = Subset(val_ds,   v_idx)
        logger.info(f"[Data] spatial split: {split_file}  "
                    f"(train={len(train_set)} val={len(val_set)})")
    else:
        # ---- fallback: random (WARNING: over-estimates metrics) ----
        if split_mode == "predefined":
            logger.warning(
                "[Data] SPLIT_MODE=predefined but no valid SPLIT_FILE. "
                "FALLING BACK TO RANDOM SPLIT — METRICS WILL BE OVERESTIMATED. "
                "Run make_spatial_split.py first."
            )
        n = len(train_ds)
        n_val = int(n * config["VAL_PERCENT"])
        g = torch.Generator().manual_seed(config["SEED"])
        perm = torch.randperm(n, generator=g).tolist()
        train_set = Subset(train_ds, perm[: n - n_val])
        val_set   = Subset(val_ds,   perm[n - n_val :])

    nw = config["NUM_WORKERS"]
    cargs = {
        "num_workers": nw,
        "pin_memory": config["PIN_MEMORY"],
        "persistent_workers": nw > 0,
    }
    if nw > 0:
        cargs["prefetch_factor"] = config["PREFETCH_FACTOR"]

    train_loader = DataLoader(train_set, batch_size=config["BATCH_SIZE"],
                              shuffle=True, **cargs)
    val_loader   = DataLoader(val_set,   batch_size=config["BATCH_SIZE"],
                              shuffle=False, **cargs)

    logger.info(
        f"[Data] domain={config['DATA_DOMAIN']}  "
        f"train={len(train_set)} val={len(val_set)}  "
        f"aug={config['AUGMENTATION']}  cache={config['CACHE_MEMORY']}  "
        f"split={split_mode}"
    )
    return train_loader, val_loader


# =========================================================================
# Training
# =========================================================================

def train():
    setup = commander_setup()

    config = {
        **setup,
        # --- fixed recipe ---
        "EPOCHS":             30000,
        "LR":                 3e-4,
        "VAL_PERCENT":        0.2,
        "DEVICE":             "cuda" if torch.cuda.is_available() else "cpu",
        "NUM_WORKERS":        1,
        "PREFETCH_FACTOR":    2,
        "PIN_MEMORY":         True,
        "BCE_WEIGHT":         0.5,
        "POS_WEIGHT":         4.0,
        "LABEL_SMOOTHING":    0.0,
        "DROPOUT":            0.1,
        "PATIENCE":           1000,
        "THRESHOLD":          0.5,
        "LOG_BATCH_INTERVAL": 10,
        # --- split control (env overrides) ---
        "SPLIT_MODE": os.environ.get("TRAIN_SPLIT_MODE", "predefined").strip().lower(),
        "SPLIT_FILE": os.environ.get("TRAIN_SPLIT_FILE", "").strip(),
    }

    seed_everything(config["SEED"])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    ckpt_dir  = os.path.join("./saved_models", config["EXP_NAME"])
    out_dir   = os.path.join("./output",       config["EXP_NAME"])
    debug_dir = os.path.join(out_dir, "val_debug_images")
    for d in [ckpt_dir, out_dir, debug_dir]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logger(out_dir)
    save_snapshot(out_dir, config)

    logger.info("=" * 72)
    logger.info(f"experiment : {config['EXP_NAME']}")
    logger.info(f"domain     : {config['DATA_DOMAIN']}")
    logger.info(f"model      : {config['MODEL_NAME']}")
    logger.info(f"channels   : {config['IN_CHANNELS']}")
    logger.info(f"batch_size : {config['BATCH_SIZE']}")
    logger.info(f"split_mode : {config['SPLIT_MODE']}")
    logger.info("=" * 72)

    train_loader, val_loader = build_dataloaders(config, logger)

    model = build_model(config["MODEL_NAME"],
                        n_channels=config["IN_CHANNELS"],
                        n_classes=1,
                        dropout=config.get("DROPOUT", 0.0)).to(config["DEVICE"])

    optimizer  = optim.Adam(model.parameters(), lr=config["LR"],
                            weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["EPOCHS"], eta_min=1e-6,
    )
    criterion  = CombinedLoss(config["BCE_WEIGHT"], config["POS_WEIGHT"],
                              label_smoothing=config.get("LABEL_SMOOTHING", 0.0))
    criterion  = criterion.to(config["DEVICE"])

    use_amp    = config["DEVICE"] == "cuda"
    scaler     = GradScaler(enabled=use_amp)

    epoch_hist: list = []
    batch_hist: list = []
    best_loss = float("inf")
    no_improve = 0
    t0 = time.time()
    prev_best_path = None
    stop_reason = "Completed All Epochs"

    for epoch in range(config["EPOCHS"]):
        t_ep = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Ep [{epoch + 1}/{config['EPOCHS']}]",
                    unit="bt", file=sys.stdout)

        for bi, (imgs, masks) in enumerate(pbar):
            imgs  = imgs[:, : config["IN_CHANNELS"], :, :]
            imgs  = imgs.to(config["DEVICE"], non_blocking=True)
            masks = masks.to(config["DEVICE"], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(imgs)
                loss    = criterion(outputs.float(), masks.float())

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"[Error] NaN/Inf loss at ep={epoch+1} batch={bi}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            cur = loss.item()
            train_loss += cur
            pbar.set_postfix(loss=f"{cur:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if bi % config["LOG_BATCH_INTERVAL"] == 0:
                batch_hist.append({
                    "epoch": epoch + 1, "batch": bi, "phase": "train",
                    "loss": cur, "lr": optimizer.param_groups[0]["lr"],
                    "model": config["MODEL_NAME"],
                    "domain": config["DATA_DOMAIN"],
                    "channels": config["IN_CHANNELS"],
                    "seed": config["SEED"],
                    "aug": int(config["AUGMENTATION"]),
                    **Monitor.snapshot(),
                })

        avg_train = train_loss / max(1, len(train_loader))

        # ---- val ----
        model.eval()
        val_loss = 0.0
        met_sum  = {"iou": 0.0, "dice": 0.0, "precision": 0.0,
                    "recall": 0.0, "f1": 0.0}

        with torch.no_grad():
            for bi, (imgs, masks) in enumerate(val_loader):
                imgs  = imgs[:, : config["IN_CHANNELS"], :, :]
                imgs  = imgs.to(config["DEVICE"], non_blocking=True)
                masks = masks.to(config["DEVICE"], non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(imgs)

                loss = criterion(outputs.float(), masks.float())
                val_loss += loss.item()

                m = compute_metrics(outputs.float(), masks, config["THRESHOLD"])
                for k in met_sum:
                    met_sum[k] += m[k]

                if bi == 0:
                    pv = (torch.sigmoid(outputs[0, 0]) > config["THRESHOLD"])
                    pv = pv.float().cpu().numpy()
                    tv = masks[0, 0].cpu().numpy()
                    cv2.imwrite(
                        os.path.join(debug_dir, f"epoch_{epoch + 1}_val.png"),
                        np.hstack([(pv * 255).astype(np.uint8),
                                   (tv * 255).astype(np.uint8)]),
                    )

        avg_val  = val_loss / max(1, len(val_loader))
        avg_met  = {k: v / max(1, len(val_loader)) for k, v in met_sum.items()}

        scheduler.step()
        t_ep = time.time() - t_ep

        rec = {
            "epoch": epoch + 1,
            "model": config["MODEL_NAME"],
            "domain": config["DATA_DOMAIN"],
            "channels": config["IN_CHANNELS"],
            "seed": config["SEED"],
            "aug": int(config["AUGMENTATION"]),
            "train_loss": avg_train,
            "val_loss": avg_val,
            "val_iou": avg_met["iou"],
            "val_dice": avg_met["dice"],
            "val_precision": avg_met["precision"],
            "val_recall": avg_met["recall"],
            "val_f1": avg_met["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(t_ep, 2),
            **Monitor.snapshot(),
        }
        epoch_hist.append(rec)

        pd.DataFrame(epoch_hist).to_csv(
            os.path.join(out_dir, "training_log_epoch.csv"), index=False)
        pd.DataFrame(batch_hist).to_csv(
            os.path.join(out_dir, "training_log_batch_detail.csv"), index=False)

        logger.info(
            f"[Ep {epoch + 1}] train_loss={avg_train:.4f}  "
            f"val_loss={avg_val:.4f}  iou={avg_met['iou']:.4f}  "
            f"f1={avg_met['f1']:.4f}  {t_ep:.1f}s"
        )

        # ---- checkpoint & early-stop ----
        if avg_val < best_loss:
            best_loss = avg_val
            no_improve = 0
            name = f"best_model_loss{best_loss:.4f}.pth"
            path = os.path.join(ckpt_dir, name)
            torch.save(model.state_dict(), path)
            if prev_best_path and prev_best_path != path and os.path.exists(prev_best_path):
                os.remove(prev_best_path)
            prev_best_path = path
            shutil.copyfile(path, os.path.join(ckpt_dir, "latest_best.pth"))
            logger.info(f"[New Best] {name}")
        else:
            no_improve += 1
            if no_improve >= config["PATIENCE"]:
                stop_reason = "Early Stopping Triggered"
                logger.info(f"[Early Stop] {config['PATIENCE']} epochs without improvement")
                break

    total = time.time() - t0
    logger.info("=" * 72)
    logger.info(f"Done: {config['EXP_NAME']}  |  {total / 3600:.2f} h")
    write_summary(epoch_hist, config, total, stop_reason, out_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()
