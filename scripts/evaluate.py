import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import build_model


@dataclass
class EvalConfig:
    experiment_name: str
    model_name: str
    data_domain: str
    in_channels: int
    checkpoint_path: str
    data_dir: str
    output_dir: str
    threshold: float
    batch_size: int
    num_workers: int
    device: str
    seed: int
    val_percent: float
    split_mode: str
    split_file: str
    metadata_csv: str
    zone_column: str
    use_tta: bool = False
    use_morphology: bool = False
    morph_kernel_size: int = 3
    morph_operation: str = "open"


class PatchDataset(Dataset):
    def __init__(self, data_dir: str, file_names: List[str], in_channels: int):
        self.input_dir = os.path.join(data_dir, "inputs")
        self.label_dir = os.path.join(data_dir, "labels")
        self.file_names = file_names
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        name = self.file_names[idx]
        img = np.load(os.path.join(self.input_dir, name)).astype(np.float32)
        mask = np.load(os.path.join(self.label_dir, name)).astype(np.float32)

        img = img[: self.in_channels, :, :]
        return (
            torch.from_numpy(img),
            torch.from_numpy(mask).unsqueeze(0),
            name,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent evaluation for flood mapping models.")

    parser.add_argument("--experiment-name", default="", help="Experiment directory name under ./saved_models and ./output")
    parser.add_argument("--model-name", default="", choices=["", "resnet34_unet", "vanilla_unet", "deeplabv3plus"])
    parser.add_argument("--data-domain", default="", choices=["", "full_basin", "floodplain_domain"])
    parser.add_argument("--in-channels", type=int, default=0)

    parser.add_argument("--checkpoint-path", default="", help="Path to model .pth; defaults to saved_models/<exp>/latest_best.pth")
    parser.add_argument("--data-dir", default="", help="Dataset directory; defaults to train_data_<domain>")
    parser.add_argument("--output-dir", default="", help="Evaluation output directory; defaults to output/<exp>/evaluation")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-percent", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["", "val", "full", "predefined"], default="")
    parser.add_argument("--split-file", default="", help="Split CSV with columns patch_name,split(train/val)")
    parser.add_argument("--metadata-csv", default="", help="Patch metadata CSV path; defaults to <data_dir>/patch_metadata.csv")
    parser.add_argument("--zone-column", default="zone_name", help="Zone column in metadata for zone-level aggregation")

    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (4-way flip)")
    parser.add_argument("--morphology", action="store_true", help="Apply morphological post-processing")
    parser.add_argument("--morph-kernel-size", type=int, default=3, help="Morphology kernel size (pixels)")
    parser.add_argument("--morph-operation", choices=["open", "close", "both"], default="open", help="Morphology operation type")

    return parser.parse_args()


def load_train_config(experiment_name: str) -> Dict:
    if not experiment_name:
        return {}

    cfg_path = os.path.join("output", experiment_name, "config.json")
    if not os.path.exists(cfg_path):
        return {}

    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_eval_config(args: argparse.Namespace, train_cfg: Dict) -> EvalConfig:
    experiment_name = args.experiment_name or train_cfg.get("EXP_NAME", "manual_eval")

    model_name = args.model_name or train_cfg.get("MODEL_NAME", "resnet34_unet")
    data_domain = args.data_domain or train_cfg.get("DATA_DOMAIN", "full_basin")

    in_channels = args.in_channels or int(train_cfg.get("IN_CHANNELS", 6))

    checkpoint_path = args.checkpoint_path
    if not checkpoint_path:
        if experiment_name and experiment_name != "manual_eval":
            checkpoint_path = os.path.join("saved_models", experiment_name, "latest_best.pth")
        else:
            raise ValueError("checkpoint path is required when experiment-name is not provided.")

    data_dir = args.data_dir or f"train_data_{data_domain}"
    metadata_csv = args.metadata_csv or os.path.join(data_dir, "patch_metadata.csv")

    if args.output_dir:
        output_dir = args.output_dir
    elif experiment_name and experiment_name != "manual_eval":
        output_dir = os.path.join("output", experiment_name, "evaluation")
    else:
        output_dir = os.path.join("output", "manual_eval", "evaluation")

    split_mode = args.split_mode or str(train_cfg.get("SPLIT_MODE", "val"))
    split_file = args.split_file or str(train_cfg.get("SPLIT_FILE", ""))

    return EvalConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        data_domain=data_domain,
        in_channels=in_channels,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed if args.seed else int(train_cfg.get("SEED", 2026)),
        val_percent=args.val_percent,
        split_mode=split_mode,
        split_file=split_file,
        metadata_csv=metadata_csv,
        zone_column=args.zone_column,
        use_tta=args.tta,
        use_morphology=args.morphology,
        morph_kernel_size=args.morph_kernel_size,
        morph_operation=args.morph_operation,
    )


def get_eval_files(
    data_dir: str,
    split_mode: str,
    seed: int,
    val_percent: float,
    split_file: str = "",
) -> List[str]:
    input_dir = os.path.join(data_dir, "inputs")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    if len(all_files) == 0:
        raise RuntimeError(f"No patch files found under: {input_dir}")

    if split_mode == "full":
        return all_files
    if split_mode == "predefined":
        if not split_file:
            raise ValueError("split_file is required when split_mode=predefined")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        split_df = pd.read_csv(split_file)
        required_cols = {"patch_name", "split"}
        if not required_cols.issubset(set(split_df.columns)):
            raise ValueError(f"Split file must include columns: {required_cols}")
        val_names = set(
            split_df.loc[
                split_df["split"].astype(str).str.lower() == "val",
                "patch_name",
            ].astype(str).tolist()
        )
        files = [f for f in all_files if f in val_names]
        if not files:
            raise ValueError("No validation patches matched from split file.")
        return sorted(files)

    n_total = len(all_files)
    n_val = int(n_total * val_percent)
    if n_val <= 0:
        raise ValueError(f"val set is empty (n_total={n_total}, val_percent={val_percent}).")

    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    val_indices = perm[n_train:]

    val_files = [all_files[i] for i in val_indices]
    val_files.sort()
    return val_files


def confusion_counts(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> Tuple[int, int, int, int]:
    pred = pred_bin.view(-1)
    target = target_bin.view(-1)

    tp = int((pred * target).sum().item())
    tn = int(((1 - pred) * (1 - target)).sum().item())
    fp = int(((1 - target) * pred).sum().item())
    fn = int((target * (1 - pred)).sum().item())
    return tp, fp, tn, fn


def metrics_from_counts(tp: float, fp: float, tn: float, fn: float) -> Dict[str, float]:
    eps = 1e-6
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
    }


def build_zone_level_results(patch_df: pd.DataFrame, metadata_csv: str, zone_column: str) -> pd.DataFrame:
    if patch_df.empty:
        return pd.DataFrame()
    if not os.path.exists(metadata_csv):
        return pd.DataFrame()

    meta_df = pd.read_csv(metadata_csv)
    if "patch_name" not in meta_df.columns or zone_column not in meta_df.columns:
        return pd.DataFrame()

    merged = patch_df.merge(
        meta_df[["patch_name", zone_column]],
        on="patch_name",
        how="left",
    )
    merged[zone_column] = merged[zone_column].fillna("unknown").replace("", "unknown")

    zone_rows = []
    for zone_value, group in merged.groupby(zone_column):
        tp = float(group["tp"].sum())
        fp = float(group["fp"].sum())
        tn = float(group["tn"].sum())
        fn = float(group["fn"].sum())
        m = metrics_from_counts(tp, fp, tn, fn)
        pred_area = float(group["pred_area"].sum())
        true_area = float(group["true_area"].sum())
        zone_rows.append(
            {
                "zone_name": str(zone_value),
                "num_patches": int(len(group)),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "iou": m["iou"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "dice": m["dice"],
                "pred_area": pred_area,
                "true_area": true_area,
                "area_bias": float((pred_area - true_area) / (true_area + 1e-6)),
            }
        )

    return pd.DataFrame(zone_rows).sort_values("zone_name")


def _tta_inference(model, imgs, device):
    """4-way test-time augmentation: original, hflip, vflip, hflip+vflip.

    Returns probability maps averaged across all flips.
    """
    probs = torch.zeros(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3], device=device)

    # Original
    probs += torch.sigmoid(model(imgs))

    # Horizontal flip
    imgs_hf = torch.flip(imgs, [3])
    probs += torch.flip(torch.sigmoid(model(imgs_hf)), [3])

    # Vertical flip
    imgs_vf = torch.flip(imgs, [2])
    probs += torch.flip(torch.sigmoid(model(imgs_vf)), [2])

    # H+V flip (180° rotation equivalent)
    imgs_hvf = torch.flip(imgs, [2, 3])
    probs += torch.flip(torch.sigmoid(model(imgs_hvf)), [2, 3])

    return probs / 4.0


def _apply_morphology(pred_np, kernel_size=3, operation="open"):
    """Apply morphological post-processing to clean binary predictions.

    - open: remove isolated false positives (erosion + dilation)
    - close: fill small holes in water bodies (dilation + erosion)
    - both: open then close
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operation == "open":
        return cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        return cv2.morphologyEx(pred_np, cv2.MORPH_CLOSE, kernel)
    elif operation == "both":
        opened = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return pred_np


def evaluate(cfg: EvalConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(cfg.output_dir, exist_ok=True)

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    if not os.path.exists(cfg.data_dir):
        raise FileNotFoundError(f"Dataset not found: {cfg.data_dir}")

    eval_files = get_eval_files(
        cfg.data_dir,
        cfg.split_mode,
        cfg.seed,
        cfg.val_percent,
        cfg.split_file,
    )

    dataset = PatchDataset(cfg.data_dir, eval_files, cfg.in_channels)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = build_model(cfg.model_name, n_channels=cfg.in_channels, n_classes=1)
    state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    global_tp = 0
    global_fp = 0
    global_tn = 0
    global_fn = 0

    total_pred_area = 0.0
    total_true_area = 0.0
    total_infer_sec = 0.0

    patch_rows = []

    with torch.no_grad():
        for imgs, masks, names in loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            masks = masks.to(cfg.device, non_blocking=True)

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            tic = time.perf_counter()

            if cfg.use_tta:
                probs = _tta_inference(model, imgs, cfg.device)
            else:
                probs = torch.sigmoid(model(imgs))

            if cfg.device == "cuda":
                torch.cuda.synchronize()
            toc = time.perf_counter()

            batch_infer_sec = toc - tic
            total_infer_sec += batch_infer_sec
            per_patch_sec = batch_infer_sec / max(1, imgs.shape[0])

            binary = (probs > cfg.threshold).float()
            targets = (masks > 0.5).float()

            if cfg.use_morphology:
                preds = binary.clone()
                for i in range(binary.shape[0]):
                    cleaned = _apply_morphology(
                        binary[i, 0].cpu().numpy().astype(np.uint8),
                        kernel_size=cfg.morph_kernel_size,
                        operation=cfg.morph_operation,
                    )
                    preds[i, 0] = torch.from_numpy(cleaned).float().to(cfg.device)
            else:
                preds = binary

            for i in range(imgs.shape[0]):
                pred_i = preds[i]
                targ_i = targets[i]

                tp, fp, tn, fn = confusion_counts(pred_i, targ_i)
                metrics = metrics_from_counts(tp, fp, tn, fn)

                pred_area = float(pred_i.sum().item())
                true_area = float(targ_i.sum().item())
                area_bias = (pred_area - true_area) / (true_area + 1e-6)

                global_tp += tp
                global_fp += fp
                global_tn += tn
                global_fn += fn

                total_pred_area += pred_area
                total_true_area += true_area

                patch_rows.append(
                    {
                        "patch_name": names[i],
                        "model_name": cfg.model_name,
                        "data_domain": cfg.data_domain,
                        "in_channels": cfg.in_channels,
                        "threshold": cfg.threshold,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "iou": metrics["iou"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "dice": metrics["dice"],
                        "pred_area": pred_area,
                        "true_area": true_area,
                        "area_bias": area_bias,
                        "inference_time_ms": per_patch_sec * 1000.0,
                    }
                )

    global_metrics = metrics_from_counts(global_tp, global_fp, global_tn, global_fn)
    global_area_bias = (total_pred_area - total_true_area) / (total_true_area + 1e-6)

    summary_row = {
        "experiment_name": cfg.experiment_name,
        "model_name": cfg.model_name,
        "data_domain": cfg.data_domain,
        "in_channels": cfg.in_channels,
        "split_mode": cfg.split_mode,
        "num_patches": len(patch_rows),
        "threshold": cfg.threshold,
        "tp": global_tp,
        "fp": global_fp,
        "tn": global_tn,
        "fn": global_fn,
        "iou": global_metrics["iou"],
        "precision": global_metrics["precision"],
        "recall": global_metrics["recall"],
        "f1": global_metrics["f1"],
        "dice": global_metrics["dice"],
        "area_bias": float(global_area_bias),
        "total_pred_area": float(total_pred_area),
        "total_true_area": float(total_true_area),
        "total_inference_sec": float(total_infer_sec),
        "avg_inference_ms": float((total_infer_sec / max(1, len(patch_rows))) * 1000.0),
        "checkpoint_path": cfg.checkpoint_path,
        "data_dir": cfg.data_dir,
        "seed": cfg.seed,
        "val_percent": cfg.val_percent,
    }

    summary_df = pd.DataFrame([summary_row])
    patch_df = pd.DataFrame(patch_rows)
    zone_df = build_zone_level_results(
        patch_df=patch_df,
        metadata_csv=cfg.metadata_csv,
        zone_column=cfg.zone_column,
    )
    return summary_df, patch_df, zone_df


def save_outputs(cfg: EvalConfig, summary_df: pd.DataFrame, patch_df: pd.DataFrame, zone_df: pd.DataFrame):
    os.makedirs(cfg.output_dir, exist_ok=True)

    summary_path = os.path.join(cfg.output_dir, "evaluation_summary.csv")
    patch_path = os.path.join(cfg.output_dir, "patch_level_results.csv")
    zone_path = os.path.join(cfg.output_dir, "zone_level_results.csv")
    cfg_path = os.path.join(cfg.output_dir, "evaluation_config.json")

    summary_df.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    patch_df.to_csv(patch_path, index=False, quoting=csv.QUOTE_MINIMAL)
    if not zone_df.empty:
        zone_df.to_csv(zone_path, index=False, quoting=csv.QUOTE_MINIMAL)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print("=" * 72)
    print("Evaluation completed")
    print(f"summary: {summary_path}")
    print(f"patches: {patch_path}")
    if not zone_df.empty:
        print(f"zones  : {zone_path}")
    print(f"config : {cfg_path}")
    print("=" * 72)
    print(summary_df.to_string(index=False))


def main():
    args = parse_args()
    train_cfg = load_train_config(args.experiment_name)
    cfg = resolve_eval_config(args, train_cfg)

    summary_df, patch_df, zone_df = evaluate(cfg)
    save_outputs(cfg, summary_df, patch_df, zone_df)


if __name__ == "__main__":
    main()
