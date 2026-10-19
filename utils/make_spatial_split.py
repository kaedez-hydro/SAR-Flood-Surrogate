import argparse
import os
import random
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create spatially independent train/val split from patch metadata."
    )
    parser.add_argument("--metadata-csv", required=True, help="Path to patch_metadata.csv")
    parser.add_argument("--output-dir", default="splits", help="Directory for split outputs")
    parser.add_argument("--split-name", default="spatial_split", help="Prefix for output files")
    parser.add_argument(
        "--strategy",
        choices=["vertical_band", "horizontal_band", "zone_block_random"],
        default="zone_block_random",
        help="Spatial split strategy",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Approximate validation ratio",
    )
    parser.add_argument(
        "--buffer-cells",
        type=int,
        default=1,
        help="Gap (in patch-grid cells) between train and val (for band strategies)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Patch stride in pixels (from preprocess.py). Used for grid-based strategies.",
    )
    parser.add_argument(
        "--block-grid",
        type=int,
        default=8,
        help="Grid cells per spatial block for zone_block_random (default 8 = 512 px).",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Use only rows with valid_flag==1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Legacy band strategies (unchanged)
# ---------------------------------------------------------------------------

def _band_threshold(values, val_ratio: float) -> Tuple[float, float]:
    sorted_vals = sorted(values.unique())
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("No spatial coordinates found for split.")
    n_val = max(1, int(round(n * val_ratio)))
    start_idx = max(0, n - n_val)
    val_start = sorted_vals[start_idx]
    val_end = sorted_vals[-1]
    return val_start, val_end


def build_band_split(
    df: pd.DataFrame, strategy: str, val_ratio: float, buffer_cells: int
) -> pd.DataFrame:
    if "patch_name" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        raise ValueError("metadata csv must include columns: patch_name, x, y")

    axis = "x" if strategy == "vertical_band" else "y"
    val_start, val_end = _band_threshold(df[axis], val_ratio)

    df = df.copy()
    df["split"] = "exclude"

    val_mask = (df[axis] >= val_start) & (df[axis] <= val_end)
    df.loc[val_mask, "split"] = "val"

    if buffer_cells > 0:
        unique_axis = sorted(df[axis].unique())
        val_positions = [v for v in unique_axis if val_start <= v <= val_end]
        if val_positions:
            pos_to_idx = {v: i for i, v in enumerate(unique_axis)}
            left_idx = max(0, pos_to_idx[val_positions[0]] - buffer_cells)
            right_idx = min(
                len(unique_axis) - 1,
                pos_to_idx[val_positions[-1]] + buffer_cells,
            )
            buffer_min = unique_axis[left_idx]
            buffer_max = unique_axis[right_idx]
            buffer_mask = (df[axis] >= buffer_min) & (df[axis] <= buffer_max)
            df.loc[~buffer_mask, "split"] = "train"
        else:
            df.loc[~val_mask, "split"] = "train"
    else:
        df.loc[~val_mask, "split"] = "train"

    return df


# ---------------------------------------------------------------------------
# Zone-block-random strategy
# ---------------------------------------------------------------------------

def build_zone_block_random(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    stride: int = 64,
    block_grid: int = 8,
) -> pd.DataFrame:
    """Within each named zone, assign spatial blocks to train/val.

    Guarantees zero pixel overlap: with block_grid >= 4, any two different
    blocks are >= 4 grid units apart in at least one dimension, so patches
    from different blocks can never overlap (overlap requires both dims < 4).

    Val blocks are selected >= 2 blocks apart for spatial distribution.
    Patches outside any named zone are assigned to train.
    """
    required = {"patch_name", "x", "y", "zone_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata csv missing columns: {missing}")

    df = df.copy()

    # Convert pixel coords to patch-grid coords (1 grid unit = stride px)
    df["gy"] = (df["y"] // stride).astype(int)
    df["gx"] = (df["x"] // stride).astype(int)
    # Group grid positions into coarse blocks
    df["block_y"] = df["gy"] // block_grid
    df["block_x"] = df["gx"] // block_grid
    df["block_id"] = list(zip(df["block_y"], df["block_x"]))

    rows = []
    # Valid zones → spatial blocking
    zone_names = sorted(
        z for z in df["zone_name"].dropna().unique()
        if str(z).strip() not in ("", "unknown", "nan")
    )
    for zone in zone_names:
        mask = df["zone_name"] == zone
        _assign_blocks(df.loc[mask], val_ratio, block_grid, rows)

    # Patches outside any named zone (NaN / empty) → all train
    # Use .isna() because NaN != NaN would silently drop them otherwise.
    outside_mask = df["zone_name"].isna() | df["zone_name"].astype(str).str.strip().isin(["", "nan", "unknown"])
    for _, row in df.loc[outside_mask].iterrows():
        rows.append({
            "patch_name": row["patch_name"],
            "x": int(row["x"]),
            "y": int(row["y"]),
            "split": "train",
        })

    return pd.DataFrame(rows).sort_values(["y", "x"]).reset_index(drop=True)


def _assign_blocks(zone_df, val_ratio, block_grid, rows):
    """Assign blocks within one zone to train / val.

    Val blocks are selected with >= 2-block separation in at least one
    dimension so they are spatially distributed. Non-val blocks go to train.

    No exclude buffer: with block_grid >= 4, any two different blocks already
    have >= 4 grid-unit separation in one dimension, so patches from different
    blocks can never overlap (overlap requires both dimensions < 4).
    """

    block_counts = zone_df.groupby("block_id").size().to_dict()
    unique_blocks = sorted(block_counts.keys())

    random.shuffle(unique_blocks)

    val_blocks = set()
    total_patches = len(zone_df)
    val_patches = 0
    target_val = int(total_patches * val_ratio)

    for bid in unique_blocks:
        by, bx = bid
        safe = True
        for vby, vbx in val_blocks:
            dy = abs(by - vby)
            dx = abs(bx - vbx)
            if dy < 2 and dx < 2:
                safe = False
                break
        if safe:
            val_blocks.add(bid)
            val_patches += block_counts[bid]
        if val_patches >= target_val:
            break

    for _, row in zone_df.iterrows():
        split = "val" if row["block_id"] in val_blocks else "train"
        rows.append({
            "patch_name": row["patch_name"],
            "x": int(row["x"]),
            "y": int(row["y"]),
            "split": split,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.metadata_csv)
    if args.valid_only and "valid_flag" in df.columns:
        df = df[df["valid_flag"] == 1].copy()

    if args.strategy in ("vertical_band", "horizontal_band"):
        split_df = build_band_split(df, args.strategy, args.val_ratio, args.buffer_cells)
    elif args.strategy == "zone_block_random":
        split_df = build_zone_block_random(
            df,
            val_ratio=args.val_ratio,
            stride=args.stride,
            block_grid=args.block_grid,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    os.makedirs(args.output_dir, exist_ok=True)
    split_csv = os.path.join(args.output_dir, f"{args.split_name}.csv")
    train_txt = os.path.join(args.output_dir, f"{args.split_name}_train.txt")
    val_txt = os.path.join(args.output_dir, f"{args.split_name}_val.txt")

    split_df[["patch_name", "split", "x", "y"]].to_csv(split_csv, index=False)

    train_names = split_df.loc[split_df["split"] == "train", "patch_name"].astype(str).tolist()
    val_names = split_df.loc[split_df["split"] == "val", "patch_name"].astype(str).tolist()

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_names) + ("\n" if train_names else ""))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_names) + ("\n" if val_names else ""))

    n_exclude = int((split_df["split"] == "exclude").sum())
    print("=" * 72)
    print("Spatial split generated")
    print(f"split csv : {split_csv}")
    print(f"train txt : {train_txt}")
    print(f"val txt   : {val_txt}")
    print(f"strategy  : {args.strategy}")
    print(
        f"counts    : train={len(train_names)}, val={len(val_names)}, "
        f"exclude={n_exclude}"
    )
    print(f"val ratio : {len(val_names) / (len(train_names) + len(val_names)):.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
