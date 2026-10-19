"""
Full-basin sliding-window inference.
Loads the best model weights, scans the entire study area, and mosaics
patch predictions into a single full-basin GeoTIFF.

Usage:
  python inference_full_basin.py --exp van_050105_6
  python inference_full_basin.py --exp res_050105_6
  python inference_full_basin.py --exp van_050105_6 --stride 128 --batch-size 32
"""
import argparse, json, os, sys, warnings
import numpy as np
import torch
from tqdm import tqdm

# Add repository root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
OUTPUT_DIR  = "./output"
SAVED_DIR   = "./saved_models"
RAW_DIR     = "./raw_data"
PATCH_SIZE  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Geotransform from label.tfw
GEO_PIXEL_W = 0.0008983153
GEO_PIXEL_H = 0.0008983153
GEO_UL_LON  = 113.5625585967
GEO_UL_LAT  = 40.0525988474

# Normalization parameters (matching preprocess.py)
NORM_PARAMS = {
    "dem_max": 2000.0, "slope_max": 45.0, "rain_max": 300.0,
    "soil_max": 0.6,   "pa_max": 200.0,
    "rain_1d_max": 150.0, "rain_3d_max": 300.0,
}
CHANNEL_ORDER = ["dem", "slope", "rain", "init", "soil", "pa", "rain_1d", "rain_3d"]
FILE_NAMES = {
    "dem": "Input_DEM.tif", "slope": "Input_Slope.tif",
    "rain": "Input_Rain_Total_GSMaP.tif", "init": "init.tif",
    "soil": "Input_Soil_SMAP.tif", "pa": "Input_Pa.tif",
    "rain_1d": "Input_Rain_1d.tif", "rain_3d": "Input_Rain_3d.tif",
    "label": "label.tif",
}


# ═══════════════════════════════════════════════════════════════════════
def load_raster(key):
    """Load a raw raster using tifffile (same as preprocess.py)."""
    import tifffile
    path = os.path.join(RAW_DIR, FILE_NAMES.get(key, ""))
    if not os.path.exists(path):
        return None
    try:
        arr = tifffile.imread(path)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return arr
    except Exception:
        return None


def normalize_channel(arr, key):
    """Normalize channel to [0,1] (matching preprocess.py)."""
    if arr is None:
        return None
    arr = arr.copy()
    if key == "dem":
        arr[arr < 0] = 0
        arr = np.log1p(arr) / np.log1p(NORM_PARAMS["dem_max"])
        return np.clip(arr, 0, 1)
    if key == "slope":
        arr[arr < 0] = 0
        return np.clip(arr / NORM_PARAMS["slope_max"], 0, 1)
    if key == "soil":
        arr[arr < 0] = 0
        return np.clip(arr / NORM_PARAMS["soil_max"], 0, 1)
    if key in ("rain", "pa", "rain_1d", "rain_3d"):
        arr[arr < 0] = 0
        max_val = NORM_PARAMS.get(f"{key}_max", 300.0)
        return np.clip(arr / max_val, 0, 1)
    if key in ("init", "label"):
        return (arr > 0.5).astype(np.float32)
    return arr


def load_model(exp_name):
    """Load model weights and config."""
    from models.models import build_model

    cfg_path = os.path.join(OUTPUT_DIR, exp_name, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found for {exp_name}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    n_channels = int(cfg.get("IN_CHANNELS", 6))
    model_name = str(cfg.get("MODEL_NAME", "vanilla_unet"))

    pth = os.path.join(SAVED_DIR, exp_name, "latest_best.pth")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Model weights not found: {pth}")

    model = build_model(model_name, n_channels=n_channels, n_classes=1)
    model.load_state_dict(torch.load(pth, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    print(f"Model: {model_name} | {n_channels}ch | device={DEVICE}")
    return model, n_channels


def infer_full_basin(exp_name, stride=64, batch_size=16):
    """Sliding-window full-basin inference, output GeoTIFF."""
    model, n_channels = load_model(exp_name)

    # ── Load and normalize all channels ───────────────────────────────
    print("Loading raw rasters ...")
    raw = {}
    for key in CHANNEL_ORDER[:n_channels]:
        arr = load_raster(key)
        if arr is not None:
            raw[key] = normalize_channel(arr, key)
            print(f"  {key}: {arr.shape} → norm [{raw[key].min():.3f}, {raw[key].max():.3f}]")
        else:
            print(f"  {key}: MISSING (will use zeros)")

    # Determine output shape from the first available channel
    ref_key = next(k for k in CHANNEL_ORDER if k in raw)
    H, W = raw[ref_key].shape
    print(f"Output grid: {H} × {W}")

    # ── Build input stack (padded to multiple of stride for clean tiling) ──
    pad_h = (PATCH_SIZE - H % stride) % stride + PATCH_SIZE
    pad_w = (PATCH_SIZE - W % stride) % stride + PATCH_SIZE
    H_pad, W_pad = H + pad_h, W + pad_w

    stack = np.zeros((n_channels, H_pad, W_pad), dtype=np.float32)
    for i, key in enumerate(CHANNEL_ORDER[:n_channels]):
        if key in raw:
            stack[i, :H, :W] = raw[key]

    # ── Sliding window inference ──────────────────────────────────────
    prob_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    count_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    y_starts = list(range(0, H_pad - PATCH_SIZE + 1, stride))
    x_starts = list(range(0, W_pad - PATCH_SIZE + 1, stride))
    total_patches = len(y_starts) * len(x_starts)
    print(f"Patches: {len(y_starts)} × {len(x_starts)} = {total_patches:,}  "
          f"(stride={stride}, batch={batch_size})")

    # Batch processing
    batch_tensors = []
    batch_positions = []

    for y in tqdm(y_starts, desc="Inference"):
        for x in x_starts:
            patch = stack[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            batch_tensors.append(torch.from_numpy(patch))
            batch_positions.append((y, x))

            if len(batch_tensors) >= batch_size:
                _flush_batch(model, batch_tensors, batch_positions,
                             prob_map, count_map, weight)
                batch_tensors.clear()
                batch_positions.clear()

    if batch_tensors:
        _flush_batch(model, batch_tensors, batch_positions,
                     prob_map, count_map, weight)

    # ── Average overlapping predictions ───────────────────────────────
    valid_mask = count_map > 0
    prob_map[valid_mask] /= count_map[valid_mask]
    prob_map = prob_map[:H, :W]       # crop to original extent
    count_map = count_map[:H, :W]
    pred_bin = (prob_map > 0.5).astype(np.float32)

    print(f"Prob range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    print(f"Flood pixels (p>0.5): {pred_bin.sum():,} / {H*W:,}  "
          f"({pred_bin.sum()/H/W*100:.2f}%)")

    # ── Save outputs ──────────────────────────────────────────────────
    out_dir = os.path.join(OUTPUT_DIR, exp_name, "inference_full_basin")
    os.makedirs(out_dir, exist_ok=True)

    _save_geotiff(os.path.join(out_dir, "probability.tif"), prob_map,
                  GEO_UL_LON, GEO_UL_LAT, GEO_PIXEL_W, GEO_PIXEL_H)
    _save_geotiff(os.path.join(out_dir, "prediction_binary.tif"), pred_bin,
                  GEO_UL_LON, GEO_UL_LAT, GEO_PIXEL_W, GEO_PIXEL_H)
    _save_geotiff(os.path.join(out_dir, "coverage_count.tif"), count_map,
                  GEO_UL_LON, GEO_UL_LAT, GEO_PIXEL_W, GEO_PIXEL_H)

    # Also save as .npy for quick loading
    np.save(os.path.join(out_dir, "probability.npy"), prob_map)
    np.save(os.path.join(out_dir, "prediction_binary.npy"), pred_bin)

    print(f"\nSaved to: {out_dir}/")
    print("  probability.tif      — per-pixel flood probability (0–1)")
    print("  prediction_binary.tif — binary flood mask (threshold=0.5)")
    print("  coverage_count.tif    — number of overlapping patches per pixel")

    # ── Summary stats ─────────────────────────────────────────────────
    pct_05 = (prob_map > 0.5).mean() * 100
    pct_03 = (prob_map > 0.3).mean() * 100
    pct_07 = (prob_map > 0.7).mean() * 100
    print(f"\nFlood coverage at different thresholds:")
    print(f"  p>0.3: {pct_03:.2f}%")
    print(f"  p>0.5: {pct_05:.2f}%")
    print(f"  p>0.7: {pct_07:.2f}%")


def _flush_batch(model, tensors, positions, prob_map, count_map, weight):
    """Run one batch through the model and accumulate probability map."""
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)

    for prob, (y, x) in zip(probs, positions):
        prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob * weight
        count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += weight


def _save_geotiff(path, arr, ul_lon, ul_lat, pix_w, pix_h):
    """Save a 2D array as GeoTIFF with georeferencing."""
    try:
        import rasterio
        from rasterio.transform import from_origin
        transform = from_origin(ul_lon, ul_lat, pix_w, pix_h)
        with rasterio.open(
            path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=1, dtype=arr.dtype.name, crs="EPSG:4326", transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(arr, 1)
    except ImportError:
        # Fallback: save as .tif via tifffile (no CRS)
        import tifffile
        tifffile.imwrite(path, arr.astype(np.float32), compress="zlib")
        print(f"  (rasterio not available, saved without CRS via tifffile)")


# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Full-basin flood inference")
    parser.add_argument("--exp", required=True, help="Experiment name (e.g. van_050105_6)")
    parser.add_argument("--stride", type=int, default=64, help="Sliding window stride")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", default=DEVICE, help="Torch device")
    args = parser.parse_args()

    import torch
    dev = args.device

    print(f"\n{'='*60}")
    print(f"Full-Basin Inference: {args.exp}")
    print(f"Stride={args.stride}  Batch={args.batch_size}  Device={dev}")
    print(f"{'='*60}\n")

    infer_full_basin(args.exp, stride=args.stride, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
