"""
Shared infrastructure for the FLIM T-cell tracking pipeline.

Contains everything the entry script (`run_pipeline.py`) needs:
  - Frame loaders (built-in PBS1/2DG datasets + a generic loader for user data)
  - CellSAM detection (lazy-loaded, cached)
  - Preprocessing, mask post-processing, MOTA evaluation, video rendering

The centroid Hungarian tracker itself lives in `run_pipeline.py`.
"""
from pathlib import Path
from functools import lru_cache
import glob
import re
import subprocess
import colorsys

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_triangle
import tifffile
import cv2


# ── color palette for mask overlays ──────────────────────────────────────────
# HSV-sweep gives ~evenly spaced hues so neighbouring track IDs look distinct.
def _make_palette(n=256):
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 0.95)
        # OpenCV uses BGR order
        out.append((int(b * 255), int(g * 255), int(r * 255)))
    return out

PALETTE = _make_palette(256)
FONT    = cv2.FONT_HERSHEY_SIMPLEX


# ── built-in dataset configs (for reproducing the paper benchmarks) ──────────
# Users bringing their own data should pass parameters through the CLI instead.
DATASETS = {
    "pbs1": dict(
        n_frames            = 15,
        fps                 = 3,
        min_area            = 50,
        max_area            = 900,
        merge_dil           = 3,
        max_dist            = 40,
        cellsam_bbox_thresh = 0.65,
    ),
    "2dg": dict(
        n_frames            = 10,
        fps                 = 2,
        min_area            = 40,
        max_area            = 2000,
        merge_dil           = 3,
        max_dist            = 40,
        # 2DG TCSPC noise creates persistent structured blobs; lower threshold
        # to recall dim cells, accepting a higher FP rate.
        cellsam_bbox_thresh = 0.50,
    ),
}

BASE_PBS1 = Path("segmentation_tracking/PBS_1")
BASE_2DG  = Path("segmentation_tracking/2DG_m1s3v1-017")


# ── built-in frame loaders ───────────────────────────────────────────────────
def load_pbs1(frame_idx):
    """Load one PBS1 intensity frame (Ch1 .ome.tif) by frame index."""
    folder = str(BASE_PBS1 / f"PBS_1-{frame_idx+1:03d}")
    files  = glob.glob(f"{folder}/*_Cycle00001_Ch1_000001.ome.tif")
    if not files:
        raise FileNotFoundError(f"No Ch1 tif in {folder}")
    return tifffile.imread(files[0]).astype(np.float32).squeeze()


def load_2dg(frame_idx):
    """Load one 2DG intensity frame by summing the TCSPC decay histogram."""
    import sdtfile
    f    = frame_idx + 1
    path = BASE_2DG / f"LifetimeData_Cycle{f:05d}_000001.sdt"
    arr  = np.array(sdtfile.SdtFile(str(path)).data[0])   # (3, H, W, 256)
    return arr[0].sum(axis=-1).astype(np.float32)


def load_gt_pbs1(frame_idx):
    path = BASE_PBS1 / f"analysis/Playground/PBS_1-{frame_idx+1:03d}_cellpose.tiff"
    return tifffile.imread(str(path))


def load_gt_2dg(frame_idx):
    f    = frame_idx + 1
    path = BASE_2DG / f"analysis/Playground/2DG_m1s3v1-017_Cycle{f:05d}_Ch1_000001_cellpose.tiff"
    return tifffile.imread(str(path))


# ── generic loader for user-supplied data ────────────────────────────────────
def _natural_key(s):
    """Sort '*_10' after '*_2' by treating digit runs as numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def load_generic_frames(path, channel=None):
    """
    Load a sequence of 2D intensity frames from a user-supplied path.

    Accepts:
      - A multi-page TIFF (shape (T, H, W) or (T, H, W, C))
      - A directory of single-frame TIFFs (sorted naturally by filename)

    Parameters
    ----------
    path : str | Path
    channel : int | None
        1-indexed channel selector. For a directory of OME-TIFFs named
        `*_Ch{N}_*.tif`, only files matching that channel are loaded — required
        when the folder mixes channels (e.g. mCherry/GFP/NADH in one folder),
        otherwise the natural sort would interleave channels as if they were
        separate frames. For a 4D `(T, H, W, C)` multi-page TIFF, picks
        `frame[..., channel-1]`. `None` keeps the legacy behaviour (first
        channel, no filtering).

    Returns a list of 2D float32 arrays (one per frame).
    """
    p = Path(path)
    if p.is_file():
        arr = tifffile.imread(str(p))
        if arr.ndim == 2:
            return [arr.astype(np.float32)]
        if arr.ndim == 3:
            return [frame.astype(np.float32) for frame in arr]
        if arr.ndim == 4:
            ci = (channel - 1) if channel is not None else 0
            return [frame[..., ci].astype(np.float32) for frame in arr]
        raise ValueError(f"Unsupported TIFF shape {arr.shape} in {p}")

    if p.is_dir():
        files = [f for f in p.iterdir() if f.suffix.lower() in {".tif", ".tiff"}]
        if channel is not None:
            ch_re = re.compile(rf"_Ch{int(channel)}_", re.IGNORECASE)
            files = [f for f in files if ch_re.search(f.name)]
            if not files:
                raise FileNotFoundError(
                    f"No .tif/.tiff files matching _Ch{channel}_ in {p}"
                )
        files = sorted(files, key=lambda f: _natural_key(f.name))
        if not files:
            raise FileNotFoundError(f"No .tif/.tiff files in {p}")
        frames = []
        for f in files:
            img = tifffile.imread(str(f)).astype(np.float32).squeeze()
            if img.ndim == 3:              # drop channel dim if present
                img = img[..., 0] if img.shape[-1] <= 4 else img[0]
            frames.append(img)
        return frames

    raise FileNotFoundError(f"No such file or directory: {p}")


# ── preprocessing ────────────────────────────────────────────────────────────
def preprocess_raw_norm(img):
    """Percentile normalize [p1, p99.8] → [0, 1]. No median, no sqrt.

    Keeps the full cell-to-background signal ratio (≈14× on PBS1, vs ≈4×
    after a √-transform) which helps CellSAM's detector lock onto cells.
    """
    arr    = img.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99.8])
    span   = hi - lo or 1.0
    return np.clip((arr - lo) / span, 0.0, 1.0).astype(np.float32)


def preprocess_raw_norm_smooth(img, sigma=0.5, sigma_bg=20.0, sigma_norm=40.0,
                               fg_pct=99.7):
    """Low-SNR-friendly preprocess with **local** background subtraction and
    **local** dynamic-range normalization. Designed for dim fluorescence where
    different parts of the frame have very different signal levels (e.g. dense
    cell cluster + dim distal cells in the same field of view).

    Steps:
      1. light Gaussian blur (`sigma`) for shot-noise suppression
      2. subtract a slowly-varying background (Gaussian blur with `sigma_bg`,
         bigger than any cell so it averages out)
      3. divide by a slowly-varying foreground envelope (`sigma_norm`) so dim
         regions get amplified to comparable contrast as bright regions
      4. linear-stretch to [0,1] at the `fg_pct` percentile

    A previous variant subtracted only a global median, which on
    high-dynamic-range frames (e.g. GFP-zmel 3.1mW) clipped dim half-frame
    cells to zero. The local normalization fixes that — verified visually."""
    arr   = gaussian_filter(img.astype(np.float32), sigma=sigma)
    bg    = gaussian_filter(arr, sigma=sigma_bg)
    sub   = np.clip(arr - bg, 0.0, None)
    sub   = gaussian_filter(sub, sigma=sigma)
    env   = gaussian_filter(sub, sigma=sigma_norm)
    floor = float(np.percentile(sub, 95)) * 0.1
    env   = np.maximum(env, floor)
    norm  = sub / env
    span  = float(np.percentile(norm, fg_pct)) or 1.0
    return np.clip(norm / span, 0.0, 1.0).astype(np.float32)


# ── display helpers ──────────────────────────────────────────────────────────
def raw_display(img):
    """Render a raw frame for video: threshold-background to black, stretch to 8-bit."""
    thresh = threshold_triangle(img)
    out    = np.where(img > thresh, img.astype(np.float32), 0.0)
    sig    = out[out > 0]
    if sig.size == 0:
        return np.zeros((*img.shape, 3), dtype=np.uint8)
    p995 = np.percentile(sig, 99.5) or 1.0
    scaled = np.clip(out / p995 * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def draw_masks(canvas, labels, tid_map=None, alpha=0.5):
    """Overlay filled translucent masks + outlines + track-ID numbers."""
    out     = canvas.copy()
    overlay = np.zeros_like(out)

    # pass 1: paint translucent fills
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        tid   = tid_map[lbl] if (tid_map and lbl in tid_map) else int(lbl)
        overlay[labels == lbl] = PALETTE[tid % len(PALETTE)]
    out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)

    # pass 2: draw solid outline + centroid text
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        tid    = tid_map[lbl] if (tid_map and lbl in tid_map) else int(lbl)
        color  = PALETTE[tid % len(PALETTE)]
        mask   = labels == lbl
        border = binary_dilation(mask, iterations=2) & ~mask
        out[border] = color
        ys, xs = np.where(mask)
        cv2.putText(out, str(tid),
                    (int(xs.mean()) - 8, int(ys.mean()) + 5),
                    FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── mask post-processing ─────────────────────────────────────────────────────
def touch_merge(labels, dil):
    """Union-find merge of labels whose dilated footprints touch.

    Guards against CellSAM occasionally splitting one cell into two fragments.
    """
    ids    = [i for i in np.unique(labels) if i != 0]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for lbl in ids:
        expanded = binary_dilation(labels == lbl, iterations=dil)
        for nb in np.unique(labels[expanded & (labels != lbl) & (labels != 0)]):
            union(lbl, int(nb))

    out, remap, counter = np.zeros_like(labels), {}, 1
    for lbl in ids:
        root = find(lbl)
        if root not in remap:
            remap[root] = counter
            counter += 1
        out[labels == lbl] = remap[root]
    return out


def area_filter(labels, min_area, max_area):
    """Drop labels whose pixel area falls outside [min_area, max_area]."""
    filtered = np.zeros_like(labels)
    new_lbl  = 1
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        area = (labels == lbl).sum()
        if min_area <= area <= max_area:
            filtered[labels == lbl] = new_lbl
            new_lbl += 1
    return filtered


# ── centroids + MOTA evaluation ──────────────────────────────────────────────
def centroids(lbl_img):
    """Return {label: (y, x)} for every non-zero label in an integer mask."""
    out = {}
    for lbl in np.unique(lbl_img):
        if lbl == 0:
            continue
        ys, xs = np.where(lbl_img == lbl)
        out[int(lbl)] = (float(ys.mean()), float(xs.mean()))
    return out


def compute_mota(pred_frames, gt_frames, max_match_dist=30):
    """MOTA = 1 - (FP + FN + IDSW) / ΣGT, matching preds↔GT by centroid Hungarian per frame."""
    total_gt = total_fp = total_fn = total_idsw = 0
    prev_match = {}

    for our_lbl, gt_lbl in zip(pred_frames, gt_frames):
        gt_cents  = centroids(gt_lbl)
        our_cents = centroids(our_lbl)
        total_gt += len(gt_cents)

        # Edge case: no GT or no predictions — everything is FP/FN
        if not gt_cents or not our_cents:
            total_fn += len(gt_cents)
            total_fp += len(our_cents)
            prev_match = {}
            continue

        gids = list(gt_cents.keys())
        dids = list(our_cents.keys())
        cost = np.full((len(gids), len(dids)), 1e9)
        for i, g in enumerate(gids):
            for j, d in enumerate(dids):
                gy, gx = gt_cents[g]
                dy, dx = our_cents[d]
                cost[i, j] = np.hypot(gy - dy, gx - dx)
        ri, ci = linear_sum_assignment(cost)
        frame_match = {gids[r]: dids[c] for r, c in zip(ri, ci)
                       if cost[r, c] <= max_match_dist}

        total_fn += len(gt_cents)  - len(frame_match)
        total_fp += len(our_cents) - len(frame_match)
        # ID switch: the same GT cell was matched to a different track ID last frame
        for g, d in frame_match.items():
            if g in prev_match and prev_match[g] != d:
                total_idsw += 1
        prev_match = frame_match

    denom = max(total_gt, 1)
    mota  = 1.0 - (total_fp + total_fn + total_idsw) / denom
    return dict(MOTA=mota, FP=total_fp, FN=total_fn, IDSW=total_idsw,
                GT=total_gt,
                Recall    = 1 - total_fn / denom,
                Precision = total_fp / denom)


# ── video rendering ──────────────────────────────────────────────────────────
def render_and_write_video(cfg, raws, tracked, tid_maps, gts=None,
                           method_label="Ours", n_interp=4):
    """Render a side-by-side video: Raw │ Tracked masks │ (optional) GT masks.

    n_interp: number of blended transition frames inserted between each real
              biological frame. Encodes at fps * (n_interp + 1) so each real
              frame is held for the same wall-clock duration.
    """
    n        = len(raws)
    H, W     = raws[0].shape
    panels   = 3 if gts is not None else 2
    out_fps  = float(cfg["fps"]) * (n_interp + 1)
    fourcc   = cv2.VideoWriter_fourcc(*"MJPG")
    writer   = cv2.VideoWriter(cfg["out_avi"], fourcc, out_fps, (W * panels, H))

    def make_panel(i):
        # raw_display is non-cheap (threshold + percentile) — compute once per frame.
        # draw_masks already takes its own copy, so passing `bright` directly is safe.
        bright = raw_display(raws[i])
        our    = draw_masks(bright, tracked[i], tid_maps[i])
        n_ours = len([l for l in np.unique(tracked[i]) if l != 0])

        # bright will get text overlays; that's fine — `our` was made from a copy.
        cv2.putText(bright, "Raw",                          (8, 22),   FONT, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(bright, f"Frame {i+1}/{n}",             (8, H-10), FONT, 0.40, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(our,    f"{method_label}  n={n_ours}",  (8, 22),   FONT, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(our,    f"Frame {i+1}/{n}",             (8, H-10), FONT, 0.40, (200,200,200), 1, cv2.LINE_AA)

        pieces = [bright, our]
        if gts is not None:
            gt     = draw_masks(bright, gts[i])
            n_gt   = len([l for l in np.unique(gts[i]) if l != 0])
            cv2.putText(gt, f"GT  n={n_gt}",      (8, 22),   FONT, 0.55, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(gt, f"Frame {i+1}/{n}",   (8, H-10), FONT, 0.40, (200,200,200), 1, cv2.LINE_AA)
            pieces.append(gt)

        return np.hstack(pieces).astype(np.float32), n_ours

    prev_composite = None
    counts = []
    for i in range(n):
        composite, n_ours = make_panel(i)
        counts.append(n_ours)

        # Insert blended transition frames for smooth playback
        if prev_composite is not None and n_interp > 0:
            for k in range(1, n_interp + 1):
                alpha = k / (n_interp + 1)
                blend = cv2.addWeighted(prev_composite, 1 - alpha, composite, alpha, 0)
                writer.write(blend.astype(np.uint8))

        writer.write(composite.astype(np.uint8))
        prev_composite = composite
        print(f"    Frame {i+1:2d}: n={n_ours}")

    writer.release()
    print(f"  Mean cells/frame: {np.mean(counts):.1f}")

    # MJPG AVI → H.264 MP4 (preferred, via ffmpeg). The AVI is an intermediate
    # because OpenCV's H.264 support is unreliable across builds. Delete the AVI
    # afterward so only the MP4 is left on disk.
    avi_path = Path(cfg["out_avi"])
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", cfg["out_avi"],
             "-vcodec", "libx264", "-crf", "18", cfg["out_mp4"]],
            check=True, capture_output=True,
        )
        avi_path.unlink(missing_ok=True)
        print(f"  Wrote {cfg['out_mp4']}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        # ffmpeg missing or failed — fall back to OpenCV's mp4v writer so the
        # output is still an .mp4. Lossier than libx264 but keeps the contract.
        print("  ffmpeg unavailable — re-encoding via OpenCV mp4v …")
        cap = cv2.VideoCapture(cfg["out_avi"])
        fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
        mp4_writer = cv2.VideoWriter(cfg["out_mp4"], fourcc_mp4, out_fps,
                                     (W * panels, H))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mp4_writer.write(frame)
        cap.release()
        mp4_writer.release()
        avi_path.unlink(missing_ok=True)
        print(f"  Wrote {cfg['out_mp4']}")


def print_mota(m, n=None, tid_maps=None):
    """Print a MOTA summary and, if tracks were provided, per-track lifetime stats."""
    print(f"  MOTA={m['MOTA']:+.3f}  Recall={m['Recall']:.3f}  "
          f"FP={m['FP']}  FN={m['FN']}  IDSW={m['IDSW']}")

    if tid_maps and n:
        # Count how many frames each track ID appears in
        track_frames = {}
        for frame_i, tmap in enumerate(tid_maps):
            for tid in tmap.values():
                track_frames.setdefault(tid, []).append(frame_i + 1)
        lifetimes = [len(v) for v in track_frames.values()]
        if lifetimes:
            full_tracks = sum(1 for l in lifetimes if l == n)
            long_tracks = sum(1 for l in lifetimes if l >= max(2, n // 2))
            print(f"  Tracks: total={len(lifetimes)}  full={full_tracks}  "
                  f"long(>={max(2, n//2)}fr)={long_tracks}  "
                  f"median_life={np.median(lifetimes):.0f}fr")


# ── CellSAM model + detection ────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_device():
    """Pick the fastest available torch device (Apple MPS > CUDA > CPU)."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_cellsam_model = None   # lazy singleton — CellSAM weights are ~400 MB

def get_cellsam_model():
    """Load (and cache) the CellSAM model. Requires DEEPCELL_ACCESS_TOKEN env var."""
    global _cellsam_model
    if _cellsam_model is not None:
        return _cellsam_model

    import os
    if not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
        raise RuntimeError(
            "CellSAM requires a DeepCell access token.\n"
            "1. Create a free account at https://users.deepcell.org\n"
            "2. Generate an access token\n"
            "3. export DEEPCELL_ACCESS_TOKEN=<your-token>\n"
            "4. Re-run this script"
        )

    from cellSAM.model import get_model
    device = get_device()
    print(f"  Loading CellSAM model (device={device}) …")
    try:
        _cellsam_model = get_model("cellsam_extra")
    except Exception:
        # Newer/older CellSAM releases host different weight names; fall back.
        print("  cellsam_extra unavailable, using cellsam_general …")
        _cellsam_model = get_model("cellsam_general")
    _cellsam_model = _cellsam_model.to(device)
    _cellsam_model.eval()
    return _cellsam_model


def segment_cellsam(img_01, min_area=50, max_area=900, bbox_threshold=0.65):
    """Run CellSAM on a single [0,1]-normalised frame and return an integer label image."""
    from cellSAM import segment_cellular_image

    device = get_device()
    model  = get_cellsam_model()

    # CellSAM expects 8-bit RGB; replicate intensity across channels.
    gray = (np.clip(img_01, 0, 1) * 255).astype(np.uint8)
    rgb  = np.stack([gray, gray, gray], axis=-1)

    mask, _, _ = segment_cellular_image(
        rgb, model=model, device=str(device), bbox_threshold=bbox_threshold
    )
    return area_filter(mask.astype(np.int32), min_area, max_area)


_cellpose_model = None  # lazy singleton — cpsam weights are ~400 MB

def get_cellpose_model():
    global _cellpose_model
    if _cellpose_model is not None:
        return _cellpose_model
    from cellpose import models
    import torch
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    print(f"  Loading Cellpose model (gpu={use_gpu}) …")
    _cellpose_model = models.CellposeModel(gpu=use_gpu)
    return _cellpose_model


def segment_cellpose(img_01, diameter=0, min_area=10, max_area=300,
                     cellprob_threshold=0.0, flow_threshold=0.4):
    """Cellpose-SAM (cpsam) on a [0,1] frame. Drop-in alternative to segment_cellsam
    for small dim densely-packed cells where CellSAM's SAM mask decoder over-fills.

    `diameter=0` (default) lets Cellpose auto-detect cell size per-frame, which
    handles fields with mixed cell scales much better than a fixed value.

    Note: Cellpose's `min_size` defaults to 15 px which silently drops sub-15px
    cells. We pass `min_size=max(min_area-1, 4)` so this filter doesn't pre-empt
    our explicit `min_area` filter."""
    model = get_cellpose_model()
    img8  = (np.clip(img_01, 0, 1) * 255).astype(np.uint8)
    cp_min_size = max(min_area - 1, 4)
    cp_diameter = None if not diameter else diameter
    masks, _, _ = model.eval(
        img8,
        diameter=cp_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=cp_min_size,
    )
    return area_filter(masks.astype(np.int32), min_area, max_area)