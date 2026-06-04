"""
FLIM T-cell tracking pipeline — command-line entry point.

Pipeline stages
---------------
  1. Load frames (built-in PBS1/2DG datasets, or user data via `--input`)
  2. Preprocess  (percentile normalisation to [0, 1])
  3. Detect      (CellSAM per-frame segmentation + area / touch-merge filters)
  4. Link        (centroid Hungarian matching + gap closing + track stitching)
  5. Evaluate    (MOTA vs. optional ground-truth masks)
  6. Render      (side-by-side MP4 video: raw │ tracked │ GT)

Usage examples
--------------
Run the built-in benchmarks (reproduces paper numbers):
    DEEPCELL_ACCESS_TOKEN=<token> python run_pipeline.py --dataset pbs1
    DEEPCELL_ACCESS_TOKEN=<token> python run_pipeline.py --dataset 2dg
    DEEPCELL_ACCESS_TOKEN=<token> python run_pipeline.py --dataset all

Run on your own data (a multi-page TIFF, or a folder of per-frame TIFFs):
    DEEPCELL_ACCESS_TOKEN=<token> python run_pipeline.py \\
        --input path/to/frames/ \\
        --output-prefix my_experiment \\
        --fps 3
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from _pipeline_common import (
    DATASETS,
    load_pbs1, load_2dg, load_gt_pbs1, load_gt_2dg, load_generic_frames,
    touch_merge, compute_mota, render_and_write_video,
    print_mota, preprocess_raw_norm, preprocess_raw_norm_smooth,
    segment_cellsam, segment_cellpose,
)


# ── centroid Hungarian tracker ───────────────────────────────────────────────
def centroid_track(raw_labels, max_dist=40, max_gap=2):
    """Link per-frame masks into tracks using nearest-centroid Hungarian matching.

    Parameters
    ----------
    raw_labels : list of 2D int arrays (one label image per frame)
    max_dist   : max centroid movement (px) allowed between consecutive frames
    max_gap    : how many frames a track may be "missing" before retirement

    Returns
    -------
    list of 2D int arrays — same shape as input, but labels are now
    temporally-consistent track IDs (tid) instead of per-frame labels.
    """
    next_tid = [1]
    active   = {}   # tid → {cy, cx, gap}
    tracked  = []

    for m in raw_labels:
        curr       = {int(p.label): p.centroid for p in regionprops(m)}
        curr_lbls  = list(curr.keys())
        track_tids = list(active.keys())

        # Cost matrix of centroid distances (tracks × detections)
        matches = {}
        if track_tids and curr_lbls:
            track_cents = np.array([(active[t]['cy'], active[t]['cx']) for t in track_tids])
            curr_cents  = np.array([curr[l] for l in curr_lbls])
            cost = cdist(track_cents, curr_cents)
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] <= max_dist:
                    matches[curr_lbls[c]] = track_tids[r]

        # Unmatched tracks age; retire them once gap exceeds max_gap
        matched_tids = set(matches.values())
        for tid in list(active):
            if tid not in matched_tids:
                active[tid]['gap'] += 1
                if active[tid]['gap'] > max_gap:
                    del active[tid]

        # Either extend a matched track, or spawn a new one for unmatched dets
        tid_map = {}
        for lbl, (cy, cx) in curr.items():
            if lbl in matches:
                tid = matches[lbl]
            else:
                tid = next_tid[0]
                next_tid[0] += 1
            active[tid]  = {'cy': cy, 'cx': cx, 'gap': 0}
            tid_map[lbl] = tid

        # Relabel this frame's mask image with track IDs
        out = np.zeros_like(m)
        for lbl, tid in tid_map.items():
            out[m == lbl] = tid
        tracked.append(out)

    return tracked


def stitch_tracks(tracked_labels, max_dist=30, max_gap=2):
    """Join tracks where one ended near where another began within `max_gap` frames.

    Recovers identity after brief CellSAM detection dropouts.
    """
    # Collect each track's first and last (frame, cy, cx)
    track_first, track_last = {}, {}
    for f, m in enumerate(tracked_labels):
        for p in regionprops(m):
            tid = int(p.label)
            if tid not in track_first:
                track_first[tid] = (f, *p.centroid)
            track_last[tid] = (f, *p.centroid)

    # Find candidate stitches: end(a) → start(b) within gap and distance
    candidates = []
    for a, (fe, ya, xa) in track_last.items():
        for b, (fs, yb, xb) in track_first.items():
            if a == b:
                continue
            gap = fs - fe
            if 1 <= gap <= max_gap:
                d = np.hypot(ya - yb, xa - xb)
                if d <= max_dist:
                    candidates.append((d, a, b))
    candidates.sort()

    # Greedy union-find merge, taking shortest stitches first
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    used_end, used_start, n = set(), set(), 0
    for d, a, b in candidates:
        if a in used_end or b in used_start:
            continue
        if find(a) == find(b):
            continue
        parent[find(b)] = find(a)
        used_end.add(a)
        used_start.add(b)
        n += 1

    if n == 0:
        return tracked_labels

    # Relabel every frame with the merged root tid
    result = []
    for m in tracked_labels:
        out = m.copy()
        for tid in np.unique(m):
            if tid == 0:
                continue
            root = find(int(tid))
            if root != int(tid):
                out[m == tid] = root
        result.append(out)
    print(f"  Track stitching: merged {n} broken track pairs")
    return result


def remove_short_tracks(tracked_labels, min_lifetime=2):
    """Drop tracks that appear in fewer than `min_lifetime` frames (likely noise)."""
    id_counts = Counter()
    for m in tracked_labels:
        for lbl in np.unique(m):
            if lbl:
                id_counts[lbl] += 1

    short_ids = {lbl for lbl, cnt in id_counts.items() if cnt < min_lifetime}
    if not short_ids:
        return tracked_labels

    result = []
    for m in tracked_labels:
        out = m.copy()
        for sid in short_ids:
            out[out == sid] = 0
        result.append(out)

    print(f"  Removed {len(short_ids)} short tracks (<{min_lifetime} fr), "
          f"kept {len(id_counts) - len(short_ids)}")
    return result


# ── top-level pipeline driver ────────────────────────────────────────────────
def run_pipeline(name, cfg, raws, gts=None):
    """Run detection + tracking on a list of raw frames.

    Parameters
    ----------
    name : str        — output filename prefix
    cfg  : dict       — pipeline knobs (see DATASETS or `build_cfg`)
    raws : list[ndarray] — raw intensity frames
    gts  : list[ndarray] | None — ground-truth label images for MOTA evaluation
    """
    n        = len(raws)
    min_area = cfg["min_area"]
    max_area = cfg["max_area"]
    bbox_thr = cfg["cellsam_bbox_thresh"]
    max_dist = cfg["max_dist"]
    dil      = cfg["merge_dil"]
    detector = cfg.get("detector", "cellsam")
    diameter = cfg.get("cell_diameter", 12)
    smooth_s = cfg.get("smooth_sigma", 0.0)

    cfg = {**cfg,
           "out_mp4": f"{name}_tracked.mp4",
           "out_avi": f"{name}_tracked.avi"}

    label = "CellSAM" if detector == "cellsam" else f"Cellpose(d={diameter})"
    print(f"\n{'='*50}")
    print(f"  {name.upper()} — {label} + Centroid Tracker")
    if detector == "cellsam":
        print(f"  {n} frames, area=[{min_area},{max_area}]  bbox_thr={bbox_thr}")
    else:
        print(f"  {n} frames, area=[{min_area},{max_area}]  diameter={diameter}")
    if smooth_s > 0:
        print(f"  Preprocess: gaussian σ={smooth_s} + percentile-norm")
    print(f"{'='*50}")

    # 1) Preprocess
    if smooth_s > 0:
        preps = [preprocess_raw_norm_smooth(r, sigma=smooth_s) for r in raws]
    else:
        preps = [preprocess_raw_norm(r) for r in raws]

    # 2) Per-frame detection (+ touch-merge of adjacent fragments)
    print(f"  {label} per-frame segmentation …")
    raw_labels = []
    for i, p in enumerate(preps):
        if detector == "cellpose":
            lbl = segment_cellpose(
                p, diameter=diameter, min_area=min_area, max_area=max_area,
                cellprob_threshold=cfg.get("cellprob_threshold", 0.0),
                flow_threshold=cfg.get("flow_threshold", 0.4),
            )
        else:
            lbl = segment_cellsam(p, min_area, max_area, bbox_thr)
        merged = touch_merge(lbl, dil) if dil > 0 else lbl
        raw_labels.append(merged)
        print(f"    Frame {i+1:2d}: {len([l for l in np.unique(merged) if l])} detections")

    # 3) Link across frames
    print("  Centroid Hungarian linking …")
    tracked = centroid_track(raw_labels, max_dist=max_dist)
    tracked = stitch_tracks(tracked, max_dist=30, max_gap=2)
    tracked = remove_short_tracks(tracked, min_lifetime=2)

    # Identity mapping — tracker already wrote track IDs into the label image
    tid_maps = [
        {int(lbl): int(lbl) for lbl in np.unique(m) if lbl != 0}
        for m in tracked
    ]

    # 4) Track lifetime summary
    id_counts = Counter(tid for tmap in tid_maps for tid in tmap.values())
    lifetimes = list(id_counts.values())
    if lifetimes:
        print(f"  Track lifetimes: avg={np.mean(lifetimes):.1f}  "
              f"max={max(lifetimes)}  "
              f"full({n}-frame)={sum(1 for v in lifetimes if v == n)}  "
              f"total_tracks={len(lifetimes)}")

    # 5) Render video
    render_and_write_video(cfg, raws, tracked, tid_maps, gts, method_label="Ours")

    # 6) MOTA (only if GT provided)
    if gts is not None:
        m = compute_mota(tracked, gts)
        print_mota(m, n=n, tid_maps=tid_maps)
        return m
    return None


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_cfg(args):
    """Construct a pipeline config dict from CLI args (user-data mode)."""
    return dict(
        fps                 = args.fps,
        min_area            = args.min_area,
        max_area            = args.max_area,
        merge_dil           = args.merge_dil,
        max_dist            = args.max_dist,
        cellsam_bbox_thresh = args.bbox_threshold,
        detector            = args.detector,
        cell_diameter       = args.cell_diameter,
        cellprob_threshold  = args.cellprob_threshold,
        flow_threshold      = args.flow_threshold,
        smooth_sigma        = args.smooth_sigma,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FLIM T-cell tracking: CellSAM detection + centroid Hungarian linking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=["pbs1", "2dg", "all"],
                     help="Run on a built-in benchmark dataset.")
    src.add_argument("--input", type=Path,
                     help="Path to your data: a multi-page TIFF, OR a folder "
                          "of per-frame TIFFs.")

    p.add_argument("--output-prefix", default=None,
                   help="Prefix for output .mp4 (default: derived from --input basename).")
    p.add_argument("--gt-dir", type=Path, default=None,
                   help="Optional: folder of integer-label GT masks (one per frame, "
                        "sorted naturally) for MOTA evaluation.")
    p.add_argument("--channel", type=int, default=None, choices=[1, 2, 3, 4],
                   help="1-indexed channel selector for multi-channel folder input "
                        "(e.g. OME-TIFFs named *_Ch1_*.tif). Required when the folder "
                        "mixes channels — otherwise channels would be interleaved as "
                        "frames. Also picks frame[..., channel-1] from a 4D multi-page TIFF.")

    # Tunable pipeline parameters (only used with --input; --dataset presets override).
    # Defaults are tuned for small dim densely-packed cells (e.g. 20x air mCherry T-cells);
    # raise --max-area / --bbox-threshold / --merge-dil / --max-dist for larger sparser cells.
    p.add_argument("--detector", choices=["cellsam", "cellpose"], default="cellsam",
                   help="Segmenter: 'cellsam' (default, used for paper benchmarks) or "
                        "'cellpose' (Cellpose-SAM cpsam — better for sub-20px low-SNR cells).")
    p.add_argument("--cell-diameter", type=int, default=0,
                   help="Expected cell diameter in pixels [0=auto-detect]. "
                        "Cellpose only. Auto handles mixed-scale fields better.")
    p.add_argument("--cellprob-threshold", type=float, default=0.0,
                   help="Cellpose cellprob threshold [0.0]. Lower = more permissive, "
                        "admits dimmer cells. Try -3 to -6 for very low-SNR data.")
    p.add_argument("--flow-threshold", type=float, default=0.4,
                   help="Cellpose flow consistency threshold [0.4]. Higher = more permissive.")
    p.add_argument("--smooth-sigma", type=float, default=0.0,
                   help="Gaussian σ applied before percentile-norm preprocess [0.0=off]. "
                        "Try 1.0 on noisy low-SNR data to suppress single-pixel speckles.")
    p.add_argument("--bbox-threshold", type=float, default=0.50,
                   help="CellSAM bbox confidence threshold [0.50]. "
                        "Lower = more detections (more FP). PBS1 preset uses 0.65.")
    p.add_argument("--min-area", type=int, default=30,
                   help="Min object area in pixels [30].")
    p.add_argument("--max-area", type=int, default=600,
                   help="Max object area in pixels [600].")
    p.add_argument("--max-dist", type=int, default=25,
                   help="Max centroid movement between frames in pixels [25].")
    p.add_argument("--merge-dil", type=int, default=1,
                   help="Dilation radius for touch-merge of adjacent fragments [1]. "
                        "Set 0 to disable; raise for sparse data prone to fragmentation.")
    p.add_argument("--fps", type=int, default=3,
                   help="Output video frames per second [3].")

    return p.parse_args(argv)


def run_builtin(name):
    """Run one of the built-in benchmark datasets."""
    cfg = DATASETS[name]
    n   = cfg["n_frames"]
    if name == "pbs1":
        raws = [load_pbs1(i)    for i in range(n)]
        gts  = [load_gt_pbs1(i) for i in range(n)]
    else:
        raws = [load_2dg(i)     for i in range(n)]
        gts  = [load_gt_2dg(i)  for i in range(n)]
    return run_pipeline(name, cfg, raws, gts)


def run_user_data(args):
    """Run the pipeline on user-supplied TIFF frames."""
    ch_msg = f" (channel={args.channel})" if args.channel else ""
    print(f"  Loading frames from {args.input}{ch_msg} …")
    raws = load_generic_frames(args.input, channel=args.channel)
    print(f"  Loaded {len(raws)} frame(s), shape={raws[0].shape}")

    gts = None
    if args.gt_dir is not None:
        print(f"  Loading GT masks from {args.gt_dir} …")
        gts = load_generic_frames(args.gt_dir)
        if len(gts) != len(raws):
            sys.exit(f"ERROR: GT frame count ({len(gts)}) != input frame count ({len(raws)})")
        # GT masks must be integer label images
        gts = [g.astype(np.int32) for g in gts]

    prefix = args.output_prefix or Path(args.input).stem or "output"
    cfg    = {**build_cfg(args), "n_frames": len(raws)}
    return run_pipeline(prefix, cfg, raws, gts)


def main(argv=None):
    args = parse_args(argv)

    if args.dataset:
        names   = ["pbs1", "2dg"] if args.dataset == "all" else [args.dataset]
        results = {n: run_builtin(n) for n in names}

        if len(results) > 1:
            print(f"\n{'='*50}")
            print("  Summary")
            print(f"{'='*50}")
            for name, m in results.items():
                if m is None:
                    continue
                print(f"  {name.upper():5s}  MOTA={m['MOTA']:+.3f}  "
                      f"FP={m['FP']}  FN={m['FN']}  IDSW={m['IDSW']}")
    else:
        run_user_data(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
