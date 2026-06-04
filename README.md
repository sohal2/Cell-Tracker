# FLIM T-Cell Tracker

Automated cell detection and tracking for FLIM (Fluorescence Lifetime Imaging
Microscopy) videos of T-cells.

## Pipeline

**Detector → Centroid Hungarian Tracker**

1. **Preprocess** — per-frame intensity normalisation. The default percentile clip
   `[p1, p99.8] → [0, 1]` preserves cell-to-background ratio for the bright PBS1/2DG
   benchmark data; for low-SNR fluorescence (e.g. GFP-zmel mCherry) pass
   `--smooth-sigma > 0` to enable a smoothed, background-subtracted variant.
2. **Detect** — pluggable per-frame segmenter (`--detector`):
   - `cellsam` (default): [CellSAM](https://github.com/vanvalenlab/cellSAM)'s
     AnchorDETR + SAM. Best on the brighter benchmark data.
   - `cellpose`: [Cellpose-SAM (cpsam)](https://github.com/MouseLand/cellpose).
     Recommended for small (≲20 px) dim densely-packed cells where CellSAM's
     SAM mask decoder produces halo-style oversized masks.
3. **Link** — centroid Hungarian matching links masks across frames, with gap
   closing (≤ 2 frames missed) and greedy track stitching.
4. **Filter** — tracks that appear in fewer than 2 frames are dropped as noise.

## Installation

Tested on Python 3.12 (macOS / Linux). CellSAM and Cellpose both ship CUDA and
Apple MPS backends; CPU also works but is slower.

```bash
# 1. Clone and create the environment
git clone https://github.com/skalalab/cell-tracker.git flim_tracker
cd flim_tracker
conda create -n flim python=3.12
conda activate flim
pip install -r requirements.txt    # installs CellSAM from GitHub + Cellpose 4

# 2. Get a free DeepCell access token (needed by CellSAM to download weights)
#    https://users.deepcell.org  →  Profile → Access tokens
export DEEPCELL_ACCESS_TOKEN=<your-token>
```

The DeepCell token is only required for `--detector cellsam` (the default).
`--detector cellpose` runs without a token; Cellpose downloads its own weights
on first use.

`ffmpeg` is recommended for H.264 MP4 encoding; if missing, the pipeline falls
back to OpenCV's `mp4v` writer (lower quality, still `.mp4`).

> **Note on the built-in `--dataset` flag:** the demo `segmentation_tracking/`
> folder is **not** included in the repo (it contains private microscopy data).
> A fresh clone can run `--input` mode on your own TIFFs immediately; the
> `--dataset pbs1`/`--dataset 2dg` benchmarks require that folder to be
> populated separately.

## Usage

### Run on your own data

Point `--input` at either a multi-page TIFF **or** a folder of per-frame
TIFFs (sorted naturally by filename):

```bash
python run_pipeline.py --input path/to/my_frames/ --output-prefix my_run
```

Output: `my_run_tracked.mp4` (Raw │ Tracked masks side-by-side).

Optional — evaluate against ground-truth label masks:

```bash
python run_pipeline.py \
    --input  path/to/my_frames/ \
    --gt-dir path/to/gt_labels/ \
    --output-prefix my_run
```

GT masks should be integer-label TIFFs (each cell a unique non-zero value),
one file per frame, with filenames that sort in the same order as the input.

### Reproduce the built-in benchmarks

```bash
python run_pipeline.py --dataset pbs1   # 15-frame PBS1 video
python run_pipeline.py --dataset 2dg    # 10-frame 2DG video
python run_pipeline.py --dataset all    # both, with summary
```

Requires the `segmentation_tracking/` data folder in the repo root.

### Multi-channel folders (e.g. GFP-zmel mCherry / GFP / NADH)

If your folder contains files like `*_Ch1_*.tif`, `*_Ch2_*.tif`, etc., pass
`--channel N` to load only that channel — otherwise the natural-sort loader
interleaves channels as if they were time points. Example for the GFP-zmel
mCherry T-cell channel:

```bash
python run_pipeline.py \
    --input segmentation_tracking/031726_GFPzmel/GFPzmel1-001/ \
    --channel 1 \
    --smooth-sigma 0.5 \
    --detector cellpose --cell-diameter 12 \
    --output-prefix gfpzmel1-001_ch1
```

The recommended GFP-zmel recipe is `--detector cellpose --smooth-sigma 0.5
--cell-diameter 12 --max-area 600 --merge-dil 0`. CellSAM under-detects on this
sub-20-px low-SNR regime because its SAM mask decoder is trained for larger,
brighter cells; Cellpose-SAM's flow-field decoder produces tightly contoured
masks instead of halos.

### All CLI options

```
--input PATH              Multi-page TIFF or folder of per-frame TIFFs
--dataset {pbs1,2dg,all}  Use a built-in benchmark preset (alternative to --input)
--output-prefix STR       Prefix for output .mp4 (default: input basename)
--gt-dir PATH             Folder of integer-label GT masks for MOTA evaluation
--channel {1,2,3,4}       1-indexed channel selector for multi-channel folders
--detector {cellsam,cellpose}  Per-frame segmenter                [cellsam]
--cell-diameter INT       Expected cell diameter (Cellpose only)  [12 px]
--cellprob-threshold FL   Cellpose cellprob cutoff                [0.0]
--flow-threshold FL       Cellpose flow consistency cutoff        [0.4]
--smooth-sigma FLOAT      Gaussian σ + bg-subtract preprocess     [0.0=off]
--bbox-threshold FLOAT    CellSAM detection confidence            [0.50]
--min-area INT            Drop detections smaller than this       [30 px]
--max-area INT            Drop detections larger than this        [600 px]
--max-dist INT            Max centroid movement between frames    [25 px]
--merge-dil INT           Dilation for touch-merging fragments    [1 px, 0=off]
--fps INT                 Output video frame rate                 [3]
```

`--dataset` presets override these defaults with values tuned for PBS1/2DG;
the values shown above only apply in `--input` mode.

## Benchmark results

| Dataset | MOTA  | Notes                                                    |
|---------|-------|----------------------------------------------------------|
| PBS1    | +0.47 | 41 tracks, avg lifetime 7.0 frames                       |
| 2DG     | −1.35 | Structural FP floor from TCSPC photon dark counts        |

The 2DG MOTA is strongly negative because the TCSPC detector's dark-count floor
produces ~25 persistent noise blobs per frame. Real cells there have SNR ≈ 1.2×
background, so any absolute SNR threshold that kills the noise also kills the
true cells. This is a data-acquisition issue, not a tracker limitation.

## Key parameters (why the defaults are what they are)

PBS1 / 2DG benchmark presets (in `DATASETS`):

| Parameter          | Default | Reason                                                                     |
|--------------------|---------|----------------------------------------------------------------------------|
| `bbox_threshold`   | 0.65    | CellSAM's default 0.40 over-detects by 2–3×. 0.50 used for noisy 2DG.      |
| `max_area`         | 900 px² | Permits larger merged blobs through the area filter.                       |
| `max_dist`         | 40 px   | Tracks can drift ≤ 40 px between frames before splitting.                  |
| `min_lifetime`     | 2 fr    | Single-frame detections are almost always noise.                           |

`--input` mode defaults are tuned for the small dim densely-packed regime
(e.g. mCherry T-cells at 20× air, ~5–20 px cells): `--min-area 30
--max-area 600 --max-dist 25 --merge-dil 1 --bbox-threshold 0.50`. Raise these
for larger sparser cells (e.g. melanoma at low magnification).

## Files

| File                  | Role                                                            |
|-----------------------|-----------------------------------------------------------------|
| `run_pipeline.py`     | CLI entry point; tracker + stitching + driver                   |
| `_pipeline_common.py` | Loaders, preprocessing, CellSAM, MOTA, video rendering          |
| `requirements.txt`    | pip dependencies                                                |
