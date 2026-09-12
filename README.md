# Motion Event-guided High-speed Hyperspectral Imaging Pipeline Demo

Starting from **aligned event data**, this demo runs the full pipeline:

```
aligned event NPZ → [1/4] event splitting → [2/4] template detection → [3/4] object matching (+ trajectory optimization)
                  → [4/4] hyperspectral frame synthesis → 61-band hyperspectral interpolated frames (.h5)
```

The motion scene is selected via **`MOTION_MODE`** in `config.py`:
- `"translation"` (default): translation scene — template matching + centroid trajectory + spectral transfer
- `"rotation"`: rotation scene — Hough circle detection + per-interval angle estimation + spectral transfer

## Files

| File | Purpose |
|---|---|
| `config.py` | **Parameter configuration file**: data paths, motion scene, multiplier, matching algorithm parameters, optimization weights, etc. Tune parameters here only |
| `demo_pipeline.py` | Entry script; dispatches the flow according to `MOTION_MODE`, with stage banners + summary |
| `stage1_event_split.py` | Event splitting: NPZ → polarity-encoded event-frame JPGs (255=ON / 128=OFF) |
| `stage2_template_detection.py` | Template detection (translation): detections → mask → Canny edge template |
| `stage3_template_matching.py` | Translation matching + trajectory optimization: edge-event template matching → IRLS sparse linear system optimization |
| `stage3_rotation_matching.py` | Rotation matching: Hough circle detection + per-frame-pair L1-loss angle estimation → joint rotation-trajectory optimization (3D: x, y, angle) |
| `stage4_hsi_reconstruction.py` | Hyperspectral frame synthesis: translation = spectral transfer; rotation = spectral rotation transfer of the circular target region |
| `stage5_hsi_visualization.py` | Pseudo-color visualization: reconstructed HSI frames → pseudo-color RGB PNGs (band-stacking mode only) |

## 0. Data preparation

Download the demo dataset from the cloud drive:

https://drive.google.com/drive/folders/1oafYxNXxwXOD2LeJ1Sr3CEf46U0YsMLK?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

> Translation scene requires: the npz, three h5 files, and two JSON files containing object detection results.
> Rotation scene requires: the npz, three h5 files, and a `{background}.png` used for locating rotating targets.

## 1. Run

```bash
python demo_event/demo_pipeline.py
```

## 2. Output structure

```
demo_event/output/
├── event_frames_10x/     # Stage 1: 1782892531128044_part000..009.jpg
├── trajectories/         # Stage 3: raw trajectory JSON/CSV + trajectory visualization + matching_log.txt
├── optimized/            # Stage 3: optimized_trajectories_{s}_{e}.csv
├── reconstructed_hsi/    # Stage 4: frame_001.h5 .. frame_010.h5 (interpolated frames)
└── visualization_rgb/    # Stage 5: RGB_frame_001_stack_sum_spectral.png .. (pseudo-color visualization)
```

Each reconstructed frame H5 contains the dataset `hsi_R` (61 bands × 2048 × 2448, float32, gzip)
and the attributes `wavelengths` (400–1000 nm at 10 nm intervals), `frame_idx`, etc.

In the rotation scene, `trajectories/` contains `rotation_trajectory_raw.json` +
`rotation_trajectory_optimized.json` + `rotate_degree.npy` (the angle sequence of the frame intervals),
`output/circle_detection.jpg` is the circle detection visualization, and the reconstructed frames are
`frame_001..009.h5` (N event frames → N-1 intervals → N-1 reconstructed frames).

## 3. Parameter tuning

All parameters are centralized in `config.py`:

- `MOTION_MODE`: motion scene selection (`"translation"` / `"rotation"`)
- `NUM_PARTS`: interpolation multiplier (default 10; change to 5/20 etc. to alter the number of splits and reconstructed frames)
- Translation matching algorithm: `SEARCH_SCALE`, `SIGMA_EVENT`, `STEP`, `STEP_COARSE`,
  `STRIP_CONSTRAINT_WEIGHT`, `SCORE_FALLBACK_THRESHOLD`, `EDGE_THICKNESS`, etc.
- Translation trajectory optimization: `OPT_LAMBDA_*`, `OPT_ROBUST_*`, `OPT_SCORE_POWER`
- Rotation scene: `ROTATE_GAMMA`, `ROTATE_MIN/MAX_RADIUS`, `ROTATE_ANGLE_MIN/STEP`,
  `ROTATE_USE_OPTIMIZATION` (EMA smoothing when False), `ROTATE_OPT_LAMBDA_*`
- Pseudo-color visualization: `VIS_R/G/B_BANDS` (stacking band ranges), `VIS_STACK_RGB_MODE` (`"single"`/`"sum"`),
  `VIS_NORM_P` (percentile normalization), `VIS_APPLY_WB` (gray-world white balance),
  `VIS_GAMMA`, `VIS_DPI` (output resolution)
