"""demo_event parameter configuration — all tunable parameters in one place,
read by every stage module.

Tune parameters here only; no need to modify any stage script.
"""

import os
import numpy as np

# ==========================================================
# Paths and data
# ==========================================================
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DEMO_DIR, "data")
OUTPUT_ROOT = os.path.join(DEMO_DIR, "output")

START_TS = "1776243611812929"
END_TS = "1776243611829857"

# Input data
NPZ_PATH = os.path.join(DATA_DIR, f"{START_TS}.npz")   # aligned event data
H5_START = os.path.join(DATA_DIR, f"{START_TS}.h5")    # HSI start boundary frame
H5_END = os.path.join(DATA_DIR, f"{END_TS}.h5")        # HSI end boundary frame
H5_BG = os.path.join(DATA_DIR, "background.h5")        # HSI background frame
PNG_START = os.path.join(DATA_DIR, "background.png")  # start-frame RGB (circle detection in rotation mode)
DETECTED_DIR = DATA_DIR  # object detection results

# ==========================================================
# Motion scene selection
# ==========================================================
# "translation" = translation scene (default): template matching + centroid
#                 trajectory + spectral transfer
# "rotation"    = rotation scene: Hough circle detection + per-interval angle
#                 estimation + spectral transfer
MOTION_MODE = "rotation"

# Per-stage output directories
EVENT_FRAMES_DIR = os.path.join(OUTPUT_ROOT, "event_frames_10x")  # Stage 1: split event frames
TRAJ_DIR = os.path.join(OUTPUT_ROOT, "trajectories")              # Stage 3: raw trajectories
OPT_DIR = os.path.join(OUTPUT_ROOT, "optimized")                  # Stage 3: optimized trajectories
RECON_DIR = os.path.join(OUTPUT_ROOT, "reconstructed_hsi")        # Stage 4: reconstructed HSI frames

# ==========================================================
# Stage 1: event splitting
# ==========================================================
NUM_PARTS = 10                   # interpolation multiplier (number of event-stream splits)
IMAGE_SIZE = (2048, 2448)        # event-frame canvas (H, W)
EVENT_SIZE = 10                  # size of each drawn event (pixels)

# ==========================================================
# Stage 3: edge-event matching algorithm parameters
# ==========================================================
SEARCH_SCALE = 0.2               # search radius = max(template w, h) × SEARCH_SCALE
SIGMA_EVENT = 3.0
STEP = 5
STEP_COARSE = 15                 # coarse-search step (two-stage search enabled when > STEP)
IMG_H, IMG_W = 2048, 2448
ACTIVE_SIGMA_MULT = 2.0
SCORE_SHARPNESS = 2.0
DENSITY_ADAPT = 0.0              # density-adaptive gain: disabled
STRIP_CONSTRAINT_WEIGHT = 0.5
SCORE_FALLBACK_THRESHOLD = 0.1   # fallback threshold (0 = fallback disabled)
EDGE_THICKNESS = 3
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
LOG_INTERVAL_FRAMES = 3

# ==========================================================
# Stage 3: trajectory optimization parameters
# ==========================================================
OPT_LAMBDA_DATA = 200.0
OPT_LAMBDA_VEL = 25.0
OPT_LAMBDA_ACC = 8.0
OPT_LAMBDA_END_GUIDE = 120.0
OPT_LAMBDA_START_BOUNDARY = 1000.0
OPT_ROBUST_ITERS = 3
OPT_ROBUST_THRESHOLD = 2.5
OPT_SCORE_POWER = 0.5

# ==========================================================
# Rotation-scene parameters (active when MOTION_MODE = "rotation")
# ==========================================================
ROTATE_GAMMA = 1.8              # gamma correction before circle detection
ROTATE_MIN_RADIUS = 400         # Hough circle detection radius range
ROTATE_MAX_RADIUS = 700
ROTATE_ANGLE_MIN = -10.0        # angle search start (degrees; increases in 0.1° steps to 0)
ROTATE_ANGLE_STEP = 0.1
ROTATE_USE_OPTIMIZATION = True  # True = joint rotation-trajectory optimization; False = EMA smoothing
ROTATE_OPT_LAMBDA_DATA = 100.0
ROTATE_OPT_LAMBDA_VEL = 50.0
ROTATE_OPT_LAMBDA_ACC = 20.0
ROTATE_OPT_LAMBDA_BOUNDARY = 500.0
ROTATE_OPT_MIN_SCORE = 1e-3
ROTATE_EMA_SPAN = 25            # EMA smoothing period count (when optimization is off)

# ==========================================================
# Stage 4: hyperspectral
# ==========================================================
NUM_BANDS = 61
WAVELENGTHS = np.arange(400, 1010, 10)   # 400–1000 nm at 10 nm intervals

# ==========================================================
# Stage 5: HSI pseudo-color visualization
# ==========================================================
VIS_DIR = os.path.join(OUTPUT_ROOT, "visualization_rgb")  # Stage 5: pseudo-color RGB output
VIS_START_NM = 400
VIS_STEP_NM = 10
VIS_R_BANDS = list(range(600, 701, 10))   # stacking scheme: R 600-700 nm
VIS_G_BANDS = list(range(500, 601, 10))   # G 500-600 nm
VIS_B_BANDS = list(range(400, 501, 10))   # B 400-500 nm
VIS_STACK_RGB_MODE = "sum"          # stacking scheme: "single" = single band; "sum" = mean over bands
VIS_NORM_P = 99.5      # percentile normalization (smaller = more robust to extreme values)
VIS_APPLY_WB = True    # gray-world white balance
VIS_GAMMA = 1.0        # gamma <1 brighter, >1 darker
VIS_DPI = 300          # output resolution
