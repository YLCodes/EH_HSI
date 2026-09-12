"""Stages 3 (rotation scene): rotation circle detection + angle estimation
+ rotation trajectory optimization.

Flow:
    1. detect_rotation_template   — Hough circle detection -> center + radius
                                    (rotation template detection)
    2. match_rotation             — per-frame-pair rotation angle estimation
                                    (L1 loss, optimal rotation angle searched
                                    in [-10°, 0) with 0.1° steps),
                                    confidence = exp(-loss std)
    3. Joint rotation-trajectory optimization (3D: x, y, angle sparse linear
       system, start boundary constraint only) or EMA smoothing (when
       ROTATE_USE_OPTIMIZATION = False)

Outputs:
    OUTPUT_ROOT/circle_detection.jpg            circle detection visualization
    TRAJ_DIR/rotation_trajectory_raw.json       raw angle trajectory
    TRAJ_DIR/rotation_trajectory_optimized.json optimized angle trajectory
    TRAJ_DIR/rotate_degree.npy                  final angle sequence (for the
                                                Stage 4 spectral transfer)
"""

import json
import os
from glob import glob

import cv2
import numpy as np

import config


# ==========================================================
# Rotation template detection
# ==========================================================
def adjust_gamma(image, gamma):
    """Gamma correction (preprocessing before circle detection)."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def detect_rotation_template(rgb_path=None, out_vis_path=None):
    """Rotation-scene template detection: Hough circle detection -> center
    + radius.

    Returns:
        (center, radius): center = (x, y), radius = int
    """
    if rgb_path is None:
        rgb_path = config.PNG_START

    color_image = cv2.imread(rgb_path)
    if color_image is None:
        raise FileNotFoundError(f"Cannot read start-frame image: {rgb_path}")
    color_image = adjust_gamma(color_image, config.ROTATE_GAMMA)

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred,
                               cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=50,
                               param1=50,
                               param2=30,
                               minRadius=config.ROTATE_MIN_RADIUS,
                               maxRadius=config.ROTATE_MAX_RADIUS)
    if circles is None:
        raise RuntimeError(
            f"No circular object detected (radius range {config.ROTATE_MIN_RADIUS}–"
            f"{config.ROTATE_MAX_RADIUS}px). The rotation scene requires a "
            f"start-frame image containing a large disk.")

    circles = np.uint16(np.around(circles))
    circle_result = circles[0][np.argmax([c[2] for c in circles[0]])]
    center = (int(circle_result[0]), int(circle_result[1]))
    radius = int(circle_result[2])
    print(f"Detected circle center: {center}, radius: {radius}")

    # Circle detection visualization
    if out_vis_path is None:
        out_vis_path = os.path.join(config.OUTPUT_ROOT, "circle_detection.jpg")
    os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)
    vis = color_image.copy()
    cv2.circle(vis, center, radius, (0, 255, 0), 5)
    cv2.imwrite(out_vis_path, vis)
    print(f"Circle detection visualization: {out_vis_path}")

    return center, radius


def get_rotation_matrix(center, angle):
    """2x3 affine matrix rotating around center by angle degrees."""
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cx, cy = center

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, cx - cx * cos_theta + cy * sin_theta],
        [sin_theta, cos_theta, cy - cx * sin_theta - cy * cos_theta]
    ], dtype=np.float32)

    return rotation_matrix


def concatenate_matrix(first, second):
    """Compose two 2x3 affine matrices (first applied, then second)."""
    if first.shape != (2, 3) or second.shape != (2, 3):
        raise ValueError("Input matrices must be 2x3.")
    M1 = np.eye(3)
    M1[:2, :3] = first
    M2 = np.eye(3)
    M2[:2, :3] = second
    result = M1 @ M2
    return result[:2, :3]


def estimate_rotation_angle(image0, image1, C, S):
    """Estimate the rotation angle between two consecutive event frames
    (L1-loss minimization).

    Args:
        image0, image1: two consecutive grayscale event frames
        C: circle center (x, y)
        S: radius (half-width of the search/comparison region)

    Returns:
        (base_theta, base_loss, std): optimal angle, L1 loss, loss standard
        deviation (confidence indicator)
    """
    image_ori = image0
    image1 = image1[(C[1] - S):(C[1] + S), (C[0] - S):(C[0] + S)]  # shrink the comparison region

    loss_0 = []  # initial acquisition of the rotation angle range
    for i in range(int(config.ROTATE_ANGLE_MIN / config.ROTATE_ANGLE_STEP), 0):
        theta = i * config.ROTATE_ANGLE_STEP
        M = get_rotation_matrix(C, theta)
        img01 = cv2.warpAffine(image_ori, M, (config.IMG_W, config.IMG_H))
        image_new = img01[(C[1] - S):(C[1] + S), (C[0] - S):(C[0] + S)]
        loss_0.append([theta, np.mean(np.abs(image1 - image_new))])  # L1 loss
    loss = sorted(loss_0, key=lambda x: x[1])
    base_theta, base_loss = loss[0]

    # Confidence evaluation based on L1-loss statistics
    loss_array = np.array(loss_0)[:, 1]
    if np.max(loss_array) > 0:
        loss_normal = loss_array / np.max(loss_array)  # normalized loss
        std = loss_normal.std()  # loss std as the confidence indicator
    else:
        std = 0.0

    return base_theta, base_loss, std


def optimize_rotation_trajectory_with_boundary(
    trajectory,
    start_pos,
    lambda_data=100.0,
    lambda_vel=50.0,
    lambda_acc=20.0,
    lambda_boundary=500.0,
    min_score=1e-3
):
    """Joint rotation-trajectory optimization (3D: x, y, angle), with the
    start boundary constraint only.

    trajectory: [{frame, x, y, angle, score}, ...]
    start_pos: (x0, y0, angle0)
    """
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve

    N = len(trajectory)
    dim = 3 * N  # x, y, angle for each frame

    A = lil_matrix((dim, dim), dtype=np.float64)
    b = np.zeros(dim, dtype=np.float64)

    def idx(k, dim_idx):
        """dim_idx: 0->x, 1->y, 2->angle"""
        return 3 * k + dim_idx

    # Data term (x, y, angle)
    for k, t in enumerate(trajectory):
        w = max(t.get("score", 1.0), min_score)
        weight = lambda_data * w

        for dim_idx in [0, 1, 2]:
            i = idx(k, dim_idx)
            A[i, i] += weight
            values = [t["x"], t["y"], t["angle"]]
            b[i] += weight * values[dim_idx]

    # Velocity smoothness term (x, y, angle)
    for k in range(2, N):
        for dim_idx in [0, 1, 2]:
            i_k = idx(k, dim_idx)
            i_k1 = idx(k - 1, dim_idx)
            i_k2 = idx(k - 2, dim_idx)

            A[i_k, i_k] += lambda_vel
            A[i_k1, i_k1] += 4 * lambda_vel
            A[i_k2, i_k2] += lambda_vel
            A[i_k, i_k1] += -2 * lambda_vel
            A[i_k1, i_k] += -2 * lambda_vel
            A[i_k1, i_k2] += -2 * lambda_vel
            A[i_k2, i_k1] += -2 * lambda_vel
            A[i_k, i_k2] += lambda_vel
            A[i_k2, i_k] += lambda_vel

    # Acceleration regularization term (x, y, angle)
    for k in range(1, N - 1):
        for dim_idx in [0, 1, 2]:
            i_km1 = idx(k - 1, dim_idx)
            i_k = idx(k, dim_idx)
            i_kp1 = idx(k + 1, dim_idx)

            A[i_km1, i_km1] += lambda_acc
            A[i_k, i_k] += 4 * lambda_acc
            A[i_kp1, i_kp1] += lambda_acc
            A[i_km1, i_k] += -2 * lambda_acc
            A[i_k, i_km1] += -2 * lambda_acc
            A[i_k, i_kp1] += -2 * lambda_acc
            A[i_kp1, i_k] += -2 * lambda_acc
            A[i_km1, i_kp1] += lambda_acc
            A[i_kp1, i_km1] += lambda_acc

    # Start boundary constraint (x, y, angle)
    for dim_idx in [0, 1, 2]:
        A[idx(0, dim_idx), idx(0, dim_idx)] += lambda_boundary
        b[idx(0, dim_idx)] += lambda_boundary * start_pos[dim_idx]

    # Solve the linear system
    A = csr_matrix(A)
    x = spsolve(A, b)

    # Organize output
    optimized_traj = []
    for k in range(N):
        optimized_traj.append({
            "frame": trajectory[k]["frame"],
            "x": float(x[idx(k, 0)]),
            "y": float(x[idx(k, 1)]),
            "angle": float(x[idx(k, 2)]),
            "score": trajectory[k].get("score", 1.0)
        })

    return optimized_traj


# ==========================================================
# Entry point
# ==========================================================
def match_rotation(center, radius, event_frames_dir=None, start_ts=None,
                   num_parts=None, traj_dir=None):
    """Rotation-scene matching: per-frame-pair angle estimation -> trajectory
    optimization (or EMA) -> final angle sequence.

    Returns:
        result dict: {rotate_degree, trajectory_raw, trajectory_optimized,
                      center, radius}
        rotate_degree: final angle of each consecutive frame interval
                       (length num_parts - 1)
    """
    if event_frames_dir is None:
        event_frames_dir = config.EVENT_FRAMES_DIR
    if start_ts is None:
        start_ts = config.START_TS
    if num_parts is None:
        num_parts = config.NUM_PARTS
    if traj_dir is None:
        traj_dir = config.TRAJ_DIR

    frame_names = sorted(glob(os.path.join(event_frames_dir, f"{start_ts}_part*.jpg")),
                         key=lambda n: int(os.path.basename(n)
                                           .rsplit('_part', 1)[1].split('.')[0]))
    if len(frame_names) < 2:
        raise RuntimeError(f"Not enough event frames (at least 2 required): {event_frames_dir}")
    frame_names = frame_names[:num_parts]

    # Per-frame-pair rotation angle estimation
    rotate_degree = []
    trajectory_raw = []
    print(f"Rotation angle estimation ({len(frame_names) - 1} interval(s))...")
    for i in range(len(frame_names) - 1):
        img0 = cv2.imread(frame_names[i], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(frame_names[i + 1], cv2.IMREAD_GRAYSCALE)
        theta, loss, std = estimate_rotation_angle(img0, img1, center, radius)

        # Confidence score (based on loss std)
        confidence = np.exp(-std) if std > 0 else 1.0

        rotate_degree.append(theta)
        trajectory_raw.append({
            "frame": i,
            "x": float(center[0]),
            "y": float(center[1]),
            "angle": float(theta),
            "score": float(confidence)
        })
        if i % config.LOG_INTERVAL_FRAMES == 0:
            print(f"  Interval {i}/{len(frame_names) - 1}: theta={theta:+.2f}°, "
                  f"loss={loss:.4f}, score={confidence:.4f}")

    rotate_degree = np.array(rotate_degree)

    # ── Trajectory optimization / EMA smoothing ──
    os.makedirs(traj_dir, exist_ok=True)
    if config.ROTATE_USE_OPTIMIZATION and len(trajectory_raw) > 0:
        start_pos = (trajectory_raw[0]["x"], trajectory_raw[0]["y"],
                     trajectory_raw[0]["angle"])
        trajectory_optimized = optimize_rotation_trajectory_with_boundary(
            trajectory=trajectory_raw,
            start_pos=start_pos,
            lambda_data=config.ROTATE_OPT_LAMBDA_DATA,
            lambda_vel=config.ROTATE_OPT_LAMBDA_VEL,
            lambda_acc=config.ROTATE_OPT_LAMBDA_ACC,
            lambda_boundary=config.ROTATE_OPT_LAMBDA_BOUNDARY,
            min_score=config.ROTATE_OPT_MIN_SCORE,
        )
        rotate_degree = np.array([t["angle"] for t in trajectory_optimized])
        print("Joint rotation-trajectory optimization complete")
    else:
        span = config.ROTATE_EMA_SPAN
        alpha = 2.0 / (span + 1)
        ema_smoothed = np.zeros_like(rotate_degree, dtype=np.float64)
        ema_smoothed[0] = rotate_degree[0]
        for i in range(1, len(rotate_degree)):
            ema_smoothed[i] = alpha * rotate_degree[i] + (1 - alpha) * ema_smoothed[i - 1]
        rotate_degree = ema_smoothed
        trajectory_optimized = [dict(t, angle=float(a))
                                for t, a in zip(trajectory_raw, rotate_degree)]
        print(f"EMA smoothing complete (span={span})")

    # ── Save ──
    with open(os.path.join(traj_dir, "rotation_trajectory_raw.json"), "w",
              encoding="utf-8") as f:
        json.dump(trajectory_raw, f, indent=2)
    with open(os.path.join(traj_dir, "rotation_trajectory_optimized.json"), "w",
              encoding="utf-8") as f:
        json.dump(trajectory_optimized, f, indent=2)
    np.save(os.path.join(traj_dir, "rotate_degree.npy"), rotate_degree)
    print(f"Rotation trajectory saved: {traj_dir}")

    return {
        "rotate_degree": rotate_degree,
        "trajectory_raw": trajectory_raw,
        "trajectory_optimized": trajectory_optimized,
        "center": center,
        "radius": radius,
    }
