"""Stage 3: (translation scene): object matching + trajectory optimization.

Outputs:
    TRAJ_DIR/trajectory_obj_{oid}.json          raw trajectories (frames 0..N)
    TRAJ_DIR/raw_trajectories_{s}_{e}.csv       raw trajectory CSV
    TRAJ_DIR/visualization_frames/              per-frame trajectory visualization
    TRAJ_DIR/summary_trajectory.jpg             trajectory summary image
    OPT_DIR/optimized_trajectories_{s}_{e}.csv  optimized trajectory CSV
"""

import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

import config
from stage2_template_detection import MatchConfig


# ==========================================================
# Event distance maps
# ==========================================================
def compute_event_distance_map(event_img):
    """Compute polarity-separated event distance maps (ON=255, OFF=128)."""
    binary_on = (event_img == 255).astype(np.uint8)
    binary_off = (event_img == 128).astype(np.uint8)
    from scipy.ndimage import distance_transform_edt
    dist_on = distance_transform_edt(1 - binary_on)
    dist_off = distance_transform_edt(1 - binary_off)
    return dist_on, dist_off


# ==========================================================
# Scoring function
# ==========================================================
def edge_event_consistency_score(dist_on, dist_off, edge_tpl, x, y, sigma,
                                 sharpness=1.0, density_adapt=0.0,
                                 polarity_mode='ON', score_mode='distance',
                                 event_img=None,
                                 edge_cy=None, edge_cx=None, n_edge=None):
    """Edge-event consistency score.

    Args:
        score_mode: 'binary'   = binary overlap hit rate (no distance map)
                    'gaussian' = Gaussian blur soft voting (no distance map)
                    'distance' = EDT distance transform + exp(-(d/σ)^k)
        edge_cy, edge_cx: precomputed relative edge-point coordinates
                          (optional; skip edge_tpl parsing when passed)
        n_edge: precomputed edge-point count (used with edge_cy/cx)
    """
    th, tw = edge_tpl.shape
    H, W = dist_on.shape

    if x < 0 or y < 0 or x + tw >= W or y + th >= H:
        return 0.0

    # ── Fast path: precomputed edge coordinates (gaussian / binary modes) ──
    if edge_cy is not None and edge_cx is not None and n_edge is not None \
            and score_mode in ('gaussian', 'binary'):
        if n_edge == 0:
            return 0.0
        abs_ys = y + edge_cy
        abs_xs = x + edge_cx

        if score_mode == 'gaussian':
            if event_img is None:
                return 0.0
            return float(np.mean(event_img[abs_ys, abs_xs]))

        # binary: event_img is the raw event frame
        if event_img is None:
            return 0.0
        vals = event_img[abs_ys, abs_xs]
        n_hits = np.sum((vals == 255) | (vals == 128))
        return float(n_hits) / n_edge

    # ── Slow path: parse edge points from edge_tpl (backward compatible,
    # used by distance mode) ──
    edge_pts = edge_tpl > 0
    n_edge_local = np.sum(edge_pts)
    if n_edge_local == 0:
        return 0.0

    # ── B1: binary overlap (slow fallback) ──
    if score_mode == 'binary':
        if event_img is None:
            return 0.0
        roi = event_img[y:y + th, x:x + tw]
        roi_events = (roi == 255) | (roi == 128)
        n_hits = np.sum(roi_events[edge_pts])
        return float(n_hits) / n_edge_local

    # ── B2: Gaussian blur (slow fallback) ──
    if score_mode == 'gaussian':
        if event_img is None:
            return 0.0
        roi = event_img[y:y + th, x:x + tw]
        return float(np.mean(roi[edge_pts]))

    # ── B3: EDT distance transform (baseline) ──
    roi_dist_on = dist_on[y:y + th, x:x + tw]
    roi_dist_off = dist_off[y:y + th, x:x + tw]

    # Polarity selection
    if polarity_mode == 'ON':
        d_edge = roi_dist_on[edge_pts].astype(np.float64)
    elif polarity_mode == 'OFF':
        d_edge = roi_dist_off[edge_pts].astype(np.float64)
    elif polarity_mode == 'BOTH':
        d_edge = np.minimum(roi_dist_on[edge_pts], roi_dist_off[edge_pts]).astype(np.float64)
    else:
        raise ValueError(f"Unknown polarity_mode: {polarity_mode}")

    s_per_edge = np.exp(-((d_edge / sigma) ** sharpness))
    active = d_edge <= config.ACTIVE_SIGMA_MULT * sigma
    n_active = np.sum(active)
    if n_active == 0:
        return 0.0
    return float(np.mean(s_per_edge[active]))


# ==========================================================
# Template matching
# ==========================================================
def match_by_edge_event(event_img, prev_cx, prev_cy, edge_tpl, search_radius,
                        sigma, tw, th, match_config: MatchConfig,
                        obj_id=0, frame_idx=0):
    """Template matching based on event consistency (contrast scoring + ON/OFF
    strip constraint)."""
    # ── Preprocessing: compute the full-frame feature map once per
    # score_mode (not per candidate) ──
    if match_config.score_mode == 'gaussian':
        binary_on = (event_img == 255).astype(np.float64)
        binary_off = (event_img == 128).astype(np.float64)

        blurred_on = cv2.GaussianBlur(binary_on, (0, 0), sigmaX=sigma)
        blurred_off = cv2.GaussianBlur(binary_off, (0, 0), sigmaX=sigma)
        preprocessed = np.maximum(blurred_on, blurred_off)
    elif match_config.score_mode == 'binary':
        preprocessed = event_img
    else:
        preprocessed = event_img  # distance mode still uses EDT

    # Only distance mode needs the EDT (not used by gaussian/binary)
    if match_config.score_mode == 'distance':
        dist_on, dist_off = compute_event_distance_map(event_img)
    else:
        dist_on = dist_off = event_img  # dummy, unused

    # ── Precompute edge-point coordinates (gaussian/binary fast path, avoids
    # re-parsing edge_tpl inside the loop) ──
    if match_config.score_mode in ('gaussian', 'binary'):
        edge_cy, edge_cx = np.where(edge_tpl > 0)
        n_edge = len(edge_cy)
    else:
        edge_cy = edge_cx = n_edge = None

    # ON/OFF event strip constraint (per object: only events within the search ROI)
    strip_ux = strip_uy = strip_len = None
    off_cx = off_cy = on_cx = on_cy = None
    if match_config.strip_constraint_weight > 0:
        # Search region = prev center ± search_radius, plus the template
        # half-width to cover the whole object
        roi_x1 = max(0, int(prev_cx - search_radius - tw // 2))
        roi_y1 = max(0, int(prev_cy - search_radius - th // 2))
        roi_x2 = min(event_img.shape[1], int(prev_cx + search_radius + tw // 2))
        roi_y2 = min(event_img.shape[0], int(prev_cy + search_radius + th // 2))
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi_events = event_img[roi_y1:roi_y2, roi_x1:roi_x2]
            on_ys, on_xs = np.where(roi_events == 255)
            off_ys, off_xs = np.where(roi_events == 128)
            if len(on_xs) > 0 and len(off_xs) > 0:
                on_cx = roi_x1 + np.mean(on_xs)
                on_cy = roi_y1 + np.mean(on_ys)
                off_cx = roi_x1 + np.mean(off_xs)
                off_cy = roi_y1 + np.mean(off_ys)
                strip_ux = on_cx - off_cx
                strip_uy = on_cy - off_cy
                strip_len = np.sqrt(strip_ux ** 2 + strip_uy ** 2)
                if strip_len > 0:
                    strip_ux /= strip_len
                    strip_uy /= strip_len

    best_score = 0.0
    best_x, best_y = prev_cx, prev_cy

    # ── Inline candidate scoring function (avoids duplicating coarse/fine
    # search code) ──
    def _eval_candidate(dx, dy):
        tl_x = int(prev_cx + dx - tw // 2)
        tl_y = int(prev_cy + dy - th // 2)
        cx_candidate = tl_x + tw // 2
        cy_candidate = tl_y + th // 2

        score_event = edge_event_consistency_score(
            dist_on, dist_off, edge_tpl, tl_x, tl_y, sigma,
            sharpness=match_config.sharpness,
            density_adapt=match_config.density_adapt,
            polarity_mode=match_config.polarity_mode,
            score_mode=match_config.score_mode,
            event_img=preprocessed,
            edge_cy=edge_cy, edge_cx=edge_cx, n_edge=n_edge,
        )

        score = score_event
        if score_event > 0 and strip_len is not None and strip_len > 0:
            vx = cx_candidate - off_cx
            vy = cy_candidate - off_cy
            proj = vx * strip_ux + vy * strip_uy
            perp = abs(vx * strip_uy - vy * strip_ux)
            perp_sigma = max(tw, th) / 4.0
            perp_penalty = np.exp(-(perp / perp_sigma) ** 2)
            proj_center = strip_len * 0.75
            proj_sigma = max(tw, th) / 2.0
            proj_penalty = np.exp(-((proj - proj_center) / proj_sigma) ** 2)
            strip_bonus = perp_penalty * proj_penalty
            score = score_event * (1 - match_config.strip_constraint_weight
                                   + match_config.strip_constraint_weight * strip_bonus)

        return score, cx_candidate, cy_candidate

    # ── Phase 1: coarse search (large step over the full range) ──
    coarse_step = max(config.STEP_COARSE, config.STEP)
    for dy in range(-search_radius, search_radius + 1, coarse_step):
        for dx in range(-search_radius, search_radius + 1, coarse_step):
            score, cx, cy = _eval_candidate(dx, dy)
            if score > best_score:
                best_score = score
                best_x, best_y = cx, cy

    # ── Phase 2: fine search (small step around the coarse best position) ──
    if coarse_step > config.STEP:
        best_dx = int(best_x - prev_cx)
        best_dy = int(best_y - prev_cy)
        fine_radius = coarse_step

        for dy in range(best_dy - fine_radius, best_dy + fine_radius + 1, config.STEP):
            for dx in range(best_dx - fine_radius, best_dx + fine_radius + 1, config.STEP):
                # Skip positions already evaluated in the coarse search
                if dx % coarse_step == 0 and dy % coarse_step == 0:
                    continue
                if abs(dx) > search_radius or abs(dy) > search_radius:
                    continue

                score, cx, cy = _eval_candidate(dx, dy)
                if score > best_score:
                    best_score = score
                    best_x, best_y = cx, cy

    return best_x, best_y, best_score


# ==========================================================
# Frame-by-frame matching
# ==========================================================
def _match_single_frame(event_img, targets, match_config, trajectories_raw,
                        pair_speed_px, frame_k, num_parts):
    """Match all objects on a single event frame (incl. low-score fallback,
    consistent with the original script behavior)."""
    for t in targets:
        obj_id = t["id"]
        prev = trajectories_raw[obj_id][-1]

        new_x, new_y, score = match_by_edge_event(
            event_img, prev["x"], prev["y"],
            t["edge_template"], t["search_radius"],
            config.SIGMA_EVENT, t["tw"], t["th"],
            match_config=match_config,
            obj_id=obj_id, frame_idx=frame_k,
        )

        # Score too low -> fall back to the initial frame position (frame 0).
        # Use the initial centroid instead of prev: prevents stationary
        # objects from drifting frame by frame due to noisy scores
        score_fallback = False
        fallback_th = config.SCORE_FALLBACK_THRESHOLD
        if score < fallback_th:
            init_pos = trajectories_raw[obj_id][0]
            new_x, new_y = init_pos["x"], init_pos["y"]
            score_fallback = True

        if frame_k % config.LOG_INTERVAL_FRAMES == 0:
            status = " FALLBACK(...)" if score_fallback else ""
            print(f"  Frame {frame_k}/{num_parts}, "
                  f"Target {obj_id}: score={score:.4f}, "
                  f"pos=({new_x:.1f}, {new_y:.1f}){status}")

        # Event density
        bx1 = max(0, int(new_x - t["tw"] // 2))
        by1 = max(0, int(new_y - t["th"] // 2))
        bx2 = min(event_img.shape[1], bx1 + t["tw"])
        by2 = min(event_img.shape[0], by1 + t["th"])
        roi = event_img[by1:by2, bx1:bx2]
        n_events = np.sum((roi == 255) | (roi == 128))
        density = n_events / (t["tw"] * t["th"])

        trajectories_raw[obj_id].append({
            "frame": frame_k + 1,
            "x": float(new_x), "y": float(new_y),
            "score": float(score),
            "density": float(density),
            "speed_px_s": pair_speed_px,
        })


def _fallback_all(targets, trajectories_raw, pair_speed_px, frame_k):
    """Missing event frame -> all objects fall back to the previous frame."""
    for t in targets:
        prev = trajectories_raw[t["id"]][-1]
        trajectories_raw[t["id"]].append({
            "frame": frame_k + 1,
            "x": float(prev["x"]), "y": float(prev["y"]),
            "score": 0.0,
            "density": 0.0,
            "speed_px_s": pair_speed_px,
        })


def match_frames(targets, match_config, trajectories_raw, pair_speed_px,
                 event_frames_dir, start_ts, num_parts=None):
    """Match all objects frame by frame over num_parts event frames (updates
    trajectories_raw in place)."""
    if num_parts is None:
        num_parts = config.NUM_PARTS

    print(f"Starting frame-by-frame matching ({num_parts} frame(s))...")

    for k in range(num_parts):
        event_path = os.path.join(
            event_frames_dir,
            f"{start_ts}_part{k:03d}.jpg")

        if not os.path.exists(event_path):
            print(f"  [WARN] Missing event frame {event_path}")
            _fallback_all(targets, trajectories_raw, pair_speed_px, k)
            continue

        event_img = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)
        if event_img is None:
            print(f"  [WARN] Cannot load event frame {event_path}")
            _fallback_all(targets, trajectories_raw, pair_speed_px, k)
            continue

        _match_single_frame(event_img, targets, match_config, trajectories_raw,
                            pair_speed_px, k, num_parts)


# ==========================================================
# Saving
# ==========================================================
def save_trajectory_csv(trajectories, csv_path, start_ts, end_ts, description):
    """Save a trajectory collection to a CSV file."""
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["pair", "frame", "object_id", "x", "y", "score",
                         "density", "speed_px_s"])
        for obj_id in sorted(trajectories.keys()):
            for entry in trajectories[obj_id]:
                if entry["frame"] == 0:
                    continue
                writer.writerow([
                    f"{start_ts}_{end_ts}",
                    entry["frame"],
                    obj_id,
                    entry["x"],
                    entry["y"],
                    entry.get("score", 0.0),
                    entry.get("density", 0.0),
                    entry.get("speed_px_s", 0.0),
                ])
    print(f"  [OK] {description} saved to: {csv_path}")


def save_trajectories(trajectories_raw, traj_dir, start_ts, end_ts):
    """Save raw trajectory JSON + CSV."""
    os.makedirs(traj_dir, exist_ok=True)

    for oid, traj in trajectories_raw.items():
        path = os.path.join(traj_dir, f"trajectory_obj_{oid}.json")
        with open(path, "w", encoding='utf-8') as f:
            json.dump(traj, f, indent=2)

    csv_path = os.path.join(traj_dir, f"raw_trajectories_{start_ts}_{end_ts}.csv")
    save_trajectory_csv(trajectories_raw, csv_path, start_ts, end_ts, "trajectory CSV")


# ==========================================================
# Trajectory optimization
# ==========================================================
def optimize_trajectory_v2(
    trajectory,
    start_pos,
    end_pos=None,
    lambda_data=200.0,
    lambda_vel=10.0,
    lambda_acc=3.0,
    lambda_end_guide=50.0,
    lambda_start_boundary=1000.0,
    robust_iters=3,
    robust_threshold=2.5,
    score_power=0.5,
    min_score=1e-3
):
    """
    Improved trajectory optimization (v2): soft end-point guidance + IRLS
    robust estimation + adaptive score weighting

    E = Σ w_data_k · ||p_k - p_k^obs||² + λ_vel · Σ ||p_k - p_{k-1}||²
      + λ_acc · Σ ||p_{k+1} - 2p_k + p_{k-1}||²
      + λ_end_guide · ||p_{N-1} - p_end||² + λ_start_boundary · ||p_0 - p_start||²

    w_data_k = λ_data · score_k^score_power · w_irls_k
    """
    import copy

    N = len(trajectory)
    if N == 0:
        raise ValueError("trajectory cannot be empty")

    has_end_guide = end_pos is not None

    # ---- Precompute the base data weight of each frame (score-adaptive) ----
    base_weights = np.ones(N, dtype=np.float64)
    for k, t in enumerate(trajectory):
        score = max(t.get("score", 1.0), min_score)
        base_weights[k] = float(score ** score_power)

    # ---- IRLS outer loop ----
    irls_weights = np.ones(N, dtype=np.float64)  # robust weights, initially all 1
    best_trajectory = None
    best_total_error = float('inf')

    actual_iters = max(1, robust_iters) if robust_iters > 0 else 1

    for iteration in range(actual_iters):
        dim = 2 * N

        # ---- Build the linear system A·x = b ----
        A = lil_matrix((dim, dim), dtype=np.float64)
        b = np.zeros(dim, dtype=np.float64)

        def idx(k, axis):
            return 2 * k + axis

        # 1. Data fidelity term
        for k, t in enumerate(trajectory):
            w = lambda_data * base_weights[k] * irls_weights[k]

            for axis in [0, 1]:
                i = idx(k, axis)
                A[i, i] += w
                b[i] += w * (t["x"] if axis == 0 else t["y"])

        # 2. Velocity smoothness term
        for k in range(1, N):
            for axis in [0, 1]:
                i_k = idx(k, axis)
                i_k1 = idx(k - 1, axis)
                A[i_k, i_k] += lambda_vel
                A[i_k1, i_k1] += lambda_vel
                A[i_k, i_k1] += -lambda_vel
                A[i_k1, i_k] += -lambda_vel

        # 3. Acceleration smoothness term
        for k in range(1, N - 1):
            for axis in [0, 1]:
                i_km1 = idx(k - 1, axis)
                i_k = idx(k, axis)
                i_kp1 = idx(k + 1, axis)

                A[i_km1, i_km1] += lambda_acc
                A[i_k, i_k] += 4 * lambda_acc
                A[i_kp1, i_kp1] += lambda_acc

                A[i_km1, i_k] += -2 * lambda_acc
                A[i_k, i_km1] += -2 * lambda_acc
                A[i_k, i_kp1] += -2 * lambda_acc
                A[i_kp1, i_k] += -2 * lambda_acc

                A[i_km1, i_kp1] += lambda_acc
                A[i_kp1, i_km1] += lambda_acc

        # 4. Hard start constraint (frame 0, same instant as the trajectory, correct)
        A[idx(0, 0), idx(0, 0)] += lambda_start_boundary
        A[idx(0, 1), idx(0, 1)] += lambda_start_boundary
        b[idx(0, 0)] += lambda_start_boundary * start_pos[0]
        b[idx(0, 1)] += lambda_start_boundary * start_pos[1]

        # 5. Soft end-point guidance (last frame only; weight far smaller than
        # v1's hard constraint)
        if has_end_guide:
            k_end = N - 1
            w_end = lambda_end_guide * irls_weights[k_end]
            A[idx(k_end, 0), idx(k_end, 0)] += w_end
            A[idx(k_end, 1), idx(k_end, 1)] += w_end
            b[idx(k_end, 0)] += w_end * end_pos[0]
            b[idx(k_end, 1)] += w_end * end_pos[1]

        # Solve
        A_csr = csr_matrix(A)
        x = spsolve(A_csr, b)

        # Extract result
        optimized_traj = []
        for k in range(N):
            optimized_traj.append({
                "frame": trajectory[k].get("frame", k),
                "x": float(x[idx(k, 0)]),
                "y": float(x[idx(k, 1)]),
                "score": trajectory[k].get("score", 1.0)
            })

        # ---- IRLS weight update ----
        if robust_iters > 0 and iteration < robust_iters - 1:
            residuals = np.zeros(N, dtype=np.float64)
            for k, t in enumerate(trajectory):
                dx = optimized_traj[k]["x"] - t["x"]
                dy = optimized_traj[k]["y"] - t["y"]
                residuals[k] = np.sqrt(dx ** 2 + dy ** 2)

            # Robust scale estimation (MAD)
            med_res = np.median(residuals)
            mad = np.median(np.abs(residuals - med_res)) * 1.4826
            mad = max(mad, 1e-6)

            # Huber-style weights
            threshold = robust_threshold * mad
            for k in range(N):
                r = residuals[k]
                if r <= threshold:
                    irls_weights[k] = 1.0
                else:
                    irls_weights[k] = threshold / max(r, 1e-8)

            # Evaluate total error (data fidelity + end-point guidance) to
            # select the best iteration
            total_data_err = np.sum(residuals * irls_weights)
            if has_end_guide:
                last_dx = optimized_traj[-1]["x"] - end_pos[0]
                last_dy = optimized_traj[-1]["y"] - end_pos[1]
                total_data_err += lambda_end_guide / lambda_data * np.sqrt(last_dx ** 2 + last_dy ** 2)

            if total_data_err < best_total_error:
                best_total_error = total_data_err
                best_trajectory = copy.deepcopy(optimized_traj)
        else:
            best_trajectory = optimized_traj

    return best_trajectory


# ==========================================================
# Trajectory optimization entry point
# ==========================================================
def _load_trajectory_from_json(json_path):
    """Load a full trajectory from a trajectory JSON (incl. frame 0), sorted
    by frame."""
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        entry['frame'] = int(entry['frame'])
        entry['x'] = float(entry['x'])
        entry['y'] = float(entry['y'])
        entry['score'] = float(entry.get('score', 0.0))
        entry['density'] = float(entry.get('density', 0.0))
        entry['speed_px_s'] = float(entry.get('speed_px_s', 0.0))
    data.sort(key=lambda e: e['frame'])
    return data


def _load_yolo_endpoints(detected_dir, ts):
    """Load detection endpoints from detected_objects_manual
    {obj_id: (x, y)}."""
    json_path = os.path.join(detected_dir, f"{ts}.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        objs = json.load(f)
    return {obj['id']: (float(obj['centroid'][0]), float(obj['centroid'][1]))
            for obj in objs}


def optimize_trajectories(traj_dir, opt_dir, detected_dir, start_ts, end_ts):
    """Load raw trajectory JSONs -> optimize via optimize_trajectory_v2 ->
    write optimized CSV.

    Returns:
        opt_csv: optimized trajectory CSV path
    """
    json_files = sorted(Path(traj_dir).glob("trajectory_obj_*.json"))
    if not json_files:
        raise RuntimeError(f"No raw trajectory JSONs found: {traj_dir}")

    start_endpoints = _load_yolo_endpoints(detected_dir, start_ts)
    end_endpoints = _load_yolo_endpoints(detected_dir, end_ts)
    print(f"Loaded start endpoints: {len(start_endpoints)} object(s) | "
          f"end endpoints: {len(end_endpoints)} object(s)")

    duration_s = (int(end_ts) - int(start_ts)) / 1_000_000.0

    optimized_rows = []
    for jf in json_files:
        obj_id = int(jf.stem.replace("trajectory_obj_", ""))

        traj = _load_trajectory_from_json(str(jf))
        if traj is None or len(traj) < 2:
            print(f"  [WARN] Object {obj_id}: insufficient trajectory data")
            continue

        # Start/end positions (detection endpoints preferred, fall back to
        # the first/last trajectory frames)
        start_pos = start_endpoints.get(obj_id, (traj[0]['x'], traj[0]['y']))
        end_pos = end_endpoints.get(obj_id, (traj[-1]['x'], traj[-1]['y']))

        n_frames = max(e['frame'] for e in traj)
        dt_per_frame = duration_s / n_frames if n_frames > 0 else 0.0

        # ── Optimize ──
        optimized = optimize_trajectory_v2(
            trajectory=traj,
            start_pos=start_pos,
            end_pos=end_pos,
            lambda_data=config.OPT_LAMBDA_DATA,
            lambda_vel=config.OPT_LAMBDA_VEL,
            lambda_acc=config.OPT_LAMBDA_ACC,
            lambda_end_guide=config.OPT_LAMBDA_END_GUIDE,
            lambda_start_boundary=config.OPT_LAMBDA_START_BOUNDARY,
            robust_iters=config.OPT_ROBUST_ITERS,
            robust_threshold=config.OPT_ROBUST_THRESHOLD,
            score_power=config.OPT_SCORE_POWER,
        )

        # ── Merge density/speed into the optimized trajectory ──
        raw_by_frame = {e['frame']: e for e in traj}
        for i, entry in enumerate(optimized):
            f = entry['frame']
            if f in raw_by_frame:
                entry['density'] = raw_by_frame[f].get('density', 0.0)
                entry['speed_px_s'] = raw_by_frame[f].get('speed_px_s', 0.0)
            else:
                entry['density'] = 0.0
                entry['speed_px_s'] = 0.0

            if entry['frame'] == 0:
                continue

            # Per-frame instantaneous speed (px/s)
            if i > 0:
                prev = optimized[i - 1]
                dx = entry['x'] - prev['x']
                dy = entry['y'] - prev['y']
                inst_speed = math.sqrt(dx ** 2 + dy ** 2) / dt_per_frame \
                    if dt_per_frame > 0 else 0.0
            else:
                inst_speed = 0.0

            optimized_rows.append([
                f"{start_ts}_{end_ts}",
                entry['frame'],
                obj_id,
                f"{entry['x']:.2f}",
                f"{entry['y']:.2f}",
                f"{entry.get('score', 0):.4f}",
                f"{entry.get('density', 0):.6f}",
                f"{inst_speed:.2f}",
                "",  # gt_error_px (no GT in the demo)
            ])

        print(f"  Object {obj_id}: {len(traj)} frame(s) -> optimization complete "
              f"(start=({start_pos[0]:.0f},{start_pos[1]:.0f}) "
              f"end=({end_pos[0]:.0f},{end_pos[1]:.0f}))")

    if not optimized_rows:
        raise RuntimeError("No trajectory was optimized successfully")

    os.makedirs(opt_dir, exist_ok=True)
    opt_csv = os.path.join(opt_dir, f"optimized_trajectories_{start_ts}_{end_ts}.csv")
    with open(opt_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "frame", "object_id", "x", "y", "score",
                         "density", "speed_px_s", "gt_error_px"])
        writer.writerows(optimized_rows)
    print(f"  [OK] Optimized trajectory: {opt_csv} ({len(optimized_rows)} row(s))")

    return opt_csv


# ==========================================================
# Trajectory visualization
# ==========================================================
def visualize_trajectory_on_frames(all_trajectories, event_frames_dir, start_ts,
                                   output_dir, img_h, img_w):
    """Draw trajectories on event frames and save as an image sequence."""
    os.makedirs(output_dir, exist_ok=True)

    all_points = {}
    all_scores = {}
    for obj_id, trajectory in all_trajectories.items():
        all_points[obj_id] = [(int(t['x']), int(t['y'])) for t in trajectory]
        all_scores[obj_id] = [t['score'] for t in trajectory]

    num_frames = len(list(all_trajectories.values())[0])

    for frame_idx in range(1, num_frames):         # skip frame 0 (start point, no dedicated event frame)
        part_num = frame_idx - 1                    # frame 1→part000, frame 2→part001, ...
        event_path = os.path.join(event_frames_dir, f"{start_ts}_part{part_num:03d}.jpg")

        if not os.path.exists(event_path):
            print(f"  [WARN] Skipping missing event frame {event_path}")
            continue

        img = cv2.imread(event_path)
        if img is None:
            print(f"  [WARN] Cannot load event frame {event_path}")
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        else:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for obj_id, points in all_points.items():
            scores = all_scores[obj_id]

            for j in range(len(points)):
                if j > frame_idx:
                    break

                pt = points[j]
                score = scores[j]
                color = (0, int(255 * score), int(255 * (1 - score)))
                cv2.circle(img, pt, 3, color, -1)

                if j > 0:
                    cv2.line(img, points[j-1], pt, color, 2)

            if frame_idx < len(points):
                current_pt = points[frame_idx]
                current_score = scores[frame_idx]
                current_color = (0, int(255 * current_score), int(255 * (1 - current_score)))
                cv2.circle(img, current_pt, 10, current_color, 3)
                cv2.circle(img, current_pt, 5, current_color, -1)
                text = f"ID:{obj_id}"
                cv2.putText(img, text, (current_pt[0] + 12, current_pt[1] - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

        frame_text = f"Frame: {frame_idx}/{num_frames-1} | Objects: {len(all_trajectories)}"
        cv2.putText(img, frame_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        output_path = os.path.join(output_dir, f"trajectory_frame_{frame_idx:03d}.jpg")
        cv2.imwrite(output_path, img)

    print(f"  [OK] Trajectory frame visualization complete: {output_dir}")


def visualize_trajectory_summary(all_trajectories, targets_info, output_path,
                                 title, img_h, img_w):
    """Generate a trajectory summary image (full trajectories on a single
    image)."""
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for obj_id, trajectory in all_trajectories.items():
        points = [(int(t['x']), int(t['y'])) for t in trajectory]
        scores = [t['score'] for t in trajectory]
        color_offset = (obj_id * 80) % 255

        for j in range(len(points)):
            pt = points[j]
            score = scores[j]
            color = (
                int((color_offset + 255 * (1 - score)) % 255),
                int((color_offset + 255 * score) % 255),
                int((color_offset + 128) % 255)
            )
            cv2.circle(img, pt, 3, color, -1)
            if j > 0:
                cv2.line(img, points[j-1], pt, color, 2)

        cv2.circle(img, points[0], 12, (0, 0, 255), 3)
        cv2.circle(img, points[0], 6, (0, 0, 255), -1)
        cv2.circle(img, points[-1], 12, (255, 0, 0), 3)
        cv2.circle(img, points[-1], 6, (255, 0, 0), -1)

        cv2.putText(img, f"Start:{obj_id}", (points[0][0]-40, points[0][1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"End:{obj_id}", (points[-1][0]-30, points[-1][1]+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    stats_text = f"Objects: {len(all_trajectories)} | Frames: {len(list(all_trajectories.values())[0])}"
    cv2.putText(img, stats_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    legend_y = 120
    cv2.circle(img, (20, legend_y), 8, (0, 0, 255), -1)
    cv2.putText(img, "Start", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(img, (120, legend_y), 8, (255, 0, 0), -1)
    cv2.putText(img, "End", (135, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(img, (200, legend_y), 8, (0, 255, 0), -1)
    cv2.putText(img, "High Score", (215, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(img, (320, legend_y), 8, (0, 0, 255), -1)
    cv2.putText(img, "Low Score", (335, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, img)
    print(f"  [OK] Trajectory summary image saved to: {output_path}")


# ==========================================================
# Entry point
# ==========================================================
def match_and_optimize(targets, match_config, trajectories_init, pair_speed_px,
                       event_frames_dir=None, traj_dir=None, opt_dir=None,
                       detected_dir=None, start_ts=None, end_ts=None,
                       num_parts=None):
    """Stage 3 main entry: frame-by-frame matching -> save raw trajectories +
    visualization -> trajectory optimization.

    Returns:
        result dict: {num_targets, num_frames, trajectories, opt_csv}
    """
    if event_frames_dir is None:
        event_frames_dir = config.EVENT_FRAMES_DIR
    if traj_dir is None:
        traj_dir = config.TRAJ_DIR
    if opt_dir is None:
        opt_dir = config.OPT_DIR
    if detected_dir is None:
        detected_dir = config.DETECTED_DIR
    if start_ts is None:
        start_ts = config.START_TS
    if end_ts is None:
        end_ts = config.END_TS
    if num_parts is None:
        num_parts = config.NUM_PARTS

    # ── 1. Frame-by-frame matching ──
    trajectories_raw = trajectories_init
    match_frames(targets, match_config, trajectories_raw, pair_speed_px,
                 event_frames_dir, start_ts, num_parts)

    # ── 2. Save raw trajectories + visualization ──
    save_trajectories(trajectories_raw, traj_dir, start_ts, end_ts)

    vis_dir = os.path.join(traj_dir, "visualization_frames")
    visualize_trajectory_on_frames(
        trajectories_raw, event_frames_dir, start_ts, vis_dir,
        config.IMG_H, config.IMG_W)
    visualize_trajectory_summary(
        trajectories_raw, targets,
        os.path.join(traj_dir, "summary_trajectory.jpg"),
        title=f"Trajectory: {start_ts} -> {end_ts} ({num_parts}x)",
        img_h=config.IMG_H, img_w=config.IMG_W)

    # ── 3. Trajectory optimization ──
    opt_csv = optimize_trajectories(traj_dir, opt_dir, detected_dir,
                                    start_ts, end_ts)

    return {
        "num_targets": len(targets),
        "num_frames": num_parts,
        "trajectories": trajectories_raw,
        "opt_csv": opt_csv,
    }
