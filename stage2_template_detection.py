"""Stage 2: Template detection.

Flow:
    1. load_detections   — load object detections of the start/end frames
                           [{id, centroid, contour}]
    2. detect_templates  — contour -> full-frame mask -> bbox crop -> Canny
                           edge extraction + dilation -> subsampled template;
                           compute search radius, object speed; initialize
                           frame-0 trajectories
"""

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

import config


# ==========================================================
# Template construction
# ==========================================================
def extract_edge_template(mask, thickness=1):
    """Extract an edge template (Canny edge detection + dilation to thicken)."""
    if mask is None or mask.size == 0:
        print(f"  [WARN] extract_edge_template: mask is empty, returning a 1x1 placeholder template")
        return np.ones((1, 1), dtype=np.uint8)
    mask = np.ascontiguousarray(mask)

    edges = cv2.Canny(mask, config.CANNY_THRESHOLD_LOW, config.CANNY_THRESHOLD_HIGH)

    if edges.size == 0:
        print(f"  [WARN] extract_edge_template: Canny result is empty (mask shape={mask.shape}), returning a 1x1 placeholder template")
        return np.ones((1, 1), dtype=np.uint8)

    if thickness <= 1:
        return (edges > 0).astype(np.uint8)

    kernel_size = 2 * thickness - 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    return (dilated_edges > 0).astype(np.uint8)


def _subsample_template(dense_template, n_target, th, tw):
    """Randomly subsample n_target edge points from the dense template
    (fixed seed for reproducibility)."""
    ys, xs = np.where(dense_template > 0)
    n_available = len(ys)
    if n_available <= n_target:
        # No subsampling needed
        return dense_template

    rng = np.random.RandomState(42)  # fixed seed for reproducibility
    indices = rng.choice(n_available, size=n_target, replace=False)
    sparse = np.zeros((th, tw), dtype=np.uint8)
    sparse[ys[indices], xs[indices]] = 1
    return sparse


@dataclass
class MatchConfig:
    """Edge-event matching parameter set."""
    sharpness: float = 1.0
    density_adapt: float = 0.0
    strip_constraint_weight: float = 0.3
    polarity_mode: str = 'ON'       # 'ON' | 'OFF' | 'BOTH'
    template_mode: str = 'mask'     # 'bbox' | 'mask' | 'edge'
    score_mode: str = 'gaussian'    # 'binary' | 'gaussian' | 'distance'


# ==========================================================
# Detection loading and template construction
# ==========================================================
def load_detections(detected_dir, ts):
    """Load object detection info of the given timestamp [{id, centroid, contour}]."""
    path = os.path.join(detected_dir, f"{ts}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Detection info not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        objects = json.load(f)
    print(f"Loaded detection info ({ts}): {len(objects)} object(s)")
    return objects


def detect_templates(start_objects, end_objects, start_ts, end_ts,
                     variant_config=None):
    """Template detection: build an edge template for each detected object in
    the start frame and initialize frame-0 trajectories.

    Args:
        start_objects: start-frame detections [{id, centroid, contour}]
        end_objects:   end-frame detections (used for speed estimation)
        start_ts, end_ts: start/end timestamps (microseconds)
        variant_config: optional matching parameter overrides dict
                        (template_mode/score_mode etc.)

    Returns:
        (targets, match_config, pair_speed_px, trajectories_init)
        targets: each object contains id/cx/cy/contour/template_mask/
                 edge_template/w/h/th/tw/search_radius
        trajectories_init: {obj_id: [frame-0 trajectory entry]}
    """
    variant_config = variant_config or {}

    targets = []
    for obj in start_objects:
        obj_id = obj["id"]
        cx, cy = obj["centroid"]
        contour = np.array(obj["contour"], dtype=np.int32)

        full_mask = np.zeros((config.IMG_H, config.IMG_W), dtype=np.uint8)
        cv2.fillPoly(full_mask, [contour], 255)
        bx, by, bw, bh = cv2.boundingRect(contour)
        template_mask = full_mask[by:by + bh, bx:bx + bw]

        search_radius = int(max(bw, bh) * config.SEARCH_SCALE)

        targets.append({
            "id": obj_id, "cx": cx, "cy": cy,
            "contour": contour,
            "template_mask": template_mask,
            "w": bw, "h": bh,
            "search_radius": search_radius,
        })
        print(f"  Object {obj_id}: center=({cx}, {cy}), size=({bw}, {bh}), "
              f"search radius={search_radius}")

    # Reference edge template (determines the edge-point count per object so
    # that computation is comparable across modes)
    _tmpl_ref = extract_edge_template(
        targets[0]["template_mask"], thickness=config.EDGE_THICKNESS) \
        if targets else np.ones((1, 1), dtype=np.uint8)
    _n_ref_pts = max(int(np.sum(_tmpl_ref > 0)), 100)

    for t in targets:
        template_mode = variant_config.get('template_mode', 'mask')
        if template_mode == 'edge':
            t["edge_template"] = extract_edge_template(
                t["template_mask"], thickness=config.EDGE_THICKNESS)
        elif template_mode == 'bbox':
            # All-ones template: random subsampling to the reference edge-point count
            dense = np.ones((t["h"], t["w"]), dtype=np.uint8)
            t["edge_template"] = _subsample_template(
                dense, _n_ref_pts, t["h"], t["w"])
        elif template_mode == 'mask':
            # Filled-mask template: random subsampling
            dense = (t["template_mask"] > 0).astype(np.uint8)
            t["edge_template"] = _subsample_template(
                dense, _n_ref_pts, t["h"], t["w"])
        else:
            raise ValueError(f"Unknown template_mode: {template_mode}")
        t["th"], t["tw"] = t["edge_template"].shape

    # ── Speed computation: centroid displacement / time difference ──
    if len(end_objects) > 0 and len(targets) > 0:
        start_cx = targets[0]["cx"]
        start_cy = targets[0]["cy"]
        end_cx = end_objects[0]["centroid"][0]
        end_cy = end_objects[0]["centroid"][1]
        dist_px = np.sqrt((end_cx - start_cx) ** 2 + (end_cy - start_cy) ** 2)
        dt_s = (float(end_ts) - float(start_ts)) / 1e6
        pair_speed_px = dist_px / dt_s if dt_s > 0 else 0.0
        print(f"  Speed: displacement={dist_px:.1f}px, time={dt_s*1000:.2f}ms, "
              f"speed={pair_speed_px:.1f}px/s")
    else:
        pair_speed_px = 0.0

    # Initialize frame-0 trajectories (start-frame centroids)
    trajectories_init = {}
    for t in targets:
        trajectories_init[t["id"]] = [{
            "frame": 0, "x": t["cx"], "y": t["cy"],
            "score": 1.0, "density": 0.0,
            "speed_px_s": pair_speed_px,
        }]

    vc = variant_config
    match_config = MatchConfig(
        sharpness=vc.get('sharpness', config.SCORE_SHARPNESS),
        density_adapt=vc.get('density_adapt', config.DENSITY_ADAPT),
        strip_constraint_weight=vc.get('strip_constraint_weight',
                                       config.STRIP_CONSTRAINT_WEIGHT),
        polarity_mode=vc.get('polarity_mode', 'ON'),
        template_mode=vc.get('template_mode', 'mask'),
        score_mode=vc.get('score_mode', 'gaussian'),
    )

    return targets, match_config, pair_speed_px, trajectories_init
