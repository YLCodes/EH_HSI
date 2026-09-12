"""Stage 4: High-speed hyperspectral frame synthesis.

Extract the 61-channel spectral signature of each object (mask region) from
the start boundary frame H5, then transfer it frame by frame along the
optimized trajectories onto the background H5 to reconstruct the
hyperspectral frames at intermediate times.

Output: RECON_DIR/frame_{k:03d}.h5 (each frame contains hsi_R (61,2048,2448)
+ metadata attributes)
"""

import csv
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

import config
import stage3_rotation_matching as stage3_rot


# ==========================================================
# Data loading
# ==========================================================
def load_hsi_data(h5_path):
    """Load hyperspectral H5 -> (61, H, W)."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    print(f"Loading: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        if 'hsi_R' not in f:
            raise KeyError(f"No 'hsi_R' dataset in H5: {h5_path}")
        hsi = f['hsi_R'][()]
    print(f"  Shape: {hsi.shape}, value range: [{hsi.min():.4f}, {hsi.max():.4f}]")
    if hsi.shape != (config.NUM_BANDS, config.IMG_H, config.IMG_W):
        raise ValueError(f"Dimension mismatch: expected ({config.NUM_BANDS},{config.IMG_H},{config.IMG_W}), "
                         f"got {hsi.shape}")
    return hsi


def load_objects_info(detected_dir, start_ts):
    """Load start-frame object detection info."""
    path = os.path.join(detected_dir, f"{start_ts}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Detection info not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        objs = json.load(f)
    print(f"Loaded detection info: {len(objs)} object(s)")
    return objs


def load_trajectories(task_dir):
    """Load optimized trajectories -> {obj_id: [{frame, x, y, score}, ...]}

    Prefer optimized_trajectories_*.csv; fall back to trajectory_obj_*.json
    when missing.
    """
    csv_files = sorted(Path(task_dir).glob("optimized_trajectories_*.csv"))
    if csv_files:
        print(f"Loaded optimized trajectories: {csv_files[0].name}")
        traj = {}
        with open(csv_files[0], 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                oid = int(row['object_id'])
                if oid not in traj:
                    traj[oid] = []
                traj[oid].append({
                    'frame': int(float(row['frame'])),
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'score': float(row.get('score', 0)),
                })
        for oid in traj:
            traj[oid].sort(key=lambda e: e['frame'])
        return traj

    # JSON fallback
    json_files = sorted(Path(task_dir).glob("trajectory_obj_*.json"))
    if json_files:
        print(f"Loaded trajectory JSONs: {len(json_files)} object(s)")
        traj = {}
        for jf in json_files:
            oid = int(jf.stem.replace("trajectory_obj_", ""))
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            traj[oid] = [{'frame': int(e['frame']), 'x': float(e['x']),
                          'y': float(e['y']), 'score': float(e.get('score', 0))}
                         for e in data]
            traj[oid].sort(key=lambda e: e['frame'])
        return traj

    raise FileNotFoundError(f"No trajectories found: {task_dir}")


# ==========================================================
# Spectral extraction and transfer
# ==========================================================
def extract_targets(initial_hsi, objects_info):
    """Extract the 61-channel spectrum + mask of each object from the initial HSI."""
    targets = []
    for obj in objects_info:
        oid = obj['id']
        contour = np.array(obj['contour'], dtype=np.int32)
        full_mask = np.zeros((config.IMG_H, config.IMG_W), dtype=np.uint8)
        cv2.fillPoly(full_mask, [contour], 255)
        x, y, w, h = cv2.boundingRect(contour)

        spectrum = initial_hsi[:, y:y+h, x:x+w]
        tpl_mask = full_mask[y:y+h, x:x+w]
        masked = spectrum * (tpl_mask[None, :, :] / 255.0)

        targets.append({'id': oid, 'x': x, 'y': y, 'w': w, 'h': h,
                        'mask': tpl_mask, 'spectrum': masked})
        print(f"  Object {oid}: position=({x},{y}), size=({w},{h})")
    return targets


def migrate_spectrum(hsi, target, new_pos):
    """Transfer an object's spectrum to a new position (replace within the mask region)."""
    w, h = target['w'], target['h']
    spectrum = target['spectrum']
    mask = target['mask']

    cx, cy = new_pos
    nx = max(0, min(int(cx - w // 2), config.IMG_W - w))
    ny = max(0, min(int(cy - h // 2), config.IMG_H - h))

    mask_3d = mask[None, :, :] / 255.0
    roi = hsi[:, ny:ny+h, nx:nx+w]
    hsi[:, ny:ny+h, nx:nx+w] = roi * (1 - mask_3d) + spectrum * mask_3d
    return hsi


def save_hsi(hsi_data, output_path, frame_idx):
    """Save a reconstructed HSI as H5."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('hsi_R', data=hsi_data, dtype=np.float32, compression='gzip')
        f.attrs['wavelengths'] = config.WAVELENGTHS
        f.attrs['frame_idx'] = frame_idx
        f.attrs['num_bands'] = config.NUM_BANDS
        f.attrs['img_height'] = config.IMG_H
        f.attrs['img_width'] = config.IMG_W


# ==========================================================
# Entry point
# ==========================================================
def reconstruct(h5_start=None, h5_bg=None, detected_dir=None,
                task_dir=None, out_dir=None):
    """Stage 4 main entry: extract spectra -> transfer along optimized
    trajectories -> save reconstructed HSI frames.

    Returns:
        out_paths: list of reconstructed frame H5 paths (frame_001..N)
    """
    if h5_start is None:
        h5_start = config.H5_START
    if h5_bg is None:
        h5_bg = config.H5_BG
    if detected_dir is None:
        detected_dir = config.DETECTED_DIR
    if task_dir is None:
        task_dir = config.OPT_DIR
    if out_dir is None:
        out_dir = config.RECON_DIR

    # ── Load ──
    initial_hsi = load_hsi_data(h5_start)
    background_hsi = load_hsi_data(h5_bg)
    objects_info = load_objects_info(detected_dir, config.START_TS)
    trajectories = load_trajectories(task_dir)
    for oid, traj in trajectories.items():
        print(f"  Object {oid}: {len(traj)} frame(s)")

    # ── Extract spectra ──
    targets = extract_targets(initial_hsi, objects_info)

    # ── Reconstruct ──
    num_frames = max(max(t['frame'] for t in traj) for traj in trajectories.values())
    print(f"Reconstructing intermediate frames ({num_frames} frame(s))...")

    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    for k in range(1, num_frames + 1):
        recon = background_hsi.copy()
        for target in targets:
            oid = target['id']
            if oid not in trajectories:
                continue
            pos = next((t for t in trajectories[oid] if t['frame'] == k), None)
            if pos is None:
                continue
            recon = migrate_spectrum(recon, target, (pos['x'], pos['y']))

        out_path = os.path.join(out_dir, f"frame_{k:03d}.h5")
        save_hsi(recon, out_path, k)
        out_paths.append(out_path)
        print(f"  [OK] Saved reconstructed frame: {out_path}")

    return out_paths


# ==========================================================
# Rotation-scene synthesis
# ==========================================================
def _warp_volume(volume, M, out_size):
    """Apply the same 2x3 affine transform to every band of a (B, H, W) volume."""
    warped = np.empty_like(volume)
    for b in range(volume.shape[0]):
        warped[b] = cv2.warpAffine(volume[b], M, out_size,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    return warped


def reconstruct_rotation(center, radius, rotate_degree,
                         h5_start=None, h5_bg=None, out_dir=None):
    """Rotation-scene hyperspectral synthesis: the circular target region's
    spectrum is cumulatively rotated frame by frame and transferred onto the
    background, then saved as .h5.

    rotate_degree: final angle of each consecutive frame interval
    (length num_parts - 1). Outputs frame_{k:03d}.h5 (k = 1..len(rotate_degree));
    frame k = the target-region spectrum after cumulatively rotating by the
    first k interval angles.

    Returns:
        out_paths: list of reconstructed frame H5 paths
    """
    if h5_start is None:
        h5_start = config.H5_START
    if h5_bg is None:
        h5_bg = config.H5_BG
    if out_dir is None:
        out_dir = config.RECON_DIR

    initial = load_hsi_data(h5_start)
    background = load_hsi_data(h5_bg)

    # Circular mask
    yy, xx = np.ogrid[:config.IMG_H, :config.IMG_W]
    dist_center = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    target_region = initial.copy()            # circular target region
    target_region[:, dist_center > radius] = 0.0
    target_back = background.copy()           # background region (circle removed)
    target_back[:, dist_center <= radius] = 0.0

    # Spectral transfer
    res_images = target_region
    out_paths = []
    print(f"Spectral transfer synthesis ({len(rotate_degree)} frame(s))...")

    for i in range(len(rotate_degree)):
        m = stage3_rot.get_rotation_matrix(center, float(rotate_degree[i]))
        res_images = _warp_volume(res_images, m, (config.IMG_W, config.IMG_H))
        recon = target_back.copy()
        recon[:, dist_center <= radius] = res_images[:, dist_center <= radius]

        out_path = os.path.join(out_dir, f"frame_{i + 1:03d}.h5")
        save_hsi(recon, out_path, i + 1)
        out_paths.append(out_path)
        print(f"  [OK] Saved reconstructed frame: {out_path}")

    return out_paths
