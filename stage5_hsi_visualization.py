"""Stage 5: Hyperspectral pseudo-color visualization.

Convert the 61-band hyperspectral frames (.h5) reconstructed in Stage 4 into
pseudo-color RGB images via band stacking:
    R = 600-700 nm, G = 500-600 nm, B = 400-500 nm (config.VIS_R/G/B_BANDS)

Common processing: percentile normalization (robust to extreme values)
+ gray-world white balance + optional gamma.
Output: VIS_DIR/RGB_{frame}_stack_{mode}_spectral.png
"""

import os
from glob import glob

import h5py
import numpy as np

import config


# ==========================================================
# Basic utilities
# ==========================================================
def robust_norm01(img, norm_p=99.5, gamma=1.0):
    """Robustly normalize a single/multi-channel image to [0, 1]."""
    hi = np.percentile(img, norm_p)
    img = img / (hi + 1e-12)
    img = np.clip(img, 0.0, 1.0)
    if gamma is not None and abs(gamma - 1.0) > 1e-6:
        img = np.power(img, 1.0 / float(gamma))
    return img


def gray_world_wb(rgb01):
    """Gray-world white balance: rgb01 float in [0, 1]."""
    m = rgb01.reshape(-1, 3).mean(axis=0) + 1e-12
    gain = m.mean() / m
    out = rgb01 * gain
    return np.clip(out, 0.0, 1.0)


def bands_to_indices(bands_nm, start_nm=400, step_nm=10):
    bands_nm = np.asarray(bands_nm).astype(np.int32)
    idx = (bands_nm - start_nm) // step_nm
    return idx.astype(np.int32)


# ==========================================================
# Band-stacking RGB
# ==========================================================
def make_rgb_by_band_stack(hsi, start_nm=400, step_nm=10,
                           r_bands=(700,), g_bands=(550,), b_bands=(450,),
                           mode="single",
                           norm_p=99.5, apply_wb=True, gamma=1.0):
    """Band-stacking RGB: pick bands per channel (mean across bands when
    mode="sum") -> (H, W, 3) float in [0, 1]."""
    C, H, W = hsi.shape

    def pick(bands):
        idx = bands_to_indices(bands, start_nm, step_nm)
        if idx.min() < 0 or idx.max() >= C:
            raise ValueError(f"band idx out of range. C={C}, idx=[{idx.min()},{idx.max()}], bands={bands}")
        cube = hsi[idx, :, :]  # (K,H,W)
        if mode == "single":
            return cube[0]
        elif mode == "sum":
            return cube.mean(axis=0)
        else:
            raise ValueError("mode must be 'single' or 'sum'")

    R = pick(np.asarray(r_bands, dtype=np.int32))
    G = pick(np.asarray(g_bands, dtype=np.int32))
    B = pick(np.asarray(b_bands, dtype=np.int32))

    rgb = np.stack([R, G, B], axis=-1).astype(np.float32)  # (H,W,3)

    # Per-channel robust normalization
    for ch in range(3):
        rgb[..., ch] = robust_norm01(rgb[..., ch], norm_p=norm_p, gamma=1.0)

    # Optional white balance
    if apply_wb:
        rgb = gray_world_wb(rgb)

    # Global gamma
    if gamma is not None and abs(gamma - 1.0) > 1e-6:
        rgb = np.power(rgb, 1.0 / float(gamma))

    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


# ==========================================================
# Saving
# ==========================================================
def save_rgb_spectral(rgb_u8, out_path):
    """Save via spectral.imshow + matplotlib (identical output path to 11_)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import spectral

    # spectral.imshow expects RGB order; do not use ::-1
    plt.figure()
    spectral.imshow(rgb_u8, (0, 1, 2))
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(rgb_u8.shape[1] / config.VIS_DPI,
                        rgb_u8.shape[0] / config.VIS_DPI)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(out_path, transparent=True, dpi=config.VIS_DPI, pad_inches=0)
    plt.close(fig)


# ==========================================================
# Entry point
# ==========================================================
def visualize_hsi_frames(recon_dir=None, out_dir=None):
    """Convert reconstructed HSI frames (.h5) to pseudo-color RGB PNGs
    (identical to the 11_ output).

    Returns:
        out_dir: visualization output directory
    """
    if recon_dir is None:
        recon_dir = config.RECON_DIR
    if out_dir is None:
        out_dir = config.VIS_DIR

    h5_files = sorted(glob(os.path.join(recon_dir, "frame_*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No reconstructed frames found: {recon_dir}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Pseudo-color visualization ({len(h5_files)} frames)...")

    for fp in h5_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        with h5py.File(fp, "r") as f:
            hsi = f["hsi_R"][()].astype(np.float32)

        rgb01 = make_rgb_by_band_stack(
            hsi,
            start_nm=config.VIS_START_NM,
            step_nm=config.VIS_STEP_NM,
            r_bands=config.VIS_R_BANDS,
            g_bands=config.VIS_G_BANDS,
            b_bands=config.VIS_B_BANDS,
            mode=config.VIS_STACK_RGB_MODE,
            norm_p=config.VIS_NORM_P,
            apply_wb=config.VIS_APPLY_WB,
            gamma=config.VIS_GAMMA)
        out_name = f"RGB_{base}_stack_{config.VIS_STACK_RGB_MODE}_spectral.png"

        rgb_u8 = (rgb01 * 255.0 + 0.5).astype(np.uint8)
        out_path = os.path.join(out_dir, out_name)
        save_rgb_spectral(rgb_u8, out_path)
        print(f"  [OK] {base} -> {out_name}")

    return out_dir
