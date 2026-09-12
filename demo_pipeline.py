"""demo_event high-speed hyperspectral frame interpolation pipeline — entry
script.

Starting from aligned event data, it sequentially runs five stages.
Select the motion scene via config.py's MOTION_MODE:

    MOTION_MODE = "translation" (default, translation scene):
        [1/5] Event splitting   stage1.split_events         NPZ -> polarity-encoded event-frame JPG
        [2/5] Template detection stage2.load_detections + stage2.detect_templates
                              detections -> mask -> Canny edge template
        [3/5] Object matching   stage3.match_and_optimize    edge-event template matching -> trajectory optimization
        [4/5] HSI synthesis     stage4.reconstruct           spectral extraction -> transfer along trajectories -> .h5
        [5/5] Pseudo-color viz  stage5.visualize_hsi_frames  reconstructed frames -> pseudo-color RGB PNG

    MOTION_MODE = "rotation" (rotation scene):
        [2/5] Template detection stage3_rot.detect_rotation_template Hough circle center + radius detection
        [3/5] Object matching   stage3_rot.match_rotation          per-frame-pair L1-loss angle estimation
                                                           -> joint rotation trajectory optimization
        [4/5] HSI synthesis     stage4.reconstruct_rotation       circular target region spectral rotation transfer -> .h5
        [5/5] Pseudo-color viz  stage5.visualize_hsi_frames  reconstructed frames -> pseudo-color RGB PNG

Final results: output/reconstructed_hsi/frame_*.h5 (61 bands) +
               output/visualization_rgb/RGB_frame_*.png (pseudo-color)

Usage:
    python demo_event/demo_pipeline.py

All parameters are centralized in config.py.
"""

import contextlib
import os
import sys
import time

# Windows terminals default to GBK encoding; explicitly switch to UTF-8 so
# that non-ASCII output displays correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
import stage1_event_split as stage1
import stage2_template_detection as stage2
import stage3_template_matching as stage3
import stage3_rotation_matching as stage3_rot
import stage4_hsi_reconstruction as stage4
import stage5_hsi_visualization as stage5


REQUIRED_FILES = [
    config.NPZ_PATH, config.H5_START, config.H5_END, config.H5_BG,
]
REQUIRED_JSONS = [
    os.path.join(config.DETECTED_DIR, f"{config.START_TS}.json"),
    os.path.join(config.DETECTED_DIR, f"{config.END_TS}.json"),
]


def banner(stage_no, title):
    """Print a stage banner."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"[{stage_no}/5] {title}")
    print(line)


def _pause():
    """Keep the window open on double-click runs; silently skipped in
    piped/redirected environments."""
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass


def ensure_data_present():
    """Verify that the input data is complete; on missing files, print copy
    instructions and exit."""
    common = [config.NPZ_PATH, config.H5_START, config.H5_BG]
    if config.MOTION_MODE == "rotation":
        # Rotation scene: needs the start RGB frame for circle detection;
        # detection JSONs and the end frame are not used
        required = common + [config.PNG_START]
    else:
        required = common + [config.H5_END] + REQUIRED_JSONS
    missing = [p for p in required
               if not os.path.exists(p)]
    if missing:
        print("\n[Error] Missing demo input data files:")
        for p in missing:
            print(f"   - {p}")
        print("\nPlease copy the data into demo_event/data/ following the "
              "checklist in README.md, then retry.")
        _pause()
        sys.exit(1)


def main():
    try:
        _run_pipeline()
    except Exception as e:
        print(f"\n[Failed] {e}")
        import traceback
        traceback.print_exc()
        _pause()
        sys.exit(1)
    _pause()


def _run_pipeline():
    t_total = time.time()

    print("=" * 70)
    print("demo_event high-speed hyperspectral frame interpolation pipeline")
    print(f"Timestamps: {config.START_TS} -> {config.END_TS}"
          f"  |  Multiplier: {config.NUM_PARTS}x")
    print(f"Motion scene: {config.MOTION_MODE}")
    print("=" * 70)
    n_recon = config.NUM_PARTS - 1 if config.MOTION_MODE == "rotation" \
        else config.NUM_PARTS
    print("Note: Stage 4 outputs 61-band full-resolution H5, ~1.05GB/frame, "
          f"{n_recon} frame(s) in total ~ {n_recon * 1.05:.1f}GB.")

    ensure_data_present()

    # ── [1/5] Event splitting ──
    banner(1, "Event splitting")
    t0 = time.time()
    num_events, t_duration = stage1.split_events()
    print(f"Event splitting complete: {num_events} events, duration {t_duration:.6f}s -> "
          f"{config.NUM_PARTS} frame(s) ({time.time() - t0:.1f}s)")

    if config.MOTION_MODE == "rotation":
        _run_rotation(t_total)
    else:
        _run_translation(t_total)


def _run_translation(t_total):
    """Translation scene: template detection -> edge-event template matching
    -> spectral transfer synthesis."""
    # ── [2/5] Template detection ──
    banner(2, "Template detection")
    t0 = time.time()
    start_objects = stage2.load_detections(config.DETECTED_DIR, config.START_TS)
    end_objects = stage2.load_detections(config.DETECTED_DIR, config.END_TS)
    targets, match_config, pair_speed_px, trajectories_init = \
        stage2.detect_templates(start_objects, end_objects,
                                config.START_TS, config.END_TS)
    print(f"Template detection complete: {len(targets)} object(s) "
          f"({time.time() - t0:.1f}s)")

    # ── [3/5] Object matching (incl. trajectory optimization) ──
    banner(3, "Object matching (incl. trajectory optimization)")
    t0 = time.time()
    os.makedirs(config.TRAJ_DIR, exist_ok=True)
    log_path = os.path.join(config.TRAJ_DIR, "matching_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_f, \
            contextlib.redirect_stdout(log_f):
        match_result = stage3.match_and_optimize(
            targets, match_config, trajectories_init, pair_speed_px)
    print(f"Object matching complete: {match_result['num_targets']} object(s) x "
          f"{match_result['num_frames']} frame(s), detailed log at {log_path}")
    print(f"Optimized trajectory: {match_result['opt_csv']} ({time.time() - t0:.1f}s)")

    # ── [4/5] Hyperspectral frame synthesis ──
    banner(4, "Hyperspectral frame synthesis")
    t0 = time.time()
    out_paths = stage4.reconstruct()
    print(f"Hyperspectral frame synthesis complete: {len(out_paths)} frame(s) "
          f"({time.time() - t0:.1f}s)")

    # ── [5/5] Hyperspectral pseudo-color visualization ──
    banner(5, "Hyperspectral pseudo-color visualization")
    t0 = time.time()
    vis_dir = stage5.visualize_hsi_frames()
    print(f"Pseudo-color visualization complete: {vis_dir} ({time.time() - t0:.1f}s)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Pipeline finished successfully!")
    print("=" * 70)
    print(f"  Motion scene:             {config.MOTION_MODE}")
    print(f"  Event split frames:       {config.EVENT_FRAMES_DIR}")
    print(f"  Raw trajectories:         {config.TRAJ_DIR}")
    print(f"  Optimized trajectories:   {match_result['opt_csv']}")
    print(f"  HSI interpolated frames:  {config.RECON_DIR} ({len(out_paths)} frame(s))")
    print(f"  Pseudo-color visualization: {vis_dir}")
    print(f"  Total time:               {time.time() - t_total:.1f}s")
    print("=" * 70)


def _run_rotation(t_total):
    """Rotation scene: Hough circle detection -> per-frame-pair angle
    estimation + optimization -> spectral transfer synthesis."""
    # ── [2/5] Template detection (rotation circle detection) ──
    banner(2, "Template detection (Hough circle detection)")
    t0 = time.time()
    center, radius = stage3_rot.detect_rotation_template()
    print(f"Template detection complete: center={center}, radius={radius} "
          f"({time.time() - t0:.1f}s)")

    # ── [3/5] Object matching (rotation angle estimation + trajectory
    # optimization) ──
    banner(3, "Object matching (rotation angle estimation + trajectory optimization)")
    t0 = time.time()
    os.makedirs(config.TRAJ_DIR, exist_ok=True)
    log_path = os.path.join(config.TRAJ_DIR, "matching_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_f, \
            contextlib.redirect_stdout(log_f):
        rot_result = stage3_rot.match_rotation(center, radius)
    print(f"Rotation matching complete: {len(rot_result['rotate_degree'])} angle interval(s), "
          f"detailed log at {log_path} ({time.time() - t0:.1f}s)")

    # ── [4/5] Hyperspectral frame synthesis (spectral transfer) ──
    banner(4, "Hyperspectral frame synthesis (spectral transfer)")
    t0 = time.time()
    out_paths = stage4.reconstruct_rotation(
        center, radius, rot_result["rotate_degree"])
    print(f"Hyperspectral frame synthesis complete: {len(out_paths)} frame(s) "
          f"({time.time() - t0:.1f}s)")

    # ── [5/5] Hyperspectral pseudo-color visualization ──
    banner(5, "Hyperspectral pseudo-color visualization")
    t0 = time.time()
    vis_dir = stage5.visualize_hsi_frames()
    print(f"Pseudo-color visualization complete: {vis_dir} ({time.time() - t0:.1f}s)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Pipeline finished successfully!")
    print("=" * 70)
    print(f"  Motion scene:             {config.MOTION_MODE}")
    print(f"  Event split frames:       {config.EVENT_FRAMES_DIR}")
    print(f"  Rotation trajectory:      {config.TRAJ_DIR} (rotation_trajectory_*.json)")
    print(f"  HSI interpolated frames:  {config.RECON_DIR} ({len(out_paths)} frame(s))")
    print(f"  Pseudo-color visualization: {vis_dir}")
    print(f"  Total time:               {time.time() - t_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
