"""Stage 1: Event splitting.

Slice the aligned event stream (NPZ) into equal time intervals and render
each as a polarity-encoded event-frame JPG (255=ON / 128=OFF), used for
per-frame matching in Stage 3.
"""

import os

import cv2
import numpy as np

import config


def split_events(npz_path=None, num_parts=None, output_dir=None):
    """
    Process a single npz file: split along the time dimension and render
    event frames.

    Args:
        npz_path:   aligned event data npz path (keys: x, y, t, p),
                    default config.NPZ_PATH
        num_parts:  number of splits, default config.NUM_PARTS
        output_dir: output directory, default config.EVENT_FRAMES_DIR

    Returns:
        (num_events, t_duration): total number of events and time span (seconds)
    """
    if npz_path is None:
        npz_path = config.NPZ_PATH
    if num_parts is None:
        num_parts = config.NUM_PARTS
    if output_dir is None:
        output_dir = config.EVENT_FRAMES_DIR

    data = np.load(npz_path)
    events = {
        'x': data['x'],
        'y': data['y'],
        't': data['t'],
        'p': data['p']
    }
    num_events = len(events['t'])
    filename = os.path.basename(npz_path).replace('.npz', '')

    # Sort by time
    sort_indices = np.argsort(events['t'])
    for key in events:
        events[key] = events[key][sort_indices]

    t_min = events['t'][0]
    t_max = events['t'][-1]
    t_duration = t_max - t_min
    time_interval = t_duration / num_parts

    os.makedirs(output_dir, exist_ok=True)

    for part_idx in range(num_parts):
        t_start = t_min + part_idx * time_interval
        t_end = t_min + (part_idx + 1) * time_interval

        mask = (events['t'] >= t_start) & (events['t'] < t_end)
        part_events = {
            'x': events['x'][mask],
            'y': events['y'][mask],
            'p': events['p'][mask]
        }

        # Event-frame rendering: 255=ON, 128=OFF
        canvas = np.zeros(config.IMAGE_SIZE, dtype=np.uint8)

        for x, y, p in zip(part_events['x'], part_events['y'], part_events['p']):
            x_int, y_int = int(x), int(y)
            half_size = config.EVENT_SIZE // 2

            if 0 <= x_int - half_size and x_int + half_size < config.IMAGE_SIZE[1] and \
               0 <= y_int - half_size and y_int + half_size < config.IMAGE_SIZE[0]:
                canvas[y_int, x_int] = 255 if p == 1 else 128

        output_filename = f"{filename}_part{part_idx:03d}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, canvas)

    return num_events, t_duration
