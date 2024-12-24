"""
This script calculates the mean and standard deviation (std) of the frames
in the Jester dataset.

To reduce memory complexity, the mean and std of the overall dataset are
calculated incrementally. For the mean, a weighted mean approach is used. For
the std, the sum of weighted vars is used, as stds cannot be summed directly.

To speed up the process, the script uses threading with a dynamic number of
workers.

Author(s): Sana Asghari, Jort van Leenen
License: GNU General Public License v3.0 (GPLv3)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def process_video(video_folder):
    """
    Process a single video folder, calculating its mean, variance,
    and pixel count.
    """
    mean = np.zeros(3, dtype=np.float64)
    var = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for file in video_folder.iterdir():
        frame = np.array(Image.open(file)) / 255.0
        pixels = frame.shape[0] * frame.shape[1]
        frame_mean = frame.mean(axis=(0, 1))
        frame_var = frame.var(axis=(0, 1))

        mean += frame_mean * pixels
        var += frame_var * pixels
        pixel_count += pixels

    return mean, var, pixel_count


def calculate_mean_std(data_root):
    """
    Calculate mean and std of the dataset using threading with a dynamic number
    of workers.
    """
    data_dir = Path(data_root)
    video_folders = [folder for folder in data_dir.iterdir() if folder.is_dir()]

    max_workers = min(32, os.cpu_count() or 1)

    total_mean = np.zeros(3, dtype=np.float64)
    total_var = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, folder): folder
                   for folder in video_folders}
        with tqdm(total=len(video_folders), desc="Processing videos",
                  unit="video") as pbar:
            for future in as_completed(futures):
                mean, var, pixel_count = future.result()
                total_mean += mean
                total_var += var
                total_pixels += pixel_count
                pbar.update(1)

    total_mean /= total_pixels
    total_std = np.sqrt(total_var / total_pixels)
    print(f"Mean: {total_mean}")
    print(f"Standard Deviation: {total_std}")


if __name__ == "__main__":
    data_root = './20bn-jester-v1'
    calculate_mean_std(data_root)
