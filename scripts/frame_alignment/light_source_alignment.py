from glob import glob
from os import makedirs
from pathlib import Path
from shutil import rmtree

import cv2

import numpy as np
import polars as pl

from scipy.signal import savgol_filter

def mean_hsv_per_frame(f: Path) -> np.array:
    cap: cv2.VideoCapture = cv2.VideoCapture(f)

    buffer: list[np.float] = []

    while cap.isOpened():
        # bool, np.array
        ret, frame = cap.read()

        if not ret:
            break
        else:
            hsv: np.array = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_mean: tuple = cv2.mean(hsv)

            buffer.append(np.max(hsv_mean))

    cap.release()

    return np.array(buffer)


## FILE TEST ##
# TODO: Move code after `## FILE TEST ##` from this file
FILE_EXT: str = "MOV"

OUTPUT_PATH:str = "data/dev/light source alignment/2025-08-01"
FILE_OUTPUT_PATH:str = f"{OUTPUT_PATH}/light_source_alignment"
TEST_VIDEO_DIR_PATH: str = f"{OUTPUT_PATH}/00*"

results: list[dict] = []

print("## Calculate Mean of HSV for each Frame ##")
for folder in glob(TEST_VIDEO_DIR_PATH):
    files: list = sorted(glob(f"{folder}/*.{FILE_EXT}"))
    folder: Path = Path(folder)

    for file_index,mov in enumerate(files):
        file_index += 1
        pth: Path = Path(mov)

        print(f"  {file_index}. '{mov}'")
        mean_hsv: np.array = mean_hsv_per_frame(pth)
        # 20 frames is 1/12 of a second
        smoothed_hsv: np.array = savgol_filter(mean_hsv, 20, 3)

        results.append(pl.from_dict({
            "name": pth.stem,
            "folder": folder.stem,
            # "folder": f"00{file_index}",
            "file_index": file_index,
            "frame": range(1, len(mean_hsv) + 1),
            "value": mean_hsv,
            "smoothed_value": smoothed_hsv,
            "path": str(pth.absolute()),
        }))

rmtree(FILE_OUTPUT_PATH, ignore_errors=True)
makedirs(FILE_OUTPUT_PATH, exist_ok=True)

print("## Write results to Parquet ##")
df: pl.DataFrame = pl.concat(results)

file_name: str = f"{FILE_OUTPUT_PATH}/light_source_alignment_output.parquet"
df.write_parquet(file_name)
