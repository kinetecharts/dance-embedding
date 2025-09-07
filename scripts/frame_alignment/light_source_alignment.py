from glob import glob
from pathlib import Path

import cv2

import altair as alt
import numpy as np
import polars as pl

from scipy.signal import savgol_filter


FILE_EXT: str = "MOV"

VIDEO_FILE_PATH: Path = Path("../../data/dev/kettlebell sport/snatch/2025-09-05")


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



print("## Calculate Mean of HSV for each Frame ##")
files: list[Path] = [Path(p) for p in sorted(glob(f"{VIDEO_FILE_PATH}/*.{FILE_EXT}"))]

results: list[dict] = []

for file_index, mov in enumerate(files):
    file_index += 1
    pth: Path = Path(mov)
    camera: str = pth.stem.split("_")[0]
    recording: int = int(pth.stem.split("-")[-1])

    print(f"  {file_index}. '{mov}'")
    mean_hsv: np.array = mean_hsv_per_frame(pth)
    # 20 frames is 1/12 of a second
    smoothed_hsv: np.array = savgol_filter(mean_hsv, 20, 3)

    results.append(
        pl.from_dict(
            {
                "name": pth.stem,
                "folder": VIDEO_FILE_PATH.stem,
                "file": mov.stem,
                "file_index": file_index,
                "camera": camera,
                "recording": recording,
                "frame": range(1, len(mean_hsv) + 1),
                "value": mean_hsv,
                "smoothed_value": smoothed_hsv,
                "path": str(pth.absolute()),
            }
        )
    )

print("## Write results to Parquet ##")
df: pl.DataFrame = pl.concat(results)

file_name: str = f"{VIDEO_FILE_PATH}/light_source_alignment_output.parquet"
df.write_parquet(file_name)

print("## Generate Plots ##")
for recording in df["recording"].unique():
    print(f"Generated plots for recording {recording}...")
    _df: pl.DataFrame = df.filter(pl.col("recording") == recording).select("camera", "frame", "value", "smoothed_value").to_pandas()

    chart_values: alt.Chart = (
        alt.Chart(
            data=_df,
            title=f"Raw Mean HSV Values for Recording {recording}",
            width=1600,
            height=900
        )
        .mark_line()
        .encode(
            x=alt.X("frame", title="Frame"),
            y=alt.Y("value", title="Mean HSV (Raw)"),
            color="camera"
        )
    )

    chart_smoothed_values: alt.Chart = (
        alt.Chart(
            data=_df,
            title=f"Savgol Filtered HSV Values for Recording {recording}",
            width=1600,
            height=900
        )
        .mark_line()
        .encode(
            x=alt.X("frame", title="Frame"),
            y=alt.Y("smoothed_value", title="Mean HSV (Smoothed)"),
            color="camera"
        )
    )

    chart_values.save(f"{VIDEO_FILE_PATH}/mean_hsv-recording_{recording}-raw.png")
    chart_smoothed_values.save(f"{VIDEO_FILE_PATH}/mean_hsv-recording_{recording}-savgol.png")