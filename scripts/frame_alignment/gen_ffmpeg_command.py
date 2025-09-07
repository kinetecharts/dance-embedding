from pathlib import Path

import polars as pl

DURATION_SECONDS: int = 120
FRAME_RATE: int = 240
TRANSPOSE:int = 1
FRAME_RANGE: tuple[int] = (0, 30*FRAME_RATE)

DATA_FILE_PATH: Path = Path("data/dev/kettlebell sport/snatch/2025-09-05/trimmed/light_source_alignment_output.parquet")

r3: pl.DataFrame = pl.read_parquet(DATA_FILE_PATH).filter(pl.col("recording") == 3)

r3 = (r3
        .select("name", "camera", "frame", "value", "smoothed_value")
        .filter(
            pl.col("frame") >= FRAME_RANGE[0],
            pl.col("frame") <= FRAME_RANGE[1]
        )
        .group_by("name", "camera").agg(pl.max("value"))
        .join(
            r3.select("name","camera", "frame", "value"),
            on=["name", "camera", "value"],
            how="inner"
        )
        .group_by("name", "camera", "value")
        .agg(pl.max("frame"))
        .with_columns(end_frame=pl.col("frame") + (DURATION_SECONDS * FRAME_RATE))
        .sort("camera")
    )

for d in r3.to_dicts():
    runtime_seconds: int = int((d['end_frame'] - d['frame']) / FRAME_RATE)

    print(f"ffmpeg -y -r {FRAME_RATE} -i {d['name']}.MOV -r {FRAME_RATE} -an -vf select=\"between(n\\,{d['frame']}\\,{d['end_frame']}),transpose={TRANSPOSE},setpts=PTS-STARTPTS\" -c:v libx264 -f mp4 {d['name']}-{runtime_seconds}sec.mp4")