# Pose Extraction - Module Usage Guide

This README provides step-by-step instructions for running the pose extraction pipeline from the `src/pose_extraction` package directory. This module focuses solely on extracting pose landmarks from videos using MediaPipe.

## Prerequisites

- **Python 3.9** (required)
- All dependencies installed (see project root README for setup)
- Activate your virtual environment:
  ```bash
  source .venv/bin/activate
  ```

## Pose Extraction: Video to 3D Pose CSV

The `pose_extraction.py` module extracts 2D and 3D pose landmarks from videos using MediaPipe and exports them to CSV format with timestamps. It also generates overlay videos showing the detected poses for review.

### Command-Line Usage

Extract poses from all videos in data/video (default):
```bash
python -m pose_extraction.main
```

Extract poses from a single video:
```bash
python -m pose_extraction.main --video data/video/your_video.mp4
```

Extract poses from all videos in a specific directory:
```bash
python -m pose_extraction.main --input-dir data/video
```

With Rerun visualization (real-time 3D pose tracking):
```bash
python -m pose_extraction.main --video data/video/your_video.mp4 --use-rerun
```

Disable overlay video generation (faster processing):
```bash
python -m pose_extraction.main --no-overlay-video
```

### Python API Usage

```python
from pose_extraction import PoseExtractor

# Initialize extractor
extractor = PoseExtractor(use_rerun=False, generate_overlay_video=True)  # Set to False to disable overlay videos

# Extract poses from a single video
pose_data = extractor.extract_pose_from_video("data/video/your_video.mp4")
print(f"Extracted pose data from {len(pose_data)} frames")

# Process all videos in a directory
extractor.process_video_directory("data/video", "data/poses")
```

### Output Files

The system generates two types of output files:

1. **CSV files**: Pose landmark data saved in `data/poses/` with the same name as the input video
2. **Overlay videos**: Videos with pose landmarks overlaid, saved in `data/video_with_pose/` with `_with_pose` suffix

### CSV Output Format

The extracted pose data is saved in CSV format with the following columns:

- `timestamp`: Frame timestamp in seconds
- `frame_number`: Frame index
- `{keypoint}_x`, `{keypoint}_y`: 2D pixel coordinates
- `{keypoint}_z`: 3D depth coordinates (relative to camera)
- `{keypoint}_confidence`: Confidence scores (0-1)

Example CSV structure:
```csv
timestamp,frame_number,nose_x,nose_y,nose_z,nose_confidence,left_eye_x,left_eye_y,left_eye_z,left_eye_confidence,...
0.0,0,320.5,240.2,0.1,0.95,315.2,235.8,0.12,0.94,...
0.033,1,321.1,239.8,0.12,0.94,316.1,235.1,0.13,0.93,...
```

### Overlay Video Features

The generated overlay videos include:
- **Pose landmarks**: Blue circles with white borders for each detected keypoint
- **Pose connections**: Green lines connecting related body parts
- **Confidence filtering**: Only high-confidence landmarks (>0.5) are displayed
- **Frame information**: Text overlay showing the number of detected landmarks
- **Original video quality**: Same resolution and frame rate as the input video

### Supported Keypoints

The system extracts 33 pose keypoints including:
- Face: nose, eyes, ears, mouth
- Upper body: shoulders, elbows, wrists, fingers
- Lower body: hips, knees, ankles, feet

## 1. Extract Poses from a Video

You can extract pose data from a single video file using the `PoseExtractor` class:

```python
from pose_extraction import PoseExtractor

extractor = PoseExtractor(use_rerun=True, generate_overlay_video=True)  # Set to False to disable features
pose_data = extractor.extract_pose_from_video("data/video/your_video.mp4")
print(f"Extracted {len(pose_data)} frames. CSV saved in data/poses/, overlay video in data/video_with_pose/")
```

## 2. Run the Full Pipeline (Recommended)

You can run the complete pose extraction pipeline using the main pipeline class:

```python
from pose_extraction.main import PoseExtractionPipeline

pipeline = PoseExtractionPipeline(use_rerun=False, generate_overlay_video=True)
results = pipeline.run_full_pipeline("data/video/your_video.mp4")
print("Pipeline complete!")
print(f"Pose CSV: {results['pose_csv_path']}")
print(f"Overlay video: {results['overlay_video_path']}")
print(f"Frames processed: {results['frame_count']}")
```

## 3. Batch Processing: All Videos in a Directory

To process all videos in `data/video/`:

```python
pipeline = PoseExtractionPipeline()
results = pipeline.process_video_directory(input_dir="data/video")
for result in results:
    print(f"Processed {result['video_name']}: {result['pose_csv_path']}")
    if result.get('overlay_video_path'):
        print(f"  Overlay video: {result['overlay_video_path']}")
```

## 4. Command-Line Usage (from project root)

You can also run the pipeline from the command line:

```bash
python -m pose_extraction.main --video data/video/your_video.mp4
# Or batch process:
python -m pose_extraction.main --input-dir data/video
```

## Notes
- All outputs (CSV files) are saved in the `data/poses/` folder.
- Overlay videos are saved in the `data/video_with_pose/` folder.
- Use `--no-overlay-video` flag for faster processing when overlay videos aren't needed.
- For more advanced usage, see the [examples/basic_usage.py](../../examples/basic_usage.py) script.
- For troubleshooting and more details, see the main project [README](../../README.md).
- This module focuses only on pose extraction. For embedding generation and motion analysis, see the separate modules. 