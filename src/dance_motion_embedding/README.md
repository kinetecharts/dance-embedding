# Dance Motion Embedding - Module Usage Guide

This README provides step-by-step instructions for running the dance motion embedding pipeline from the `src/dance_motion_embedding` package directory. You can use the modules directly or via the command-line interface.

## Prerequisites

- **Python 3.9** (required)
- All dependencies installed (see project root README for setup)
- Activate your virtual environment:
  ```bash
  source .venv/bin/activate
  ```

## Pose Extraction: Video to 3D Pose CSV

The `pose_extraction.py` module extracts 2D and 3D pose landmarks from videos using MediaPipe and exports them to CSV format with timestamps.

### Command-Line Usage

Extract poses from a single video:
```bash
python -m dance_motion_embedding.pose_extraction --video data/video/your_video.mp4
```

Extract poses from all videos in a directory:
```bash
python -m dance_motion_embedding.pose_extraction --input-dir data/video
```

With Rerun visualization (real-time 3D pose tracking):
```bash
python -m dance_motion_embedding.pose_extraction --video data/video/your_video.mp4 --use-rerun
```

### Python API Usage

```python
from dance_motion_embedding import PoseExtractor

# Initialize extractor
extractor = PoseExtractor(use_rerun=False)  # Set to True for visualization

# Extract poses from a single video
pose_data = extractor.extract_pose_from_video("data/video/your_video.mp4")
print(f"Extracted {len(pose_data)} frames")

# Process all videos in a directory
extractor.process_video_directory("data/video", "data/poses")
```

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

### Supported Keypoints

The system extracts 33 pose keypoints including:
- Face: nose, eyes, ears, mouth
- Upper body: shoulders, elbows, wrists, fingers
- Lower body: hips, knees, ankles, feet

### Output Files

- **CSV files**: Saved in `data/poses/` with the same name as the input video
- **Rerun visualization**: Real-time 3D pose tracking (if enabled)

## 1. Extract Poses from a Video

You can extract pose data from a single video file using the `PoseExtractor` class:

```python
from dance_motion_embedding import PoseExtractor

extractor = PoseExtractor(use_rerun=True)  # Set to False to disable Rerun visualization
pose_data = extractor.extract_pose_from_video("data/video/your_video.mp4")
print(f"Extracted {len(pose_data)} frames. CSV saved in data/poses/")
```

## 2. Generate Embeddings from Pose Data

Generate vector embeddings for poses or segments from a CSV file:

```python
from dance_motion_embedding import EmbeddingGenerator

generator = EmbeddingGenerator(model_type="transformer", device="cpu")
embeddings = generator.process_csv_file("data/poses/your_video.csv", embedding_type="segment")
print(f"Embeddings shape: {embeddings.shape}. Saved in data/embeddings/")
```

## 3. Analyze Motion Embeddings

Analyze and visualize the generated embeddings:

```python
from dance_motion_embedding import MotionAnalyzer

analyzer = MotionAnalyzer(method="umap")
embeddings = analyzer.load_embeddings("data/embeddings/your_video_segment.npy")
results = analyzer.analyze_motion_patterns(embeddings)
print("Analysis complete. Visualizations saved in data/analysis/")
```

## 4. Run the Full Pipeline (Recommended)

You can run the entire pipeline (extraction → embedding → analysis) using the main pipeline class:

```python
from dance_motion_embedding.main import DanceMotionEmbeddingPipeline

pipeline = DanceMotionEmbeddingPipeline(use_rerun=False, model_type="transformer", device="cpu")
results = pipeline.run_full_pipeline("data/video/your_video.mp4", embedding_type="segment")
print("Pipeline complete!")
print(f"Pose CSV: {results['pose_csv_path']}")
print(f"Embeddings: {results['embeddings_path']}")
print(f"Analysis: {results['analysis_dir']}")
```

## 5. Batch Processing: All Videos in a Directory

To process all videos in `data/video/`:

```python
pipeline = DanceMotionEmbeddingPipeline()
pipeline.process_video_directory(input_dir="data/video", embedding_type="segment", analyze=True)
```

## 6. Command-Line Usage (from project root)

You can also run the pipeline from the command line:

```bash
python -m dance_motion_embedding.main --video data/video/your_video.mp4
# Or batch process:
python -m dance_motion_embedding.main --input-dir data/video
```

## Notes
- All outputs (CSV, embeddings, analysis) are saved in the `data/` subfolders.
- For more advanced usage, see the [examples/basic_usage.py](../../examples/basic_usage.py) script.
- For troubleshooting and more details, see the main project [README](../../README.md). 