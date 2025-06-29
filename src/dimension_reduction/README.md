# Dimension Reduction for Dance Motion Analysis

This module provides interactive visualization of dance pose data in reduced-dimensional spaces, allowing you to see how dance movements translate into trajectories in 2D or 3D space.

## Features

- **Multiple Reduction Methods**: PCA, t-SNE, UMAP
- **2D/3D Visualization**: Choose between 2D plots or 3D interactive plots
- **Video Synchronization**: Playback video while showing corresponding data point movement
- **Interactive Controls**: Select videos, methods, and dimensions
- **Real-time Trajectory**: See how dance movements create paths in reduced space

## Quick Start

### Command Line Usage

```bash
# Basic usage with default settings
python -m dimension_reduction.main

# Specify video and pose CSV
python -m dimension_reduction.main --video data/video/dance.mp4 --pose-csv data/poses/dance.csv

# Choose specific reduction method and dimensions
python -m dimension_reduction.main --method umap --dimensions 3d --video data/video/dance.mp4

# Create combined visualization with video player
python -m dimension_reduction.main --video data/video/dance.mp4 --combined

# Create standalone video player with timeline
python -m dimension_reduction.main --video data/video/dance.mp4 --create-video-player
```

### Python API Usage

```python
from dimension_reduction import DimensionReductionVisualizer

# Initialize visualizer
viz = DimensionReductionVisualizer()

# Load data and create visualization
viz.load_data("data/video/kurt.mov", "data/poses/kurt.csv")
viz.create_visualization(method="umap", dimensions="3d")

# Create different types of visualizations
viz.create_interactive_plot(save_html=True)  # Interactive plot with timeline
viz.create_video_player_html(save_html=True)  # Standalone video player
viz.create_combined_visualization(save_html=True)  # Combined plot + video
viz.show()  # Display interactive plot
```

## Supported Methods

### 1. PCA (Principal Component Analysis)
- **Best for**: Linear relationships, quick exploration
- **Use case**: Initial data exploration, finding main movement directions
- **Parameters**: `n_components` (2 or 3)

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Best for**: Preserving local structure, clustering similar poses
- **Use case**: Finding pose clusters, identifying similar dance moves
- **Parameters**: `perplexity` (default: 30), `learning_rate` (default: 200)

### 3. UMAP (Uniform Manifold Approximation and Projection)
- **Best for**: Preserving both local and global structure
- **Use case**: Best overall visualization, maintaining temporal relationships
- **Parameters**: `n_neighbors` (default: 15), `min_dist` (default: 0.1)

## Data Requirements

### Video File
- **Formats**: MP4, AVI, MOV, WebM, MKV
- **Resolution**: Any (will be resized for display)
- **Duration**: Any length (processing time scales with duration)

### Pose CSV File
Must contain columns in this format:
```csv
timestamp,frame_number,nose_x,nose_y,nose_z,nose_confidence,left_eye_x,left_eye_y,...
0.0,0,320.5,240.2,0.1,0.95,315.2,235.8,...
0.033,1,321.1,239.8,0.12,0.94,316.1,235.1,...
```

**Required columns**:
- `timestamp`: Frame timestamp in seconds
- `frame_number`: Sequential frame number
- `{keypoint}_x`, `{keypoint}_y`: 2D coordinates for each keypoint
- `{keypoint}_z`: 3D coordinates (optional, for 3D analysis)
- `{keypoint}_confidence`: Confidence scores (optional)

## Visualization Features

### 1. Interactive Plot with Timeline
- **2D/3D Scatter Plot**: Shows pose data points in reduced space
- **Color-coded Time Progression**: Points colored by timestamp
- **Movement Trajectory**: Red line showing movement path
- **Timeline Slider**: Click to jump to specific time points
- **Hover Information**: Frame number and timestamp on hover

### 2. Standalone Video Player
- **HTML5 Video Player**: Native browser video controls
- **Custom Timeline**: Clickable timeline with pose frame markers
- **Step Controls**: Forward/backward step through pose frames
- **Keyboard Shortcuts**: Space (play/pause), arrows (step), Home/End (seek)
- **Real-time Frame Display**: Shows current pose frame number

### 3. Combined Visualization
- **Side-by-side Layout**: Plot and video player together
- **Synchronized Playback**: Video and plot highlight current position
- **Interactive Timeline**: Click timeline to seek video and highlight plot point
- **Statistics Panel**: Real-time display of analysis information
- **Responsive Design**: Adapts to different screen sizes

### 4. Video Synchronization Features
- **Frame Highlighting**: Current video frame highlighted in reduced space
- **Time Indicator**: Shows current position in trajectory
- **Pose Frame Stepping**: Step through actual pose frames, not just video frames
- **Real-time Updates**: Plot updates as video plays
- **Click-to-Seek**: Click on plot points to jump to video time

### 5. Timeline Controls
- **Visual Timeline**: Shows all pose frames with color progression
- **Current Time Marker**: Blue line showing current position
- **Click Navigation**: Click anywhere on timeline to seek
- **Frame Information**: Hover to see frame number and timestamp
- **Smooth Transitions**: Animated marker movement

## Configuration

### Method Parameters

```python
# PCA parameters
pca_params = {
    "n_components": 3,  # 2 or 3
    "random_state": 42
}

# t-SNE parameters
tsne_params = {
    "n_components": 3,  # 2 or 3
    "perplexity": 30,
    "learning_rate": 200,
    "random_state": 42
}

# UMAP parameters
umap_params = {
    "n_components": 3,  # 2 or 3
    "n_neighbors": 15,
    "min_dist": 0.1,
    "random_state": 42
}
```

### Visualization Settings

```python
viz_settings = {
    "point_size": 5,
    "trajectory_width": 2,
    "color_scheme": "viridis",  # or "plasma", "inferno", "magma"
    "background_color": "white",
    "show_confidence": True,
    "confidence_threshold": 0.5
}
```

## Examples

### Example 1: Quick Exploration
```bash
# Use UMAP for best overall visualization
python -m dimension_reduction.main --method umap --video data/video/dance.mp4
```

### Example 2: Detailed Analysis with Video Player
```bash
# Create combined visualization with synchronized video playback
python -m dimension_reduction.main --method umap --dimensions 2d --video data/video/dance.mp4 --combined
```

### Example 3: Standalone Video Player
```bash
# Create video player with timeline for detailed frame-by-frame analysis
python -m dimension_reduction.main --video data/video/dance.mp4 --create-video-player
```

### Example 4: 3D Analysis with Frame Extraction
```bash
# 3D UMAP with extracted video frames for advanced analysis
python -m dimension_reduction.main --method umap --dimensions 3d --video data/video/dance.mp4 --extract-frames
```

### Example 5: Complete Analysis Pipeline
```bash
# Full analysis with all visualization types
python -m dimension_reduction.main \
    --video data/video/dance.mp4 \
    --pose-csv data/poses/dance.csv \
    --method umap \
    --dimensions 2d \
    --combined \
    --create-video-player \
    --extract-frames
```

## Output Files

The module generates several output files:

- **`data/analysis/dimension_reduction/`**: Main output directory
  - `{video_name}_{method}_{dimensions}.html`: Interactive visualization with timeline
  - `{video_name}_combined_visualization.html`: Combined plot and video player
  - `{video_name}_video_player.html`: Standalone video player with timeline
  - `{video_name}_{method}_{dimensions}.png`: Static plot image
  - `{video_name}_reduced_data.json`: Reduced coordinates data

- **`data/analysis/video_frames/`**: Extracted video frames (if `--extract-frames` used)
  - `frame_000000.jpg`, `frame_000001.jpg`, etc.: Individual video frames

## Troubleshooting

### Common Issues

1. **"No pose data found"**: Ensure pose CSV file exists and has correct format
2. **"Video file not found"**: Check video path and file format
3. **"Memory error"**: Reduce video resolution or use shorter video
4. **"Slow processing"**: Use PCA for faster processing, UMAP for better quality

### Performance Tips

- **For long videos**: Use PCA or reduce video resolution
- **For high quality**: Use UMAP with default parameters
- **For clustering**: Use t-SNE with appropriate perplexity
- **For real-time**: Use 2D visualization instead of 3D

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: PCA and t-SNE
- `umap-learn`: UMAP dimensionality reduction
- `plotly`: Interactive visualizations
- `opencv-python`: Video processing
- `matplotlib`: Static plots

## Contributing

To add new dimension reduction methods:

1. Add method to `reduction_methods.py`
2. Update `main.py` argument parser
3. Add method parameters to configuration
4. Update documentation

## License

This module is part of the Dance Motion Embedding System. 