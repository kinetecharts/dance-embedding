# Recall Module - Live Pose Matching and Video Playback

A real-time pose matching system that uses live camera input or video to find similar poses across multiple dance videos and automatically displays matching segments with side-by-side visualization.

## 🎯 Overview

The Recall module enables:
- **Live pose tracking** from camera or video input with real-time OpenCV visualization
- **Real-time pose matching** against a database of dance poses
- **Multi-video search** across all available pose CSV files
- **Side-by-side display** of live pose vs matched reference pose
- **Normalized pose comparison** to handle different heights, angles, and orientations
- **Performance metrics** including FPS and match statistics

## 🏗️ Architecture

### Core Components

1. **Live Pose Tracker**: Captures pose data from camera or video input
2. **Pose Normalizer**: Removes scale, translation, and rotation differences
3. **Pose Matcher**: Finds similar poses across multiple CSV files
4. **Video Player**: Manages display of matched video segments
5. **Display System**: Side-by-side OpenCV windows for live and matched poses

### Data Flow

```
Camera/Video Input → Pose Extraction → Normalization → Database Search → Video Display
                                        ↓
                                   OpenCV Visualization
                                        ↓
                              Live Tracking + Matches + Display
```

### Display Features

- **Live Pose Tracking**: Real-time 2D visualization of camera/video input poses
- **Match Visualization**: Display of matched video frame with pose overlay
- **Side-by-Side Layout**: Live video on left, matched video on right
- **Skeleton Visualization**: Pose connections and body structure
- **Real-time Metrics**: FPS, similarity scores, and system status
- **Interactive Controls**: Keyboard shortcuts with visual feedback

## 🚀 Quick Start

### 1. Setup Pose Database

Ensure you have pose CSV files in `data/poses/`:
```bash
# Extract poses from your dance videos first
python -m pose_extraction.main --video data/video/dance1.mp4
python -m pose_extraction.main --video data/video/dance2.mp4
# ... more videos
```

### 2. Run Live Recall

```bash
# Live camera mode
python -m recall.main --mode camera --top-n 3 --match-interval 1.0 --playback-duration 3.0

# Video input mode
python -m recall.main --mode video --input data/video/test.mp4 --top-n 2 --match-interval 2.0

# Adjust matching frequency
python -m recall.main --mode camera --top-n 4 --match-interval 0.5 --playback-duration 5.0
```

### 3. Interactive Controls

- **Q**: Quit the application
- **P**: Pause/resume matching
- **R**: Reset match display
- **1-9**: Select top-N matches

## 📊 Pose Matching Algorithm

### Normalization Steps

1. **Translation**: Subtract root joint (left hip) position
2. **Scale**: Divide by mean torso length (shoulder-hip distance)
3. **Rotation**: Align to principal axes (optional, for advanced matching)

### Similarity Metrics

- **Euclidean Distance**: Standard L2 distance between normalized poses
- **Cosine Similarity**: Angle-based similarity (rotation-invariant)
- **Weighted Distance**: Joint-specific weights for important keypoints

### Matching Strategy

1. **Frame Selection**: Every N frames (configurable), extract current pose
2. **Database Search**: Compare against all poses in all CSV files
3. **Ranking**: Sort by similarity score
4. **Selection**: Randomly pick from top-N matches
5. **Display**: Show matched video frame with pose overlay
6. **Visualization**: Display live and matched poses side-by-side

## ⚙️ Configuration

### Command Line Options

```bash
python -m recall.main [OPTIONS]

Options:
  --mode {camera,video}          Input mode: camera or video file
  --input PATH                   Input video file (required for video mode)
  --top-n INTEGER               Number of top matches to consider (default: 3)
  --match-interval FLOAT        Interval between matches in seconds (default: 2.0)
  --playback-duration FLOAT     Duration to display each match (default: 3.0)
  --pose-dir PATH               Directory containing pose CSV files (default: data/poses)
  --video-dir PATH              Directory containing video files (default: data/video)
```

### Advanced Configuration

```python
from recall import RecallSystem, RecallConfig

# Custom configuration
config = RecallConfig(
    top_n=3,
    match_interval=1.0,
    playback_duration=3.0,
    pose_dir="data/poses",
    video_dir="data/video"
)

system = RecallSystem(config)
system.run_live()
```

## 📁 Module Structure

```
src/recall/
├── __init__.py              # Main package
├── main.py                  # CLI interface
├── recall_system.py         # Core recall system
├── pose_matcher.py          # Pose matching algorithms
├── pose_normalizer.py       # Pose normalization
├── video_player.py          # Video playback and display
├── data_structures.py       # Data classes and structures
├── config.py               # Configuration management
└── README.md               # This file
```

## 🎮 Usage Examples

### Basic Live Camera Mode

```bash
# Start with default settings
python -m recall.main --mode camera
```

### Advanced Video Analysis

```bash
# Analyze a video with high-frequency matching
python -m recall.main \
    --mode video \
    --input data/video/analysis.mp4 \
    --top-n 4 \
    --match-interval 1.0 \
    --playback-duration 5.0
```

### Custom Pose Database

```bash
# Use custom pose directory
python -m recall.main \
    --pose-dir data/custom_poses \
    --video-dir data/custom_videos \
    --top-n 3 \
    --mode camera
```

## 🔧 Development

### Adding New Display Features

```python
# In video_player.py
def display_live_frame(self, frame: np.ndarray, pose_data: Optional[PoseData] = None, 
                      match_info: Optional[Match] = None):
    """Display live video frame with match info and matched pose side by side"""
    # Create side-by-side display
    canvas_width = 1280
    canvas_height = 480
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Left side: Live video with red pose overlay
    # Right side: Matched video with green pose overlay
    # ... implementation details
```

### Extending Pose Normalization

```python
# In pose_normalizer.py
def advanced_normalization(pose):
    # Add your normalization steps
    pose = normalize_translation(pose)
    pose = normalize_scale(pose)
    pose = normalize_rotation(pose)  # New step
    return pose
```

## 📈 Performance Considerations

### Optimization Tips

1. **Pre-compute normalized poses**: Cache normalized poses for faster matching
2. **Use approximate nearest neighbors**: For large databases, use ANN algorithms
3. **Parallel matching**: Match against multiple files simultaneously
4. **Frame skipping**: Adjust `match_interval` based on performance needs

### Memory Management

- **Pose caching**: Keep frequently accessed poses in memory
- **Video buffering**: Pre-load video segments for smooth playback
- **Display optimization**: Limit OpenCV window updates for better performance
- **Garbage collection**: Clear old matches and visualizations

### Performance Tips

1. **Optimize for Real-time**: Use `--match-interval 1.0` or higher for better performance
2. **Reduce Top-N**: Use `--top-n 3` instead of higher values for faster matching
3. **Camera Quality**: Ensure good lighting and clear camera view for better pose detection
4. **Database Size**: The system works best with 3-10 reference videos in the database

## 🧪 Testing

```bash
# Run recall module tests
python -m pytest src/recall/tests/

# Test with sample data
python -m recall.main --mode video --input tests/sample_video.mp4

# Test camera mode
python -m recall.main --mode camera --top-n 3
```

## 🔮 Future Enhancements

### Planned Features

1. **3D Pose Comparison**: Side-by-side 3D pose comparison with overlay mode
2. **Real-time Analytics**: Pose similarity heatmaps and movement trajectories
3. **Interactive Controls**: Click-to-select poses in display space
4. **Multi-camera Support**: Multiple camera feeds in synchronized views
5. **Animation Transitions**: Smooth pose transition animations

### Advanced Features

1. **Temporal Matching**: Match pose sequences, not just single frames
2. **Style Transfer**: Transfer movement style from matched videos
3. **Learning-based Matching**: Use neural networks for better similarity
4. **Real-time Feedback**: Visual indicators for match quality
5. **Custom Pose Databases**: Support for user-defined pose collections

### Research Directions

- **Pose Embeddings**: Use learned pose representations
- **Motion Prediction**: Predict future poses from current matches
- **Style Classification**: Categorize dance styles automatically
- **Performance Metrics**: Quantitative evaluation of matching quality
- **3D Pose Synthesis**: Generate intermediate poses for smooth transitions

## 🚨 Troubleshooting

### Common Issues

**No matches found**:
- Ensure pose CSV files exist in `data/poses/`
- Check that video files are in `data/video/`
- Verify pose extraction was completed successfully

**Poor performance**:
- Reduce `--top-n` value
- Increase `--match-interval`
- Close other applications to free up CPU/GPU resources

**Camera not working**:
- Ensure camera permissions are granted
- Try a different camera if available
- Check camera is not being used by another application

**Display issues**:
- Ensure OpenCV is properly installed
- Check for OpenCV window conflicts on macOS
- Try different window management settings

### Data Requirements

The Dance Recall System requires:

1. **Pose CSV Files**: Extracted pose data in `data/poses/` directory
2. **Video Files**: Original video files in `data/video/` directory
3. **File Naming**: Pose CSV files should match video file names (e.g., `Dai2.csv` for `Dai2.mov`)

## 📞 Support

For questions and issues:
- Check the main project README
- Review the configuration examples
- Test with sample data first
- Report bugs with detailed logs
- Check OpenCV documentation for display issues

---

**Note**: This module is in active development. API may change in future releases. OpenCV is required for the display functionality. 