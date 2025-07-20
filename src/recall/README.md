# Recall Module - Live Pose Matching and Video Playback

A real-time pose matching system that uses live camera input or video to find similar poses across multiple dance videos and automatically plays matching segments, with **comprehensive Rerun 3D visualization** for both live tracking and video playback.

## 🎯 Overview

The Recall module enables:
- **Live pose tracking** from camera or video input with **real-time Rerun visualization**
- **Real-time pose matching** against a database of dance poses
- **Multi-video search** across all available pose CSV files
- **Automatic video playback** of the most similar segments
- **Synchronized Rerun visualization** of video playback poses
- **Normalized pose comparison** to handle different heights, angles, and orientations
- **Side-by-side 3D comparison** of live pose vs matched video poses

## 🏗️ Architecture

### Core Components

1. **Live Pose Tracker**: Captures pose data from camera or video input with **Rerun visualization**
2. **Pose Normalizer**: Removes scale, translation, and rotation differences
3. **Pose Matcher**: Finds similar poses across multiple CSV files
4. **Video Player**: Manages playback of matched video segments with **synchronized Rerun poses**
5. **Rerun Visualizer**: **Comprehensive 3D visualization** of all pose streams

### Data Flow

```
Camera/Video Input → Pose Extraction → Normalization → Database Search → Video Playback
                                        ↓
                                   Rerun Visualization
                                        ↓
                              Live Tracking + Matches + Playback
```

### Rerun Visualization Features

- **Live Pose Tracking**: Real-time 3D visualization of camera/video input poses
- **Match Visualization**: Display of top-N matched poses with similarity scores
- **Video Playback Poses**: Synchronized 3D poses from matched video segments
- **Multi-view Layout**: Separate visualization spaces for different components
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

### 2. Run Live Recall with Rerun

```bash
# Live camera mode with Rerun visualization
python -m recall.main --mode camera --top-n 3

# Video input mode with Rerun visualization
python -m recall.main --mode video --input data/video/test.mp4 --top-n 2

# Adjust matching frequency and visualization
python -m recall.main --mode camera --match-every 30 --top-n 4 --visualization-layout multi_view
```

### 3. Interactive Controls

- **Space**: Pause/resume matching (with visual status)
- **R**: Reset video players (clears playback visualization)
- **1-9**: Adjust top-N matches (with visual feedback)
- **V**: Toggle visualization layout
- **Q**: Quit

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
5. **Playback**: Start video playback from matched timestamp
6. **Visualization**: Display all poses in Rerun with synchronized timing

## ⚙️ Configuration

### Command Line Options

```bash
python -m recall.main [OPTIONS]

Options:
  --mode {camera,video}     Input mode (default: camera)
  --input PATH              Video file path (for video mode)
  --top-n INT               Number of top matches to consider (default: 2)
  --match-every INT         Match every N frames (default: 30)
  --similarity {euclidean,cosine,weighted}  Similarity metric (default: euclidean)
  --pose-dir PATH           Directory containing pose CSV files (default: data/poses)
  --video-dir PATH          Directory containing video files (default: data/video)
  --use-rerun               Enable Rerun 3D visualization (default: True)
  --no-rerun                Disable Rerun visualization
  --rerun-port INT          Custom Rerun port (default: auto)
  --visualization-layout {single_view,multi_view,side_by_side}  Layout type (default: multi_view)
  --confidence-threshold FLOAT  Minimum pose confidence (default: 0.5)
  --normalize-rotation      Enable rotation normalization (default: False)
```

### Advanced Configuration

```python
from recall import RecallSystem, RecallConfig

# Custom configuration with Rerun settings
config = RecallConfig(
    top_n=3,
    match_every=15,
    similarity_metric='weighted',
    confidence_threshold=0.7,
    normalize_rotation=True,
    use_rerun=True,
    rerun_spawn=True,
    rerun_port=9090,
    visualization_layout='multi_view',
    joint_weights={
        'nose': 1.0,
        'shoulders': 1.2,
        'hips': 1.2,
        'hands': 0.8,
        'feet': 0.8
    }
)

system = RecallSystem(config)
system.run_live()
```

## 📁 Module Structure

```
src/recall/
├── __init__.py              # Main package
├── main.py                  # CLI interface
├── recall_system.py         # Core recall system with Rerun
├── pose_tracker.py          # Live pose tracking with Rerun
├── pose_matcher.py          # Pose matching algorithms
├── pose_normalizer.py       # Pose normalization
├── video_player.py          # Multi-video playback with Rerun
├── rerun_visualizer.py      # Comprehensive 3D visualization
├── config.py               # Configuration management
└── README.md               # This file
```

## 🎮 Usage Examples

### Basic Live Camera Mode with Rerun

```bash
# Start with default settings and Rerun visualization
python -m recall.main
```

### Advanced Video Analysis with Custom Visualization

```bash
# Analyze a video with high-frequency matching and custom Rerun layout
python -m recall.main \
    --mode video \
    --input data/video/analysis.mp4 \
    --top-n 4 \
    --match-every 10 \
    --similarity weighted \
    --visualization-layout side_by_side \
    --rerun-port 9090
```

### Custom Pose Database with Rerun

```bash
# Use custom pose directory with Rerun visualization
python -m recall.main \
    --pose-dir data/custom_poses \
    --video-dir data/custom_videos \
    --top-n 3 \
    --use-rerun
```

## 🔧 Development

### Adding New Rerun Visualization Features

```python
# In rerun_visualizer.py
def visualize_pose_comparison(self, live_pose: PoseData, matched_pose: PoseData):
    """Visualize side-by-side pose comparison"""
    # Live pose in red
    self.rr.log("comparison/live", rr.Points3D(
        positions=live_pose.landmarks,
        colors=[[255, 0, 0]] * len(live_pose.landmarks)
    ))
    
    # Matched pose in blue (offset for side-by-side view)
    offset_poses = matched_pose.landmarks + np.array([2.0, 0, 0])
    self.rr.log("comparison/matched", rr.Points3D(
        positions=offset_poses,
        colors=[[0, 0, 255]] * len(matched_pose.landmarks)
    ))
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

### Rerun-Specific Optimizations

1. **Batch Updates**: Buffer pose updates for efficient Rerun transmission
2. **Frame Rate Control**: Limit visualization updates to maintain performance
3. **Memory Management**: Clear old visualizations to prevent memory buildup
4. **GPU Acceleration**: Use hardware acceleration when available

### General Optimization Tips

1. **Pre-compute normalized poses**: Cache normalized poses for faster matching
2. **Use approximate nearest neighbors**: For large databases, use ANN algorithms
3. **Parallel matching**: Match against multiple files simultaneously
4. **Frame skipping**: Adjust `match_every` based on performance needs

### Memory Management

- **Pose caching**: Keep frequently accessed poses in memory
- **Video buffering**: Pre-load video segments for smooth playback
- **Rerun cleanup**: Clear old visualizations periodically
- **Garbage collection**: Clear old matches and visualizations

## 🧪 Testing

```bash
# Run recall module tests
python -m pytest src/recall/tests/

# Test with sample data and Rerun visualization
python -m recall.main --mode video --input tests/sample_video.mp4 --use-rerun

# Test without Rerun for performance testing
python -m recall.main --mode video --input tests/sample_video.mp4 --no-rerun
```

## 🔮 Future Enhancements

### Planned Rerun Features

1. **3D Pose Comparison**: Side-by-side 3D pose comparison with overlay mode
2. **Real-time Analytics**: Pose similarity heatmaps and movement trajectories
3. **Interactive Controls**: Click-to-select poses in 3D space
4. **Multi-camera Support**: Multiple camera feeds in synchronized Rerun views
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

## 📞 Support

For questions and issues:
- Check the main project README
- Review the configuration examples
- Test with sample data first
- Report bugs with detailed logs
- Check Rerun documentation for visualization issues

---

**Note**: This module is in active development. API may change in future releases. Rerun visualization requires the `rerun-sdk` package to be installed. 