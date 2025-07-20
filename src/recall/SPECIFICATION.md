# Recall Module Technical Specification

## 🎯 System Requirements

### Functional Requirements

1. **Live Pose Tracking**
   - Capture pose data from camera input in real-time
   - Support video file input as alternative source
   - Extract pose landmarks using MediaPipe
   - Maintain pose confidence scores
   - **Real-time Rerun visualization** of live pose tracking

2. **Pose Normalization**
   - Remove translation by centering on root joint
   - Normalize scale using torso length
   - Optional rotation normalization for advanced matching
   - Handle missing/invalid keypoints gracefully

3. **Multi-Database Matching**
   - Search across all pose CSV files in specified directory
   - Support configurable similarity metrics
   - Rank matches by similarity score
   - Random selection from top-N matches

4. **Video Playback with Rerun Visualization**
   - Play matched video segments from correct timestamps
   - Support multiple simultaneous video players
   - **Synchronized Rerun visualization** of video playback poses
   - Handle video file loading and buffering
   - **Side-by-side comparison** of live pose vs matched video poses

5. **Comprehensive Rerun Visualization**
   - **Live pose tracking** in 3D space with real-time updates
   - **Matched pose visualization** showing top-N matches
   - **Video playback poses** synchronized with actual video timestamps
   - **Similarity score display** and match quality indicators
   - **Interactive controls** for system management
   - **Multi-view layout** showing live input, matches, and playback

### Non-Functional Requirements

- **Performance**: Real-time operation (30+ FPS for pose tracking)
- **Latency**: <100ms from pose detection to video playback start
- **Memory**: Efficient handling of large pose databases
- **Scalability**: Support for 1000+ pose files
- **Reliability**: Graceful handling of missing files and errors
- **Visualization**: Smooth 60+ FPS Rerun rendering for all pose streams

## 🏗️ Architecture Design

### Core Classes

```python
class RecallSystem:
    """Main system orchestrator"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.pose_tracker = PoseTracker(config)
        self.pose_matcher = PoseMatcher(config)
        self.video_player = VideoPlayer(config)
        self.visualizer = RerunVisualizer(config)
        self.running = True
        self.paused = False
    
    def run_live(self):
        """Main live processing loop with Rerun visualization"""
    
    def run_video(self, video_path: str):
        """Process video file input with Rerun visualization"""

class PoseTracker:
    """Live pose tracking from camera/video with Rerun visualization"""
    
    def __init__(self, config: RecallConfig):
        self.mp_pose = mp.solutions.pose.Pose()
        self.cap = None
        self.visualizer = RerunVisualizer(config)
    
    def start_camera(self):
        """Initialize camera capture"""
    
    def start_video(self, video_path: str):
        """Initialize video capture"""
    
    def get_next_pose(self) -> Optional[PoseData]:
        """Get next pose frame and visualize in Rerun"""
    
    def visualize_live_pose(self, pose: PoseData):
        """Visualize live pose in Rerun"""

class PoseMatcher:
    """Pose matching against database"""
    
    def __init__(self, config: RecallConfig):
        self.pose_database = PoseDatabase(config.pose_dir)
        self.normalizer = PoseNormalizer(config)
    
    def find_matches(self, pose: PoseData, top_n: int) -> List[Match]:
        """Find top-N similar poses"""
    
    def random_select(self, matches: List[Match], count: int) -> List[Match]:
        """Randomly select from matches"""

class PoseNormalizer:
    """Pose normalization for comparison"""
    
    def normalize(self, pose: PoseData) -> NormalizedPose:
        """Apply normalization pipeline"""
    
    def normalize_translation(self, pose: PoseData) -> PoseData:
        """Remove translation by centering on root"""
    
    def normalize_scale(self, pose: PoseData) -> PoseData:
        """Normalize scale using torso length"""
    
    def normalize_rotation(self, pose: PoseData) -> PoseData:
        """Align to principal axes (optional)"""

class VideoPlayer:
    """Multi-video playback management with Rerun visualization"""
    
    def __init__(self, config: RecallConfig):
        self.players = {}  # video_path -> player
        self.config = config
        self.visualizer = RerunVisualizer(config)
    
    def play_match(self, match: Match):
        """Start playback for a match and visualize in Rerun"""
    
    def stop_all(self):
        """Stop all video players"""
    
    def get_available_videos(self) -> List[str]:
        """Get list of available video files"""
    
    def visualize_playback_pose(self, pose: PoseData, video_name: str):
        """Visualize video playback pose in Rerun"""

class RerunVisualizer:
    """Comprehensive 3D visualization using Rerun"""
    
    def __init__(self, config: RecallConfig):
        self.rr = rr.init("recall-system", spawn=True)
        self.config = config
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup Rerun visualization layout and components"""
        # Create multi-view layout
        self.rr.log("view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        
        # Setup different visualization spaces
        self.rr.log("live_pose", rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[255, 0, 0]]
        ))
        
        self.rr.log("matches", rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[0, 255, 0]]
        ))
        
        self.rr.log("playback", rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[0, 0, 255]]
        ))
    
    def visualize_live_pose(self, pose: PoseData):
        """Visualize live pose in 3D"""
        self.rr.log("live_pose", rr.Points3D(
            positions=pose.landmarks,
            colors=[[255, 0, 0]] * len(pose.landmarks),
            radii=[0.02] * len(pose.landmarks)
        ))
    
    def visualize_matches(self, live_pose: PoseData, matches: List[Match]):
        """Visualize live pose and matched poses"""
        # Clear previous matches
        self.rr.log("matches", rr.Clear())
        
        # Visualize live pose
        self.visualize_live_pose(live_pose)
        
        # Visualize matched poses with different colors
        colors = [[0, 255, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255]]
        for i, match in enumerate(matches):
            color = colors[i % len(colors)]
            self.rr.log(f"match_{i}", rr.Points3D(
                positions=match.normalized_pose.coordinates,
                colors=[color] * len(match.normalized_pose.coordinates),
                radii=[0.015] * len(match.normalized_pose.coordinates)
            ))
    
    def visualize_playback_pose(self, pose: PoseData, video_name: str):
        """Visualize video playback pose"""
        self.rr.log(f"playback_{video_name}", rr.Points3D(
            positions=pose.landmarks,
            colors=[[0, 0, 255]] * len(pose.landmarks),
            radii=[0.02] * len(pose.landmarks)
        ))
    
    def show_similarity_scores(self, scores: List[float]):
        """Display similarity scores as text"""
        score_text = " | ".join([f"Match {i+1}: {score:.3f}" for i, score in enumerate(scores)])
        self.rr.log("similarity_scores", rr.TextLog(score_text))
    
    def show_system_status(self, status: str):
        """Display system status"""
        self.rr.log("status", rr.TextLog(status))
    
    def clear_playback(self):
        """Clear playback visualizations"""
        self.rr.log("playback", rr.Clear())

class PoseDatabase:
    """Database of pose CSV files"""
    
    def __init__(self, pose_dir: str):
        self.pose_files = self._load_pose_files(pose_dir)
        self.cached_poses = {}  # filename -> List[NormalizedPose]
    
    def search_all(self, query_pose: NormalizedPose) -> List[Match]:
        """Search across all pose files"""
    
    def _load_pose_files(self, pose_dir: str) -> List[str]:
        """Load all CSV files from directory"""
```

### Data Structures

```python
@dataclass
class PoseData:
    """Raw pose data from MediaPipe"""
    landmarks: np.ndarray  # (33, 3) or (33, 2)
    confidence: np.ndarray  # (33,)
    timestamp: float
    frame_number: int

@dataclass
class NormalizedPose:
    """Normalized pose for comparison"""
    coordinates: np.ndarray  # (33, 3) normalized coordinates
    original_pose: PoseData
    normalization_params: Dict[str, Any]

@dataclass
class Match:
    """A pose match result"""
    pose_file: str
    video_file: str
    timestamp: float
    frame_number: int
    similarity_score: float
    normalized_pose: NormalizedPose

@dataclass
class RecallConfig:
    """Configuration for recall system"""
    mode: str = "camera"  # "camera" or "video"
    input_path: Optional[str] = None
    top_n: int = 2
    match_every: int = 30
    similarity_metric: str = "euclidean"
    pose_dir: str = "data/poses"
    video_dir: str = "data/video"
    use_rerun: bool = True
    confidence_threshold: float = 0.5
    normalize_rotation: bool = False
    joint_weights: Optional[Dict[str, float]] = None
    # Rerun visualization settings
    rerun_spawn: bool = True
    rerun_port: Optional[int] = None
    visualization_layout: str = "multi_view"  # "single_view", "multi_view", "side_by_side"
```

## 🔧 Implementation Details

### Rerun Visualization Pipeline

```python
class RerunVisualizationManager:
    """Manages comprehensive Rerun visualization"""
    
    def __init__(self, config: RecallConfig):
        self.rr = rr.init("recall-system", spawn=config.rerun_spawn, port=config.rerun_port)
        self.config = config
        self.setup_layout()
    
    def setup_layout(self):
        """Setup visualization layout based on configuration"""
        if self.config.visualization_layout == "multi_view":
            self.setup_multi_view_layout()
        elif self.config.visualization_layout == "side_by_side":
            self.setup_side_by_side_layout()
        else:
            self.setup_single_view_layout()
    
    def setup_multi_view_layout(self):
        """Setup multi-view layout with separate spaces"""
        # Live pose view
        self.rr.log("live_pose", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        self.rr.log("live_pose", rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[255, 0, 0]]
        ))
        
        # Matches view
        self.rr.log("matches", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        self.rr.log("matches", rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[0, 255, 0]]
        ))
        
        # Playback view
        self.rr.log("playback", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        self.rr.log("playback", rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[0, 0, 255]]
        ))
    
    def visualize_live_tracking(self, pose: PoseData):
        """Visualize live pose tracking"""
        self.rr.log("live_pose", rr.Points3D(
            positions=pose.landmarks,
            colors=[[255, 0, 0]] * len(pose.landmarks),
            radii=[0.02] * len(pose.landmarks)
        ))
        
        # Add pose connections (skeleton)
        self.visualize_skeleton(pose.landmarks, "live_pose", [255, 0, 0])
    
    def visualize_matches(self, live_pose: PoseData, matches: List[Match]):
        """Visualize live pose and matched poses"""
        # Clear previous matches
        self.rr.log("matches", rr.Clear())
        
        # Visualize live pose
        self.visualize_live_tracking(live_pose)
        
        # Visualize matched poses
        colors = [[0, 255, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255]]
        for i, match in enumerate(matches):
            color = colors[i % len(colors)]
            self.rr.log(f"match_{i}", rr.Points3D(
                positions=match.normalized_pose.coordinates,
                colors=[color] * len(match.normalized_pose.coordinates),
                radii=[0.015] * len(match.normalized_pose.coordinates)
            ))
            
            # Add skeleton for matched poses
            self.visualize_skeleton(match.normalized_pose.coordinates, f"match_{i}", color)
    
    def visualize_video_playback(self, pose: PoseData, video_name: str):
        """Visualize video playback pose"""
        self.rr.log(f"playback_{video_name}", rr.Points3D(
            positions=pose.landmarks,
            colors=[[0, 0, 255]] * len(pose.landmarks),
            radii=[0.02] * len(pose.landmarks)
        ))
        
        # Add skeleton for playback pose
        self.visualize_skeleton(pose.landmarks, f"playback_{video_name}", [0, 0, 255])
    
    def visualize_skeleton(self, landmarks: np.ndarray, entity_path: str, color: List[int]):
        """Visualize pose skeleton with connections"""
        # MediaPipe pose connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6),  # Face
            (0, 9), (9, 10), (10, 11), (0, 9), (9, 10), (10, 12)  # Neck
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_pos = landmarks[start_idx]
                end_pos = landmarks[end_idx]
                
                self.rr.log(f"{entity_path}/skeleton", rr.LineStrips3D(
                    positions=[[start_pos, end_pos]],
                    colors=[color]
                ))
    
    def show_metrics(self, similarity_scores: List[float], fps: float, match_count: int):
        """Display real-time metrics"""
        metrics_text = f"FPS: {fps:.1f} | Matches: {match_count} | Scores: {[f'{s:.3f}' for s in similarity_scores]}"
        self.rr.log("metrics", rr.TextLog(metrics_text))
```

### Enhanced Video Player with Rerun

```python
class VideoPlayerWithRerun:
    """Video player with integrated Rerun visualization"""
    
    def __init__(self, config: RecallConfig):
        self.players = {}  # video_path -> player
        self.config = config
        self.visualizer = RerunVisualizationManager(config)
        self.pose_cache = {}  # video_path -> List[PoseData]
    
    def play_match(self, match: Match):
        """Start playback for a match with Rerun visualization"""
        video_path = os.path.join(self.config.video_dir, match.video_file)
        
        if video_path not in self.players:
            # Create new player and load pose data
            self.players[video_path] = self._create_player(video_path)
            self.pose_cache[video_path] = self._load_pose_data(match.pose_file)
        
        player = self.players[video_path]
        poses = self.pose_cache[video_path]
        
        # Start playback from match timestamp
        player.set_position(match.timestamp)
        player.play()
        
        # Start pose visualization thread
        self._start_pose_visualization(video_path, poses, match.timestamp)
    
    def _start_pose_visualization(self, video_path: str, poses: List[PoseData], start_time: float):
        """Start thread to visualize poses during video playback"""
        def visualize_poses():
            current_time = start_time
            pose_idx = 0
            
            while pose_idx < len(poses) and poses[pose_idx].timestamp < current_time:
                pose_idx += 1
            
            while pose_idx < len(poses):
                pose = poses[pose_idx]
                self.visualizer.visualize_video_playback(pose, Path(video_path).stem)
                time.sleep(0.033)  # ~30 FPS
                pose_idx += 1
        
        thread = threading.Thread(target=visualize_poses)
        thread.daemon = True
        thread.start()
```

### Main Processing Loop with Rerun

```python
def run_live_with_rerun(self):
    """Main live processing loop with comprehensive Rerun visualization"""
    self.visualizer = RerunVisualizationManager(self.config)
    
    frame_count = 0
    start_time = time.time()
    
    while self.running:
        if self.paused:
            time.sleep(0.1)
            continue
        
        # Get live pose
        pose = self.pose_tracker.get_next_pose()
        if pose is None:
            continue
        
        # Visualize live tracking
        self.visualizer.visualize_live_tracking(pose)
        
        # Match every N frames
        if frame_count % self.config.match_every == 0:
            matches = self.pose_matcher.find_matches(pose, self.config.top_n)
            selected_matches = self.pose_matcher.random_select(matches, 2)
            
            # Visualize matches
            self.visualizer.visualize_matches(pose, selected_matches)
            
            # Play matched videos
            for match in selected_matches:
                self.video_player.play_match(match)
            
            # Show metrics
            fps = frame_count / (time.time() - start_time)
            similarity_scores = [m.similarity_score for m in selected_matches]
            self.visualizer.show_metrics(similarity_scores, fps, len(selected_matches))
        
        frame_count += 1
        time.sleep(0.033)  # ~30 FPS
```

## 🎮 User Interface

### Interactive Controls with Rerun

```python
def setup_controls_with_rerun(system: RecallSystem):
    """Setup keyboard controls with Rerun integration"""
    
    def on_key_press(key):
        if key == 'space':
            system.toggle_pause()
            status = "PAUSED" if system.paused else "RUNNING"
            system.visualizer.show_system_status(status)
        elif key == 'r':
            system.reset_players()
            system.visualizer.clear_playback()
            system.visualizer.show_system_status("RESET")
        elif key in '123456789':
            top_n = int(key)
            system.set_top_n(top_n)
            system.visualizer.show_system_status(f"Top-N: {top_n}")
        elif key == 'q':
            system.quit()
        elif key == 'v':
            system.visualizer.toggle_view_mode()
    
    # Register keyboard handler
    keyboard.on_press(on_key_press)
```

### Rerun Visualization Layouts

```python
def setup_visualization_layouts(self):
    """Setup different Rerun visualization layouts"""
    
    # Single view - all poses in same space
    self.rr.log("single_view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    
    # Multi-view - separate spaces for different components
    self.rr.log("live_view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    self.rr.log("match_view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    self.rr.log("playback_view", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    
    # Side-by-side comparison
    self.rr.log("comparison", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
```

## 📊 Performance Optimization

### Rerun-Specific Optimizations

```python
class OptimizedRerunVisualizer:
    """Optimized Rerun visualization for real-time performance"""
    
    def __init__(self, config: RecallConfig):
        self.rr = rr.init("recall-system", spawn=True)
        self.config = config
        self.pose_buffer = []
        self.last_update = 0
        self.update_interval = 1.0 / 30  # 30 FPS max
    
    def buffer_pose(self, pose: PoseData, entity_path: str):
        """Buffer pose for batch update"""
        self.pose_buffer.append((pose, entity_path))
        
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.flush_buffer()
            self.last_update = current_time
    
    def flush_buffer(self):
        """Flush buffered poses to Rerun"""
        if not self.pose_buffer:
            return
        
        # Batch update all poses
        for pose, entity_path in self.pose_buffer:
            self.rr.log(entity_path, rr.Points3D(
                positions=pose.landmarks,
                colors=[[255, 0, 0]] * len(pose.landmarks),
                radii=[0.02] * len(pose.landmarks)
            ))
        
        self.pose_buffer.clear()
```

## 🧪 Testing Strategy

### Rerun Visualization Tests

```python
def test_rerun_visualization():
    """Test Rerun visualization components"""
    config = RecallConfig(use_rerun=True)
    visualizer = RerunVisualizationManager(config)
    
    # Test pose visualization
    test_pose = create_test_pose()
    visualizer.visualize_live_tracking(test_pose)
    
    # Test match visualization
    test_matches = create_test_matches()
    visualizer.visualize_matches(test_pose, test_matches)
    
    # Test metrics display
    visualizer.show_metrics([0.8, 0.6], 30.0, 2)

def test_video_playback_visualization():
    """Test video playback with Rerun"""
    config = RecallConfig(use_rerun=True)
    player = VideoPlayerWithRerun(config)
    
    # Test match playback
    test_match = create_test_match()
    player.play_match(test_match)
    
    # Verify pose visualization started
    assert len(player.visualization_threads) > 0
```

## 🔮 Future Enhancements

### Advanced Rerun Features

1. **3D Pose Comparison**
   - Side-by-side 3D pose comparison
   - Overlay mode for pose differences
   - Animation of pose transitions

2. **Real-time Analytics**
   - Pose similarity heatmaps
   - Movement trajectory visualization
   - Performance metrics dashboard

3. **Interactive Controls**
   - Click-to-select poses in 3D space
   - Drag-and-drop pose matching
   - Real-time parameter adjustment

4. **Multi-camera Support**
   - Multiple camera feeds in Rerun
   - Synchronized multi-view visualization
   - Cross-camera pose matching

### Performance Improvements

1. **GPU Acceleration**
   - GPU-accelerated pose rendering
   - Hardware-accelerated video decoding
   - Parallel pose processing

2. **Streaming Optimization**
   - Adaptive quality based on performance
   - Dynamic frame rate adjustment
   - Intelligent pose caching

3. **Memory Management**
   - Efficient pose data structures
   - Smart visualization cleanup
   - Optimized Rerun data transmission 