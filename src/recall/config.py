"""Configuration management for the recall module."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path


@dataclass
class RecallConfig:
    """Configuration for the recall system"""
    mode: str = "camera"  # "camera" or "video"
    input_video: Optional[str] = None  # Path to input video file
    camera_id: int = 0  # Camera device ID to use
    top_n: int = 5
    match_every: int = 60  # frames
    match_interval: float = 2.0  # seconds
    match_playback_duration: float = 2.0  # seconds
    similarity_metric: str = "euclidean"  # "euclidean", "cosine", "weighted"
    pose_dir: str = "data/poses"
    video_dir: str = "data/video"
    video_with_pose_dir: str = "data/video_with_pose"
    confidence_threshold: float = 0.5
    normalize_rotation: bool = False
    max_cache_size: int = 1000
    parallel_workers: int = 4
    video_player: str = "opencv"  # "opencv", "vlc", "mpv"
    
    # OSC streaming configuration (now handled by JSON config)
    osc_enabled: bool = False
    
    # Joint weights for weighted similarity
    joint_weights: Optional[Dict[str, float]] = field(default_factory=lambda: {
        'nose': 1.0,
        'left_eye': 0.8,
        'right_eye': 0.8,
        'left_ear': 0.8,
        'right_ear': 0.8,
        'left_shoulder': 1.2,
        'right_shoulder': 1.2,
        'left_elbow': 1.0,
        'right_elbow': 1.0,
        'left_wrist': 0.8,
        'right_wrist': 0.8,
        'left_hip': 1.2,
        'right_hip': 1.2,
        'left_knee': 1.0,
        'right_knee': 1.0,
        'left_ankle': 0.8,
        'right_ankle': 0.8
    })
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.mode not in ["camera", "video"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'camera' or 'video'")
        
        if self.mode == "video" and not self.input_video:
            raise ValueError("input_video must be specified for video mode")
        
        if self.similarity_metric not in ["euclidean", "cosine", "weighted"]:
            raise ValueError(f"Invalid similarity_metric: {self.similarity_metric}")
        
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")
        
        if self.match_every < 1:
            raise ValueError(f"match_every must be >= 1, got {self.match_every}")
        
        if self.match_playback_duration <= 0:
            raise ValueError(f"match_playback_duration must be > 0, got {self.match_playback_duration}")
        
        if self.match_interval <= 0:
            raise ValueError(f"match_interval must be > 0, got {self.match_interval}")
        
        # Ensure directories exist
        Path(self.pose_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_with_pose_dir).mkdir(parents=True, exist_ok=True)
    
    def get_pose_files(self) -> List[Path]:
        """Get list of pose CSV files"""
        pose_dir = Path(self.pose_dir)
        return list(pose_dir.glob("*.csv"))
    
    def get_video_files(self) -> List[Path]:
        """Get list of video files"""
        video_dir = Path(self.video_dir)
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        video_files = []
        for ext in extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
        return video_files
    
    def get_video_with_pose_files(self) -> List[Path]:
        """Get list of video files with pose overlay"""
        video_dir = Path(self.video_with_pose_dir)
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        video_files = []
        for ext in extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
        return video_files
    
    def get_video_path_for_pose(self, pose_file: Path) -> Optional[Path]:
        """Get corresponding video path for a pose file"""
        video_dir = Path(self.video_dir)
        base_name = pose_file.stem
        
        # Try different video extensions
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            video_path = video_dir / f"{base_name}{ext}"
            if video_path.exists():
                return video_path
        
        return None
    
    def get_video_with_pose_path_for_pose(self, pose_file: Path) -> Optional[Path]:
        """Get corresponding video with pose path for a pose file"""
        video_dir = Path(self.video_with_pose_dir)
        base_name = pose_file.stem
        
        # Try different video extensions with "_with_pose" suffix
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            video_path = video_dir / f"{base_name}_with_pose{ext}"
            if video_path.exists():
                return video_path
        
        return None 