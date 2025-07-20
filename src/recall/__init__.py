"""Recall Module - Live Pose Matching and Video Playback

A real-time pose matching system that uses live camera input or video to find 
similar poses across multiple dance videos and automatically plays matching segments.
"""

from .recall_system import RecallSystem, RecallSystemWithKeyboard, create_recall_system
from .config import RecallConfig
from .data_structures import PoseData, NormalizedPose, Match, PoseConnection, get_pose_connections, get_landmark_name, create_pose_from_mediapipe
from .pose_tracker import PoseTracker
from .pose_matcher import PoseMatcher, CachedPoseMatcher, create_pose_matcher
from .pose_normalizer import PoseNormalizer, normalize_pose_batch, compute_pose_statistics
from .video_player import VideoPlayer, VideoPlayerWithControls, create_video_player

__version__ = "0.1.0"
__all__ = [
    # Main system
    "RecallSystem",
    "RecallSystemWithKeyboard", 
    "create_recall_system",
    
    # Configuration
    "RecallConfig",
    
    # Data structures
    "PoseData",
    "NormalizedPose", 
    "Match",
    "PoseConnection",
    "get_pose_connections",
    "get_landmark_name",
    "create_pose_from_mediapipe",
    
    # Components
    "PoseTracker",
    
    "PoseMatcher",
    "CachedPoseMatcher", 
    "create_pose_matcher",
    
    "PoseNormalizer",
    "normalize_pose_batch",
    "compute_pose_statistics",
    
    "VideoPlayer",
    "VideoPlayerWithControls",
    "create_video_player"
] 