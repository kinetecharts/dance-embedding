"""Data structures for the recall module."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class PoseData:
    """Raw pose data from MediaPipe"""
    landmarks: np.ndarray  # (33, 3) or (33, 2)
    confidence: np.ndarray  # (33,)
    timestamp: float
    frame_number: int
    
    def __post_init__(self):
        """Validate pose data after initialization"""
        if self.landmarks.shape[0] != 33:
            raise ValueError(f"Expected 33 landmarks, got {self.landmarks.shape[0]}")
        
        if self.confidence.shape[0] != 33:
            raise ValueError(f"Expected 33 confidence scores, got {self.confidence.shape[0]}")
        
        if self.landmarks.shape[1] not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D landmarks, got shape {self.landmarks.shape}")
    
    @property
    def is_3d(self) -> bool:
        """Check if pose has 3D coordinates"""
        return self.landmarks.shape[1] == 3
    
    def get_landmark(self, index: int) -> np.ndarray:
        """Get landmark at specific index"""
        if 0 <= index < 33:
            return self.landmarks[index]
        raise ValueError(f"Invalid landmark index: {index}")
    
    def get_confidence(self, index: int) -> float:
        """Get confidence score for specific landmark"""
        if 0 <= index < 33:
            return self.confidence[index]
        raise ValueError(f"Invalid landmark index: {index}")


@dataclass
class NormalizedPose:
    """Normalized pose for comparison"""
    coordinates: np.ndarray  # (33, 3) normalized coordinates
    original_pose: PoseData
    normalization_params: Dict[str, Any]
    
    def __post_init__(self):
        """Validate normalized pose data"""
        if self.coordinates.shape != (33, 3):
            raise ValueError(f"Expected normalized coordinates shape (33, 3), got {self.coordinates.shape}")
    
    def get_landmark(self, index: int) -> np.ndarray:
        """Get normalized landmark at specific index"""
        if 0 <= index < 33:
            return self.coordinates[index]
        raise ValueError(f"Invalid landmark index: {index}")


@dataclass
class Match:
    """A pose match result"""
    pose_file: str
    video_file: str
    timestamp: float
    frame_number: int
    similarity_score: float
    normalized_pose: NormalizedPose
    pose_index: int = 0  # Index of the pose in the pose file
    
    def __post_init__(self):
        """Validate match data"""
        # Note: Similarity scores can be outside -1 to 1 range for Euclidean distance
        # This validation is removed to allow for different similarity metrics
        pass


@dataclass
class PoseConnection:
    """Represents a connection between two pose landmarks"""
    start_idx: int
    end_idx: int
    name: str
    
    def __post_init__(self):
        """Validate connection data"""
        if not (0 <= self.start_idx < 33 and 0 <= self.end_idx < 33):
            raise ValueError(f"Invalid landmark indices: {self.start_idx}, {self.end_idx}")


# MediaPipe pose connections for skeleton visualization
POSE_CONNECTIONS = [
    PoseConnection(11, 12, "shoulders"),
    PoseConnection(11, 13, "left_upper_arm"),
    PoseConnection(13, 15, "left_forearm"),
    PoseConnection(12, 14, "right_upper_arm"),
    PoseConnection(14, 16, "right_forearm"),
    PoseConnection(11, 23, "left_torso"),
    PoseConnection(12, 24, "right_torso"),
    PoseConnection(23, 24, "hips"),
    PoseConnection(23, 25, "left_thigh"),
    PoseConnection(25, 27, "left_shin"),
    PoseConnection(27, 29, "left_foot"),
    PoseConnection(27, 31, "left_heel"),
    PoseConnection(24, 26, "right_thigh"),
    PoseConnection(26, 28, "right_shin"),
    PoseConnection(28, 30, "right_foot"),
    PoseConnection(28, 32, "right_heel"),
    PoseConnection(0, 1, "nose_left_eye"),
    PoseConnection(1, 2, "left_eye_left_ear"),
    PoseConnection(2, 3, "left_ear_left_cheek"),
    PoseConnection(3, 7, "left_cheek_mouth"),
    PoseConnection(0, 4, "nose_right_eye"),
    PoseConnection(4, 5, "right_eye_right_ear"),
    PoseConnection(5, 6, "right_ear_right_cheek"),
    PoseConnection(0, 9, "nose_neck"),
    PoseConnection(9, 10, "neck_left_shoulder"),
    PoseConnection(10, 11, "left_shoulder"),
    PoseConnection(9, 10, "neck_right_shoulder"),
    PoseConnection(10, 12, "right_shoulder"),
]


def get_pose_connections() -> List[PoseConnection]:
    """Get list of pose connections for skeleton visualization"""
    return POSE_CONNECTIONS


def get_landmark_name(index: int) -> str:
    """Get landmark name by index"""
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    if 0 <= index < len(landmark_names):
        return landmark_names[index]
    return f"landmark_{index}"


def create_pose_from_mediapipe(landmarks, confidence_scores, timestamp: float, frame_number: int) -> PoseData:
    """Create PoseData from MediaPipe pose landmarks"""
    # Convert MediaPipe landmarks to numpy array
    if hasattr(landmarks, 'landmark'):
        # MediaPipe pose landmarks object
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        confidences = np.array([lm.visibility for lm in landmarks.landmark])
    else:
        # Already numpy array
        coords = np.array(landmarks)
        confidences = np.array(confidence_scores)
    
    return PoseData(
        landmarks=coords,
        confidence=confidences,
        timestamp=timestamp,
        frame_number=frame_number
    ) 