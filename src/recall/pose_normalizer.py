"""Pose normalization for comparison."""

import numpy as np
from typing import Dict, Any, Optional
from .data_structures import PoseData, NormalizedPose
from .config import RecallConfig
import logging

logger = logging.getLogger(__name__)


class PoseNormalizer:
    """Pose normalization for comparison"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.root_joint_idx = 23  # left_hip as root joint
    
    def normalize(self, pose: PoseData) -> NormalizedPose:
        """Apply complete normalization pipeline"""
        coords = pose.landmarks.copy()
        
        # Handle missing data
        coords = np.nan_to_num(coords, nan=0.0)
        
        # Ensure 3D coordinates
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(coords.shape[0])])
        
        # Apply normalization steps
        coords = self.normalize_translation(coords)
        coords = self.normalize_scale(coords)
        
        if self.config.normalize_rotation:
            coords = self.normalize_rotation(coords)
        
        return NormalizedPose(
            coordinates=coords,
            original_pose=pose,
            normalization_params={
                'root_joint': pose.landmarks[self.root_joint_idx],
                'torso_length': self._compute_torso_length(coords),
                'normalized_rotation': self.config.normalize_rotation
            }
        )
    
    def normalize_translation(self, coords: np.ndarray) -> np.ndarray:
        """Remove translation by centering on root joint"""
        root_joint = coords[self.root_joint_idx, :]
        return coords - root_joint[None, :]
    
    def normalize_scale(self, coords: np.ndarray) -> np.ndarray:
        """Normalize scale using torso length"""
        torso_length = self._compute_torso_length(coords)
        if torso_length > 1e-6:
            return coords / torso_length
        return coords
    
    def normalize_rotation(self, coords: np.ndarray) -> np.ndarray:
        """Align to principal axes using PCA"""
        # Use shoulders and hips to define principal axes
        key_points = [11, 12, 23, 24]  # shoulders and hips
        key_coords = coords[key_points, :]
        
        # Compute principal components
        mean_coords = np.mean(key_coords, axis=0)
        centered_coords = key_coords - mean_coords
        
        if np.linalg.matrix_rank(centered_coords) < 2:
            return coords  # Not enough variation for rotation
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_coords.T)
        
        # Get eigenvectors (principal components)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sort_idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_idx]
        eigenvecs = eigenvecs[:, sort_idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvecs) < 0:
            eigenvecs[:, 2] *= -1
        
        # Apply rotation
        rotation_matrix = eigenvecs.T
        return coords @ rotation_matrix.T
    
    def _compute_torso_length(self, coords: np.ndarray) -> float:
        """Compute torso length as mean of shoulder-hip distances"""
        # Left shoulder to left hip
        left_torso = np.linalg.norm(coords[11, :] - coords[23, :])
        # Right shoulder to right hip
        right_torso = np.linalg.norm(coords[12, :] - coords[24, :])
        
        return (left_torso + right_torso) / 2.0
    
    def _compute_shoulder_width(self, coords: np.ndarray) -> float:
        """Compute shoulder width"""
        return np.linalg.norm(coords[11, :] - coords[12, :])
    
    def _compute_hip_width(self, coords: np.ndarray) -> float:
        """Compute hip width"""
        return np.linalg.norm(coords[23, :] - coords[24, :])
    
    def denormalize(self, normalized_pose: NormalizedPose, 
                   target_scale: float = 1.0,
                   target_translation: Optional[np.ndarray] = None) -> np.ndarray:
        """Denormalize pose to original scale and position"""
        coords = normalized_pose.coordinates.copy()
        
        # Apply inverse transformations
        if self.config.normalize_rotation:
            # Inverse rotation (transpose of rotation matrix)
            rotation_matrix = self._get_rotation_matrix(normalized_pose)
            coords = coords @ rotation_matrix
        
        # Scale back
        original_torso_length = normalized_pose.normalization_params.get('torso_length', 1.0)
        coords = coords * (original_torso_length * target_scale)
        
        # Translate back
        original_root = normalized_pose.normalization_params.get('root_joint', np.zeros(3))
        if target_translation is not None:
            coords = coords + target_translation[None, :]
        else:
            coords = coords + original_root[None, :]
        
        return coords
    
    def _get_rotation_matrix(self, normalized_pose: NormalizedPose) -> np.ndarray:
        """Get rotation matrix from normalization parameters"""
        # This would need to be stored during normalization
        # For now, return identity matrix
        return np.eye(3)
    
    def get_normalization_info(self, pose: PoseData) -> Dict[str, Any]:
        """Get information about pose normalization"""
        coords = pose.landmarks.copy()
        coords = np.nan_to_num(coords, nan=0.0)
        
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(coords.shape[0])])
        
        return {
            'torso_length': self._compute_torso_length(coords),
            'shoulder_width': self._compute_shoulder_width(coords),
            'hip_width': self._compute_hip_width(coords),
            'root_joint': coords[self.root_joint_idx, :],
            'confidence_mean': np.mean(pose.confidence),
            'confidence_min': np.min(pose.confidence),
            'is_3d': pose.is_3d
        }


def normalize_pose_batch(poses: list[PoseData], config: RecallConfig) -> list[NormalizedPose]:
    """Normalize a batch of poses"""
    normalizer = PoseNormalizer(config)
    return [normalizer.normalize(pose) for pose in poses]


def compute_pose_statistics(poses: list[PoseData]) -> Dict[str, Any]:
    """Compute statistics across a batch of poses"""
    if not poses:
        return {}
    
    normalizer = PoseNormalizer(RecallConfig())
    
    torso_lengths = []
    shoulder_widths = []
    hip_widths = []
    confidences = []
    
    for pose in poses:
        info = normalizer.get_normalization_info(pose)
        torso_lengths.append(info['torso_length'])
        shoulder_widths.append(info['shoulder_width'])
        hip_widths.append(info['hip_width'])
        confidences.append(info['confidence_mean'])
    
    return {
        'num_poses': len(poses),
        'torso_length_mean': np.mean(torso_lengths),
        'torso_length_std': np.std(torso_lengths),
        'shoulder_width_mean': np.mean(shoulder_widths),
        'shoulder_width_std': np.std(shoulder_widths),
        'hip_width_mean': np.mean(hip_widths),
        'hip_width_std': np.std(hip_widths),
        'confidence_mean': np.mean(confidences),
        'confidence_std': np.std(confidences)
    } 