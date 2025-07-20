"""Pose matching and similarity computation."""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict
from pathlib import Path
import random
import time

from .data_structures import PoseData, Match, NormalizedPose
from .pose_normalizer import PoseNormalizer
from .config import RecallConfig

logger = logging.getLogger(__name__)


class PoseMatcher:
    """Pose matching with multiple similarity metrics"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.pose_normalizer = PoseNormalizer(config)
        self.pose_cache = {}  # pose_file -> List[NormalizedPose]
        self.video_cache = {}  # pose_file -> video_file
        self.loaded_files = []  # List of loaded pose files
        
        # Load poses (limit to first 3 + input video pose)
        self._load_poses()
    
    def _load_poses(self):
        """Load poses from CSV files (limit to first 3 + input video pose)"""
        pose_files = list(Path(self.config.pose_dir).glob("*.csv"))
        
        # Sort files and take first 3
        pose_files.sort()
        first_three = pose_files[:3]
        
        # Also include the pose file that corresponds to the input video if specified
        input_pose_file = None
        if hasattr(self.config, 'input_video') and self.config.input_video:
            input_video_name = Path(self.config.input_video).stem
            for pose_file in pose_files:
                if pose_file.stem.lower() == input_video_name.lower():
                    input_pose_file = pose_file
                    break
        
        # Combine the files to load
        files_to_load = first_three.copy()
        if input_pose_file and input_pose_file not in files_to_load:
            files_to_load.append(input_pose_file)
        
        logger.info(f"Loading pose files: {[f.stem for f in files_to_load]}")
        
        for pose_file in files_to_load:
            try:
                poses = self._load_poses_from_csv(pose_file)
                logger.info(f"Loaded {len(poses)} poses from {pose_file.stem}")
                
                if poses:
                    # Normalize poses
                    normalized_poses = [self.pose_normalizer.normalize(pose) for pose in poses]
                    logger.info(f"Normalized {len(normalized_poses)} poses from {pose_file.stem}")
                    
                    # Store in cache
                    self.pose_cache[pose_file] = normalized_poses
                    self.loaded_files.append(pose_file)
                    
                    # Find corresponding video file
                    video_file = self._find_video_file(pose_file)
                    if video_file:
                        self.video_cache[pose_file] = video_file
                        logger.info(f"Loaded {len(normalized_poses)} poses from {pose_file.stem} -> {video_file}")
                    else:
                        logger.warning(f"No video file found for {pose_file.stem}")
                else:
                    logger.warning(f"No poses loaded from {pose_file.stem}")
                
            except Exception as e:
                logger.error(f"Error loading poses from {pose_file}: {e}")
        
        logger.info(f"Successfully loaded {len(self.loaded_files)} pose files with {sum(len(poses) for poses in self.pose_cache.values())} total poses")
    
    def _load_poses_from_csv(self, pose_file: Path) -> List[PoseData]:
        """Load poses from CSV file"""
        try:
            df = pd.read_csv(pose_file)
            
            # Define the landmark names in the correct order
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
            
            poses = []
            for _, row in df.iterrows():
                # Extract landmarks (33 landmarks, 3 coordinates each)
                landmarks = []
                for landmark_name in landmark_names:
                    x = row.get(f'{landmark_name}_x', 0.0)
                    y = row.get(f'{landmark_name}_y', 0.0)
                    z = row.get(f'{landmark_name}_z', 0.0)
                    landmarks.append([x, y, z])
                
                # Extract confidence scores
                confidence = []
                for landmark_name in landmark_names:
                    conf = row.get(f'{landmark_name}_confidence', 1.0)
                    confidence.append(conf)
                
                # Create pose data
                pose_data = PoseData(
                    landmarks=np.array(landmarks),
                    confidence=np.array(confidence),
                    timestamp=row.get('timestamp', 0.0),
                    frame_number=row.get('frame_number', 0)
                )
                poses.append(pose_data)
            
            return poses
            
        except Exception as e:
            logger.error(f"Error loading poses from {pose_file}: {e}")
            return []
    
    def _find_video_file(self, pose_file: Path) -> Optional[str]:
        """Find corresponding video file for pose file"""
        base_name = pose_file.stem
        
        # Try different video extensions
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            video_path = Path(self.config.video_dir) / f"{base_name}{ext}"
            if video_path.exists():
                return f"{base_name}{ext}"
        
        return None
    
    def find_matches(self, query_pose: PoseData, top_n: int = 5) -> List[Match]:
        """Find top-N matches for a query pose"""
        try:
            logger.info(f"Finding matches for pose with {len(query_pose.landmarks)} landmarks")
            
            # Normalize query pose
            normalized_query = self.pose_normalizer.normalize(query_pose)
            
            all_matches = []
            
            # Search through all loaded pose files
            for pose_file in self.loaded_files:
                if pose_file not in self.pose_cache:
                    logger.warning(f"Pose file {pose_file} not in cache")
                    continue
                
                poses = self.pose_cache[pose_file]
                video_file = self.video_cache.get(pose_file, pose_file.stem + ".mp4")
                
                logger.info(f"Searching in {pose_file.stem} with {len(poses)} poses")
                
                # Limit search to a subset of poses for performance
                # Sample every 10th pose to reduce computation
                sample_indices = range(0, len(poses), 10)
                sampled_poses = [poses[i] for i in sample_indices]
                
                # Compute similarities with sampled poses
                for i, pose in enumerate(sampled_poses):
                    original_index = sample_indices[i]
                    similarity = self._compute_similarity(normalized_query, pose)
                    
                    match = Match(
                        pose_file=pose_file.name,
                        video_file=video_file,
                        timestamp=pose.original_pose.timestamp,
                        frame_number=pose.original_pose.frame_number,
                        similarity_score=similarity,
                        normalized_pose=pose,
                        pose_index=original_index
                    )
                    all_matches.append(match)
            
            logger.info(f"Found {len(all_matches)} total matches")
            
            # Sort by similarity score (higher is better)
            all_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top-N matches
            top_matches = all_matches[:top_n]
            
            # Log similarity score statistics
            if all_matches:
                scores = [m.similarity_score for m in all_matches]
                logger.info(f"Similarity scores - min: {min(scores):.3f}, max: {max(scores):.3f}, mean: {np.mean(scores):.3f}")
                logger.info(f"Top {len(top_matches)} scores: {[f'{m.similarity_score:.3f}' for m in top_matches]}")
            
            return top_matches
            
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return []
    
    def _compute_similarity(self, pose1: NormalizedPose, pose2: NormalizedPose) -> float:
        """Compute similarity between two poses"""
        try:
            if self.config.similarity_metric == "euclidean":
                return self._euclidean_similarity(pose1, pose2)
            elif self.config.similarity_metric == "cosine":
                return self._cosine_similarity(pose1, pose2)
            elif self.config.similarity_metric == "weighted":
                return self._weighted_similarity(pose1, pose2)
            else:
                return self._euclidean_similarity(pose1, pose2)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def _euclidean_similarity(self, pose1: NormalizedPose, pose2: NormalizedPose) -> float:
        """Compute Euclidean distance-based similarity"""
        try:
            # Compute Euclidean distance between landmarks
            diff = pose1.coordinates - pose2.coordinates
            distance = np.sqrt(np.sum(diff ** 2, axis=1))
            
            # Weight by confidence
            confidence = np.minimum(pose1.original_pose.confidence, pose2.original_pose.confidence)
            weighted_distance = np.sum(distance * confidence) / np.sum(confidence)
            
            # Convert to similarity (negative distance, so higher is better)
            return -weighted_distance
            
        except Exception as e:
            logger.error(f"Error in Euclidean similarity: {e}")
            return 0.0
    
    def _cosine_similarity(self, pose1: NormalizedPose, pose2: NormalizedPose) -> float:
        """Compute cosine similarity between poses"""
        try:
            # Flatten landmarks
            vec1 = pose1.coordinates.flatten()
            vec2 = pose2.coordinates.flatten()
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error in cosine similarity: {e}")
            return 0.0
    
    def _weighted_similarity(self, pose1: NormalizedPose, pose2: NormalizedPose) -> float:
        """Compute weighted similarity based on joint importance"""
        try:
            # Use joint weights from config
            weights = self.config.joint_weights
            
            total_similarity = 0.0
            total_weight = 0.0
            
            for i in range(len(pose1.coordinates)):
                # Get joint name (simplified)
                joint_name = f"joint_{i}"
                weight = weights.get(joint_name, 1.0)
                
                # Compute distance for this joint
                diff = pose1.coordinates[i] - pose2.coordinates[i]
                distance = np.sqrt(np.sum(diff ** 2))
                
                # Weight by confidence and joint importance
                confidence = min(pose1.original_pose.confidence[i], pose2.original_pose.confidence[i])
                weighted_distance = distance * confidence * weight
                
                total_similarity += weighted_distance
                total_weight += weight * confidence
            
            if total_weight == 0:
                return 0.0
            
            # Convert to similarity (negative weighted distance)
            return -total_similarity / total_weight
            
        except Exception as e:
            logger.error(f"Error in weighted similarity: {e}")
            return 0.0
    
    def random_select(self, matches: List[Match], count: int) -> List[Match]:
        """Randomly select from top matches with diversity"""
        if not matches:
            return []
        
        # Add randomization seed based on time and add some noise
        current_time = time.time()
        random.seed(int(current_time * 1000) % 10000 + int(current_time * 100) % 100)
        
        # Take top matches but add some randomization
        top_matches = matches[:min(len(matches), count * 3)]  # Consider more matches
        
        # Randomly select with preference for different files and timestamps
        selected = []
        used_files = set()
        used_timestamps = set()
        
        # First, try to select from different files
        for match in top_matches:
            if len(selected) >= count:
                break
            
            # Check if this file and timestamp combination is different
            file_timestamp_key = (match.pose_file, round(match.timestamp, 1))
            if match.pose_file not in used_files and file_timestamp_key not in used_timestamps:
                selected.append(match)
                used_files.add(match.pose_file)
                used_timestamps.add(file_timestamp_key)
        
        # If we need more, add random selections with timestamp diversity
        remaining = [m for m in top_matches if m not in selected]
        if remaining and len(selected) < count:
            # Group by timestamp ranges to ensure diversity
            timestamp_groups = {}
            for match in remaining:
                time_group = int(match.timestamp / 2.0)  # Group by 2-second intervals
                if time_group not in timestamp_groups:
                    timestamp_groups[time_group] = []
                timestamp_groups[time_group].append(match)
            
            # Select from different time groups
            time_groups = list(timestamp_groups.keys())
            random.shuffle(time_groups)
            
            for time_group in time_groups:
                if len(selected) >= count:
                    break
                
                group_matches = timestamp_groups[time_group]
                if group_matches:
                    selected_match = random.choice(group_matches)
                    selected.append(selected_match)
        
        # If still need more, just add random selections
        if len(selected) < count:
            remaining = [m for m in top_matches if m not in selected]
            if remaining:
                additional = random.sample(remaining, min(len(remaining), count - len(selected)))
                selected.extend(additional)
        
        # Shuffle the final selection
        random.shuffle(selected)
        
        logger.info(f"Random selection: considered {len(top_matches)} matches, selected {len(selected)} with diversity")
        
        return selected[:count]


class CachedPoseMatcher(PoseMatcher):
    """Pose matcher with caching for better performance"""
    
    def __init__(self, config: RecallConfig):
        super().__init__(config)
        self.similarity_cache = {}  # (pose1_id, pose2_id) -> similarity
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _compute_similarity(self, pose1: NormalizedPose, pose2: NormalizedPose) -> float:
        """Compute similarity with caching"""
        # Create cache key
        key = (id(pose1), id(pose2))
        
        if key in self.similarity_cache:
            self.cache_hits += 1
            return self.similarity_cache[key]
        
        self.cache_misses += 1
        similarity = super()._compute_similarity(pose1, pose2)
        
        # Cache the result
        self.similarity_cache[key] = similarity
        
        # Limit cache size
        if len(self.similarity_cache) > self.config.max_cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.similarity_cache.keys())[:100]
            for k in keys_to_remove:
                del self.similarity_cache[k]
        
        return similarity


def create_pose_matcher(config: RecallConfig, use_cache: bool = True) -> PoseMatcher:
    """Create pose matcher with optional caching"""
    if use_cache:
        return CachedPoseMatcher(config)
    else:
        return PoseMatcher(config) 