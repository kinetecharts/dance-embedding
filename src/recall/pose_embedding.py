"""Pose embedding system using LanceDB for efficient similarity search."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import lancedb
from lancedb.embeddings import get_registry
import mediapipe as mp

from .data_structures import PoseData, NormalizedPose
from .pose_normalizer import PoseNormalizer

logger = logging.getLogger(__name__)


@dataclass
class PoseEmbedding:
    """A pose embedding with metadata"""
    embedding: np.ndarray
    video_file: str
    timestamp: float
    frame_number: int
    pose_index: int
    original_pose: PoseData
    normalization_params: Dict[str, Any]


class PoseEmbeddingGenerator:
    """Generate rotation and translation invariant pose embeddings"""
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        
        # Create a default config for the normalizer
        from .config import RecallConfig
        default_config = RecallConfig()
        self.normalizer = PoseNormalizer(default_config)
        
    def generate_embedding(self, pose_data: PoseData) -> np.ndarray:
        """
        Generate a simple embedding from pose data.
        
        This creates an embedding that is:
        1. Translation invariant (centered on root joint)
        2. Scale invariant (normalized by torso length)
        3. Simple and direct from the 99 pose coordinates
        
        Args:
            pose_data: Raw pose data from MediaPipe
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Step 1: Normalize pose (translation and scale)
        normalized_pose = self.normalizer.normalize(pose_data)
        
        # Step 2: Extract simple features from the 99 coordinates
        features = self._extract_simple_features(normalized_pose)
        
        # Step 3: Create embedding vector
        embedding = self._create_embedding_vector(features)
        
        return embedding
    
    def _extract_simple_features(self, normalized_pose: NormalizedPose) -> np.ndarray:
        """Extract simple features from normalized pose coordinates"""
        coords = normalized_pose.coordinates
        
        # Flatten the 33 landmarks × 3 coordinates = 99 values
        # But let's focus on key body parts for better matching
        key_landmarks = [
            0,   # nose
            11, 12,  # shoulders
            13, 14,  # elbows
            15, 16,  # wrists
            23, 24,  # hips
            25, 26,  # knees
            27, 28,  # ankles
        ]
        
        # Extract coordinates for key landmarks
        key_coords = coords[key_landmarks].flatten()  # 13 landmarks × 3 coords = 39 values
        
        # Add some simple derived features
        # 1. Torso length (shoulder to hip distance)
        torso_length = np.linalg.norm(coords[12] - coords[11])
        
        # 2. Arm lengths
        left_arm_length = np.linalg.norm(coords[13] - coords[11]) + np.linalg.norm(coords[15] - coords[13])
        right_arm_length = np.linalg.norm(coords[14] - coords[12]) + np.linalg.norm(coords[16] - coords[14])
        
        # 3. Leg lengths
        left_leg_length = np.linalg.norm(coords[25] - coords[23]) + np.linalg.norm(coords[27] - coords[25])
        right_leg_length = np.linalg.norm(coords[26] - coords[24]) + np.linalg.norm(coords[28] - coords[26])
        
        # Combine all features
        features = np.concatenate([
            key_coords,  # 39 values
            [torso_length, left_arm_length, right_arm_length, left_leg_length, right_leg_length]  # 5 values
        ])
        
        return features  # Total: 44 values
    
    def _create_embedding_vector(self, features: np.ndarray) -> np.ndarray:
        """Create a fixed-size embedding vector from features"""
        # Pad or truncate to target dimension
        if len(features) < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros(self.embedding_dim - len(features))
            embedding = np.concatenate([features, padding])
        elif len(features) > self.embedding_dim:
            # Truncate
            embedding = features[:self.embedding_dim]
        else:
            embedding = features
        
        # Normalize to unit length for cosine similarity
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding


class LanceDBPoseDatabase:
    """LanceDB-based pose database for efficient similarity search"""
    
    def __init__(self, db_path: str = "data/pose_database.lancedb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.table_name = "poses"
        
        # Initialize embedding generator
        self.embedding_generator = PoseEmbeddingGenerator()
        
        # Check if table exists
        if self.table_name not in self.db.table_names():
            self._create_table()
    
    def _create_table(self):
        """Create the poses table with proper schema"""
        import pyarrow as pa
        
        schema = pa.schema([
            ("embedding", pa.list_(pa.float32(), 32)),  # Smaller embedding dimension
            ("video_file", pa.string()),
            ("timestamp", pa.float32()),
            ("frame_number", pa.int32()),
            ("pose_index", pa.int32()),
            ("normalization_params", pa.string())  # JSON as string
        ])
        
        # Create empty table
        self.db.create_table(self.table_name, schema=schema)
        logger.info(f"Created LanceDB table: {self.table_name}")
    
    def load_poses_from_csv(self, csv_file: Path, video_file: str) -> int:
        """
        Load poses from CSV file into LanceDB
        
        Args:
            csv_file: Path to CSV file with pose data
            video_file: Name of corresponding video file
            
        Returns:
            Number of poses loaded
        """
        logger.info(f"Loading poses from {csv_file} for video {video_file}")
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
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
        
        # Extract pose data
        poses = []
        for idx, row in df.iterrows():
            # Extract landmarks (33 landmarks, 3 coordinates each)
            landmarks = []
            confidences = []
            
            for landmark_name in landmark_names:
                x = row.get(f'{landmark_name}_x', 0.0)
                y = row.get(f'{landmark_name}_y', 0.0)
                z = row.get(f'{landmark_name}_z', 0.0)
                confidence = row.get(f'{landmark_name}_confidence', 1.0)
                
                landmarks.append([x, y, z])
                confidences.append(confidence)
            
            # Create PoseData
            pose_data = PoseData(
                landmarks=np.array(landmarks),
                confidence=np.array(confidences),
                timestamp=row.get('timestamp', idx * 0.033),  # Assume 30fps if not provided
                frame_number=row.get('frame_number', idx)
            )
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(pose_data)
            
            # Store in LanceDB
            record = {
                "embedding": embedding.tolist(),
                "video_file": video_file,
                "timestamp": pose_data.timestamp,
                "frame_number": pose_data.frame_number,
                "pose_index": idx,
                "normalization_params": "{}"  # Empty JSON as string
            }
            
            poses.append(record)
        
        # Insert into LanceDB
        table = self.db.open_table(self.table_name)
        table.add(poses)
        
        logger.info(f"Loaded {len(poses)} poses from {csv_file}")
        return len(poses)
    
    def load_all_poses(self, pose_dir: str = "data/poses", video_dir: str = "data/video") -> int:
        """
        Load all pose CSV files into LanceDB
        
        Args:
            pose_dir: Directory containing pose CSV files
            video_dir: Directory containing video files
            
        Returns:
            Total number of poses loaded
        """
        pose_dir = Path(pose_dir)
        video_dir = Path(video_dir)
        
        if not pose_dir.exists():
            logger.warning(f"Pose directory {pose_dir} does not exist")
            return 0
        
        total_poses = 0
        
        for csv_file in pose_dir.glob("*.csv"):
            # Find corresponding video file
            video_name = csv_file.stem
            video_file = None
            
            # Look for video file with various extensions
            for ext in ['.mp4', '.mov', '.avi', '.mkv']:
                potential_video = video_dir / f"{video_name}{ext}"
                if potential_video.exists():
                    video_file = f"{video_name}{ext}"
                    break
            
            if video_file is None:
                logger.warning(f"No video file found for {csv_file}")
                continue
            
            poses_loaded = self.load_poses_from_csv(csv_file, video_file)
            total_poses += poses_loaded
        
        logger.info(f"Total poses loaded: {total_poses}")
        return total_poses
    
    def find_similar_poses(self, query_pose: PoseData, top_k: int = 5, target_videos: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find similar poses using cosine similarity
        
        Args:
            query_pose: Pose to find matches for
            top_k: Number of top matches to return
            target_videos: Optional list of video file names to filter results (e.g., ["Dai3.mp4", "Francine.MOV"])
            
        Returns:
            List of match dictionaries with similarity scores
        """
        # Generate embedding for query pose
        query_embedding = self.embedding_generator.generate_embedding(query_pose)
        
        # Search in LanceDB using cosine similarity
        table = self.db.open_table(self.table_name)
        
        try:
            # Try L2 distance first (more reliable than cosine in LanceDB)
            results = table.search(query_embedding.tolist()).metric("l2").limit(top_k * 3).to_pandas()
            
            # Check if LanceDB returned valid distances
            if results.empty or results["_distance"].isna().all():
                logger.warning("LanceDB L2 search returned NaN distances, trying cosine...")
                # Try cosine as fallback
                results = table.search(query_embedding.tolist()).metric("cosine").limit(top_k * 3).to_pandas()
                
                if results.empty or results["_distance"].isna().all():
                    logger.warning("LanceDB cosine search also failed, falling back to manual computation")
                    return self._find_similar_poses_manual(query_embedding, top_k, target_videos)
            
            # Convert distances to similarity scores and filter out NaN results
            matches = []
            for _, row in results.iterrows():
                distance = row["_distance"]
                if pd.isna(distance):
                    continue
                
                # Filter by target videos if specified
                video_file = row["video_file"]
                if target_videos is not None:
                    if video_file not in target_videos:
                        continue
                
                # For L2 distance, convert to similarity (lower distance = higher similarity)
                # Normalize L2 distance to [0,1] range and invert
                # Assuming normalized vectors, max L2 distance is ~2.0
                if distance <= 2.0:
                    similarity_score = 1.0 - (distance / 2.0)
                else:
                    similarity_score = 0.0
                
                match = {
                    "video_file": video_file,
                    "timestamp": row["timestamp"],
                    "frame_number": row["frame_number"],
                    "pose_index": row["pose_index"],
                    "similarity_score": similarity_score,
                    "distance": distance
                }
                matches.append(match)
            
            # Sort by similarity score (highest first) and return top_k
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"LanceDB search failed: {e}, falling back to manual computation")
            return self._find_similar_poses_manual(query_embedding, top_k, target_videos)
    
    def _find_similar_poses_manual(self, query_embedding: np.ndarray, top_k: int = 5, target_videos: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Manual fallback for finding similar poses when LanceDB fails
        """
        logger.info("Using manual similarity computation")
        
        # Get all poses from database
        table = self.db.open_table(self.table_name)
        all_poses = table.to_pandas()
        
        # Compute similarities manually
        similarities = []
        for _, row in all_poses.iterrows():
            video_file = row["video_file"]
            
            # Filter by target videos if specified
            if target_videos is not None:
                if video_file not in target_videos:
                    continue
            
            db_embedding = np.array(row["embedding"])
            
            # Compute cosine similarity
            cosine_sim = np.dot(query_embedding, db_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
            
            similarities.append({
                "video_file": video_file,
                "timestamp": row["timestamp"],
                "frame_number": row["frame_number"],
                "pose_index": row["pose_index"],
                "similarity_score": cosine_sim,
                "distance": 1.0 - cosine_sim
            })
        
        # Sort by similarity score (highest first) and return top_k
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:top_k]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the pose database"""
        table = self.db.open_table(self.table_name)
        
        # Get total count
        total_poses = len(table)
        
        # Get unique videos
        videos = table.to_pandas()["video_file"].unique()
        
        return {
            "total_poses": total_poses,
            "unique_videos": len(videos),
            "video_files": videos.tolist()
        }
    
    def clear_database(self):
        """Clear all data from the database"""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            self._create_table()
            logger.info("Cleared pose database")


def create_pose_database(pose_dir: str = "data/poses", video_dir: str = "data/video", 
                        db_path: str = "data/pose_database.lancedb") -> LanceDBPoseDatabase:
    """
    Create and populate a LanceDB pose database
    
    Args:
        pose_dir: Directory containing pose CSV files
        video_dir: Directory containing video files
        db_path: Path for LanceDB database
        
    Returns:
        Populated LanceDBPoseDatabase instance
    """
    database = LanceDBPoseDatabase(db_path)
    database.load_all_poses(pose_dir, video_dir)
    return database 