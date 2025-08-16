"""Pose matching using LanceDB for efficient similarity search."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import random
import time

from .data_structures import PoseData, NormalizedPose, Match
from .pose_normalizer import PoseNormalizer
from .pose_embedding import LanceDBPoseDatabase, PoseEmbeddingGenerator

logger = logging.getLogger(__name__)


class LanceDBPoseMatcher:
    """Pose matcher using LanceDB for efficient similarity search"""
    
    def __init__(self, database: LanceDBPoseDatabase, config: Dict[str, Any]):
        self.database = database
        self.config = config
        self.embedding_generator = PoseEmbeddingGenerator()
        
        # Create a default config for the normalizer
        from .config import RecallConfig
        default_config = RecallConfig()
        self.normalizer = PoseNormalizer(default_config)
        
        # Performance tracking
        self.total_matches = 0
        self.match_times = []
        
    def find_matches(self, pose_data: PoseData, top_n: int = 3) -> List[Match]:
        """
        Find similar poses using LanceDB vector search
        
        Args:
            pose_data: Current pose to find matches for
            top_n: Number of top matches to return
            
        Returns:
            List of Match objects sorted by similarity
        """
        start_time = time.time()
        
        try:
            # Use LanceDB to find similar poses
            matches = self.database.find_similar_poses(pose_data, top_k=top_n)
            
            # Convert to Match objects
            match_objects = []
            for match_dict in matches:
                # Create a dummy NormalizedPose for the match
                # In a real implementation, you might want to load the actual pose data
                dummy_normalized = NormalizedPose(
                    coordinates=np.zeros((33, 3)),  # Placeholder
                    original_pose=pose_data,  # Placeholder
                    normalization_params=match_dict.get("normalization_params", {})
                )
                
                match = Match(
                    pose_file=f"{Path(match_dict['video_file']).stem}.csv",
                    video_file=match_dict['video_file'],
                    timestamp=match_dict['timestamp'],
                    frame_number=match_dict['frame_number'],
                    similarity_score=match_dict['similarity_score'],
                    normalized_pose=dummy_normalized,
                    pose_index=match_dict['pose_index']
                )
                match_objects.append(match)
            
            # Track performance
            match_time = time.time() - start_time
            self.match_times.append(match_time)
            self.total_matches += 1
            
            logger.debug(f"Found {len(match_objects)} matches in {match_time:.3f}s")
            
            return match_objects
            
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            return []
    
    def select_random_match(self, matches: List[Match]) -> Optional[Match]:
        """
        Select a random match from the top matches with diversity
        
        Args:
            matches: List of matches sorted by similarity
            
        Returns:
            Randomly selected match or None
        """
        if not matches:
            return None
        
        # For now, just return the best match
        # Could implement diversity-based selection later
        return matches[0]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.match_times:
            return {
                "total_matches": 0,
                "avg_match_time": 0.0,
                "min_match_time": 0.0,
                "max_match_time": 0.0
            }
        
        return {
            "total_matches": self.total_matches,
            "avg_match_time": np.mean(self.match_times),
            "min_match_time": np.min(self.match_times),
            "max_match_time": np.max(self.match_times)
        }


class PoseMatcher:
    """Legacy pose matcher - kept for backward compatibility"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Create a default config for the normalizer
        from .config import RecallConfig
        default_config = RecallConfig()
        self.normalizer = PoseNormalizer(default_config)
        
        # Initialize LanceDB database
        self.lancedb_matcher = None
        self._initialize_lancedb()
        
        # Performance tracking
        self.total_matches = 0
        self.match_times = []
    
    def _initialize_lancedb(self):
        """Initialize LanceDB database"""
        try:
            pose_dir = self.config.get('pose_dir', 'data/poses')
            video_dir = self.config.get('video_dir', 'data/video')
            db_path = self.config.get('db_path', 'data/pose_database.lancedb')
            
            # Check if database exists, if not create it
            db_path_obj = Path(db_path)
            if not db_path_obj.exists():
                logger.info("Creating new LanceDB pose database...")
                from .pose_embedding import create_pose_database
                database = create_pose_database(pose_dir, video_dir, db_path)
            else:
                logger.info("Loading existing LanceDB pose database...")
                database = LanceDBPoseDatabase(db_path)
            
            self.lancedb_matcher = LanceDBPoseMatcher(database, self.config)
            
            # Print database stats
            stats = database.get_database_stats()
            logger.info(f"LanceDB Database Stats: {stats['total_poses']} poses from {stats['unique_videos']} videos")
            
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            self.lancedb_matcher = None
    
    def find_matches(self, pose_data: PoseData, top_n: int = 3) -> List[Match]:
        """
        Find similar poses using LanceDB
        
        Args:
            pose_data: Current pose to find matches for
            top_n: Number of top matches to return
            
        Returns:
            List of Match objects sorted by similarity
        """
        if self.lancedb_matcher is None:
            logger.error("LanceDB matcher not initialized")
            return []
        
        return self.lancedb_matcher.find_matches(pose_data, top_n)
    
    def select_random_match(self, matches: List[Match]) -> Optional[Match]:
        """
        Select a random match from the top matches
        
        Args:
            matches: List of matches sorted by similarity
            
        Returns:
            Randomly selected match or None
        """
        if self.lancedb_matcher is None:
            return None
        
        return self.lancedb_matcher.select_random_match(matches)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.lancedb_matcher is None:
            return {"error": "LanceDB matcher not initialized"}
        
        return self.lancedb_matcher.get_performance_stats() 