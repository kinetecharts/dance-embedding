#!/usr/bin/env python3
"""Script to rebuild LanceDB database with 32-dimensional embeddings"""

import logging
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def rebuild_database():
    """Rebuild the LanceDB database with 32-dimensional embeddings"""
    from recall.pose_embedding import LanceDBPoseDatabase
    
    # Clear existing database
    db_path = "data/pose_database.lancedb"
    if Path(db_path).exists():
        shutil.rmtree(db_path)
        logger.info(f"Cleared existing database: {db_path}")
    
    # Create new database with 32-dimensional embeddings
    logger.info("Creating new LanceDB database with 32-dimensional embeddings...")
    db = LanceDBPoseDatabase(db_path)
    
    # Load all poses
    pose_dir = "data/poses"
    video_dir = "data/video"
    
    total_poses = db.load_all_poses(pose_dir, video_dir)
    
    # Get database stats
    stats = db.get_database_stats()
    logger.info(f"Database rebuilt successfully!")
    logger.info(f"Total poses: {stats['total_poses']}")
    logger.info(f"Unique videos: {stats['unique_videos']}")
    logger.info(f"Videos: {stats['video_files']}")

if __name__ == "__main__":
    rebuild_database() 