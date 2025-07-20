#!/usr/bin/env python3
"""Focused test script for dai* videos with Dai2.csv exact matching"""

import logging
import numpy as np
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dai_focused_matching():
    """Test exact matching with only dai* videos"""
    from recall.pose_embedding import LanceDBPoseDatabase
    from recall.pose_tracker import PoseTracker
    from recall.config import RecallConfig
    
    # Clear existing database
    db_path = "data/pose_database_dai_test.lancedb"
    if Path(db_path).exists():
        shutil.rmtree(db_path)
        logger.info(f"Cleared existing database: {db_path}")
    
    # Initialize database with custom path
    logger.info("Creating new LanceDB database for dai* videos...")
    db = LanceDBPoseDatabase(db_path)
    
    # Load only dai* videos
    pose_dir = Path("data/poses")
    video_dir = Path("data/video")
    
    # Load both dai* and Dai* files (case-sensitive)
    dai_csv_files = list(pose_dir.glob("dai*.csv")) + list(pose_dir.glob("Dai*.csv"))
    logger.info(f"Found {len(dai_csv_files)} dai*/Dai* CSV files: {[f.stem for f in dai_csv_files]}")
    
    total_poses = 0
    for csv_file in dai_csv_files:
        # Find corresponding video file
        video_name = csv_file.stem
        video_file = None
        
        # Look for video file with various extensions
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.MOV']:
            potential_video = video_dir / f"{video_name}{ext}"
            if potential_video.exists():
                video_file = f"{video_name}{ext}"
                break
        
        if video_file is None:
            logger.warning(f"No video file found for {csv_file}")
            continue
        
        logger.info(f"Loading {csv_file} -> {video_file}")
        poses_loaded = db.load_poses_from_csv(csv_file, video_file)
        total_poses += poses_loaded
    
    logger.info(f"Total poses loaded: {total_poses}")
    
    # Get database stats
    stats = db.get_database_stats()
    logger.info(f"Database has {stats['total_poses']} poses from {stats['unique_videos']} videos")
    logger.info(f"Videos in database: {stats['video_files']}")
    
    # Check if Dai2.csv is in the database
    dai2_poses = db.db.open_table('poses').to_pandas()
    dai2_video_poses = dai2_poses[dai2_poses['video_file'] == 'Dai2.mov']
    logger.info(f"Found {len(dai2_video_poses)} poses for Dai2.mov in database")
    
    if len(dai2_video_poses) == 0:
        logger.error("No poses found for Dai2.mov in database!")
        return
    
    # Initialize pose tracker
    config = RecallConfig()
    tracker = PoseTracker(config)
    
    # Start Dai2.mov video
    logger.info("Starting Dai2.mov video...")
    if not tracker.start_video("data/video/Dai2.mov"):
        logger.error("Failed to start Dai2.mov video")
        return
    
    # Test a few poses from Dai2.mov
    test_count = 0
    max_tests = 3
    
    while test_count < max_tests:
        result = tracker.get_next_pose()
        if result is None:
            continue
            
        pose_data, frame = result
        if pose_data is None:
            continue
        
        test_count += 1
        logger.info(f"\n--- Test {test_count}: Frame {pose_data.frame_number}, Time {pose_data.timestamp:.2f}s ---")
        
        # Generate embedding for the query pose
        query_embedding = db.embedding_generator.generate_embedding(pose_data)
        logger.info(f"Query embedding shape: {query_embedding.shape}, norm: {np.linalg.norm(query_embedding):.6f}")
        
        # Find similar poses
        matches = db.find_similar_poses(pose_data, top_k=10)
        
        logger.info(f"Top 10 matches:")
        for i, match in enumerate(matches):
            logger.info(f"  {i+1}. {match['video_file']} at {match['timestamp']:.2f}s (score: {match['similarity_score']:.6f})")
        
        # Check if the top match is from Dai2.mov
        if matches:
            top_match = matches[0]
            if top_match['video_file'] == 'Dai2.mov':
                logger.info("✅ SUCCESS: Top match is from Dai2.mov!")
            else:
                logger.warning(f"❌ FAILURE: Top match is from {top_match['video_file']}, not Dai2.mov")
                
                # Check if any Dai2.mov matches are in the results
                dai2_matches = [m for m in matches if m['video_file'] == 'Dai2.mov']
                if dai2_matches:
                    logger.info(f"Found {len(dai2_matches)} Dai2.mov matches in results:")
                    for m in dai2_matches:
                        logger.info(f"  - Dai2.mov at {m['timestamp']:.2f}s (score: {m['similarity_score']:.6f})")
                else:
                    logger.error("No Dai2.mov matches found at all!")
        
        # Check similarity score distribution
        scores = [m['similarity_score'] for m in matches]
        logger.info(f"Score range: {min(scores):.6f} to {max(scores):.6f}")
        
        if test_count >= max_tests:
            break
    
    tracker.release()
    logger.info("Test completed!")

if __name__ == "__main__":
    test_dai_focused_matching() 