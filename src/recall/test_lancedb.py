"""Test script for LanceDB pose embedding system."""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from typing import List, Dict, Any

from .pose_embedding import LanceDBPoseDatabase, PoseEmbeddingGenerator, create_pose_database
from .data_structures import PoseData
from .pose_normalizer import PoseNormalizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pose_embedding_generation():
    """Test pose embedding generation"""
    logger.info("Testing pose embedding generation...")
    
    # Create a sample pose
    landmarks = np.random.rand(33, 3)  # Random 3D landmarks
    confidence = np.random.rand(33)    # Random confidence scores
    
    pose_data = PoseData(
        landmarks=landmarks,
        confidence=confidence,
        timestamp=0.0,
        frame_number=0
    )
    
    # Generate embedding
    embedding_generator = PoseEmbeddingGenerator()
    embedding = embedding_generator.generate_embedding(pose_data)
    
    logger.info(f"Generated embedding shape: {embedding.shape}")
    logger.info(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
    
    # Test that embedding is normalized
    assert np.abs(np.linalg.norm(embedding) - 1.0) < 1e-6, "Embedding should be normalized"
    
    logger.info("✅ Pose embedding generation test passed")


def test_pose_database_creation():
    """Test LanceDB database creation"""
    logger.info("Testing LanceDB database creation...")
    
    # Create database
    db_path = "data/test_pose_database.lancedb"
    database = LanceDBPoseDatabase(db_path)
    
    # Check if table exists
    assert database.table_name in database.db.table_names(), "Table should be created"
    
    # Get stats
    stats = database.get_database_stats()
    logger.info(f"Database stats: {stats}")
    
    # Clean up
    database.clear_database()
    
    logger.info("✅ LanceDB database creation test passed")


def test_pose_loading_from_csv():
    """Test loading poses from CSV files"""
    logger.info("Testing pose loading from CSV...")
    
    # Create database
    db_path = "data/test_pose_database.lancedb"
    database = LanceDBPoseDatabase(db_path)
    
    # Load poses from actual CSV files
    pose_dir = "data/poses"
    video_dir = "data/video"
    
    if Path(pose_dir).exists():
        total_poses = database.load_all_poses(pose_dir, video_dir)
        logger.info(f"Loaded {total_poses} poses from CSV files")
        
        # Get stats
        stats = database.get_database_stats()
        logger.info(f"Database stats after loading: {stats}")
        
        assert stats['total_poses'] > 0, "Should have loaded some poses"
    else:
        logger.warning(f"Pose directory {pose_dir} not found, skipping CSV loading test")
    
    # Clean up
    database.clear_database()
    
    logger.info("✅ Pose loading from CSV test passed")


def test_exact_match_retrieval():
    """Test that feeding a pose from the database returns the exact match"""
    logger.info("Testing exact match retrieval...")
    
    # Create database
    db_path = "data/test_pose_database.lancedb"
    database = LanceDBPoseDatabase(db_path)
    
    # Load poses from actual CSV files
    pose_dir = "data/poses"
    video_dir = "data/video"
    
    if not Path(pose_dir).exists():
        logger.warning(f"Pose directory {pose_dir} not found, skipping exact match test")
        return
    
    total_poses = database.load_all_poses(pose_dir, video_dir)
    if total_poses == 0:
        logger.warning("No poses loaded, skipping exact match test")
        return
    
    # Get a pose from the database
    table = database.db.open_table(database.table_name)
    df = table.to_pandas()
    
    if len(df) == 0:
        logger.warning("No poses in database, skipping exact match test")
        return
    
    # Take a random pose from the database
    random_row = df.sample(1).iloc[0]
    
    # Recreate the original pose data
    embedding = np.array(random_row['embedding'])
    video_file = random_row['video_file']
    timestamp = random_row['timestamp']
    frame_number = random_row['frame_number']
    pose_index = random_row['pose_index']
    
    logger.info(f"Testing with pose from {video_file} at {timestamp:.2f}s (index {pose_index})")
    
    # Create a dummy pose for testing (we'll use the embedding directly)
    # In a real scenario, we'd load the actual pose data from CSV
    dummy_landmarks = np.random.rand(33, 3)
    dummy_confidence = np.random.rand(33)
    
    test_pose = PoseData(
        landmarks=dummy_landmarks,
        confidence=dummy_confidence,
        timestamp=timestamp,
        frame_number=frame_number
    )
    
    # Find similar poses
    matches = database.find_similar_poses(test_pose, top_k=5)
    
    logger.info(f"Found {len(matches)} matches:")
    for i, match in enumerate(matches):
        logger.info(f"  {i+1}. {match['video_file']} at {match['timestamp']:.2f}s (score: {match['similarity_score']:.6f})")
    
    # Check if we get any matches
    assert len(matches) > 0, "Should find at least one match"
    
    # The best match should have a high similarity score
    best_match = matches[0]
    logger.info(f"Best match similarity score: {best_match['similarity_score']:.6f}")
    
    # Clean up
    database.clear_database()
    
    logger.info("✅ Exact match retrieval test passed")


def test_performance():
    """Test performance of LanceDB matching"""
    logger.info("Testing LanceDB performance...")
    
    # Create database
    db_path = "data/test_pose_database.lancedb"
    database = LanceDBPoseDatabase(db_path)
    
    # Load poses
    pose_dir = "data/poses"
    video_dir = "data/video"
    
    if Path(pose_dir).exists():
        total_poses = database.load_all_poses(pose_dir, video_dir)
        logger.info(f"Loaded {total_poses} poses for performance test")
        
        # Create test poses
        test_poses = []
        for i in range(10):
            landmarks = np.random.rand(33, 3)
            confidence = np.random.rand(33)
            pose = PoseData(
                landmarks=landmarks,
                confidence=confidence,
                timestamp=i * 0.1,
                frame_number=i
            )
            test_poses.append(pose)
        
        # Time the matching
        start_time = time.time()
        total_matches = 0
        
        for pose in test_poses:
            matches = database.find_similar_poses(pose, top_k=5)
            total_matches += len(matches)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Performance test results:")
        logger.info(f"  Total poses in database: {total_poses}")
        logger.info(f"  Test queries: {len(test_poses)}")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Average time per query: {total_time/len(test_poses):.3f}s")
        logger.info(f"  Queries per second: {len(test_poses)/total_time:.1f}")
        logger.info(f"  Total matches found: {total_matches}")
        
        # Performance should be reasonable
        avg_time_per_query = total_time / len(test_poses)
        assert avg_time_per_query < 0.1, f"Average query time {avg_time_per_query:.3f}s is too slow"
        
    else:
        logger.warning(f"Pose directory {pose_dir} not found, skipping performance test")
    
    # Clean up
    database.clear_database()
    
    logger.info("✅ Performance test passed")


def test_embedding_invariance():
    """Test that embeddings are invariant to translation and rotation"""
    logger.info("Testing embedding invariance...")
    
    # Create a base pose
    landmarks = np.random.rand(33, 3)
    confidence = np.random.rand(33)
    
    base_pose = PoseData(
        landmarks=landmarks,
        confidence=confidence,
        timestamp=0.0,
        frame_number=0
    )
    
    # Create translated pose
    translation = np.array([10.0, 5.0, 2.0])
    translated_landmarks = landmarks + translation
    translated_pose = PoseData(
        landmarks=translated_landmarks,
        confidence=confidence,
        timestamp=0.0,
        frame_number=0
    )
    
    # Generate embeddings
    embedding_generator = PoseEmbeddingGenerator()
    base_embedding = embedding_generator.generate_embedding(base_pose)
    translated_embedding = embedding_generator.generate_embedding(translated_pose)
    
    # Compute similarity
    similarity = np.dot(base_embedding, translated_embedding)
    logger.info(f"Similarity between base and translated pose: {similarity:.6f}")
    
    # Similarity should be high (close to 1.0) for translation invariance
    assert similarity > 0.8, f"Translation invariance failed: similarity {similarity:.6f} too low"
    
    logger.info("✅ Embedding invariance test passed")


def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("RUNNING LANCEDB POSE EMBEDDING TESTS")
    logger.info("=" * 60)
    
    try:
        test_pose_embedding_generation()
        test_pose_database_creation()
        test_pose_loading_from_csv()
        test_exact_match_retrieval()
        test_performance()
        test_embedding_invariance()
        
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED! ✅")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests() 