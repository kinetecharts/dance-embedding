#!/usr/bin/env python3
"""Debug script to test exact matching with dai.mov"""

import logging
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dai_exact_matching():
    """Test exact matching with dai.mov"""
    from recall.pose_embedding import LanceDBPoseDatabase
    from recall.pose_tracker import PoseTracker
    from recall.config import RecallConfig
    
    # Initialize database
    logger.info("Loading LanceDB database...")
    db = LanceDBPoseDatabase()
    
    # Get database stats
    stats = db.get_database_stats()
    logger.info(f"Database has {stats['total_poses']} poses from {stats['unique_videos']} videos")
    
    # Check if dai.mov is in the database
    dai_poses = db.db.open_table('poses').to_pandas()
    dai_video_poses = dai_poses[dai_poses['video_file'] == 'dai.mov']
    logger.info(f"Found {len(dai_video_poses)} poses for dai.mov in database")
    
    if len(dai_video_poses) == 0:
        logger.error("No poses found for dai.mov in database!")
        return
    
    # Check the embeddings in the database
    logger.info("\n--- Checking database embeddings ---")
    sample_embeddings = dai_poses['embedding'].head(3)
    for i, embedding in enumerate(sample_embeddings):
        embedding_array = np.array(embedding)
        logger.info(f"Sample embedding {i+1}: shape={embedding_array.shape}, norm={np.linalg.norm(embedding_array):.6f}")
        logger.info(f"  Has NaN: {np.isnan(embedding_array).any()}")
        logger.info(f"  Has Inf: {np.isinf(embedding_array).any()}")
        logger.info(f"  Min: {np.min(embedding_array):.6f}, Max: {np.max(embedding_array):.6f}")
    
    # Check embeddings from other videos too
    other_video_poses = dai_poses[dai_poses['video_file'] != 'dai.mov'].head(3)
    logger.info(f"\nChecking embeddings from other videos:")
    for i, (_, row) in enumerate(other_video_poses.iterrows()):
        embedding_array = np.array(row['embedding'])
        logger.info(f"Other video embedding {i+1} ({row['video_file']}): shape={embedding_array.shape}, norm={np.linalg.norm(embedding_array):.6f}")
        logger.info(f"  Has NaN: {np.isnan(embedding_array).any()}")
        logger.info(f"  Has Inf: {np.isinf(embedding_array).any()}")
    
    # Initialize pose tracker
    config = RecallConfig()
    tracker = PoseTracker(config)
    
    # Start dai.mov video
    logger.info("\nStarting dai.mov video...")
    if not tracker.start_video("data/video/dai.mov"):
        logger.error("Failed to start dai.mov video")
        return
    
    # Test a few poses from dai.mov
    test_count = 0
    max_tests = 2
    
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
        logger.info(f"Query embedding has NaN: {np.isnan(query_embedding).any()}")
        logger.info(f"Query embedding has Inf: {np.isinf(query_embedding).any()}")
        
        # Search in LanceDB using cosine similarity
        table = db.db.open_table('poses')
        
        # Try cosine similarity first
        logger.info("Trying cosine similarity search...")
        results_cosine = table.search(query_embedding.tolist()).metric("cosine").limit(10).to_pandas()
        
        logger.info(f"Cosine similarity results (first 5):")
        for i, (_, row) in enumerate(results_cosine.head().iterrows()):
            distance = row["_distance"]
            video_file = row["video_file"]
            timestamp = row["timestamp"]
            logger.info(f"  {i+1}. {video_file} at {timestamp:.2f}s (distance: {distance})")
        
        # Try Euclidean distance
        logger.info("\nTrying L2 distance search...")
        results_euclidean = table.search(query_embedding.tolist()).metric("l2").limit(10).to_pandas()
        
        logger.info(f"Euclidean distance results (first 5):")
        for i, (_, row) in enumerate(results_euclidean.head().iterrows()):
            distance = row["_distance"]
            video_file = row["video_file"]
            timestamp = row["timestamp"]
            logger.info(f"  {i+1}. {video_file} at {timestamp:.2f}s (distance: {distance})")
        
        # Use Euclidean results for further testing
        results = results_euclidean
        
        # Manual similarity computation test
        logger.info("\n--- Manual similarity computation test ---")
        
        # Get a few embeddings from dai.mov in the database
        dai_embeddings = dai_poses['embedding'].head(5).tolist()
        logger.info(f"Testing manual similarity with {len(dai_embeddings)} dai.mov embeddings")
        
        for i, db_embedding in enumerate(dai_embeddings):
            db_embedding_array = np.array(db_embedding)
            
            # Compute cosine similarity manually
            cosine_sim = np.dot(query_embedding, db_embedding_array) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding_array))
            
            # Compute L2 distance manually
            l2_distance = np.linalg.norm(query_embedding - db_embedding_array)
            
            logger.info(f"  Manual test {i+1}: cosine_sim={cosine_sim:.6f}, l2_distance={l2_distance:.6f}")
        
        # Test with a few embeddings from other videos
        other_embeddings = dai_poses[dai_poses['video_file'] != 'dai.mov']['embedding'].head(3).tolist()
        logger.info(f"Testing manual similarity with {len(other_embeddings)} other video embeddings")
        
        for i, db_embedding in enumerate(other_embeddings):
            db_embedding_array = np.array(db_embedding)
            
            # Compute cosine similarity manually
            cosine_sim = np.dot(query_embedding, db_embedding_array) / (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding_array))
            
            # Compute L2 distance manually
            l2_distance = np.linalg.norm(query_embedding - db_embedding_array)
            
            logger.info(f"  Other video test {i+1}: cosine_sim={cosine_sim:.6f}, l2_distance={l2_distance:.6f}")
        
        # Check if any dai.mov poses are in the raw results
        dai_matches_raw = results[results['video_file'] == 'dai.mov']
        logger.info(f"Raw results contain {len(dai_matches_raw)} dai.mov poses")
        
        if len(dai_matches_raw) > 0:
            logger.info("Dai.mov poses in raw results:")
            for _, row in dai_matches_raw.head().iterrows():
                distance = row["_distance"]
                timestamp = row["timestamp"]
                logger.info(f"  - dai.mov at {timestamp:.2f}s (distance: {distance})")
        
        # Now test the find_similar_poses method
        matches = db.find_similar_poses(pose_data, top_k=20)  # Get more matches
        
        logger.info(f"Processed matches (top 10):")
        for i, match in enumerate(matches[:10]):
            logger.info(f"  {i+1}. {match['video_file']} at {match['timestamp']:.2f}s (score: {match['similarity_score']:.6f})")
        
        # Check if any dai.mov matches are in the results
        dai_matches = [m for m in matches if m['video_file'] == 'dai.mov']
        logger.info(f"Found {len(dai_matches)} dai.mov matches in top {len(matches)} results")
        
        if len(dai_matches) > 0:
            logger.info("Dai.mov matches found:")
            for i, m in enumerate(dai_matches[:5]):  # Show top 5 dai.mov matches
                logger.info(f"  {i+1}. dai.mov at {m['timestamp']:.2f}s (score: {m['similarity_score']:.6f}, rank: {matches.index(m)+1})")
        else:
            logger.error("No dai.mov matches found at all!")
            
            # Check what the highest scoring dai.mov match would be
            all_dai_poses = dai_poses.copy()
            all_dai_poses['similarity'] = all_dai_poses['embedding'].apply(
                lambda x: np.dot(query_embedding, np.array(x)) / (np.linalg.norm(query_embedding) * np.linalg.norm(np.array(x)))
            )
            best_dai_match = all_dai_poses.loc[all_dai_poses['similarity'].idxmax()]
            logger.info(f"Best dai.mov match would be: {best_dai_match['timestamp']:.2f}s (score: {best_dai_match['similarity']:.6f})")
        
        # Check if the top match is from dai.mov
        if matches:
            top_match = matches[0]
            if top_match['video_file'] == 'dai.mov':
                logger.info("✅ SUCCESS: Top match is from dai.mov!")
            else:
                logger.warning(f"❌ FAILURE: Top match is from {top_match['video_file']}, not dai.mov")
        
        # Check similarity score distribution
        scores = [m['similarity_score'] for m in matches]
        logger.info(f"Score range: {min(scores):.6f} to {max(scores):.6f}")
        
        if test_count >= max_tests:
            break
    
    tracker.release()
    logger.info("Test completed!")

if __name__ == "__main__":
    test_dai_exact_matching() 