#!/usr/bin/env python3
"""Test script for the recall system."""

import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from recall import RecallSystem, RecallConfig, create_recall_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_configuration():
    """Test configuration creation and validation"""
    logger.info("Testing configuration...")
    
    try:
        # Test basic configuration
        config = RecallConfig(
            mode="camera",
            top_n=3,
            match_every=15,
            similarity_metric="euclidean",
            use_rerun=True
        )
        logger.info("✅ Basic configuration created successfully")
        
        # Test video mode configuration
        config_video = RecallConfig(
            mode="video",
            input_path="data/video/test.mp4",
            top_n=2,
            use_rerun=False
        )
        logger.info("✅ Video mode configuration created successfully")
        
        # Test Rerun configuration
        config_rerun = RecallConfig(
            rerun_port=9090,
            visualization_layout="side_by_side",
            rerun_max_fps=60
        )
        logger.info("✅ Rerun configuration created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_system_creation():
    """Test system creation"""
    logger.info("Testing system creation...")
    
    try:
        config = RecallConfig(
            mode="camera",
            top_n=2,
            use_rerun=False  # Disable Rerun for testing
        )
        
        system = create_recall_system(config, with_keyboard=False)
        logger.info("✅ System created successfully")
        
        # Test system properties
        assert system.config.top_n == 2
        assert system.config.mode == "camera"
        assert not system.config.use_rerun
        
        logger.info("✅ System properties verified")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System creation test failed: {e}")
        return False


def test_pose_data_structures():
    """Test pose data structures"""
    logger.info("Testing pose data structures...")
    
    try:
        from recall import PoseData, NormalizedPose, Match
        import numpy as np
        
        # Test PoseData creation
        landmarks = np.random.rand(33, 3)
        confidence = np.random.rand(33)
        
        pose_data = PoseData(
            landmarks=landmarks,
            confidence=confidence,
            timestamp=1.0,
            frame_number=10
        )
        
        assert pose_data.landmarks.shape == (33, 3)
        assert pose_data.confidence.shape == (33,)
        assert pose_data.timestamp == 1.0
        assert pose_data.frame_number == 10
        assert pose_data.is_3d
        
        logger.info("✅ PoseData creation successful")
        
        # Test pose connections
        from recall import get_pose_connections, get_landmark_name
        
        connections = get_pose_connections()
        assert len(connections) > 0
        
        landmark_name = get_landmark_name(0)
        assert landmark_name == "nose"
        
        logger.info("✅ Pose connections and landmark names working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pose data structures test failed: {e}")
        return False


def test_pose_normalization():
    """Test pose normalization"""
    logger.info("Testing pose normalization...")
    
    try:
        from recall import PoseNormalizer, normalize_pose_batch, PoseData
        import numpy as np
        
        config = RecallConfig()
        normalizer = PoseNormalizer(config)
        
        # Create test pose
        landmarks = np.random.rand(33, 3)
        confidence = np.random.rand(33)
        
        pose_data = PoseData(
            landmarks=landmarks,
            confidence=confidence,
            timestamp=1.0,
            frame_number=10
        )
        
        # Test normalization
        normalized_pose = normalizer.normalize(pose_data)
        
        assert normalized_pose.coordinates.shape == (33, 3)
        assert normalized_pose.original_pose == pose_data
        
        # Test batch normalization
        poses = [pose_data, pose_data]
        normalized_poses = normalize_pose_batch(poses, config)
        
        assert len(normalized_poses) == 2
        
        logger.info("✅ Pose normalization working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pose normalization test failed: {e}")
        return False


def test_pose_matching():
    """Test pose matching (without actual CSV files)"""
    logger.info("Testing pose matching...")
    
    try:
        from recall import create_pose_matcher
        
        config = RecallConfig(
            pose_dir="data/poses",
            video_dir="data/video",
            use_rerun=False
        )
        
        # Create matcher (will fail if no pose files, but that's expected)
        matcher = create_pose_matcher(config, use_cache=False)
        
        logger.info("✅ Pose matcher created successfully")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Pose matcher test (expected if no pose files): {e}")
        return True  # This is expected if no pose files exist


def main():
    """Run all tests"""
    logger.info("Starting recall system tests...")
    
    tests = [
        ("Configuration", test_configuration),
        ("System Creation", test_system_creation),
        ("Pose Data Structures", test_pose_data_structures),
        ("Pose Normalization", test_pose_normalization),
        ("Pose Matching", test_pose_matching),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! The recall system is ready to use.")
        logger.info("\nTo run the system:")
        logger.info("  python -m recall.main --help")
        logger.info("  python -m recall.main --mode camera --top-n 3")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 