"""Test that all modules can be imported correctly."""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import_pose_extraction():
    """Test that pose extraction module can be imported."""
    try:
        from dance_motion_embedding.pose_extraction import PoseExtractor, VideoSource, PoseData
        assert PoseExtractor is not None
        assert VideoSource is not None
        assert PoseData is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import pose_extraction: {e}")


def test_import_embedding_generator():
    """Test that embedding generator module can be imported."""
    try:
        from dance_motion_embedding.embedding_generator import (
            EmbeddingGenerator, PoseTransformer, SegmentLSTM, PoseDataset
        )
        assert EmbeddingGenerator is not None
        assert PoseTransformer is not None
        assert SegmentLSTM is not None
        assert PoseDataset is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import embedding_generator: {e}")


def test_import_motion_analyzer():
    """Test that motion analyzer module can be imported."""
    try:
        from dance_motion_embedding.motion_analyzer import MotionAnalyzer
        assert MotionAnalyzer is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import motion_analyzer: {e}")


def test_import_main():
    """Test that main module can be imported."""
    try:
        from dance_motion_embedding.main import DanceMotionEmbeddingPipeline
        assert DanceMotionEmbeddingPipeline is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import main: {e}")


def test_import_package():
    """Test that the main package can be imported."""
    try:
        from dance_motion_embedding import PoseExtractor, EmbeddingGenerator, MotionAnalyzer
        assert PoseExtractor is not None
        assert EmbeddingGenerator is not None
        assert MotionAnalyzer is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import main package: {e}")


if __name__ == "__main__":
    # Run all tests
    test_import_pose_extraction()
    test_import_embedding_generator()
    test_import_motion_analyzer()
    test_import_main()
    test_import_package()
    print("All import tests passed!") 