"""Test that all modules can be imported correctly."""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import_pose_extraction():
    """Test that pose extraction module can be imported."""
    try:
        from pose_extraction.pose_extraction import PoseExtractor, VideoSource, PoseData
        assert PoseExtractor is not None
        assert VideoSource is not None
        assert PoseData is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import pose_extraction: {e}")


def test_import_main():
    """Test that main module can be imported."""
    try:
        from pose_extraction.main import PoseExtractionPipeline
        assert PoseExtractionPipeline is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import main: {e}")


def test_import_package():
    """Test that the main package can be imported."""
    try:
        from pose_extraction import PoseExtractor, PoseExtractionPipeline
        assert PoseExtractor is not None
        assert PoseExtractionPipeline is not None
    except ImportError as e:
        raise AssertionError(f"Failed to import main package: {e}")


if __name__ == "__main__":
    # Run all tests
    test_import_pose_extraction()
    test_import_main()
    test_import_package()
    print("All import tests passed!") 