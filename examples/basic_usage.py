#!/usr/bin/env python3
"""Basic usage example for the pose extraction system."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pose_extraction import PoseExtractor
from pose_extraction.main import PoseExtractionPipeline


def example_pose_extraction():
    """Example of pose extraction from video."""
    print("=== Pose Extraction Example ===")
    
    # Initialize pose extractor
    extractor = PoseExtractor(use_rerun=False, generate_overlay_video=True)  # Set to False to disable features
    
    # Example video path (you'll need to provide your own video)
    video_path = "data/video/example_dance.mp4"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a video file in data/video/example_dance.mp4")
        return
    
    # Extract poses
    pose_data = extractor.extract_pose_from_video(video_path)
    print(f"Extracted pose data from {len(pose_data)} frames")
    print(f"Pose data saved to: data/poses/example_dance.csv")
    print(f"Overlay video saved to: data/video_with_pose/example_dance_with_pose.mp4")


def example_full_pipeline():
    """Example of running the complete pose extraction pipeline."""
    print("\n=== Full Pipeline Example ===")
    
    # Initialize pipeline
    pipeline = PoseExtractionPipeline(use_rerun=False, generate_overlay_video=True)
    
    # Example video path
    video_path = "data/video/example_dance.mp4"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a video file in data/video/example_dance.mp4")
        return
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(video_path)
    
    print("Full pipeline completed successfully!")
    print(f"Results:")
    print(f"  - Pose data: {results['pose_csv_path']}")
    print(f"  - Overlay video: {results['overlay_video_path']}")
    print(f"  - Frames processed: {results['frame_count']}")


def example_batch_processing():
    """Example of batch processing multiple videos."""
    print("\n=== Batch Processing Example ===")
    
    # Initialize pipeline
    pipeline = PoseExtractionPipeline(use_rerun=False, generate_overlay_video=True)
    
    # Process all videos in data/video directory
    results = pipeline.process_video_directory("data/video")
    
    if results:
        print(f"Processed {len(results)} videos:")
        for result in results:
            print(f"  - {result['video_name']}: {result['pose_csv_path']}")
            if result.get('overlay_video_path'):
                print(f"    Overlay video: {result['overlay_video_path']}")
    else:
        print("No videos found in data/video/ directory")
        print("Please place some video files in data/video/")


def example_with_rerun():
    """Example of pose extraction with Rerun visualization."""
    print("\n=== Rerun Visualization Example ===")
    
    # Initialize extractor with Rerun enabled
    extractor = PoseExtractor(use_rerun=True, generate_overlay_video=True)
    
    # Example video path
    video_path = "data/video/example_dance.mp4"
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a video file in data/video/example_dance.mp4")
        return
    
    print("Starting pose extraction with Rerun visualization...")
    print("You should see a Rerun window with real-time 3D pose tracking")
    
    # Extract poses with visualization
    pose_data = extractor.extract_pose_from_video(video_path)
    print(f"Extracted {len(pose_data)} frames with visualization")
    print(f"Overlay video also generated for review")


def example_without_overlay():
    """Example of pose extraction without overlay video (faster)."""
    print("\n=== Fast Processing Example (No Overlay) ===")
    
    # Initialize extractor without overlay video generation
    extractor = PoseExtractor(use_rerun=False, generate_overlay_video=False)
    
    # Example video path
    video_path = "data/video/example_dance.mp4"
    
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a video file in data/video/example_dance.mp4")
        return
    
    print("Starting fast pose extraction (no overlay video)...")
    
    # Extract poses without overlay video
    pose_data = extractor.extract_pose_from_video(video_path)
    print(f"Extracted {len(pose_data)} frames (CSV only, no overlay video)")


def main():
    """Run all examples."""
    print("Pose Extraction System - Basic Usage Examples")
    print("=" * 50)
    
    # Create necessary directories
    Path("data/video").mkdir(parents=True, exist_ok=True)
    Path("data/poses").mkdir(parents=True, exist_ok=True)
    Path("data/video_with_pose").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    example_pose_extraction()
    example_full_pipeline()
    example_batch_processing()
    example_with_rerun()
    example_without_overlay()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the system with your own data:")
    print("1. Place video files in data/video/")
    print("2. Run: python -m pose_extraction.main")
    print("3. Check results in data/poses/ and data/video_with_pose/")


if __name__ == "__main__":
    main() 