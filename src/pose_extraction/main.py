"""Main script for the pose extraction system."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from .pose_extraction import PoseExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoseExtractionPipeline:
    """Pipeline for pose extraction from videos."""
    
    def __init__(self, use_rerun: bool = False):
        """Initialize the pipeline.
        
        Args:
            use_rerun: Whether to use Rerun for visualization during pose extraction
        """
        self.pose_extractor = PoseExtractor(use_rerun=use_rerun)
        
    def extract_poses_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract pose data from a video file.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the CSV file. If None, auto-generates.
            
        Returns:
            Path to the generated CSV file
        """
        logger.info(f"Extracting poses from video: {video_path}")
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"data/poses/{video_name}.csv"
        
        self.pose_extractor.extract_pose_from_video(video_path, output_path)
        return output_path
    
    def run_full_pipeline(self, video_path: str) -> dict:
        """Run the complete pose extraction pipeline.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Dictionary containing results and file paths
        """
        logger.info("Starting pose extraction pipeline")
        
        results = {
            'video_path': video_path,
        }
        
        # Extract poses
        logger.info("Extracting poses from video")
        pose_csv_path = self.extract_poses_from_video(video_path)
        results['pose_csv_path'] = pose_csv_path
        
        logger.info("Pose extraction completed successfully")
        return results
    
    def process_video_directory(self, input_dir: str = "data/video") -> list[dict]:
        """Process all videos in a directory.
        
        Args:
            input_dir: Directory containing video files
            
        Returns:
            List of results for each processed video
        """
        logger.info(f"Processing all videos in directory: {input_dir}")
        
        # Process all videos
        self.pose_extractor.process_video_directory(input_dir)
        
        # Get list of generated CSV files
        results = []
        poses_dir = Path("data/poses")
        for csv_file in poses_dir.glob("*.csv"):
            results.append({
                'pose_csv_path': str(csv_file),
                'video_name': csv_file.stem
            })
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Pose Extraction System - Extract pose landmarks from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract poses from a single video
  python -m pose_extraction.main --video data/video/dance.mp4
  
  # Extract poses from all videos in a directory
  python -m pose_extraction.main --input-dir data/video
  
  # Use Rerun for real-time visualization
  python -m pose_extraction.main --video data/video/dance.mp4 --use-rerun
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Path to single video file")
    input_group.add_argument("--input-dir", type=str, help="Directory containing video files")
    
    # Processing options
    parser.add_argument("--use-rerun", action="store_true", help="Use Rerun for visualization")
    parser.add_argument("--output-dir", type=str, help="Output directory for CSV files")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PoseExtractionPipeline(use_rerun=args.use_rerun)
    
    try:
        if args.video:
            # Process single video
            results = pipeline.run_full_pipeline(args.video)
            print(f"Processed video: {args.video}")
            print(f"Pose data saved to: {results['pose_csv_path']}")
        
        else:
            # Process directory
            results = pipeline.process_video_directory(args.input_dir)
            print(f"Processed {len(results)} videos from: {args.input_dir}")
            for result in results:
                print(f"  - {result['video_name']}: {result['pose_csv_path']}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 