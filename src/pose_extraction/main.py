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


def get_unprocessed_videos(video_dir: Path, pose_dir: Path) -> list[Path]:
    """Get list of videos that haven't been processed yet."""
    video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    video_files = [f for f in video_dir.glob("*") if f.is_file() and f.suffix.lower() in video_extensions]
    processed_bases = {f.stem for f in pose_dir.glob("*.csv")}
    unprocessed = [f for f in video_files if f.stem not in processed_bases]
    return unprocessed


class PoseExtractionPipeline:
    """Pipeline for pose extraction from videos."""
    
    def __init__(self, use_rerun: bool = False, generate_overlay_video: bool = True):
        """Initialize the pipeline.
        
        Args:
            use_rerun: Whether to use Rerun for visualization during pose extraction
            generate_overlay_video: Whether to generate videos with pose overlays
        """
        self.pose_extractor = PoseExtractor(use_rerun=use_rerun, generate_overlay_video=generate_overlay_video)
        
    def extract_poses_from_video(self, video_path: str, output_path: Optional[str] = None, 
                                overlay_video_path: Optional[str] = None) -> dict:
        """Extract pose data from a video file.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the CSV file. If None, auto-generates.
            overlay_video_path: Path to save the overlay video. If None, auto-generates.
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info(f"Extracting poses from video: {video_path}")
        
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"data/poses/{video_name}.csv"
        
        # Only set overlay_video_path if we're generating overlay videos
        if overlay_video_path is None and self.pose_extractor.generate_overlay_video:
            video_name = Path(video_path).stem
            overlay_video_path = f"data/video_with_pose/{video_name}_with_pose.mp4"
        
        pose_data = self.pose_extractor.extract_pose_from_video(video_path, output_path, overlay_video_path)
        
        return {
            'pose_csv_path': output_path,
            'overlay_video_path': overlay_video_path if self.pose_extractor.generate_overlay_video else None,
            'frame_count': len(pose_data)
        }
    
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
        extraction_results = self.extract_poses_from_video(video_path)
        results.update(extraction_results)
        
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
            video_name = csv_file.stem
            overlay_video_path = f"data/video_with_pose/{video_name}_with_pose.mp4"
            
            results.append({
                'pose_csv_path': str(csv_file),
                'video_name': video_name,
                'overlay_video_path': overlay_video_path if Path(overlay_video_path).exists() else None
            })
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Pose Extraction System - Extract pose landmarks from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract poses from all videos in data/video (default)
  python -m pose_extraction.main
  
  # Extract poses from a single video
  python -m pose_extraction.main --video data/video/dance.mp4
  
  # Extract poses from all videos in a specific directory
  python -m pose_extraction.main --input-dir data/video
  
  # Use Rerun for real-time visualization
  python -m pose_extraction.main --video data/video/dance.mp4 --use-rerun
  
  # Disable overlay video generation (faster processing)
  python -m pose_extraction.main --no-overlay-video
        """
    )
    
    # Input options - make input-dir the default
    parser.add_argument("--video", type=str, help="Path to single video file")
    parser.add_argument("--input-dir", type=str, default="data/video", help="Directory containing video files")
    
    # Processing options
    parser.add_argument("--use-rerun", action="store_true", help="Use Rerun for visualization")
    parser.add_argument("--no-overlay-video", action="store_true", help="Disable generation of overlay videos")
    parser.add_argument("--output-dir", type=str, default="data/poses", help="Output directory for CSV files")
    parser.add_argument("--model-path", type=str, help="Path to MediaPipe model")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PoseExtractionPipeline(use_rerun=args.use_rerun, generate_overlay_video=not args.no_overlay_video)
    
    try:
        if args.video:
            # Process single video
            results = pipeline.run_full_pipeline(args.video)
            print(f"Processed video: {args.video}")
            print(f"Pose data saved to: {results['pose_csv_path']}")
            if results.get('overlay_video_path'):
                print(f"Overlay video saved to: {results['overlay_video_path']}")
            print(f"Extracted {results.get('frame_count', 0)} frames")
        
        else:
            # Process directory with unprocessed videos logic (default behavior)
            video_dir = Path(args.input_dir)
            pose_dir = Path(args.output_dir)
            
            if not video_dir.exists():
                logger.error(f"Input directory {args.input_dir} does not exist")
                return
            
            pose_dir.mkdir(parents=True, exist_ok=True)
            
            unprocessed_videos = get_unprocessed_videos(video_dir, pose_dir)
            if not unprocessed_videos:
                print("All videos already processed. No new files to process.")
                return
            
            print(f"Videos to process: {[f.name for f in unprocessed_videos]}")
            
            # Process unprocessed videos
            results = []
            for video_file in unprocessed_videos:
                logger.info(f"Processing {video_file.name}")
                try:
                    output_file = pose_dir / f"{video_file.stem}.csv"
                    overlay_file = Path("data/video_with_pose") / f"{video_file.stem}_with_pose.mp4" if not args.no_overlay_video else None
                    
                    extraction_results = pipeline.extract_poses_from_video(str(video_file), str(output_file), str(overlay_file) if overlay_file else None)
                    results.append(extraction_results)
                except Exception as e:
                    logger.error(f"Error processing {video_file.name}: {e}")
                    continue
            
            # Show skipped files
            processed_bases = {f.stem for f in pose_dir.glob("*.csv")}
            skipped = [f for f in video_dir.glob("*") if f.is_file() and f.stem in processed_bases]
            if skipped:
                print(f"Skipped already processed: {[f.name for f in skipped]}")
            
            print(f"Processed {len(results)} videos from: {args.input_dir}")
            for result in results:
                video_name = result.get('video_name', '<unknown>')
                print(f"  - {video_name}: {result.get('pose_csv_path', '')}")
                if result.get('overlay_video_path'):
                    print(f"    Overlay video: {result['overlay_video_path']}")
                print(f"    Frames: {result.get('frame_count', 0)}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 