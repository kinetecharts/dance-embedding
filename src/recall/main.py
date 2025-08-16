"""CLI interface for the recall module."""

import argparse
import logging
import sys
from pathlib import Path

from .recall_system import RecallSystem
from .config import RecallConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Live pose matching and video playback system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live camera mode
  python -m recall.main
  
  # Video input mode
  python -m recall.main --mode video --input data/video/dai2.mov
  
  # Custom matching settings
  python -m recall.main --mode camera --top-n 3 --match-every 15
  
  # Advanced similarity metric
  python -m recall.main --similarity weighted --normalize-rotation
  
  # Custom directories
  python -m recall.main --pose-dir data/custom_poses --video-dir data/custom_videos --top-n 3
        """
    )
    
    # Input settings
    parser.add_argument(
        "--mode", 
        choices=["camera", "video"], 
        default="camera",
        help="Input mode (default: camera)"
    )
    parser.add_argument(
        "--input", 
        type=str,
        help="Video file path (required for video mode)"
    )
    
    # Matching settings
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=2,
        help="Number of top matches to consider (default: 2)"
    )
    parser.add_argument(
        "--match-every", 
        type=int, 
        default=60,
        help="Match every N frames (default: 60)"
    )
    parser.add_argument(
        "--match-interval", 
        type=float, 
        default=2.0,
        help="Match every N seconds (default: 2.0)"
    )
    parser.add_argument(
        "--playback-duration", 
        type=float, 
        default=2.0,
        help="Duration to play matched videos in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--similarity", 
        choices=["euclidean", "cosine", "weighted"],
        default="euclidean",
        help="Similarity metric (default: euclidean)"
    )
    
    # File paths
    parser.add_argument(
        "--pose-dir", 
        type=str, 
        default="data/poses",
        help="Directory containing pose CSV files (default: data/poses)"
    )
    parser.add_argument(
        "--video-dir", 
        type=str, 
        default="data/video",
        help="Directory containing video files (default: data/video)"
    )
    
    # Processing settings
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.5,
        help="Minimum pose confidence (default: 0.5)"
    )
    parser.add_argument(
        "--normalize-rotation", 
        action="store_true",
        help="Enable rotation normalization"
    )
    
    # Performance settings
    parser.add_argument(
        "--max-cache-size", 
        type=int, 
        default=1000,
        help="Maximum number of cached poses (default: 1000)"
    )
    parser.add_argument(
        "--parallel-workers", 
        type=int, 
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    # Video player settings
    parser.add_argument(
        "--video-player", 
        choices=["opencv", "vlc", "mpv"],
        default="opencv",
        help="Video player type (default: opencv)"
    )
    
    return parser


def validate_args(args) -> None:
    """Validate command line arguments."""
    if args.mode == "video" and not args.input:
        logger.error("--input is required for video mode")
        sys.exit(1)
    
    if args.mode == "video" and not Path(args.input).exists():
        logger.error(f"Input video file not found: {args.input}")
        sys.exit(1)
    
    if not Path(args.pose_dir).exists():
        logger.error(f"Pose directory not found: {args.pose_dir}")
        sys.exit(1)
    
    if not Path(args.video_dir).exists():
        logger.error(f"Video directory not found: {args.video_dir}")
        sys.exit(1)
    
    pose_files = list(Path(args.pose_dir).glob("*.csv"))
    if not pose_files:
        logger.error(f"No pose CSV files found in {args.pose_dir}")
        logger.info("Please run pose extraction first: python -m pose_extraction.main")
        sys.exit(1)
    
    logger.info(f"Found {len(pose_files)} pose files")


def create_config(args) -> RecallConfig:
    """Create configuration from command line arguments."""
    config = RecallConfig(
        mode=args.mode,
        input_video=args.input if args.mode == "video" else None,
        top_n=args.top_n,
        match_every=args.match_every,
        match_interval=args.match_interval,
        match_playback_duration=args.playback_duration,
        similarity_metric=args.similarity,
        pose_dir=args.pose_dir,
        video_dir=args.video_dir,
        confidence_threshold=args.confidence_threshold,
        normalize_rotation=args.normalize_rotation,
        max_cache_size=args.max_cache_size,
        parallel_workers=args.parallel_workers,
        video_player=args.video_player
    )
    
    return config


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Create configuration
        config = create_config(args)
        
        # Log configuration
        logger.info("Recall System Configuration:")
        logger.info(f"  Mode: {config.mode}")
        logger.info(f"  Top-N: {config.top_n}")
        logger.info(f"  Match every: {config.match_every} frames")
        logger.info(f"  Match interval: {config.match_interval} seconds")
        logger.info(f"  Playback duration: {config.match_playback_duration} seconds")
        logger.info(f"  Similarity metric: {config.similarity_metric}")
        logger.info(f"  Pose directory: {config.pose_dir}")
        logger.info(f"  Video directory: {config.video_dir}")
        logger.info(f"  Video with pose directory: {config.video_with_pose_dir}")
        logger.info(f"  Confidence threshold: {config.confidence_threshold}")
        logger.info(f"  Normalize rotation: {config.normalize_rotation}")
        
        # Create and run recall system
        system = RecallSystem(config)
        
        if config.mode == "camera":
            logger.info("Starting live camera mode...")
            logger.info("If camera access fails, try video mode instead:")
            logger.info("  python -m recall.main --mode video --input data/video/dai2.mov")
            
            # Try camera mode first
            try:
                system.run_live()
            except Exception as e:
                if "camera" in str(e).lower() or "Failed to start camera" in str(e):
                    logger.warning("Camera mode failed, trying video mode as fallback...")
                    # Try with a default video file
                    import os
                    video_dir = Path(config.video_dir)
                    video_files = list(video_dir.glob("*.mov")) + list(video_dir.glob("*.mp4"))
                    
                    if video_files:
                        fallback_video = str(video_files[0])
                        logger.info(f"Using fallback video: {fallback_video}")
                        system.run_video(fallback_video)
                    else:
                        logger.error("No video files found for fallback")
                        raise e
                else:
                    raise e
        else:
            logger.info(f"Starting video mode: {config.input_video}")
            system.run_video(config.input_video)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        if "camera" in str(e).lower():
            logger.info("Camera access failed. Try video mode instead:")
            logger.info("  python -m recall.main --mode video --input data/video/dai2.mov")
        sys.exit(1)


if __name__ == "__main__":
    main() 