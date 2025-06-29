"""Main script for the dance motion embedding system."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from .pose_extraction import PoseExtractor
from .embedding_generator import EmbeddingGenerator
from .motion_analyzer import MotionAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DanceMotionEmbeddingPipeline:
    """Complete pipeline for dance motion embedding."""
    
    def __init__(self, use_rerun: bool = False, model_type: str = "transformer", 
                 device: str = "cpu"):
        """Initialize the pipeline.
        
        Args:
            use_rerun: Whether to use Rerun for visualization during pose extraction
            model_type: Type of embedding model ("transformer" or "lstm")
            device: Device to run models on
        """
        self.pose_extractor = PoseExtractor(use_rerun=use_rerun)
        self.embedding_generator = EmbeddingGenerator(model_type=model_type, device=device)
        self.motion_analyzer = MotionAnalyzer()
        
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
    
    def generate_embeddings_from_poses(self, pose_csv_path: str, 
                                     output_path: Optional[str] = None,
                                     embedding_type: str = "segment") -> str:
        """Generate embeddings from pose data.
        
        Args:
            pose_csv_path: Path to the pose CSV file
            output_path: Path to save embeddings. If None, auto-generates.
            embedding_type: Type of embedding ("pose" or "segment")
            
        Returns:
            Path to the generated embeddings file
        """
        logger.info(f"Generating {embedding_type} embeddings from: {pose_csv_path}")
        
        if output_path is None:
            csv_name = Path(pose_csv_path).stem
            output_path = f"data/embeddings/{csv_name}_{embedding_type}.npy"
        
        self.embedding_generator.process_csv_file(pose_csv_path, output_path, embedding_type)
        return output_path
    
    def analyze_embeddings(self, embeddings_path: str, 
                          output_dir: Optional[str] = None) -> dict:
        """Analyze motion embeddings.
        
        Args:
            embeddings_path: Path to the embeddings file
            output_dir: Directory to save analysis results. If None, auto-generates.
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing embeddings from: {embeddings_path}")
        
        if output_dir is None:
            embeddings_name = Path(embeddings_path).stem
            output_dir = f"data/analysis/{embeddings_name}"
        
        embeddings = self.motion_analyzer.load_embeddings(embeddings_path)
        results = self.motion_analyzer.analyze_motion_patterns(embeddings, output_dir)
        return results
    
    def run_full_pipeline(self, video_path: str, embedding_type: str = "segment",
                         analyze: bool = True) -> dict:
        """Run the complete pipeline from video to analysis.
        
        Args:
            video_path: Path to the input video file
            embedding_type: Type of embedding to generate
            analyze: Whether to perform motion analysis
            
        Returns:
            Dictionary containing all results and file paths
        """
        logger.info("Starting full dance motion embedding pipeline")
        
        results = {
            'video_path': video_path,
            'embedding_type': embedding_type
        }
        
        # Step 1: Extract poses
        logger.info("Step 1: Extracting poses from video")
        pose_csv_path = self.extract_poses_from_video(video_path)
        results['pose_csv_path'] = pose_csv_path
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings from poses")
        embeddings_path = self.generate_embeddings_from_poses(pose_csv_path, 
                                                             embedding_type=embedding_type)
        results['embeddings_path'] = embeddings_path
        
        # Step 3: Analyze embeddings (optional)
        if analyze:
            logger.info("Step 3: Analyzing motion patterns")
            analysis_results = self.analyze_embeddings(embeddings_path)
            results['analysis_results'] = analysis_results
            results['analysis_dir'] = f"data/analysis/{Path(embeddings_path).stem}"
        
        logger.info("Pipeline completed successfully")
        return results
    
    def process_video_directory(self, input_dir: str = "data/video", 
                               embedding_type: str = "segment",
                               analyze: bool = True) -> list[dict]:
        """Process all videos in a directory.
        
        Args:
            input_dir: Directory containing video files
            embedding_type: Type of embedding to generate
            analyze: Whether to perform motion analysis
            
        Returns:
            List of results for each processed video
        """
        logger.info(f"Processing all videos in directory: {input_dir}")
        
        # Process all videos
        self.pose_extractor.process_video_directory(input_dir)
        
        # Generate embeddings for all pose files
        self.embedding_generator.process_directory(embedding_type=embedding_type)
        
        # Analyze embeddings if requested
        results = []
        if analyze:
            embeddings_dir = Path("data/embeddings")
            for embedding_file in embeddings_dir.glob(f"*_{embedding_type}.npy"):
                try:
                    analysis_results = self.analyze_embeddings(str(embedding_file))
                    results.append({
                        'embeddings_path': str(embedding_file),
                        'analysis_results': analysis_results
                    })
                except Exception as e:
                    logger.error(f"Error analyzing {embedding_file}: {e}")
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Dance Motion Embedding System - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video
  python -m dance_motion_embedding.main --video data/video/dance.mp4
  
  # Process all videos in a directory
  python -m dance_motion_embedding.main --input-dir data/video
  
  # Generate pose embeddings only (no analysis)
  python -m dance_motion_embedding.main --video data/video/dance.mp4 --no-analyze
  
  # Use LSTM model and GPU
  python -m dance_motion_embedding.main --video data/video/dance.mp4 --model lstm --device cuda
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Path to single video file")
    input_group.add_argument("--input-dir", type=str, help="Directory containing video files")
    
    # Processing options
    parser.add_argument("--embedding-type", type=str, default="segment", 
                       choices=["pose", "segment"], help="Type of embedding to generate")
    parser.add_argument("--model", type=str, default="transformer", 
                       choices=["transformer", "lstm"], help="Embedding model type")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on")
    parser.add_argument("--use-rerun", action="store_true", help="Use Rerun for visualization")
    parser.add_argument("--no-analyze", action="store_true", help="Skip motion analysis")
    
    # Output options
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DanceMotionEmbeddingPipeline(
        use_rerun=args.use_rerun,
        model_type=args.model,
        device=args.device
    )
    
    try:
        if args.video:
            # Process single video
            results = pipeline.run_full_pipeline(
                args.video,
                embedding_type=args.embedding_type,
                analyze=not args.no_analyze
            )
            print(f"Processed video: {args.video}")
            print(f"Pose data saved to: {results['pose_csv_path']}")
            print(f"Embeddings saved to: {results['embeddings_path']}")
            if not args.no_analyze:
                print(f"Analysis results saved to: {results['analysis_dir']}")
        
        else:
            # Process directory
            results = pipeline.process_video_directory(
                args.input_dir,
                embedding_type=args.embedding_type,
                analyze=not args.no_analyze
            )
            print(f"Processed {len(results)} videos from: {args.input_dir}")
            if not args.no_analyze:
                print(f"Analysis results saved to: data/analysis/")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main() 