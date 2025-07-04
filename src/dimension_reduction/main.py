"""Main CLI interface for dimension reduction visualization."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import glob

from .visualizer import DimensionReductionVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_video_files(video_dir: str = "data/video") -> list[Path]:
    """Find video files in the specified directory.
    
    Args:
        video_dir: Directory to search for videos
        
    Returns:
        List of video file paths
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return []
    
    video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    video_files = [f for f in video_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in video_extensions]
    
    return sorted(video_files)


def find_pose_csv_files(pose_dir: str = "data/poses") -> list[Path]:
    """Find pose CSV files in the specified directory.
    
    Args:
        pose_dir: Directory to search for pose CSV files
        
    Returns:
        List of pose CSV file paths
    """
    pose_dir = Path(pose_dir)
    if not pose_dir.exists():
        return []
    
    csv_files = [f for f in pose_dir.iterdir() 
                if f.is_file() and f.suffix.lower() == '.csv']
    
    return sorted(csv_files)


def interactive_file_selection() -> tuple[Optional[Path], Optional[Path]]:
    """Interactive file selection for video and pose CSV.
    
    Returns:
        Tuple of (video_path, pose_csv_path) or (None, None) if cancelled
    """
    print("\n=== Dimension Reduction Visualization ===\n")
    
    # Find available video files
    video_files = find_video_files()
    if not video_files:
        print("No video files found in data/video/")
        print("Please place video files in data/video/ directory")
        return None, None
    
    print("Available video files:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file.name}")
    
    # Select video file
    while True:
        try:
            choice = input(f"\nSelect video file (1-{len(video_files)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None, None
            
            video_idx = int(choice) - 1
            if 0 <= video_idx < len(video_files):
                selected_video = video_files[video_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(video_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Find corresponding pose CSV file
    video_name = selected_video.stem
    pose_csv_path = Path("data/poses") / f"{video_name}.csv"
    
    if not pose_csv_path.exists():
        print(f"\nPose CSV file not found: {pose_csv_path}")
        print("Available pose CSV files:")
        
        pose_files = find_pose_csv_files()
        if not pose_files:
            print("  No pose CSV files found in data/poses/")
            print("  Please run pose extraction first")
            return None, None
        
        for i, pose_file in enumerate(pose_files, 1):
            print(f"  {i}. {pose_file.name}")
        
        # Select pose CSV file
        while True:
            try:
                choice = input(f"\nSelect pose CSV file (1-{len(pose_files)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None, None
                
                pose_idx = int(choice) - 1
                if 0 <= pose_idx < len(pose_files):
                    selected_pose_csv = pose_files[pose_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(pose_files)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        selected_pose_csv = pose_csv_path
    
    return selected_video, selected_pose_csv


def select_reduction_method() -> str:
    """Interactive selection of reduction method.
    
    Returns:
        Selected reduction method
    """
    methods = ['pca', 'tsne', 'umap']
    
    print("\nAvailable reduction methods:")
    print("  1. PCA - Principal Component Analysis (fast, linear)")
    print("  2. t-SNE - t-Distributed Stochastic Neighbor Embedding (clustering)")
    print("  3. UMAP - Uniform Manifold Approximation and Projection (best overall)")
    
    while True:
        try:
            choice = input(f"\nSelect method (1-3) [default: 3 (UMAP)]: ").strip()
            if not choice:
                return 'umap'
            
            method_idx = int(choice) - 1
            if 0 <= method_idx < len(methods):
                return methods[method_idx]
            else:
                print("Please enter a number between 1 and 3")
        except ValueError:
            print("Please enter a valid number")


def select_dimensions() -> str:
    """Interactive selection of output dimensions.
    
    Returns:
        Selected dimensions ('2d' or '3d')
    """
    print("\nOutput dimensions:")
    print("  1. 2D - Two-dimensional visualization (faster, easier to view)")
    print("  2. 3D - Three-dimensional visualization (more detailed, interactive)")
    
    while True:
        try:
            choice = input(f"\nSelect dimensions (1-2) [default: 1 (2D)]: ").strip()
            if not choice:
                return '2d'
            
            if choice == '1':
                return '2d'
            elif choice == '2':
                return '3d'
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")


def get_unanalyzed_pose_files(pose_dir, reduced_dir, methods):
    pose_dir = Path(pose_dir)
    reduced_dir = Path(reduced_dir)
    pose_files = [f for f in pose_dir.glob("*.csv") if f.is_file()]
    unanalyzed = []
    for pose_file in pose_files:
        base = pose_file.stem
        missing_methods = []
        for method in methods:
            reduced_file = reduced_dir / f"{base}_{method}_reduced.csv"
            if not reduced_file.exists():
                missing_methods.append(method)
        if missing_methods:
            unanalyzed.append((pose_file, missing_methods))
    return unanalyzed


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Interactive dimension reduction visualization for dance pose data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - generates CSV only (fastest)
  python -m dimension_reduction.main --video data/video/dance.mp4 --pose-csv data/poses/dance.csv
  
  # Interactive mode
  python -m dimension_reduction.main
  
  # Command line mode with HTML output
  python -m dimension_reduction.main --video data/video/dance.mp4 --pose-csv data/poses/dance.csv --method umap --dimensions 3d --save-html
  
  # Quick PCA analysis with static plot
  python -m dimension_reduction.main --video data/video/dance.mp4 --method pca --dimensions 2d --save-png
        """
    )
    
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--pose-csv", type=str, help="Path to pose CSV file")
    parser.add_argument("--method", type=str, choices=['pca', 'tsne', 'umap'], 
                       default='umap', help="Dimension reduction method")
    parser.add_argument("--dimensions", type=str, choices=['2d', '3d'], 
                       default='2d', help="Output dimensions")
    parser.add_argument("--use-3d-coords", action="store_true", 
                       help="Use 3D coordinates from pose data (if available)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Minimum confidence threshold for keypoints")
    parser.add_argument("--output-dir", type=str, 
                       default="data/dimension_reduction",
                       help="Output directory for results")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Disable interactive plot display")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output files")
    parser.add_argument("--create-video-player", action="store_true",
                       help="Create a standalone video player with timeline synchronization")
    parser.add_argument("--extract-frames", action="store_true",
                       help="Extract video frames for synchronization")
    parser.add_argument("--combined", action="store_true",
                       help="Create combined visualization with plot and video player side by side")
    parser.add_argument("--save-html", action="store_true",
                       help="Save interactive HTML plot")
    parser.add_argument("--save-png", action="store_true",
                       help="Save static PNG plot")
    
    args = parser.parse_args()
    
    if not args.video and not args.pose_csv:
        # Batch mode: process all pose files that haven't been analyzed for all methods
        methods = ['umap', 'tsne', 'pca']
        pose_dir = Path("data/poses")
        reduced_dir = Path("data/dimension_reduction")
        unanalyzed = get_unanalyzed_pose_files(pose_dir, reduced_dir, methods)
        if not unanalyzed:
            print("All pose files have been analyzed for all methods.")
            return
        print(f"Pose files to analyze: {[f[0].name for f in unanalyzed]}")
        for pose_file, missing_methods in unanalyzed:
            for method in missing_methods:
                print(f"Analyzing {pose_file.name} with {method.upper()}...")
                # Use default video path guess (data/video/{base}.mp4 or similar)
                video_guess = None
                video_dir = Path("data/video")
                for ext in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
                    candidate = video_dir / f"{pose_file.stem}{ext}"
                    if candidate.exists():
                        video_guess = candidate
                        break
                visualizer = DimensionReductionVisualizer(output_dir="data/dimension_reduction")
                visualizer.load_data(str(video_guess) if video_guess else "", str(pose_file))
                visualizer.create_visualization(method=method, dimensions="2d")
                visualizer.save_reduced_data(save_csv=True)
        print("Batch dimension reduction complete.")
        return
    
    try:
        # Initialize visualizer
        logger.info("Initializing dimension reduction visualizer...")
        visualizer = DimensionReductionVisualizer(output_dir=args.output_dir)
        
        # Load data
        logger.info(f"Loading data from {args.video} and {args.pose_csv}")
        visualizer.load_data(args.video, args.pose_csv)
        
        # Create visualization
        logger.info(f"Creating {args.method.upper()} visualization in {args.dimensions.upper()}")
        visualizer.create_visualization(
            method=args.method,
            dimensions=args.dimensions,
            use_3d_coords=args.use_3d_coords,
            confidence_threshold=args.confidence_threshold
        )
        
        # Save CSV data (default behavior)
        if not args.no_save:
            logger.info("Saving reduced data...")
            visualizer.save_reduced_data(save_csv=True)
        
        # Create plots only if explicitly requested
        if args.save_html:
            logger.info("Creating interactive plot...")
            visualizer.create_interactive_plot(save_html=True)
        
        if args.save_png:
            logger.info("Creating static plot...")
            visualizer.create_static_plot(save_png=True)
        
        if args.create_video_player:
            logger.info("Creating video player with timeline synchronization...")
            visualizer.create_video_player_html(save_html=True)
        
        # Extract video frames if requested
        if args.extract_frames:
            logger.info("Extracting video frames...")
            visualizer.extract_video_frames()
        
        # Create combined visualization if requested
        if args.combined:
            logger.info("Creating combined visualization...")
            visualizer.create_combined_visualization(save_html=True)
        
        # Show interactive plot only if explicitly requested
        if not args.no_interactive and (args.create_video_player or args.combined or args.save_html):
            logger.info("Displaying interactive visualization...")
            visualizer.show()
        
        logger.info("Dimension reduction visualization complete!")
        
        # Print summary
        video_name = Path(args.video).stem
        print(f"\n=== Results Summary ===")
        print(f"Video: {video_name}")
        print(f"Method: {args.method.upper()}")
        print(f"Dimensions: {args.dimensions.upper()}")
        print(f"Frames processed: {len(visualizer.pose_data)}")
        print(f"Output directory: {args.output_dir}")
        
        if not args.no_save:
            print(f"\nFiles created:")
            print(f"  - {video_name}_{args.method}_reduced.csv")
        
        if args.save_html:
            print(f"  - {video_name}_{args.method}_interactive_{args.dimensions}.html")
        
        if args.save_png:
            print(f"  - {video_name}_{args.method}_static_2d.png")
        
        if args.create_video_player:
            print(f"  - {video_name}_video_player.html")
        
        if args.extract_frames:
            print(f"  - Video frames extracted to data/analysis/video_frames/")
        
        if args.combined:
            print(f"  - {video_name}_combined_visualization.html")
    
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 