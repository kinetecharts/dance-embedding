#!/usr/bin/env python3
"""Basic usage example for the dance motion embedding system."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dance_motion_embedding import PoseExtractor, EmbeddingGenerator, MotionAnalyzer
from dance_motion_embedding.main import DanceMotionEmbeddingPipeline


def example_pose_extraction():
    """Example of pose extraction from video."""
    print("=== Pose Extraction Example ===")
    
    # Initialize pose extractor
    extractor = PoseExtractor(use_rerun=False)  # Set to True for visualization
    
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


def example_embedding_generation():
    """Example of embedding generation from pose data."""
    print("\n=== Embedding Generation Example ===")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_type="transformer", device="cpu")
    
    # Example pose CSV path
    pose_csv_path = "data/poses/example_dance.csv"
    
    # Check if pose data exists
    if not Path(pose_csv_path).exists():
        print(f"Pose data file not found: {pose_csv_path}")
        print("Please run pose extraction first")
        return
    
    # Generate embeddings
    embeddings = generator.process_csv_file(pose_csv_path, embedding_type="segment")
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print(f"Embeddings saved to: data/embeddings/example_dance_segment.npy")


def example_motion_analysis():
    """Example of motion analysis."""
    print("\n=== Motion Analysis Example ===")
    
    # Initialize motion analyzer
    analyzer = MotionAnalyzer(method="umap")
    
    # Example embeddings path
    embeddings_path = "data/embeddings/example_dance_segment.npy"
    
    # Check if embeddings exist
    if not Path(embeddings_path).exists():
        print(f"Embeddings file not found: {embeddings_path}")
        print("Please run embedding generation first")
        return
    
    # Load embeddings
    embeddings = analyzer.load_embeddings(embeddings_path)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Perform analysis
    results = analyzer.analyze_motion_patterns(embeddings)
    print("Motion analysis completed")
    print(f"Analysis results saved to: data/analysis/example_dance_segment/")


def example_full_pipeline():
    """Example of running the complete pipeline."""
    print("\n=== Full Pipeline Example ===")
    
    # Initialize pipeline
    pipeline = DanceMotionEmbeddingPipeline(
        use_rerun=False,
        model_type="transformer",
        device="cpu"
    )
    
    # Example video path
    video_path = "data/video/example_dance.mp4"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a video file in data/video/example_dance.mp4")
        return
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(video_path, embedding_type="segment")
    
    print("Full pipeline completed successfully!")
    print(f"Results:")
    print(f"  - Pose data: {results['pose_csv_path']}")
    print(f"  - Embeddings: {results['embeddings_path']}")
    print(f"  - Analysis: {results['analysis_dir']}")


def example_custom_analysis():
    """Example of custom motion analysis."""
    print("\n=== Custom Analysis Example ===")
    
    # Initialize analyzer
    analyzer = MotionAnalyzer(method="tsne")  # Use t-SNE instead of UMAP
    
    # Example embeddings path
    embeddings_path = "data/embeddings/example_dance_segment.npy"
    
    if not Path(embeddings_path).exists():
        print(f"Embeddings file not found: {embeddings_path}")
        return
    
    # Load embeddings
    embeddings = analyzer.load_embeddings(embeddings_path)
    
    # Custom dimensionality reduction
    embeddings_3d = analyzer.reduce_dimensions(embeddings, n_components=3, perplexity=20)
    print(f"Reduced embeddings to 3D using t-SNE")
    
    # Custom clustering
    cluster_labels = analyzer.cluster_embeddings(embeddings, method="kmeans", n_clusters=3)
    print(f"Clustered embeddings into {len(set(cluster_labels))} clusters")
    
    # Custom visualization
    fig = analyzer.visualize_embeddings_3d(
        embeddings_3d, 
        labels=cluster_labels,
        title="Custom Dance Motion Analysis",
        save_path="data/analysis/custom_analysis.html"
    )
    print("Custom analysis visualization saved to: data/analysis/custom_analysis.html")


def main():
    """Run all examples."""
    print("Dance Motion Embedding System - Basic Usage Examples")
    print("=" * 50)
    
    # Create necessary directories
    Path("data/video").mkdir(parents=True, exist_ok=True)
    Path("data/poses").mkdir(parents=True, exist_ok=True)
    Path("data/embeddings").mkdir(parents=True, exist_ok=True)
    Path("data/analysis").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    example_pose_extraction()
    example_embedding_generation()
    example_motion_analysis()
    example_full_pipeline()
    example_custom_analysis()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the system with your own data:")
    print("1. Place video files in data/video/")
    print("2. Run: python -m dance_motion_embedding.main --input-dir data/video")
    print("3. Check results in data/poses/, data/embeddings/, and data/analysis/")


if __name__ == "__main__":
    main() 