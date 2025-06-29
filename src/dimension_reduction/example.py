#!/usr/bin/env python3
"""Example usage of the dimension reduction module."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dimension_reduction import DimensionReductionVisualizer


def example_basic_usage():
    """Basic example of dimension reduction visualization."""
    print("=== Basic Dimension Reduction Example ===")
    
    # Initialize visualizer
    visualizer = DimensionReductionVisualizer()
    
    # Example file paths (you'll need to provide your own files)
    video_path = "data/video/example_dance.mp4"
    pose_csv_path = "data/poses/example_dance.csv"
    
    # Check if files exist
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        print("Please place a video file in data/video/example_dance.mp4")
        return
    
    if not Path(pose_csv_path).exists():
        print(f"Pose CSV file not found: {pose_csv_path}")
        print("Please run pose extraction first to generate pose CSV")
        return
    
    # Load data
    print("Loading data...")
    visualizer.load_data(video_path, pose_csv_path)
    
    # Create UMAP visualization in 2D
    print("Creating UMAP visualization...")
    visualizer.create_visualization(
        method='umap',
        dimensions='2d',
        use_3d_coords=False,
        confidence_threshold=0.5
    )
    
    # Create and save plots
    print("Creating plots...")
    fig = visualizer.create_interactive_plot(save_html=True)
    visualizer.create_static_plot(save_png=True)
    visualizer.save_reduced_data(save_json=True)
    
    # Display interactive plot
    print("Displaying interactive visualization...")
    visualizer.show()
    
    print("Example completed!")


def example_comparison_methods():
    """Example comparing different reduction methods."""
    print("\n=== Method Comparison Example ===")
    
    # Initialize visualizer
    visualizer = DimensionReductionVisualizer()
    
    # Example file paths
    video_path = "data/video/example_dance.mp4"
    pose_csv_path = "data/poses/example_dance.csv"
    
    if not Path(video_path).exists() or not Path(pose_csv_path).exists():
        print("Required files not found. Skipping comparison example.")
        return
    
    # Load data
    visualizer.load_data(video_path, pose_csv_path)
    
    # Compare different methods
    methods = ['pca', 'tsne', 'umap']
    
    for method in methods:
        print(f"\nCreating {method.upper()} visualization...")
        
        # Create visualization
        visualizer.create_visualization(
            method=method,
            dimensions='2d',
            use_3d_coords=False,
            confidence_threshold=0.5
        )
        
        # Save results
        fig = visualizer.create_interactive_plot(save_html=True)
        visualizer.save_reduced_data(save_json=True)
        
        print(f"{method.upper()} visualization saved!")
    
    print("Method comparison completed!")


def example_3d_visualization():
    """Example of 3D visualization."""
    print("\n=== 3D Visualization Example ===")
    
    # Initialize visualizer
    visualizer = DimensionReductionVisualizer()
    
    # Example file paths
    video_path = "data/video/example_dance.mp4"
    pose_csv_path = "data/poses/example_dance.csv"
    
    if not Path(video_path).exists() or not Path(pose_csv_path).exists():
        print("Required files not found. Skipping 3D example.")
        return
    
    # Load data
    visualizer.load_data(video_path, pose_csv_path)
    
    # Create 3D UMAP visualization
    print("Creating 3D UMAP visualization...")
    visualizer.create_visualization(
        method='umap',
        dimensions='3d',
        use_3d_coords=True,  # Use 3D coordinates if available
        confidence_threshold=0.5
    )
    
    # Create and save 3D plot
    fig = visualizer.create_interactive_plot(save_html=True)
    visualizer.save_reduced_data(save_json=True)
    
    # Display 3D plot
    print("Displaying 3D visualization...")
    visualizer.show()
    
    print("3D visualization completed!")


def example_custom_parameters():
    """Example with custom parameters."""
    print("\n=== Custom Parameters Example ===")
    
    # Initialize visualizer
    visualizer = DimensionReductionVisualizer()
    
    # Example file paths
    video_path = "data/video/example_dance.mp4"
    pose_csv_path = "data/poses/example_dance.csv"
    
    if not Path(video_path).exists() or not Path(pose_csv_path).exists():
        print("Required files not found. Skipping custom parameters example.")
        return
    
    # Load data
    visualizer.load_data(video_path, pose_csv_path)
    
    # Create t-SNE with custom parameters
    print("Creating t-SNE with custom parameters...")
    visualizer.create_visualization(
        method='tsne',
        dimensions='2d',
        use_3d_coords=False,
        confidence_threshold=0.7,  # Higher confidence threshold
        perplexity=50,  # Custom perplexity
        learning_rate=300,  # Custom learning rate
        n_iter=2000  # More iterations
    )
    
    # Create and save plot
    fig = visualizer.create_interactive_plot(save_html=True)
    visualizer.save_reduced_data(save_json=True)
    
    print("Custom parameters example completed!")


if __name__ == "__main__":
    print("Dimension Reduction Examples")
    print("=" * 40)
    
    # Run examples
    example_basic_usage()
    example_comparison_methods()
    example_3d_visualization()
    example_custom_parameters()
    
    print("\nAll examples completed!")
    print("\nTo run the interactive CLI:")
    print("  python -m dimension_reduction.main") 