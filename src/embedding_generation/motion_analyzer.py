"""Motion analysis module for pose embeddings."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Final, Optional

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap

# Configure logging
logger = logging.getLogger(__name__)


class MotionAnalyzer:
    """Analyzes motion embeddings for patterns and similarities."""
    
    def __init__(self, method: str = "umap"):
        """Initialize the motion analyzer.
        
        Args:
            method: Dimensionality reduction method ("umap", "tsne", "pca")
        """
        self.method = method
        self.reducer = None
        self.embeddings_3d = None
        self.cluster_labels = None
        
    def load_embeddings(self, embeddings_path: str) -> np.ndarray:
        """Load embeddings from file.
        
        Args:
            embeddings_path: Path to embeddings file
            
        Returns:
            Array of embeddings
        """
        embeddings_path = Path(embeddings_path)
        
        if embeddings_path.suffix == '.npy':
            embeddings = np.load(embeddings_path)
        elif embeddings_path.suffix == '.csv':
            embeddings = pd.read_csv(embeddings_path).values
        else:
            raise ValueError(f"Unsupported file format: {embeddings_path.suffix}")
        
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 3, 
                         **kwargs) -> np.ndarray:
        """Reduce dimensionality of embeddings for visualization.
        
        Args:
            embeddings: Input embeddings
            n_components: Number of components to reduce to
            **kwargs: Additional arguments for the reduction method
            
        Returns:
            Reduced embeddings
        """
        if self.method == "umap":
            self.reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.method == "tsne":
            self.reducer = TSNE(
                n_components=n_components,
                perplexity=kwargs.get('perplexity', 30),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.method == "pca":
            self.reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")
        
        self.embeddings_3d = self.reducer.fit_transform(embeddings)
        logger.info(f"Reduced embeddings to {n_components} dimensions using {self.method}")
        
        return self.embeddings_3d
    
    def cluster_embeddings(self, embeddings: np.ndarray, method: str = "kmeans", 
                          n_clusters: int = 5, **kwargs) -> np.ndarray:
        """Cluster embeddings to identify motion patterns.
        
        Args:
            embeddings: Input embeddings
            method: Clustering method ("kmeans" or "dbscan")
            n_clusters: Number of clusters (for kmeans)
            **kwargs: Additional arguments for clustering
            
        Returns:
            Cluster labels
        """
        if method == "kmeans":
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=kwargs.get('random_state', 42)
            )
        elif method == "dbscan":
            clusterer = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        self.cluster_labels = clusterer.fit_predict(embeddings)
        
        # Calculate silhouette score for kmeans
        if method == "kmeans" and len(np.unique(self.cluster_labels)) > 1:
            silhouette = silhouette_score(embeddings, self.cluster_labels)
            logger.info(f"Clustering silhouette score: {silhouette:.3f}")
        
        logger.info(f"Clustered embeddings into {len(np.unique(self.cluster_labels))} clusters")
        return self.cluster_labels
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix between embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        
        logger.info(f"Calculated similarity matrix with shape: {similarity_matrix.shape}")
        return similarity_matrix
    
    def find_similar_motions(self, query_embedding: np.ndarray, embeddings: np.ndarray, 
                           k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Find most similar motions to a query embedding.
        
        Args:
            query_embedding: Query embedding
            embeddings: All embeddings to search in
            k: Number of similar motions to return
            
        Returns:
            Tuple of (indices, similarities)
        """
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top k indices
        indices = np.argsort(similarities)[::-1][:k]
        
        return indices, similarities[indices]
    
    def visualize_embeddings_3d(self, embeddings_3d: np.ndarray, 
                               labels: Optional[np.ndarray] = None,
                               title: str = "Dance Motion Embeddings",
                               save_path: Optional[str] = None) -> go.Figure:
        """Create 3D visualization of embeddings.
        
        Args:
            embeddings_3d: 3D embeddings
            labels: Optional labels for coloring
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if labels is not None:
            # Color by labels
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=embeddings_3d[mask, 0],
                    y=embeddings_3d[mask, 1],
                    z=embeddings_3d[mask, 2],
                    mode='markers',
                    name=f'Cluster {label}',
                    marker=dict(size=5, opacity=0.7)
                ))
        else:
            # Single color
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                mode='markers',
                marker=dict(size=5, opacity=0.7)
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved 3D visualization to {save_path}")
        
        return fig
    
    def visualize_similarity_matrix(self, similarity_matrix: np.ndarray,
                                  save_path: Optional[str] = None) -> go.Figure:
        """Visualize similarity matrix as a heatmap.
        
        Args:
            similarity_matrix: Similarity matrix
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            colorscale='Viridis',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title="Motion Similarity Matrix",
            xaxis_title="Motion Index",
            yaxis_title="Motion Index",
            width=600,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved similarity matrix to {save_path}")
        
        return fig
    
    def create_motion_timeline(self, embeddings: np.ndarray, 
                             timestamps: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> go.Figure:
        """Create a timeline visualization of motion embeddings.
        
        Args:
            embeddings: Motion embeddings
            timestamps: Optional timestamps for x-axis
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        if timestamps is None:
            timestamps = np.arange(len(embeddings))
        
        # Reduce to 2D for timeline
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        fig = go.Figure()
        
        # Plot trajectory
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=embeddings_2d[:, 0],
            mode='lines+markers',
            name='Component 1',
            line=dict(width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=embeddings_2d[:, 1],
            mode='lines+markers',
            name='Component 2',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title="Motion Timeline",
            xaxis_title="Time",
            yaxis_title="Embedding Components",
            width=800,
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved motion timeline to {save_path}")
        
        return fig
    
    def analyze_motion_patterns(self, embeddings: np.ndarray, 
                              output_dir: str = "data/analysis") -> dict[str, Any]:
        """Perform comprehensive motion pattern analysis.
        
        Args:
            embeddings: Input embeddings
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary containing analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. Dimensionality reduction
        logger.info("Performing dimensionality reduction...")
        embeddings_3d = self.reduce_dimensions(embeddings)
        results['embeddings_3d'] = embeddings_3d
        
        # 2. Clustering
        logger.info("Performing clustering...")
        cluster_labels = self.cluster_embeddings(embeddings)
        results['cluster_labels'] = cluster_labels
        
        # 3. Similarity analysis
        logger.info("Calculating similarity matrix...")
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        results['similarity_matrix'] = similarity_matrix
        
        # 4. Visualizations
        logger.info("Creating visualizations...")
        
        # 3D embedding plot
        fig_3d = self.visualize_embeddings_3d(
            embeddings_3d, 
            cluster_labels,
            save_path=str(output_path / "embeddings_3d.html")
        )
        results['fig_3d'] = fig_3d
        
        # Similarity matrix
        fig_sim = self.visualize_similarity_matrix(
            similarity_matrix,
            save_path=str(output_path / "similarity_matrix.html")
        )
        results['fig_sim'] = fig_sim
        
        # Motion timeline
        fig_timeline = self.create_motion_timeline(
            embeddings,
            save_path=str(output_path / "motion_timeline.html")
        )
        results['fig_timeline'] = fig_timeline
        
        # 5. Save numerical results
        np.save(output_path / "embeddings_3d.npy", embeddings_3d)
        np.save(output_path / "cluster_labels.npy", cluster_labels)
        np.save(output_path / "similarity_matrix.npy", similarity_matrix)
        
        # Save as CSV for easy inspection
        pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z']).to_csv(
            output_path / "embeddings_3d.csv", index=False
        )
        pd.DataFrame({'cluster': cluster_labels}).to_csv(
            output_path / "cluster_labels.csv", index=False
        )
        
        logger.info(f"Analysis complete. Results saved to {output_dir}")
        
        return results
    
    def compare_motions(self, embeddings1: np.ndarray, embeddings2: np.ndarray,
                       labels1: str = "Motion 1", labels2: str = "Motion 2",
                       save_path: Optional[str] = None) -> go.Figure:
        """Compare two sets of motion embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels1: Label for first set
            labels2: Label for second set
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        # Reduce both to 3D for comparison
        combined_embeddings = np.vstack([embeddings1, embeddings2])
        combined_3d = self.reduce_dimensions(combined_embeddings)
        
        # Split back
        n1 = len(embeddings1)
        embeddings1_3d = combined_3d[:n1]
        embeddings2_3d = combined_3d[n1:]
        
        fig = go.Figure()
        
        # Plot first set
        fig.add_trace(go.Scatter3d(
            x=embeddings1_3d[:, 0],
            y=embeddings1_3d[:, 1],
            z=embeddings1_3d[:, 2],
            mode='markers',
            name=labels1,
            marker=dict(size=5, color='blue', opacity=0.7)
        ))
        
        # Plot second set
        fig.add_trace(go.Scatter3d(
            x=embeddings2_3d[:, 0],
            y=embeddings2_3d[:, 1],
            z=embeddings2_3d[:, 2],
            mode='markers',
            name=labels2,
            marker=dict(size=5, color='red', opacity=0.7)
        ))
        
        fig.update_layout(
            title=f"Motion Comparison: {labels1} vs {labels2}",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved motion comparison to {save_path}")
        
        return fig


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze motion embeddings")
    parser.add_argument("--input", type=str, required=True, help="Path to embeddings file")
    parser.add_argument("--output", type=str, default="data/analysis", help="Output directory")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne", "pca"],
                       help="Dimensionality reduction method")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    
    args = parser.parse_args()
    
    analyzer = MotionAnalyzer(method=args.method)
    
    # Load embeddings
    embeddings = analyzer.load_embeddings(args.input)
    
    # Perform analysis
    results = analyzer.analyze_motion_patterns(embeddings, args.output)
    
    print(f"Analysis complete. Results saved to {args.output}")


if __name__ == "__main__":
    main() 