"""Dimension reduction methods for pose data."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)


class ReductionMethods:
    """Implements various dimension reduction methods for pose data."""
    
    def __init__(self):
        """Initialize the reduction methods."""
        self.fitted_models = {}
    
    def preprocess_pose_data(self, pose_data: pd.DataFrame, use_3d: bool = False, 
                           confidence_threshold: float = 0.5) -> np.ndarray:
        """Preprocess pose data for dimension reduction.
        
        Args:
            pose_data: DataFrame with pose keypoint data
            use_3d: Whether to use 3D coordinates (z-axis)
            confidence_threshold: Minimum confidence score for keypoints
            
        Returns:
            Preprocessed pose data as numpy array
        """
        # Extract coordinate columns
        coord_columns = []
        for col in pose_data.columns:
            if col.endswith('_x') or col.endswith('_y'):
                coord_columns.append(col)
            elif use_3d and col.endswith('_z'):
                coord_columns.append(col)
        
        # Extract confidence columns
        confidence_columns = [col for col in pose_data.columns if col.endswith('_confidence')]
        
        # Create feature matrix
        features = pose_data[coord_columns].values
        
        # Apply confidence filtering if available
        if confidence_columns:
            confidence_scores = pose_data[confidence_columns].values
            
            # Create expanded confidence mask to match feature dimensions
            # Each keypoint has 2 (x,y) or 3 (x,y,z) coordinates
            coords_per_keypoint = 3 if use_3d else 2
            expanded_confidence_mask = np.repeat(confidence_scores < confidence_threshold, coords_per_keypoint, axis=1)
            
            # Replace low-confidence coordinates with NaN
            features[expanded_confidence_mask] = np.nan
        
        # Handle missing values (NaN) by interpolation
        features = self._interpolate_missing_values(features)
        
        # Normalize features
        features = self._normalize_features(features)
        
        return features
    
    def _interpolate_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Interpolate missing values in pose data.
        
        Args:
            data: Pose data with potential NaN values
            
        Returns:
            Data with interpolated values
        """
        # Forward fill, then backward fill
        data_filled = pd.DataFrame(data).ffill().bfill().values
        
        # If still has NaN, fill with zeros
        data_filled = np.nan_to_num(data_filled, nan=0.0)
        
        return data_filled
    
    def _normalize_features(self, data: np.ndarray) -> np.ndarray:
        """Normalize pose features.
        
        Args:
            data: Raw pose data
            
        Returns:
            Normalized pose data
        """
        # Center around mean
        data_centered = data - np.mean(data, axis=0)
        
        # Scale by standard deviation
        std = np.std(data_centered, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        data_normalized = data_centered / std
        
        return data_normalized
    
    def pca(self, data: np.ndarray, n_components: int = 2, 
            **kwargs) -> Tuple[np.ndarray, PCA]:
        """Apply Principal Component Analysis.
        
        Args:
            data: Input data
            n_components: Number of components to keep
            **kwargs: Additional PCA parameters
            
        Returns:
            Tuple of (reduced_data, pca_model)
        """
        pca = PCA(n_components=n_components, random_state=kwargs.get('random_state', 42))
        reduced_data = pca.fit_transform(data)
        
        logger.info(f"PCA: Explained variance ratio: {pca.explained_variance_ratio_}")
        logger.info(f"PCA: Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
        
        self.fitted_models['pca'] = pca
        return reduced_data, pca
    
    def tsne(self, data: np.ndarray, n_components: int = 2, 
             **kwargs) -> Tuple[np.ndarray, TSNE]:
        """Apply t-SNE dimensionality reduction.
        
        Args:
            data: Input data
            n_components: Number of components to keep
            **kwargs: Additional t-SNE parameters
            
        Returns:
            Tuple of (reduced_data, tsne_model)
        """
        tsne_params = {
            'n_components': n_components,
            'perplexity': kwargs.get('perplexity', 30),
            'learning_rate': kwargs.get('learning_rate', 200),
            'random_state': kwargs.get('random_state', 42),
            'n_iter': kwargs.get('n_iter', 1000)
        }
        
        tsne = TSNE(**tsne_params)
        reduced_data = tsne.fit_transform(data)
        
        logger.info(f"t-SNE: Final KL divergence: {tsne.kl_divergence_:.3f}")
        
        self.fitted_models['tsne'] = tsne
        return reduced_data, tsne
    
    def umap_reduction(self, data: np.ndarray, n_components: int = 2, 
                      **kwargs) -> Tuple[np.ndarray, umap.UMAP]:
        """Apply UMAP dimensionality reduction.
        
        Args:
            data: Input data
            n_components: Number of components to keep
            **kwargs: Additional UMAP parameters
            
        Returns:
            Tuple of (reduced_data, umap_model)
        """
        umap_params = {
            'n_components': n_components,
            'n_neighbors': kwargs.get('n_neighbors', 15),
            'min_dist': kwargs.get('min_dist', 0.1),
            'random_state': kwargs.get('random_state', 42),
            'metric': kwargs.get('metric', 'euclidean')
        }
        
        umap_model = umap.UMAP(**umap_params)
        reduced_data = umap_model.fit_transform(data)
        
        logger.info(f"UMAP: Number of components: {n_components}")
        
        self.fitted_models['umap'] = umap_model
        return reduced_data, umap_model
    
    def reduce_dimensions(self, data: np.ndarray, method: str = 'umap', 
                         n_components: int = 2, **kwargs) -> Tuple[np.ndarray, Any]:
        """Apply dimension reduction using the specified method.
        
        Args:
            data: Input data
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components to keep
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (reduced_data, model)
        """
        method = method.lower()
        
        if method == 'pca':
            return self.pca(data, n_components, **kwargs)
        elif method == 'tsne':
            return self.tsne(data, n_components, **kwargs)
        elif method == 'umap':
            return self.umap_reduction(data, n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {method}. "
                           f"Supported methods: pca, tsne, umap")
    
    def get_method_parameters(self, method: str) -> Dict[str, Any]:
        """Get default parameters for a reduction method.
        
        Args:
            method: Reduction method name
            
        Returns:
            Dictionary of default parameters
        """
        if method == 'pca':
            return {
                'n_components': 2,
                'random_state': 42
            }
        elif method == 'tsne':
            return {
                'n_components': 2,
                'perplexity': 30,
                'learning_rate': 200,
                'random_state': 42,
                'n_iter': 1000
            }
        elif method == 'umap':
            return {
                'n_components': 2,
                'n_neighbors': 15,
                'min_dist': 0.1,
                'random_state': 42,
                'metric': 'euclidean'
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save_model(self, model: Any, filepath: str) -> None:
        """Save a fitted model to disk.
        
        Args:
            model: Fitted model to save
            filepath: Path to save the model
        """
        import joblib
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        import joblib
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model 