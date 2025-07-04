"""Embedding generation module for pose data."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configure logging
logger = logging.getLogger(__name__)

# Constants
POSE_KEYPOINTS: Final = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


class PoseDataset(Dataset):
    """Dataset for pose data."""
    
    def __init__(self, pose_data: np.ndarray, sequence_length: int = 150):
        """Initialize the dataset.
        
        Args:
            pose_data: Array of pose data (frames, features)
            sequence_length: Length of sequences to extract
        """
        self.pose_data = pose_data
        self.sequence_length = sequence_length
        
    def __len__(self) -> int:
        """Return the number of sequences."""
        return max(0, len(self.pose_data) - self.sequence_length + 1)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sequence of pose data."""
        sequence = self.pose_data[idx:idx + self.sequence_length]
        
        # Pad if necessary
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        
        return torch.FloatTensor(sequence)


class PoseTransformer(nn.Module):
    """Transformer-based pose embedding model."""
    
    def __init__(self, input_dim: int = 66, hidden_dim: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, embedding_dim: int = 512):
        """Initialize the transformer model.
        
        Args:
            input_dim: Input dimension (number of pose features)
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, pose_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.
        
        Args:
            pose_sequence: Input pose sequence (batch_size, seq_len, input_dim)
            
        Returns:
            Pose embedding (batch_size, embedding_dim)
        """
        x = self.input_projection(pose_sequence)
        x = self.positional_encoding(x)
        
        encoded = self.transformer(x)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        
        return self.output_projection(pooled)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class SegmentLSTM(nn.Module):
    """LSTM-based segment embedding model."""
    
    def __init__(self, input_dim: int = 66, hidden_dim: int = 512, num_layers: int = 3, 
                 embedding_dim: int = 1024, dropout: float = 0.2):
        """Initialize the LSTM model.
        
        Args:
            input_dim: Input dimension (number of pose features)
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            embedding_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(hidden_dim * 2, embedding_dim)
        
    def forward(self, pose_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM.
        
        Args:
            pose_sequence: Input pose sequence (batch_size, seq_len, input_dim)
            
        Returns:
            Segment embedding (batch_size, embedding_dim)
        """
        lstm_out, _ = self.lstm(pose_sequence)
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attended.mean(dim=1)
        
        return self.output_projection(pooled)


class EmbeddingGenerator:
    """Generates embeddings from pose data."""
    
    def __init__(self, model_type: str = "transformer", device: str = "cpu"):
        """Initialize the embedding generator.
        
        Args:
            model_type: Type of model ("transformer" or "lstm")
            device: Device to run the model on
        """
        self.model_type = model_type
        self.device = torch.device(device)
        
        # Initialize model based on type
        if model_type == "transformer":
            self.model = PoseTransformer().to(self.device)
        elif model_type == "lstm":
            self.model = SegmentLSTM().to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.eval()
        
    def preprocess_pose_data(self, pose_data: pd.DataFrame) -> np.ndarray:
        """Preprocess pose data for embedding generation.
        
        Args:
            pose_data: DataFrame containing pose data
            
        Returns:
            Preprocessed pose data as numpy array
        """
        # Extract 2D coordinates
        features = []
        for keypoint in POSE_KEYPOINTS:
            x_col = f"{keypoint}_x"
            y_col = f"{keypoint}_y"
            
            if x_col in pose_data.columns and y_col in pose_data.columns:
                x = pose_data[x_col].fillna(0).values
                y = pose_data[y_col].fillna(0).values
                features.extend([x, y])
        
        # Stack features
        pose_array = np.column_stack(features)
        
        # Normalize pose data
        pose_array = self._normalize_pose_data(pose_array)
        
        return pose_array
    
    def _normalize_pose_data(self, pose_data: np.ndarray) -> np.ndarray:
        """Normalize pose data.
        
        Args:
            pose_data: Raw pose data
            
        Returns:
            Normalized pose data
        """
        # Center around hip center (keypoints 23 and 24)
        hip_center_x = (pose_data[:, 46] + pose_data[:, 48]) / 2  # left_hip_x + right_hip_x
        hip_center_y = (pose_data[:, 47] + pose_data[:, 49]) / 2  # left_hip_y + right_hip_y
        
        # Center all keypoints
        pose_data[:, 0::2] -= hip_center_x[:, np.newaxis]  # X coordinates
        pose_data[:, 1::2] -= hip_center_y[:, np.newaxis]  # Y coordinates
        
        # Scale by shoulder-to-hip distance
        shoulder_hip_distance = np.sqrt(
            (pose_data[:, 22] - pose_data[:, 46])**2 +  # left_shoulder_x - left_hip_x
            (pose_data[:, 23] - pose_data[:, 47])**2    # left_shoulder_y - left_hip_y
        )
        
        # Avoid division by zero
        shoulder_hip_distance = np.maximum(shoulder_hip_distance, 1e-6)
        
        # Scale all coordinates
        pose_data[:, 0::2] /= shoulder_hip_distance[:, np.newaxis]  # X coordinates
        pose_data[:, 1::2] /= shoulder_hip_distance[:, np.newaxis]  # Y coordinates
        
        return pose_data
    
    def generate_pose_embeddings(self, pose_data: pd.DataFrame, 
                                sequence_length: int = 150) -> np.ndarray:
        """Generate embeddings for individual poses.
        
        Args:
            pose_data: DataFrame containing pose data
            sequence_length: Length of sequences to process
            
        Returns:
            Array of pose embeddings
        """
        # Preprocess pose data
        pose_array = self.preprocess_pose_data(pose_data)
        
        # Create dataset
        dataset = PoseDataset(pose_array, sequence_length)
        
        if len(dataset) == 0:
            logger.warning("No valid sequences found in pose data")
            return np.array([])
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                batch_embeddings = self.model(batch)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def generate_segment_embeddings(self, pose_data: pd.DataFrame, 
                                   segment_length: int = 150) -> np.ndarray:
        """Generate embeddings for motion segments.
        
        Args:
            pose_data: DataFrame containing pose data
            segment_length: Length of segments in frames (5 seconds at 30fps = 150)
            
        Returns:
            Array of segment embeddings
        """
        return self.generate_pose_embeddings(pose_data, segment_length)
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str) -> None:
        """Save embeddings to file.
        
        Args:
            embeddings: Array of embeddings
            output_path: Path to save embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy array
        np.save(output_path.with_suffix('.npy'), embeddings)
        
        # Also save as CSV for easy inspection
        df = pd.DataFrame(embeddings)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        logger.info(f"Saved embeddings to {output_path}")
    
    def load_embeddings(self, input_path: str) -> np.ndarray:
        """Load embeddings from file.
        
        Args:
            input_path: Path to embeddings file
            
        Returns:
            Array of embeddings
        """
        input_path = Path(input_path)
        
        if input_path.suffix == '.npy':
            embeddings = np.load(input_path)
        elif input_path.suffix == '.csv':
            embeddings = pd.read_csv(input_path).values
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        return embeddings
    
    def process_csv_file(self, csv_path: str, output_path: str | None = None, 
                        embedding_type: str = "segment") -> np.ndarray:
        """Process a CSV file and generate embeddings.
        
        Args:
            csv_path: Path to CSV file containing pose data
            output_path: Path to save embeddings. If None, auto-generates.
            embedding_type: Type of embedding ("pose" or "segment")
            
        Returns:
            Array of embeddings
        """
        # Load pose data
        pose_data = pd.read_csv(csv_path)
        
        # Generate embeddings
        if embedding_type == "pose":
            embeddings = self.generate_pose_embeddings(pose_data)
        elif embedding_type == "segment":
            embeddings = self.generate_segment_embeddings(pose_data)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # Save embeddings
        if output_path is None:
            csv_name = Path(csv_path).stem
            output_path = f"data/embeddings/{csv_name}_{embedding_type}.npy"
        
        self.save_embeddings(embeddings, output_path)
        
        return embeddings
    
    def process_directory(self, input_dir: str = "data/poses", 
                         output_dir: str = "data/embeddings",
                         embedding_type: str = "segment") -> None:
        """Process all CSV files in a directory.
        
        Args:
            input_dir: Directory containing CSV files
            output_dir: Directory to save embeddings
            embedding_type: Type of embedding to generate
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory {input_dir} does not exist")
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all CSV files
        csv_files = list(input_path.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {input_dir}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}")
            try:
                output_file = output_path / f"{csv_file.stem}_{embedding_type}.npy"
                self.process_csv_file(str(csv_file), str(output_file), embedding_type)
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                continue
        
        logger.info("Finished processing all CSV files")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings from pose data")
    parser.add_argument("--input", type=str, help="Path to CSV file or directory")
    parser.add_argument("--output", type=str, help="Output path for embeddings")
    parser.add_argument("--type", type=str, default="segment", choices=["pose", "segment"],
                       help="Type of embedding to generate")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm"],
                       help="Model type to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    generator = EmbeddingGenerator(model_type=args.model, device=args.device)
    
    if Path(args.input).is_file():
        # Process single file
        generator.process_csv_file(args.input, args.output, args.type)
    else:
        # Process directory
        generator.process_directory(args.input, args.output, args.type)


if __name__ == "__main__":
    main() 