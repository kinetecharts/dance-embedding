"""Dance Motion Embedding System.

A comprehensive system for converting dance videos into pose time series data
using MediaPipe's AI pose estimation, generating vector embeddings for poses
and movement segments, and enabling motion analysis in high-dimensional space.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .pose_extraction import PoseExtractor
from .embedding_generator import EmbeddingGenerator
from .motion_analyzer import MotionAnalyzer

__all__ = [
    "PoseExtractor",
    "EmbeddingGenerator", 
    "MotionAnalyzer",
] 