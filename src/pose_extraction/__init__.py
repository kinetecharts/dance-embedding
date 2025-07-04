"""Pose Extraction System.

A focused system for extracting pose landmarks from dance videos
using MediaPipe's AI pose estimation and exporting to CSV format.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .pose_extraction import PoseExtractor
from .main import PoseExtractionPipeline

__all__ = [
    "PoseExtractor",
    "PoseExtractionPipeline",
] 