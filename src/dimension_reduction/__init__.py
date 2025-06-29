"""Dimension Reduction for Dance Motion Analysis.

A module for visualizing dance pose data in reduced-dimensional spaces
with interactive video synchronization.
"""

__version__ = "0.1.0"
__author__ = "Dance Motion Embedding Team"

from .visualizer import DimensionReductionVisualizer
from .reduction_methods import ReductionMethods

__all__ = [
    "DimensionReductionVisualizer",
    "ReductionMethods",
] 