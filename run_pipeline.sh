#!/bin/bash
set -e

echo "=== Running Pose Extraction for all videos ==="
python -m pose_extraction.main --input-dir data/video

echo "=== Running Dimension Reduction for all pose CSVs ==="
python -m dimension_reduction.main

echo "=== Pipeline complete! ===" 