#!/usr/bin/env python3
"""Test script for OSC streaming of hand joints from pose data."""

import time
import logging
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recall.osc_streamer import create_osc_streamer, OSCConfig
from recall.data_structures import PoseData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_pose_data():
    """Create test pose data with hand movements"""
    # Create 33 landmarks (MediaPipe pose format)
    landmarks = np.zeros((33, 3))
    confidence = np.ones(33)
    
    # Set some hand landmarks to simulate movement
    # Left hand (landmark 15)
    landmarks[15] = [0.1, 0.8, 0.2]  # Left wrist
    landmarks[4] = [0.05, 0.75, 0.25]   # Left thumb tip
    landmarks[8] = [0.15, 0.7, 0.2]     # Left index tip
    landmarks[12] = [0.2, 0.65, 0.2]    # Left middle tip
    landmarks[16] = [0.25, 0.7, 0.2]    # Left ring tip
    landmarks[20] = [0.3, 0.75, 0.2]    # Left pinky tip
    
    # Right hand (landmark 16)
    landmarks[16] = [0.9, 0.8, 0.2]  # Right wrist
    landmarks[4] = [0.95, 0.75, 0.25]  # Right thumb tip
    landmarks[8] = [0.85, 0.7, 0.2]    # Right index tip
    landmarks[12] = [0.8, 0.65, 0.2]   # Right middle tip
    landmarks[16] = [0.75, 0.7, 0.2]   # Right ring tip
    landmarks[20] = [0.7, 0.75, 0.2]   # Right pinky tip
    
    return PoseData(
        landmarks=landmarks,
        confidence=confidence,
        timestamp=time.time(),
        frame_number=0
    )


def test_osc_streaming():
    """Test OSC streaming with simulated hand movements"""
    logger.info("Testing OSC streaming of hand joints...")
    
    # Create OSC streamer
    config = OSCConfig(
        host="127.0.0.1",
        port=6448,
        enabled=True,
        stream_rate=10.0,  # 10 Hz for testing
        hand_joints_only=True,
        include_confidence=True,
        include_3d=True
    )
    
    streamer = create_osc_streamer(
        host=config.host,
        port=config.port,
        stream_rate=config.stream_rate,
        enabled=config.enabled
    )
    
    if not streamer:
        logger.error("Failed to create OSC streamer")
        return
    
    logger.info(f"✅ OSC streamer created: {config.host}:{config.port}")
    logger.info("Streaming hand joints at 10 Hz...")
    logger.info("Use an OSC receiver (like TouchOSC, Max/MSP, etc.) to see the data")
    logger.info("Press Ctrl+C to stop")
    
    try:
        frame_count = 0
        while True:
            # Create test pose data with some movement
            pose_data = create_test_pose_data()
            
            # Add some movement to simulate real-time tracking
            t = time.time()
            pose_data.landmarks[15][0] = 0.1 + 0.1 * np.sin(t * 2)  # Left wrist x
            pose_data.landmarks[15][1] = 0.8 + 0.05 * np.cos(t * 3)  # Left wrist y
            pose_data.landmarks[16][0] = 0.9 - 0.1 * np.sin(t * 2)   # Right wrist x
            pose_data.landmarks[16][1] = 0.8 + 0.05 * np.cos(t * 3)  # Right wrist y
            
            # Update timestamp and frame number
            pose_data.timestamp = t
            pose_data.frame_number = frame_count
            
            # Stream the pose data
            streamer.stream_pose(pose_data)
            
            frame_count += 1
            
            # Log every 10 frames
            if frame_count % 10 == 0:
                logger.info(f"Streamed frame {frame_count}, left wrist: {pose_data.landmarks[15][:2]}, right wrist: {pose_data.landmarks[16][:2]}")
            
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        logger.info("Stopping OSC streaming...")
    finally:
        streamer.close()
        logger.info("OSC streamer closed")


if __name__ == "__main__":
    test_osc_streaming()
