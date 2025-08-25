#!/usr/bin/env python3
"""Test script for advanced OSC streaming with body-relative coordinates."""

import time
import logging
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recall.advanced_osc_streamer import create_advanced_osc_streamer
from recall.data_structures import PoseData

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_pose_data(frame_number: int, time_offset: float = 0.0) -> PoseData:
    """Create test pose data with realistic hand movements"""
    
    # Base landmarks (33 MediaPipe pose landmarks)
    landmarks = np.zeros((33, 3))
    
    # Set body landmarks (shoulders, hips, etc.)
    # Simulate different body orientations based on frame number
    if frame_number < 50:
        # Frame 0-49: Facing camera (shoulders horizontal)
        landmarks[11] = [0.4, 0.3, 0.4]  # Left shoulder
        landmarks[12] = [0.6, 0.3, 0.4]  # Right shoulder
    elif frame_number < 100:
        # Frame 50-99: Turned slightly right (shoulders at angle)
        landmarks[11] = [0.4, 0.3, 0.3]  # Left shoulder (closer)
        landmarks[12] = [0.6, 0.3, 0.5]  # Right shoulder (farther)
    else:
        # Frame 100+: Turned slightly left (shoulders at angle)
        landmarks[11] = [0.4, 0.3, 0.5]  # Left shoulder (farther)
        landmarks[12] = [0.6, 0.3, 0.3]  # Right shoulder (closer)
    # Left hip (23)
    landmarks[23] = [0.4, 0.7, 0.5]
    # Right hip (24)
    landmarks[24] = [0.6, 0.7, 0.5]
    
    # Head landmarks
    landmarks[0] = [0.5, 0.1, 0.4]  # Nose
    landmarks[7] = [0.3, 0.1, 0.4]  # Left ear
    landmarks[8] = [0.7, 0.1, 0.4]  # Right ear
    
    # Simulate hand movement over time
    t = frame_number * 0.1 + time_offset
    
    # Left hand (15) - circular motion
    left_hand_x = 0.3 + 0.2 * np.sin(t)
    left_hand_y = 0.4 + 0.1 * np.cos(t * 2)
    left_hand_z = 0.3 + 0.1 * np.sin(t * 0.5)
    landmarks[15] = [left_hand_x, left_hand_y, left_hand_z]
    
    # Right hand (16) - different motion pattern
    right_hand_x = 0.7 + 0.15 * np.cos(t * 1.5)
    right_hand_y = 0.4 + 0.15 * np.sin(t * 1.2)
    right_hand_z = 0.3 + 0.1 * np.cos(t * 0.8)
    landmarks[16] = [right_hand_x, right_hand_y, right_hand_z]
    
    # Confidence scores (all high for test)
    confidence = np.ones(33) * 0.95
    
    return PoseData(
        landmarks=landmarks,
        confidence=confidence,
        timestamp=time.time() + time_offset,
        frame_number=frame_number
    )


def test_advanced_osc_streaming():
    """Test the advanced OSC streaming system"""
    
    logger.info("🧪 Testing Advanced OSC Streaming System")
    logger.info("=" * 50)
    
    # Test configuration
    test_config = {
        "enabled": True,
        "stream_rate": 10.0,  # 10 Hz for testing
        "streams": {
            "pose_data": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 6448,
                "address": "/pose/data",
                "z_filter": {
                    "velocity_fast_rise": 0.8,
                    "velocity_slow_decay": 0.95,
                    "acceleration_fast_rise": 0.9,
                    "acceleration_slow_decay": 0.98
                }
            }
        }
    }
    
    # Create OSC streamer
    logger.info("Creating advanced OSC streamer...")
    streamer = create_advanced_osc_streamer(test_config)
    
    if not streamer:
        logger.error("❌ Failed to create OSC streamer")
        return
    
    logger.info("✅ OSC streamer created successfully")
    logger.info(f"📡 Streaming to {len(streamer.clients)} port:")
    
    for stream_name, client in streamer.clients.items():
        if client:
            stream_config = test_config["streams"][stream_name]
            logger.info(f"   {stream_name}: {stream_config['host']}:{stream_config['port']} -> {stream_config['address']}")
            logger.info(f"   Data: 21 values in single message")
    
    # Test streaming
    logger.info("\n🎭 Starting pose streaming test...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Create test pose data
            pose_data = create_test_pose_data(frame_count, time.time() - start_time)
            
            # Stream pose data
            streamer.stream_pose(pose_data)
            
            # Log every 10 frames
            if frame_count % 10 == 0:
                logger.info(f"Frame {frame_count}: Streamed pose data")
                
                # Show some calculated values
                body_scale = streamer._calculate_body_scale(pose_data)
                chest_center = streamer._get_chest_center(pose_data)
                body_yaw, body_pitch = streamer._calculate_body_orientation(pose_data)
                
                logger.info(f"   Body scale: {body_scale:.3f}")
                logger.info(f"   Chest center: [{chest_center[0]:.3f}, {chest_center[1]:.3f}, {chest_center[2]:.3f}]")
                logger.info(f"   Body orientation: Yaw={body_yaw:.1f}°, Pitch={body_pitch:.1f}°")
            
            frame_count += 1
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping test...")
    
    finally:
        # Cleanup
        streamer.close()
        logger.info("✅ OSC streamer closed")


if __name__ == "__main__":
    test_advanced_osc_streaming()
