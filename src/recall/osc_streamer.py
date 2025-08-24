"""OSC streaming for pose landmarks, specifically focused on hand joints."""

import time
import logging
from typing import Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

try:
    from pythonosc import udp_client
    from pythonosc.osc_message_builder import OscMessageBuilder
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    logging.warning("python-osc not available. Install with: pip install python-osc")

from .data_structures import PoseData

logger = logging.getLogger(__name__)


@dataclass
class OSCConfig:
    """Configuration for OSC streaming"""
    host: str = "127.0.0.1"
    port: int = 8000
    enabled: bool = True
    stream_rate: float = 30.0  # Hz
    hand_joints_only: bool = True
    
    # OSC address patterns for different joints
    left_hand_prefix: str = "/pose/left_hand"
    right_hand_prefix: str = "/pose/right_hand"
    
    # Include confidence scores
    include_confidence: bool = True
    
    # Include 3D coordinates (if available)
    include_3d: bool = True


class HandOSCStreamer:
    """Streams hand joint data via OSC protocol"""
    
    def __init__(self, config: OSCConfig):
        self.config = config
        self.client = None
        self.last_stream_time = 0
        self.stream_interval = 1.0 / config.stream_rate if config.stream_rate > 0 else 0
        
        # MediaPipe hand landmark indices
        self.hand_landmarks = {
            'left_hand': {
                'wrist': 15,
                'thumb_tip': 4,
                'thumb_ip': 3,
                'thumb_mcp': 2,
                'thumb_cmc': 1,
                'index_tip': 8,
                'index_dip': 7,
                'index_pip': 6,
                'index_mcp': 5,
                'middle_tip': 12,
                'middle_dip': 11,
                'middle_pip': 10,
                'middle_mcp': 9,
                'ring_tip': 16,
                'ring_dip': 15,
                'ring_pip': 14,
                'ring_mcp': 13,
                'pinky_tip': 20,
                'pinky_dip': 19,
                'pinky_pip': 18,
                'pinky_mcp': 17
            },
            'right_hand': {
                'wrist': 16,
                'thumb_tip': 4,
                'thumb_ip': 3,
                'thumb_mcp': 2,
                'thumb_cmc': 1,
                'index_tip': 8,
                'index_dip': 7,
                'index_pip': 6,
                'index_mcp': 5,
                'middle_tip': 12,
                'middle_dip': 11,
                'middle_pip': 10,
                'middle_mcp': 9,
                'ring_tip': 16,
                'ring_dip': 15,
                'ring_pip': 14,
                'ring_mcp': 13,
                'pinky_tip': 20,
                'pinky_dip': 19,
                'pinky_pip': 18,
                'pinky_mcp': 17
            }
        }
        
        if config.enabled and OSC_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OSC UDP client"""
        try:
            self.client = udp_client.SimpleUDPClient(self.config.host, self.config.port)
            logger.info(f"✅ OSC client initialized: {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Failed to initialize OSC client: {e}")
            self.client = None
    
    def _should_stream(self) -> bool:
        """Check if enough time has passed to stream again"""
        if not self.config.enabled or self.client is None:
            return False
        
        current_time = time.time()
        if current_time - self.last_stream_time >= self.stream_interval:
            self.last_stream_time = current_time
            return True
        return False
    
    def _send_hand_joint(self, hand: str, joint_name: str, coordinates: np.ndarray, confidence: float):
        """Send a single hand joint via OSC"""
        if not self.client:
            return
        
        # Build OSC message
        builder = OscMessageBuilder()
        
        # Set address pattern
        prefix = self.config.left_hand_prefix if hand == 'left_hand' else self.config.right_hand_prefix
        address = f"{prefix}/{joint_name}"
        builder.address = address
        
        # Add coordinates
        if self.config.include_3d and len(coordinates) >= 3:
            builder.add_arg(coordinates[0])  # x
            builder.add_arg(coordinates[1])  # y
            builder.add_arg(coordinates[2])  # z
        else:
            builder.add_arg(coordinates[0])  # x
            builder.add_arg(coordinates[1])  # y
        
        # Add confidence if enabled
        if self.config.include_confidence:
            builder.add_arg(confidence)
        
        # Send message
        try:
            msg = builder.build()
            self.client.send(msg)
        except Exception as e:
            logger.error(f"Failed to send OSC message for {address}: {e}")
    
    def stream_hands(self, pose_data: PoseData):
        """Stream hand joint data from pose landmarks"""
        if not self._should_stream():
            return
        
        if pose_data is None or pose_data.landmarks is None:
            return
        
        try:
            # Stream left hand joints
            for joint_name, landmark_idx in self.hand_landmarks['left_hand'].items():
                if landmark_idx < len(pose_data.landmarks):
                    coords = pose_data.landmarks[landmark_idx]
                    confidence = pose_data.confidence[landmark_idx] if landmark_idx < len(pose_data.confidence) else 1.0
                    self._send_hand_joint('left_hand', joint_name, coords, confidence)
            
            # Stream right hand joints
            for joint_name, landmark_idx in self.hand_landmarks['right_hand'].items():
                if landmark_idx < len(pose_data.landmarks):
                    coords = pose_data.landmarks[landmark_idx]
                    confidence = pose_data.confidence[landmark_idx] if landmark_idx < len(pose_data.confidence) else 1.0
                    self._send_hand_joint('right_hand', joint_name, coords, confidence)
            
            logger.debug(f"Streamed hand joints via OSC to {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Error streaming hands via OSC: {e}")
    
    def close(self):
        """Clean up OSC client"""
        if self.client:
            self.client = None
            logger.info("OSC client closed")


class PoseOSCStreamer:
    """Main OSC streaming interface for pose data"""
    
    def __init__(self, config: OSCConfig):
        self.config = config
        self.hand_streamer = HandOSCStreamer(config)
    
    def stream_pose(self, pose_data: PoseData):
        """Stream pose data via OSC"""
        if self.config.hand_joints_only:
            self.hand_streamer.stream_hands(pose_data)
        # Future: Add other body parts streaming here
    
    def close(self):
        """Clean up all streamers"""
        self.hand_streamer.close()


# Convenience function to create OSC streamer
def create_osc_streamer(host: str = "127.0.0.1", port: int = 8000, 
                       stream_rate: float = 30.0, enabled: bool = True) -> PoseOSCStreamer:
    """Create an OSC streamer with default configuration"""
    config = OSCConfig(
        host=host,
        port=port,
        stream_rate=stream_rate,
        enabled=enabled
    )
    return PoseOSCStreamer(config)
