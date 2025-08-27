"""Advanced OSC streaming for pose landmarks with body-relative coordinates and multiple streams."""

import time
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass
import math

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
class StreamConfig:
    """Configuration for a single OSC stream"""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    address: str = "/pose/stream"
    include_confidence: bool = False
    z_filter: Optional[Dict[str, float]] = None


@dataclass
class OSCStreamsConfig:
    """Configuration for all OSC streams"""
    enabled: bool = False
    stream_rate: float = 30.0
    streams: Dict[str, StreamConfig] = None
    
    def __post_init__(self):
        if self.streams is None:
            self.streams = {}


class ZFilter:
    """Z-transform filter for smoothing signals with fast rise, slow decay"""
    
    def __init__(self, fast_rise: float = 0.8, slow_decay: float = 0.95):
        self.fast_rise = fast_rise
        self.slow_decay = slow_decay
        self.filtered_value = 0.0
        self.last_input = 0.0
    
    def update(self, input_value: float) -> float:
        """Update filter with new input value"""
        if input_value > self.filtered_value:
            # Fast rise when input increases
            self.filtered_value = (self.fast_rise * input_value + 
                                 (1 - self.fast_rise) * self.filtered_value)
        else:
            # Slow decay when input decreases
            self.filtered_value = (self.slow_decay * self.filtered_value + 
                                 (1 - self.slow_decay) * input_value)
        
        self.last_input = input_value
        return self.filtered_value


class AdvancedOSCStreamer:
    """Advanced OSC streamer with body-relative coordinates and multiple streams"""
    
    def __init__(self, config: OSCStreamsConfig):
        self.config = config
        self.clients: Dict[str, Optional[udp_client.UDPClient]] = {}
        self.z_filters: Dict[str, ZFilter] = {}
        self.last_stream_time = 0
        self.stream_interval = 1.0 / config.stream_rate if config.stream_rate > 0 else 0
        
        # Previous pose data for velocity/acceleration calculations
        self.previous_hand_positions = {'left': None, 'right': None}
        self.previous_hand_velocities = {'left': None, 'right': None}
        self.previous_foot_positions = {'left_foot': None, 'right_foot': None}
        self.previous_foot_positions_2 = {'left_foot': None, 'right_foot': None}  # For acceleration calculation
        
        # Initialize clients and filters
        self._initialize_clients()
        self._initialize_filters()
    
    def _initialize_clients(self):
        """Initialize OSC clients for each enabled stream"""
        if not OSC_AVAILABLE:
            logger.warning("OSC not available, streamer will not function")
            return
        
        for stream_name, stream_config in self.config.streams.items():
            if stream_config.enabled:
                try:
                    client = udp_client.UDPClient(stream_config.host, stream_config.port)
                    self.clients[stream_name] = client
                    logger.info(f"✅ OSC client initialized for {stream_name} at {stream_config.host}:{stream_config.port}")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize OSC client for {stream_name}: {e}")
                    self.clients[stream_name] = None
            else:
                self.clients[stream_name] = None
    
    def _initialize_filters(self):
        """Initialize Z-filters for movement and acceleration streams"""
        for stream_name, stream_config in self.config.streams.items():
            if (stream_config.enabled and stream_config.z_filter and 
                stream_name == 'pose_data'):
                z_config = stream_config.z_filter
                # Create a single Z-filter for the pose_data stream
                self.z_filters[stream_name] = ZFilter(
                    fast_rise=z_config.get('velocity_fast_rise', 0.8),
                    slow_decay=z_config.get('velocity_slow_decay', 0.95)
                )
                logger.info(f"✅ Z-filter initialized for {stream_name}")
    
    def _calculate_body_scale(self, pose_data: PoseData) -> float:
        """Calculate body scale based on torso length (shoulder to hip distance)"""
        if not pose_data.is_3d:
            return 1.0
        
        # Get shoulder and hip landmarks
        left_shoulder = pose_data.landmarks[11]  # Left shoulder
        right_shoulder = pose_data.landmarks[12]  # Right shoulder
        left_hip = pose_data.landmarks[23]  # Left hip
        right_hip = pose_data.landmarks[24]  # Right hip
        
        # Calculate shoulder and hip centers
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Calculate torso length
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        # Return normalized scale (1.0 = standard torso length)
        return torso_length
    
    def _get_chest_center(self, pose_data: PoseData) -> np.ndarray:
        """Get chest center point (midpoint between shoulders and hips)"""
        if not pose_data.is_3d:
            return np.array([0.5, 0.5, 0.5])  # Default center
        
        left_shoulder = pose_data.landmarks[11]
        right_shoulder = pose_data.landmarks[12]
        left_hip = pose_data.landmarks[23]
        right_hip = pose_data.landmarks[24]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Chest is midpoint between shoulders and hips
        chest_center = (shoulder_center + hip_center) / 2
        return chest_center
    
    def _calculate_body_orientation(self, pose_data: PoseData) -> Tuple[float, float]:
        """Calculate body orientation (yaw and pitch relative to camera)"""
        if not pose_data.is_3d:
            return (0.0, 0.0)
        
        # Use shoulders to determine body orientation
        left_shoulder = pose_data.landmarks[11]
        right_shoulder = pose_data.landmarks[12]
        
        # Calculate shoulder vector (right to left)
        shoulder_vector = right_shoulder - left_shoulder
        
        # For yaw: when person faces camera, shoulders should be horizontal
        # We want 0° when facing camera, positive when turning right, negative when turning left
        # Use the X component (left-right) and Z component (forward-backward)
        yaw = math.atan2(shoulder_vector[0], shoulder_vector[2])
        
        # Rotate coordinate system 90° so front = 0°
        yaw += math.pi/2
        
        # For pitch: when person faces camera, shoulders should be level
        # We want 0° when level, positive when leaning forward, negative when leaning back
        # Use the Y component (up-down) and Z component (forward-backward)
        pitch = math.atan2(shoulder_vector[1], abs(shoulder_vector[2]))
        
        # Convert to degrees
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        
        # Normalize yaw to -180 to 180 range
        while yaw_deg > 180:
            yaw_deg -= 360
        while yaw_deg < -180:
            yaw_deg += 360
        
        return (yaw_deg, pitch_deg)
    
    def _calculate_head_rotation(self, pose_data: PoseData, body_yaw: float) -> Tuple[float, float]:
        """Calculate head rotation relative to body orientation"""
        if not pose_data.is_3d:
            return (0.0, 0.0)
        
        # Use nose and ears to determine head orientation
        nose = pose_data.landmarks[0]
        left_ear = pose_data.landmarks[7]
        right_ear = pose_data.landmarks[8]
        
        # Calculate head center and direction
        head_center = (left_ear + right_ear) / 2
        head_direction = nose - head_center
        
        # Yaw relative to body (head turning left/right)
        # When head is aligned with body, this should be 0°
        head_yaw = math.atan2(head_direction[0], head_direction[2])
        head_yaw_deg = math.degrees(head_yaw) - body_yaw
        
        # Rotate head coordinate system 180° so front = 0°
        head_yaw_deg += 180
        
        # Pitch (head nodding up/down)
        # When head is level with body, this should be 0°
        # Use abs(head_direction[2]) to avoid division by zero issues
        head_pitch = math.atan2(head_direction[1], abs(head_direction[2]))
        head_pitch_deg = math.degrees(head_pitch)
        
        # Normalize yaw to -180 to 180
        while head_yaw_deg > 180:
            head_yaw_deg -= 360
        while head_yaw_deg < -180:
            head_yaw_deg += 360
        
        return (head_yaw_deg, head_pitch_deg)
    
    def _calculate_hand_movement(self, pose_data: PoseData) -> Dict[str, float]:
        """Calculate hand movement magnitude using Z-filter"""
        if not pose_data.is_3d:
            return {'left': 0.0, 'right': 0.0}
        
        left_wrist = pose_data.landmarks[15]
        right_wrist = pose_data.landmarks[16]
        
        movements = {}
        
        for hand, wrist_pos in [('left', left_wrist), ('right', right_wrist)]:
            if self.previous_hand_positions[hand] is not None:
                # Calculate distance moved
                distance = np.linalg.norm(wrist_pos - self.previous_hand_positions[hand])
                movements[hand] = distance
            else:
                movements[hand] = 0.0
            
            self.previous_hand_positions[hand] = wrist_pos.copy()
        
        return movements
    
    def _calculate_hand_acceleration(self, pose_data: PoseData) -> Dict[str, float]:
        """Calculate hand acceleration using Z-filter"""
        if not pose_data.is_3d:
            return {'left': 0.0, 'right': 0.0}
        
        left_wrist = pose_data.landmarks[15]
        right_wrist = pose_data.landmarks[16]
        
        accelerations = {}
        
        for hand, wrist_pos in [('left', left_wrist), ('right', right_wrist)]:
            if self.previous_hand_positions[hand] is not None:
                # Calculate velocity
                velocity = wrist_pos - self.previous_hand_positions[hand]
                
                if self.previous_hand_velocities[hand] is not None:
                    # Calculate acceleration (change in velocity)
                    acceleration = np.linalg.norm(velocity - self.previous_hand_velocities[hand])
                    accelerations[hand] = acceleration
                else:
                    accelerations[hand] = 0.0
                
                self.previous_hand_velocities[hand] = velocity.copy()
            else:
                accelerations[hand] = 0.0
            
            self.previous_hand_positions[hand] = wrist_pos.copy()
        
        return accelerations
    
    def _send_osc_message(self, stream_name: str, address: str, *args):
        """Send OSC message to specified stream"""
        if stream_name not in self.clients or self.clients[stream_name] is None:
            logger.debug(f"No OSC client available for {stream_name}")
            return
        
        try:
            builder = OscMessageBuilder()
            builder.address = address
            
            for arg in args:
                if isinstance(arg, (int, float)):
                    builder.add_arg(arg)
                else:
                    builder.add_arg(float(arg))
            
            msg = builder.build()
            self.clients[stream_name].send(msg)
            logger.debug(f"✅ OSC message sent to {stream_name}: {address} with {len(args)} values")
            
        except Exception as e:
            logger.error(f"Failed to send OSC message to {stream_name}: {e}")
    
    def stream_pose(self, pose_data: PoseData):
        """Stream pose data to all enabled OSC streams"""
        if not self.config.enabled:
            logger.debug("OSC streaming disabled")
            return
        
        current_time = time.time()
        if current_time - self.last_stream_time < self.stream_interval:
            return
        
        self.last_stream_time = current_time
        
        logger.debug(f"Streaming pose data to {len(self.clients)} OSC streams")
        logger.debug(f"Available streams: {list(self.config.streams.keys())}")
        
        # Calculate body-relative measurements
        body_scale = self._calculate_body_scale(pose_data)
        chest_center = self._get_chest_center(pose_data)
        body_yaw, body_pitch = self._calculate_body_orientation(pose_data)
        
        # Calculate head rotation using body yaw
        head_yaw, head_pitch = self._calculate_head_rotation(pose_data, body_yaw)
        
        logger.debug(f"Body scale: {body_scale}, Chest center: {chest_center}")
        logger.debug(f"Body yaw: {body_yaw:.2f}°, Body pitch: {body_pitch:.2f}°")
        
        # Send to all enabled streams
        if pose_data.is_3d:
            # Get hand positions (relative to chest, normalized by body scale)
            left_wrist = pose_data.landmarks[15]
            right_wrist = pose_data.landmarks[16]
            
            # Get foot positions (relative to chest, normalized by body scale)
            left_ankle = pose_data.landmarks[27]
            right_ankle = pose_data.landmarks[28]
            
            left_hand_pos = (left_wrist - chest_center) / body_scale
            right_hand_pos = (right_wrist - chest_center) / body_scale
            left_foot_pos = (left_ankle - chest_center) / body_scale
            right_foot_pos = (right_ankle - chest_center) / body_scale
            
            # Get torso position in frame coordinates
            torso_pos = chest_center
            
            # Calculate movement and acceleration (now including feet)
            movements = self._calculate_hand_movement(pose_data)
            accelerations = self._calculate_hand_acceleration(pose_data)
            
            # Apply Z-filters if available
            velocity_magnitude = 0.0
            acceleration_magnitude = 0.0
            
            if 'pose_data' in self.z_filters:
                # Use velocity filter for overall movement (hands + feet)
                left_movement = movements.get('left', 0.0)
                right_movement = movements.get('right', 0.0)
                # Add foot movement to total
                left_foot_movement = np.linalg.norm(left_ankle - self._get_previous_position('left_foot'))
                right_foot_movement = np.linalg.norm(right_ankle - self._get_previous_position('right_foot'))
                total_movement = left_movement + right_movement + left_foot_movement + right_foot_movement
                velocity_magnitude = self.z_filters['pose_data'].update(total_movement)
                
                # Use acceleration filter for overall acceleration (hands + feet)
                left_accel = accelerations.get('left', 0.0)
                right_accel = accelerations.get('right', 0.0)
                # Add foot acceleration to total
                left_foot_accel = np.linalg.norm(left_ankle - 2 * self._get_previous_position('left_foot') + self._get_previous_position('left_foot', 2))
                right_foot_accel = np.linalg.norm(right_ankle - 2 * self._get_previous_position('right_foot') + self._get_previous_position('right_foot', 2))
                total_acceleration = left_accel + right_accel + left_foot_accel + right_foot_accel
                acceleration_magnitude = self.z_filters['pose_data'].update(total_acceleration)
            
            # Pack all 21 values into single OSC message
            # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
            # 1-3: left hand x,y,z (body-relative)
            # 4-6: right hand x,y,z (body-relative)
            # 7-9: left foot x,y,z (body-relative)
            # 10-12: right foot x,y,z (body-relative)
            # 13-14: torso rotation yaw,pitch (degrees)
            # 15-16: head rotation yaw,pitch (relative to torso, degrees)
            # 17-19: torso position x,y,z (frame coordinates)
            # 20: velocity magnitude (Z-filtered)
            # 21: acceleration magnitude (Z-filtered)
            
            osc_data = [
                left_hand_pos[0], left_hand_pos[1], left_hand_pos[2],      # 1-3: left hand
                right_hand_pos[0], right_hand_pos[1], right_hand_pos[2],   # 4-6: right hand
                left_foot_pos[0], left_foot_pos[1], left_foot_pos[2],      # 7-9: left foot
                right_foot_pos[0], right_foot_pos[1], right_foot_pos[2],   # 10-12: right foot
                body_yaw, body_pitch,                                      # 13-14: torso rotation
                head_yaw, head_pitch,                                      # 15-16: head rotation
                torso_pos[0], torso_pos[1], torso_pos[2],                  # 17-19: torso position
                velocity_magnitude,                                        # 20: velocity
                acceleration_magnitude                                     # 21: acceleration
            ]
            
            # Send to all enabled streams
            for stream_name, stream_config in self.config.streams.items():
                if stream_config.enabled:
                    logger.debug(f"Sending OSC message to {stream_name} at {stream_config.address}: {len(osc_data)} values")
                    self._send_osc_message(stream_name, stream_config.address, *osc_data)
            
            # Update previous positions for next frame
            self._update_previous_positions(left_ankle, right_ankle)
    
    def _get_previous_position(self, limb: str, frames_back: int = 1) -> np.ndarray:
        """Get previous position for a limb (for velocity/acceleration calculations)"""
        if frames_back == 1:
            if limb in self.previous_foot_positions:
                return self.previous_foot_positions[limb] if self.previous_foot_positions[limb] is not None else np.array([0.0, 0.0, 0.0])
            else:
                return self.previous_hand_positions[limb] if self.previous_hand_positions[limb] is not None else np.array([0.0, 0.0, 0.0])
        elif frames_back == 2:
            if limb in self.previous_foot_positions_2:
                return self.previous_foot_positions_2[limb] if self.previous_foot_positions_2[limb] is not None else np.array([0.0, 0.0, 0.0])
            else:
                return self.previous_hand_positions[limb] if self.previous_hand_positions[limb] is not None else np.array([0.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 0.0])
    
    def _update_previous_positions(self, left_ankle: np.ndarray, right_ankle: np.ndarray):
        """Update previous positions for feet (shift back by 2 frames for acceleration)"""
        # Shift positions back by 2 frames for acceleration calculation
        if self.previous_foot_positions['left_foot'] is not None:
            self.previous_foot_positions_2['left_foot'] = self.previous_foot_positions['left_foot'].copy()
        if self.previous_foot_positions['right_foot'] is not None:
            self.previous_foot_positions_2['right_foot'] = self.previous_foot_positions['right_foot'].copy()
        
        # Update current positions
        self.previous_foot_positions['left_foot'] = left_ankle.copy()
        self.previous_foot_positions['right_foot'] = right_ankle.copy()
    
    def close(self):
        """Close all OSC clients"""
        for stream_name, client in self.clients.items():
            if client is not None:
                try:
                    # python-osc UDPClient doesn't have a close method
                    # Just clear the reference
                    logger.info(f"✅ OSC client reference cleared for {stream_name}")
                except Exception as e:
                    logger.error(f"❌ Error clearing OSC client for {stream_name}: {e}")
        
        self.clients.clear()


def create_advanced_osc_streamer(config_data: Dict[str, Any]) -> Optional[AdvancedOSCStreamer]:
    """Create an AdvancedOSCStreamer from configuration data"""
    if not config_data.get("enabled", False):
        return None
    
    # Parse stream configurations
    streams_config = OSCStreamsConfig()
    streams_config.enabled = config_data.get("enabled", False)
    streams_config.stream_rate = config_data.get("stream_rate", 30.0)
    
    streams_data = config_data.get("streams", {})
    for stream_name, stream_data in streams_data.items():
        stream_config = StreamConfig(
            enabled=stream_data.get("enabled", True),
            host=stream_data.get("host", "127.0.0.1"),
            port=stream_data.get("port", 8000),
            address=stream_data.get("address", f"/pose/{stream_name}"),
            include_confidence=stream_data.get("include_confidence", False),
            z_filter=stream_data.get("z_filter")
        )
        streams_config.streams[stream_name] = stream_config
    
    return AdvancedOSCStreamer(streams_config)
