"""Live pose tracking from camera or video input."""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Tuple
import logging
from pathlib import Path

from .data_structures import PoseData, create_pose_from_mediapipe
from .config import RecallConfig

logger = logging.getLogger(__name__)


class PoseTracker:
    """Live pose tracking from camera/video input"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = None
        self.is_camera = False
        self.frame_count = 0
        self.start_time = None
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """Initialize camera capture"""
        import time  # Import at the top of the method
        
        try:
            logger.info(f"Attempting to open camera {camera_id}...")
            self.cap = cv2.VideoCapture(camera_id)
            
            # Check if camera opened immediately
            if self.cap.isOpened():
                logger.info(f"Camera {camera_id} opened successfully on first try")
            else:
                # Try with timeout
                logger.info(f"Camera {camera_id} not ready, waiting up to 5 seconds...")
                start_time = time.time()
                timeout = 5.0
                
                while not self.cap.isOpened() and (time.time() - start_time) < timeout:
                    logger.info(f"Retrying camera {camera_id}...")
                    time.sleep(0.5)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(camera_id)
                
                if not self.cap.isOpened():
                    logger.error(f"Failed to open camera {camera_id} after {timeout}s timeout")
                    logger.info("Please check camera permissions and try again")
                    return False
            
            # Test reading a frame
            ret, test_frame = self.cap.read()
            if not ret:
                logger.error("Camera opened but cannot read frames")
                self.cap.release()
                return False
            
            logger.info(f"Camera test frame shape: {test_frame.shape}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            self.is_camera = True
            self.frame_count = 0
            self.start_time = time.time()
            logger.info(f"✅ Camera {camera_id} initialized successfully")
            logger.info("Press 'q' in the video window to quit")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def start_video(self, video_path: str) -> bool:
        """Initialize video capture"""
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return False
            
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False
            
            self.is_camera = False
            self.frame_count = 0
            self.start_time = time.time()
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Started video capture: {video_path}")
            logger.info(f"Video properties: {fps:.2f} FPS, {frame_count} frames, {duration:.2f}s duration")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video: {e}")
            return False
    
    def _process_frame(self, frame: np.ndarray) -> Optional[PoseData]:
        """Process a single frame to extract pose"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.mp_pose.process(rgb_frame)
            
            if results.pose_landmarks is None:
                return None
            
            # Get timestamp
            if self.start_time is None:
                timestamp = 0.0
            else:
                timestamp = time.time() - self.start_time
            
            # Create pose data
            pose_data = create_pose_from_mediapipe(
                results.pose_landmarks,
                [lm.visibility for lm in results.pose_landmarks.landmark],
                timestamp,
                self.frame_count
            )
            
            # Check confidence threshold
            if np.mean(pose_data.confidence) < self.config.confidence_threshold:
                return None
            
            return pose_data
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def display_frame(self, frame: np.ndarray, pose_data: Optional[PoseData] = None, matches: list = None):
        """Display frame with pose overlay and match information"""
        try:
            display_frame = frame.copy()
            
            if pose_data is not None:
                # Draw pose landmarks
                for i, (landmark, confidence) in enumerate(zip(pose_data.landmarks, pose_data.confidence)):
                    if confidence > 0.5:  # Only draw confident landmarks
                        x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                        cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw skeleton connections
                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
                    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
                    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6),  # Face
                    (0, 9), (9, 10), (10, 11), (0, 9), (9, 10), (10, 12)  # Neck
                ]
                
                for start_idx, end_idx in connections:
                    if (start_idx < len(pose_data.landmarks) and end_idx < len(pose_data.landmarks) and
                        pose_data.confidence[start_idx] > 0.5 and pose_data.confidence[end_idx] > 0.5):
                        start_pos = pose_data.landmarks[start_idx]
                        end_pos = pose_data.landmarks[end_idx]
                        start_x, start_y = int(start_pos[0] * frame.shape[1]), int(start_pos[1] * frame.shape[0])
                        end_x, end_y = int(end_pos[0] * frame.shape[1]), int(end_pos[1] * frame.shape[0])
                        cv2.line(display_frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
            
            # Add frame info
            cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add match information
            if matches:
                y_offset = 60
                cv2.putText(display_frame, f"Matches: {len(matches)}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                for i, match in enumerate(matches[:3]):  # Show top 3 matches
                    y_offset += 25
                    score = match.similarity_score
                    video_name = Path(match.video_file).stem
                    cv2.putText(display_frame, f"Match {i+1}: {video_name} ({score:.3f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(display_frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Live Pose Tracking", display_frame)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested via 'q' key")
                raise KeyboardInterrupt
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
    
    def get_next_pose(self) -> Optional[PoseData]:
        """Get next pose frame"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            if self.is_camera:
                logger.warning("Failed to read camera frame")
            else:
                logger.info("End of video reached")
            return None
        
        # Process frame
        pose_data = self._process_frame(frame)
        if pose_data is not None:
            self.frame_count += 1
        
        return pose_data, frame
    
    def get_frame_info(self) -> Tuple[int, float, float]:
        """Get current frame information"""
        if self.cap is None:
            return 0, 0.0, 0.0
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) if not self.is_camera else 30.0
        current_time = time.time() - self.start_time if self.start_time else 0.0
        
        return self.frame_count, current_time, fps
    
    def seek_to_time(self, timestamp: float) -> bool:
        """Seek to specific timestamp (video only)"""
        if self.is_camera or self.cap is None:
            return False
        
        try:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_count = frame_number
            return True
        except Exception as e:
            logger.error(f"Error seeking to timestamp {timestamp}: {e}")
            return False
    
    def get_video_properties(self) -> Optional[dict]:
        """Get video properties (video only)"""
        if self.is_camera or self.cap is None:
            return None
        
        try:
            return {
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
            }
        except Exception as e:
            logger.error(f"Error getting video properties: {e}")
            return None
    
    def is_ended(self) -> bool:
        """Check if video has ended (video only)"""
        if self.is_camera:
            return False
        
        if self.cap is None:
            return True
        
        return self.frame_count >= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def release(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if hasattr(self, 'mp_pose'):
            self.mp_pose.close()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Ensure windows are closed
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release() 