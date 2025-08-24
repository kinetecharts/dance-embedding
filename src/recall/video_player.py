"""Multi-video playback management with synchronized dual-window display."""

import cv2
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd

from .data_structures import PoseData, Match, create_pose_from_mediapipe
from .config import RecallConfig

logger = logging.getLogger(__name__)


class VideoPlayer:
    """Multi-video playback management with synchronized dual-window display"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.players = {}  # video_path -> player
        self.pose_cache = {}  # video_path -> List[PoseData]
        self.frame_cache = {}  # (video_path, timestamp) -> frame
        self.playback_threads = {}  # video_path -> thread
        self.running = True
        self.current_match = None
        self.match_start_time = None
        self.target_video_stems = []  # Initialize empty target videos list
        
        # Don't preload poses here - wait until target_video_stems is set
    
    def _preload_all_poses(self):
        """Pre-load all pose data to avoid repeated CSV loading"""
        logger.info("Pre-loading all pose data for better performance...")
        pose_dir = Path(self.config.pose_dir)
        
        # Get the list of videos we actually want to use for matching
        target_videos = self._get_target_video_stems()
        
        if target_videos:
            logger.info(f"Pre-loading poses for {len(target_videos)} target videos: {target_videos}")
            for csv_file in pose_dir.glob("*.csv"):
                video_stem = csv_file.stem
                # Only load poses for videos we want to use
                if video_stem in target_videos:
                    poses = self._load_poses_from_csv(csv_file)
                    self.pose_cache[video_stem] = poses
                    logger.info(f"Pre-loaded {len(poses)} poses for {video_stem}")
                else:
                    logger.debug(f"Skipping poses for {video_stem} (not in target videos)")
        else:
            # Fallback: load all poses if no target videos specified
            logger.info("No target videos specified, loading all pose data...")
            for csv_file in pose_dir.glob("*.csv"):
                video_stem = csv_file.stem
                poses = self._load_poses_from_csv(csv_file)
                self.pose_cache[video_stem] = poses
                logger.info(f"Pre-loaded {len(poses)} poses for {video_stem}")
        
        logger.info(f"Pre-loaded pose data for {len(self.pose_cache)} videos")
    
    def _get_target_video_stems(self) -> List[str]:
        """Get the list of video stems we want to use for matching"""
        try:
            # Check if target_video_stems was set during creation
            if hasattr(self, 'target_video_stems') and self.target_video_stems:
                return self.target_video_stems
            
            # Fallback: try to get from config if available
            if hasattr(self, 'config') and hasattr(self.config, 'video_dir'):
                return []
            return []
        except Exception as e:
            logger.debug(f"Could not determine target videos: {e}")
            return []
    
    def play_match(self, match: Match):
        """Start playback for a match with synchronized dual-window display"""
        try:
            # Store current match info for display
            self.current_match = match
            self.match_start_time = time.time()
            
            logger.info(f"🎬 Started match display for {Path(match.video_file).stem} at {match.timestamp:.2f}s")
            
            # Try to enable video playback
            video_path = self._find_video_file(match.video_file)
            if video_path:
                self._create_player(video_path)
                
                # Seek to the matched timestamp
                if video_path in self.players:
                    player = self.players[video_path]
                    self._seek_to_timestamp(player, match.timestamp)
                    logger.info(f"Seeked to timestamp {match.timestamp:.2f}s for {Path(match.video_file).stem}")
                
                logger.info(f"Video playback enabled for {Path(match.video_file).stem}")
            else:
                logger.warning(f"Video file not found for {match.video_file}, showing pose only")
            
        except Exception as e:
            logger.error(f"Error starting match playback: {e}")
            # Don't clear current_match on error - keep it for pose display
    
    def _create_player(self, video_path: str):
        """Create video player instance"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return
            
            self.players[video_path] = cap
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Created player for {Path(video_path).name}: {fps:.2f} FPS, {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error creating video player for {video_path}: {e}")
    
    def _find_video_file(self, video_filename: str) -> Optional[str]:
        """Find the full path to a video file"""
        try:
            video_stem = Path(video_filename).stem
            logger.info(f"Looking for video file with stem: {video_stem}")
            
            # Try different video extensions
            for ext in [".mp4", ".avi", ".mov", ".MOV", ".mkv", ".webm"]:
                video_path = Path(self.config.video_dir) / f"{video_stem}{ext}"
                if video_path.exists():
                    logger.info(f"Found video file: {video_path}")
                    return str(video_path)
            
            # If not found, try the exact filename
            video_path = Path(self.config.video_dir) / video_filename
            if video_path.exists():
                logger.info(f"Found video file with exact name: {video_path}")
                return str(video_path)
            
            # Try case-insensitive search
            video_dir = Path(self.config.video_dir)
            for video_file in video_dir.iterdir():
                if video_file.is_file() and video_file.stem.lower() == video_stem.lower():
                    logger.info(f"Found video file (case-insensitive): {video_file}")
                    return str(video_file)
            
            logger.warning(f"No video file found for {video_filename}")
            logger.info(f"Available video files: {[f.name for f in Path(self.config.video_dir).iterdir() if f.is_file()]}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding video file for {video_filename}: {e}")
            return None
    
    def _load_pose_data(self, video_path: str, pose_file: str):
        """Load pose data from CSV file"""
        try:
            # Fix path construction - use just the filename, not full path
            pose_filename = Path(pose_file).name  # Get just the filename
            pose_path = Path(self.config.pose_dir) / pose_filename
            
            if not pose_path.exists():
                logger.error(f"Pose file not found: {pose_path}")
                return
            
            poses = self._load_poses_from_csv(pose_path)
            self.pose_cache[video_path] = poses
            
            logger.info(f"Loaded {len(poses)} poses for {Path(video_path).name}")
            
        except Exception as e:
            logger.error(f"Error loading pose data for {video_path}: {e}")
    
    def _load_poses_from_csv(self, pose_file: Path) -> List[PoseData]:
        """Load poses from CSV file"""
        try:
            df = pd.read_csv(pose_file)
            
            # Define the landmark names in the correct order
            landmark_names = [
                "nose", "left_eye_inner", "left_eye", "left_eye_outer",
                "right_eye_inner", "right_eye", "right_eye_outer",
                "left_ear", "right_ear", "mouth_left", "mouth_right",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_pinky", "right_pinky",
                "left_index", "right_index", "left_thumb", "right_thumb",
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle", "left_heel", "right_heel",
                "left_foot_index", "right_foot_index"
            ]
            
            poses = []
            for _, row in df.iterrows():
                # Extract landmarks (33 landmarks, 3 coordinates each)
                landmarks = []
                for landmark_name in landmark_names:
                    x = row.get(f'{landmark_name}_x', 0.0)
                    y = row.get(f'{landmark_name}_y', 0.0)
                    z = row.get(f'{landmark_name}_z', 0.0)
                    landmarks.append([x, y, z])
                
                # Extract confidence scores
                confidence = []
                for landmark_name in landmark_names:
                    conf = row.get(f'{landmark_name}_confidence', 1.0)
                    confidence.append(conf)
                
                # Create pose data
                pose_data = PoseData(
                    landmarks=np.array(landmarks),
                    confidence=np.array(confidence),
                    timestamp=row.get('timestamp', 0.0),
                    frame_number=row.get('frame_number', 0)
                )
                poses.append(pose_data)
            
            return poses
            
        except Exception as e:
            logger.error(f"Error loading poses from {pose_file}: {e}")
            return []
    
    def _seek_to_timestamp(self, player, timestamp: float):
        """Seek video player to specific timestamp"""
        try:
            fps = player.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            player.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        except Exception as e:
            logger.error(f"Error seeking to timestamp {timestamp}: {e}")
    
    def _start_synchronized_playback(self, video_path: str, start_time: float):
        """Start synchronized playback with dual-window display"""
        # Stop existing thread if running
        if video_path in self.playback_threads:
            try:
                self.playback_threads[video_path].join(timeout=0.1)
            except:
                pass
        
        def play_video():
            try:
                player = self.players.get(video_path)
                if not player:
                    logger.error(f"No player found for {video_path}")
                    return
                
                poses = self.pose_cache.get(video_path, [])
                video_name = Path(video_path).stem
                
                # Reset video to beginning
                player.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Find starting pose index
                pose_idx = 0
                while pose_idx < len(poses) and poses[pose_idx].timestamp < start_time:
                    pose_idx += 1
                
                # Seek to approximate frame
                fps = player.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    start_frame = int(start_time * fps)
                    player.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Play video for specified duration
                frame_count = 0
                max_frames = int(self.config.match_playback_duration * 30)  # 30 FPS
                playback_start_time = time.time()
                
                logger.info(f"Starting playback for {video_name} at {start_time:.2f}s for {self.config.match_playback_duration}s")
                
                while frame_count < max_frames and self.running:
                    ret, frame = player.read()
                    if not ret:
                        logger.info(f"End of video reached for {video_name}")
                        break
                    
                    # Get current pose if available
                    current_pose = None
                    if pose_idx < len(poses):
                        current_pose = poses[pose_idx]
                    
                    # Display frame with info
                    self._display_match_frame(frame, current_pose, video_name, frame_count, max_frames)
                    
                    # Sleep to match video frame rate
                    if fps > 0:
                        time.sleep(1.0 / fps)
                    
                    pose_idx += 1
                    frame_count += 1
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Clear current match after playback
                self.current_match = None
                self.match_start_time = None
                
                logger.info(f"Finished synchronized playback for {video_name} ({frame_count} frames)")
                
            except Exception as e:
                logger.error(f"Error in synchronized video playback: {e}")
                # Don't clear current match on error - keep it for pose display
                # self.current_match = None
                # self.match_start_time = None
        
        # Start playback thread
        thread = threading.Thread(target=play_video, daemon=True)
        thread.start()
        self.playback_threads[video_path] = thread
    
    def _display_match_frame(self, frame: np.ndarray, pose_data: Optional[PoseData], 
                           video_name: str, frame_count: int, max_frames: int):
        """Display matched video frame with info"""
        try:
            # Create window for matched video
            window_name = f"Matched Video - {video_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(window_name, 700, 100)  # Position to the right of main window
            
            # Resize frame to reasonable size
            display_frame = cv2.resize(frame, (640, 480))
            
            # Add overlay with video info
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            cv2.putText(display_frame, f"Matched Video: {video_name}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}/{max_frames}", (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw pose landmarks on video frame if available
            if pose_data is not None:
                for i, (landmark, confidence) in enumerate(zip(pose_data.landmarks, pose_data.confidence)):
                    if confidence > 0.5:
                        x, y = int(landmark[0]), int(landmark[1])
                        cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)  # Green for matched pose
            
            # Display the frame
            cv2.imshow(window_name, display_frame)
            
        except Exception as e:
            logger.error(f"Error displaying matched video frame: {e}")
    
    def display_live_frame(self, frame: np.ndarray, pose_data: Optional[PoseData] = None, 
                          match_info: Optional[Match] = None):
        """Display live video frame with match info and matched video side by side"""
        try:
            # Create a larger canvas to hold both live video and matched video
            canvas_width = 1280  # Double the original width
            canvas_height = 480  # Keep original height
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Resize live frame to fit in left half
            live_width = canvas_width // 2
            live_height = canvas_height
            live_frame_resized = cv2.resize(frame, (live_width, live_height))
            
            # Draw pose landmarks on live frame
            if pose_data is not None:
                # Draw pose landmarks
                for i, (landmark, confidence) in enumerate(zip(pose_data.landmarks, pose_data.confidence)):
                    if confidence > 0.5:  # Only draw confident landmarks
                        x, y = int(landmark[0] * live_width), int(landmark[1] * live_height)
                        cv2.circle(live_frame_resized, (x, y), 3, (0, 0, 255), -1)  # Red for live pose
                
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
                        start_x, start_y = int(start_pos[0] * live_width), int(start_pos[1] * live_height)
                        end_x, end_y = int(end_pos[0] * live_width), int(end_pos[1] * live_height)
                        cv2.line(live_frame_resized, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)  # Red for live pose
            
            # Add live video info overlay
            overlay = live_frame_resized.copy()
            cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, live_frame_resized, 0.3, 0, live_frame_resized)
            
            cv2.putText(live_frame_resized, "Live Camera", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add detailed match info if available
            if match_info:
                video_name = Path(match_info.video_file).stem
                score = match_info.similarity_score
                timestamp = match_info.timestamp
                
                cv2.putText(live_frame_resized, f"🎬 MATCH FOUND!", (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(live_frame_resized, f"Video: {video_name}", (20, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(live_frame_resized, f"Time: {timestamp:.2f}s", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(live_frame_resized, f"Score: {score:.3f}", (20, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Add playback status
                if self.match_start_time:
                    elapsed = time.time() - self.match_start_time
                    remaining = max(0, self.config.match_playback_duration - elapsed)
                    cv2.putText(live_frame_resized, f"Playing: {remaining:.1f}s left", (20, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                # Show "No match" when no match is active
                cv2.putText(live_frame_resized, "No match", (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Place live frame on left side of canvas
            canvas[:, :live_width] = live_frame_resized
            
            # Create matched video display on right side
            matched_video_canvas = np.zeros((live_height, live_width, 3), dtype=np.uint8)
            
            # Try to get the current matched video frame
            if match_info:
                try:
                    # Find the video file path
                    video_path = self._find_video_file(match_info.video_file)
                    if video_path:
                        # Get the frame at the matched timestamp
                        matched_frame = self._get_frame_at_timestamp(video_path, match_info.timestamp)
                        
                        if matched_frame is not None:
                            # Resize to fit the right side
                            matched_frame_resized = cv2.resize(matched_frame, (live_width, live_height))
                            
                            # Add overlay with video info
                            overlay = matched_frame_resized.copy()
                            cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.7, matched_frame_resized, 0.3, 0, matched_frame_resized)
                            
                            cv2.putText(matched_frame_resized, f"Matched Video: {Path(match_info.video_file).stem}", (20, 35), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(matched_frame_resized, f"Time: {match_info.timestamp:.2f}s", (20, 55), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Draw pose landmarks on matched video if available
                            video_name = Path(match_info.video_file).stem
                            if video_name in self.pose_cache and match_info.pose_index < len(self.pose_cache[video_name]):
                                matched_pose_data = self.pose_cache[video_name][match_info.pose_index]
                                for i, (landmark, confidence) in enumerate(zip(matched_pose_data.landmarks, matched_pose_data.confidence)):
                                    if confidence > 0.5:
                                        # Scale coordinates to match video frame
                                        x = int(landmark[0] * live_width / 1280)  # Scale from original video width
                                        y = int(landmark[1] * live_height / 720)  # Scale from original video height
                                        x = max(0, min(x, live_width - 1))
                                        y = max(0, min(y, live_height - 1))
                                        cv2.circle(matched_frame_resized, (x, y), 4, (0, 255, 0), -1)  # Green for matched pose
                            
                            matched_video_canvas = matched_frame_resized
                        else:
                            # If no frame available, show placeholder
                            cv2.putText(matched_video_canvas, "Video Frame Unavailable", (20, 35), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    else:
                        # Video file not found
                        cv2.putText(matched_video_canvas, "Video File Not Found", (20, 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                except Exception as e:
                    logger.error(f"Error reading matched video frame: {e}")
                    cv2.putText(matched_video_canvas, "Video Error", (20, 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Show "No match" when no match is active
                cv2.putText(matched_video_canvas, "No match", (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Place matched video on right side of canvas
            canvas[:, live_width:] = matched_video_canvas
            
            # Add separator line between live and matched video
            cv2.line(canvas, (live_width, 0), (live_width, canvas_height), (255, 255, 255), 2)
            
            # Add controls info at bottom
            cv2.putText(canvas, "Press 'q' to quit", (10, canvas_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Create and position window
            cv2.namedWindow("Dance Recall System", cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow("Dance Recall System", 100, 100)  # Position window
            cv2.resizeWindow("Dance Recall System", canvas_width, canvas_height)  # Set size
            
            # Display frame
            cv2.imshow("Dance Recall System", canvas)
            
            # Handle key presses and return the key value
            key = cv2.waitKey(1) & 0xFF
            return key
            
        except Exception as e:
            logger.error(f"Error displaying live frame: {e}")
            return -1
    
    def display_matched_pose(self, match: Match, live_pose: Optional[PoseData] = None):
        """Display matched pose in a separate window for comparison - DISABLED"""
        # Disabled since we now show matched pose in the main window
        pass
    
    def clear_matched_pose_window(self):
        """Clear the matched pose window"""
        try:
            cv2.destroyWindow("Matched Pose")
            cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Error clearing matched pose window: {e}")
    
    def cleanup(self):
        """Clean up video player resources"""
        try:
            self.stop_all()
            logger.info("Video player cleanup complete")
        except Exception as e:
            logger.error(f"Error during video player cleanup: {e}")
    
    def stop_all(self):
        """Stop all video players"""
        self.running = False
        
        # Stop playback threads
        for thread in self.playback_threads.values():
            thread.join(timeout=1.0)
        self.playback_threads.clear()
        
        # Release video players
        for player in self.players.values():
            player.release()
        self.players.clear()
        
        # Clear pose cache
        self.pose_cache.clear()
        
        # Clear current match
        self.current_match = None
        self.match_start_time = None
        
        # Clear frame cache
        self.frame_cache.clear()
        
        # Close video windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        logger.info("Stopped all video players")
    
    def get_available_videos(self) -> List[str]:
        """Get list of available video files"""
        video_dir = Path(self.config.video_dir)
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        video_files = []
        
        for ext in extensions:
            video_files.extend([f.name for f in video_dir.glob(f"*{ext}")])
        
        return video_files
    
    def get_playing_videos(self) -> List[str]:
        """Get list of currently playing videos"""
        return list(self.players.keys())
    
    def get_video_progress(self, video_path: str) -> Optional[float]:
        """Get current playback progress for a video"""
        if video_path not in self.players:
            return None
        
        try:
            player = self.players[video_path]
            current_frame = player.get(cv2.CAP_PROP_POS_FRAMES)
            fps = player.get(cv2.CAP_PROP_FPS)
            return current_frame / fps if fps > 0 else 0.0
        except Exception as e:
            logger.error(f"Error getting video progress: {e}")
            return None
    
    def _get_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Get a specific frame from a video at a given timestamp"""
        try:
            # Check if frame is already in cache
            cache_key = (video_path, round(timestamp, 2))  # Round to 2 decimal places for caching
            if cache_key in self.frame_cache:
                return self.frame_cache[cache_key]

            # Limit cache size to prevent memory issues
            if len(self.frame_cache) > 100:  # Keep only last 100 frames
                # Remove oldest entries
                oldest_keys = list(self.frame_cache.keys())[:50]
                for key in oldest_keys:
                    del self.frame_cache[key]

            # Create a temporary video capture for this specific frame
            temp_cap = cv2.VideoCapture(video_path)
            if not temp_cap.isOpened():
                logger.error(f"Failed to open video for frame extraction: {video_path}")
                return None
            
            # Get video properties
            fps = temp_cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.error(f"Invalid FPS for video: {video_path}")
                temp_cap.release()
                return None
            
            # Calculate frame number
            frame_number = int(timestamp * fps)
            
            # Seek to the frame
            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            ret, frame = temp_cap.read()
            
            # Release the temporary capture
            temp_cap.release()
            
            if ret:
                self.frame_cache[cache_key] = frame  # Cache the frame
                return frame
            else:
                logger.error(f"Failed to read frame at timestamp {timestamp}s from {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting frame at timestamp {timestamp}s from {video_path}: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_all()


class VideoPlayerWithControls(VideoPlayer):
    """Video player with additional control features"""
    
    def __init__(self, config: RecallConfig):
        super().__init__(config)
        self.playback_speeds = {}  # video_path -> speed
        self.loop_modes = {}  # video_path -> loop
    
    def set_playback_speed(self, video_path: str, speed: float):
        """Set playback speed for a video"""
        self.playback_speeds[video_path] = speed
        logger.info(f"Set playback speed for {Path(video_path).name}: {speed}x")
    
    def set_loop_mode(self, video_path: str, loop: bool):
        """Set loop mode for a video"""
        self.loop_modes[video_path] = loop
        logger.info(f"Set loop mode for {Path(video_path).name}: {loop}")


def create_video_player(config: RecallConfig, with_controls: bool = False, target_videos: Optional[List[Path]] = None) -> VideoPlayer:
    """Create video player with optional control features and target video filtering"""
    if with_controls:
        player = VideoPlayerWithControls(config)
    else:
        player = VideoPlayer(config)
    
    # Set target videos if provided
    if target_videos is not None:
        player.target_video_stems = [v.stem for v in target_videos]
        logger.info(f"Video player configured with target videos: {player.target_video_stems}")
    
    # Now preload poses with the target videos set
    player._preload_all_poses()
    
    return player 