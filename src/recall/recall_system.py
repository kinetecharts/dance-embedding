"""Main recall system orchestrator with dual-window display."""

import time
import threading
import logging
from typing import List, Optional
from pathlib import Path
import cv2

from .config import RecallConfig
from .data_structures import PoseData, Match
from .pose_tracker import PoseTracker
from .pose_matcher import PoseMatcher
from .video_player import create_video_player
from .advanced_osc_streamer import create_advanced_osc_streamer
from .json_config_loader import create_config_loader
from .video_recorder import MacVideoRecorder

logger = logging.getLogger(__name__)


class RecallSystem:
    """Main system orchestrator for live pose matching and video playback with dual-window display"""
    
    def __init__(self, config: RecallConfig, osc_only: bool = False):
        self.config = config
        self.osc_only = osc_only
        self.running = True
        self.paused = False
        
        if osc_only:
            # Lightweight OSC-only mode - no video loading or matching
            logger.info("🚀 Initializing lightweight OSC-only mode")
            
            # Initialize JSON config loader for OSC configuration only
            self.config_loader = create_config_loader()
            logger.info("✅ JSON config loader initialized for OSC")
            
            # Initialize only essential components
            self.pose_tracker = PoseTracker(config)
            
            # Initialize advanced OSC streamer if enabled
            self.osc_streamer = None
            if config.osc_enabled:
                # Get OSC configuration from JSON config loader
                osc_config = self.config_loader.config_data.get("osc_streaming", {})
                self.osc_streamer = create_advanced_osc_streamer(osc_config)
                if self.osc_streamer:
                    logger.info("✅ Advanced OSC streaming enabled with multiple streams")
                else:
                    logger.warning("⚠️ OSC streaming configuration invalid or disabled")
            
            # Initialize video recorder if enabled
            self.video_recorder = None
            if config.record_video:
                self.video_recorder = MacVideoRecorder(config, config.record_dir)
                logger.info("✅ Video recorder initialized")
            
            # Skip heavy components
            self.pose_matcher = None
            self.video_player = None
            self.target_video_files = []
            
            logger.info("✅ Lightweight OSC-only system initialized")
            
        else:
            # Full mode with video matching
            logger.info("🎬 Initializing full recall system with video matching")
            
            # Initialize JSON config loader for video file selection
            self.config_loader = create_config_loader()
            logger.info("✅ JSON config loader initialized")
            
            # Get filtered video files for matching
            self.target_video_files = self.config_loader.get_video_files_for_matching(config.video_dir)
            logger.info(f"Target videos for matching: {[f.stem for f in self.target_video_files]}")
            
            # Initialize components
            self.pose_tracker = PoseTracker(config)
            
            # Create pose matcher with target video filtering
            target_video_names = [f.name for f in self.target_video_files] if self.target_video_files else None
            self.pose_matcher = PoseMatcher(config.__dict__, target_videos=target_video_names)
            
            # Create video player with target video information
            self.video_player = create_video_player(config, with_controls=True, target_videos=self.target_video_files)
            
            # Initialize advanced OSC streamer if enabled
            self.osc_streamer = None
            if config.osc_enabled:
                # Get OSC configuration from JSON config loader
                osc_config = self.config_loader.config_data.get("osc_streaming", {})
                self.osc_streamer = create_advanced_osc_streamer(osc_config)
                if self.osc_streamer:
                    logger.info("✅ Advanced OSC streaming enabled with multiple streams")
                else:
                    logger.warning("⚠️ OSC streaming configuration invalid or disabled")
            
            # Initialize video recorder if enabled
            self.video_recorder = None
            if config.record_video:
                self.video_recorder = MacVideoRecorder(config, config.record_dir)
                logger.info("✅ Video recorder initialized")
            
            logger.info("✅ Full recall system initialized with LanceDB-based pose matching")
        
        # State tracking
        self.current_pose = None
        self.current_matches = []
        self.matches_history = []
        self.frame_count = 0
        self.start_time = None
        self.last_match_time = 0
        

        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = 0
        self.current_fps = 0.0
    
    def run_live(self):
        """Main live processing loop with dual-window display"""
        if self.osc_only:
            logger.info("Starting lightweight OSC-only mode - camera input only")
        else:
            logger.info("Starting live camera mode with LanceDB pose matching")
        
        # Start camera
        if not self.pose_tracker.start_camera(self.config.camera_id):
            logger.error("Failed to start camera")
            return
        
        self.start_time = time.time()
        logger.info("✅ Camera started successfully")
        
        # Start video recording if enabled
        if self.video_recorder:
            # Get frame dimensions from camera
            frame_width = int(self.pose_tracker.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.pose_tracker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use the actual camera frame rate to avoid video stretching/slowing
            recording_fps = self.config.record_fps  # Use full 30 FPS to match real-time speed
            
            logger.info(f"🎬 Attempting to start video recording: {frame_width}x{frame_height} @ {recording_fps} FPS, {self.config.record_quality} quality")
            
            if self.video_recorder.start_recording(
                frame_width, frame_height, 
                recording_fps,  # Use capped FPS to match processing speed
                self.config.record_quality
            ):
                logger.info("✅ Video recording started")
            else:
                logger.error("Failed to start video recording")
                self.video_recorder = None
        
        # Create initial live camera window (both modes need this)
        logger.info("Creating live camera window...")
        try:
            # Get initial frame to create window
            result = self.pose_tracker.get_next_pose()
            if result is not None:
                pose_data, frame = result
                logger.info(f"Got initial frame: {frame.shape}")
                
                if self.osc_only:
                    # OSC-only mode: create a simple display window
                    cv2.imshow("Live Camera - OSC Only", frame)
                    logger.info("✅ Live camera window created successfully (OSC-only mode)")
                else:
                    # Full mode: use video player display
                    self.video_player.display_live_frame(frame, pose_data, None)
                    logger.info("✅ Live camera window created successfully (full mode)")
                
                # Force window to appear
                cv2.waitKey(100)
                logger.info("Forced window display")
            else:
                logger.warning("No initial frame available")
        except Exception as e:
            logger.error(f"Error creating live camera window: {e}")
        
        if self.osc_only:
            logger.info("🚀 OSC-only mode: Press 'q' in video window to quit")
            logger.info("📡 Streaming pose data to configured OSC endpoints")
        else:
            logger.info("Press 'q' in any video window to quit")
            logger.info(f"🎯 Matching every {self.config.match_interval} seconds")
            logger.info(f"🎬 Playing matched videos for {self.config.match_playback_duration} seconds")
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Get live pose and frame
                result = self.pose_tracker.get_next_pose()
                if result is None:
                    continue
                
                pose_data, frame = result
                if pose_data is None:
                    # No pose detected
                    if self.osc_only:
                        # In OSC-only mode, still show the frame even without pose
                        # Add frame to video recording if active (record every frame, not just pose frames)
                        if self.video_recorder and self.video_recorder.is_recording():
                            self.video_recorder.record_frame(frame, None)
                        
                        cv2.imshow("Live Camera - OSC Only", frame)
                        
                        # Handle key press for OSC-only mode
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("Q pressed - quitting")
                            break
                        
                        # Update FPS and show basic metrics
                        self._update_fps()
                        if self.frame_count % 30 == 0:  # Show metrics every 30 frames
                            self._show_basic_metrics()
                        
                        # Don't sleep - let it run as fast as possible for smooth video
                        continue
                    else:
                        # Show frame in full mode
                        key = self.video_player.display_live_frame(frame)
                        if key == ord('q'):
                            logger.info("Q pressed - quitting")
                            break
                        continue
                
                self.current_pose = pose_data
                self.frame_count += 1
                
                # Stream pose data via OSC if enabled
                if self.osc_streamer:
                    logger.debug("Calling OSC streamer with pose data")
                    self.osc_streamer.stream_pose(pose_data)
                else:
                    logger.debug("No OSC streamer available")
                
                if self.osc_only:
                    # OSC-only mode: display frame and stream data, no matching
                    # Draw pose landmarks on frame
                    display_frame = self._draw_pose_on_frame(frame, pose_data)
                    
                    # Add frame to video recording if active (record every frame, not just pose frames)
                    if self.video_recorder and self.video_recorder.is_recording():
                        self.video_recorder.record_frame(display_frame, pose_data)
                    
                    # Display live frame with pose visualization (no recording indicator)
                    cv2.imshow("Live Camera - OSC Only", display_frame)
                    
                    # Handle key press for OSC-only mode
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Q pressed - quitting")
                        break
                    elif key == ord('v'):  # V to start/stop video recording
                        if self.video_recorder and self.video_recorder.is_recording():
                            logger.info("Stopping video recording...")
                            self.video_recorder.stop_recording()
                        elif self.video_recorder and not self.config.record_video:
                            # Only allow manual start if --record-video flag was not passed
                            logger.info("Starting video recording...")
                            frame_width = int(self.pose_tracker.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            frame_height = int(self.pose_tracker.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            self.video_recorder.start_recording(frame_width, frame_height, 
                                                             self.config.record_fps, 
                                                             self.config.record_quality)
                        elif self.video_recorder and self.config.record_video:
                            logger.info("Recording already started automatically with --record-video flag")
                            logger.info("Press 'V' again to stop recording")
                    elif key == ord('h'):  # H to show help
                        self._show_recording_help()
                    
                    # Update FPS and show basic metrics
                    self._update_fps()
                    if self.frame_count % 30 == 0:  # Show metrics every 30 frames
                        self._show_basic_metrics()
                    
                    # Don't sleep - let it run as fast as possible for smooth video
                else:
                    # Full mode: perform matching and display
                    # Check if it's time to match (every 2 seconds)
                    current_time = time.time()
                    if current_time - self.last_match_time >= self.config.match_interval:
                        logger.info(f"🎯 Performing match at {current_time:.1f}s")
                        self._perform_matching(pose_data)
                        self.last_match_time = current_time
                    
                    # Display live frame with current match info
                    current_match = self.video_player.current_match
                    key = self.video_player.display_live_frame(frame, pose_data, current_match)
                    
                    # Add frame to video recording if active (record every frame, not just pose frames)
                    if self.video_recorder and self.video_recorder.is_recording():
                        self.video_recorder.record_frame(frame, pose_data)
                    
                    # Handle key press
                    if key == ord('q'):
                        logger.info("Q pressed - quitting")
                        break
                    
                    # Update FPS
                    self._update_fps()
                    
                    # Show metrics in terminal
                    self._show_metrics()
                    
                    # Don't sleep - let it run as fast as possible for smooth video
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in live processing loop: {e}")
        finally:
            self._cleanup()
    
    def run_video(self, video_path: str, max_frames: Optional[int] = None):
        """Process video file input with dual-window display"""
        logger.info(f"Starting video mode with LanceDB pose matching: {video_path}")
        
        # Start video
        if not self.pose_tracker.start_video(video_path):
            logger.error("Failed to start video")
            return
        
        self.start_time = time.time()
        logger.info("✅ Video started successfully")
        logger.info("Press 'q' in any video window to quit")
        logger.info(f"🎯 Matching every {self.config.match_interval} seconds")
        logger.info(f"🎬 Playing matched videos for {self.config.match_playback_duration} seconds")
        
        try:
            while self.running and not self.pose_tracker.is_ended():
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Check max frames limit
                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached max frames limit: {max_frames}")
                    break
                
                # Get pose from video
                result = self.pose_tracker.get_next_pose()
                if result is None:
                    continue
                
                pose_data, frame = result
                if pose_data is None:
                    # No pose detected, still show frame
                    key = self.video_player.display_live_frame(frame)
                    if key == ord('q'):
                        logger.info("Q pressed - quitting")
                        break
                    continue
                
                self.current_pose = pose_data
                self.frame_count += 1
                
                # Stream pose data via OSC if enabled
                if self.osc_streamer:
                    logger.debug("Calling OSC streamer with pose data")
                    self.osc_streamer.stream_pose(pose_data)
                else:
                    logger.debug("No OSC streamer available")
                
                # Check if it's time to match (every 2 seconds)
                current_time = time.time()
                if current_time - self.last_match_time >= self.config.match_interval:
                    self._perform_matching(pose_data)
                    self.last_match_time = current_time
                
                # Display live frame with current match info
                current_match = self.video_player.current_match
                key = self.video_player.display_live_frame(frame, pose_data, current_match)
                
                # Handle key press
                if key == ord('q'):
                    logger.info("Q pressed - quitting")
                    break
                
                # Update FPS
                self._update_fps()
                
                # Show metrics in terminal
                self._show_metrics()
                
                # Don't sleep - let it run as fast as possible for smooth video
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in video processing loop: {e}")
        finally:
            self._cleanup()
    
    def _perform_matching(self, pose: PoseData):
        """Perform pose matching and video playback"""
        try:
            # Find matches using LanceDB
            matches = self.pose_matcher.find_matches(pose, self.config.top_n)
            
            if not matches:
                logger.warning("No matches found")
                return
            
            # Log match details
            logger.info(f"Found {len(matches)} matches:")
            for i, match in enumerate(matches):
                logger.info(f"  {i+1}. {match.video_file} at {match.timestamp:.2f}s (score: {match.similarity_score:.3f})")
            
            # Select a match (for now, just take the best one)
            selected_match = self.pose_matcher.select_random_match(matches)
            if selected_match:
                logger.info(f"Selected match: {selected_match.video_file} at {selected_match.timestamp:.2f}s")
                
                # Start video playback for the match
                self.video_player.play_match(selected_match)
                
                # Store in history
                self.current_matches = matches
                self.matches_history.append(matches)
                
                # Log performance stats
                stats = self.pose_matcher.get_performance_stats()
                if "avg_match_time" in stats:
                    logger.info(f"Match performance: {stats['avg_match_time']:.3f}s avg, {stats['total_matches']} total")
            
        except Exception as e:
            logger.error(f"Error in pose matching: {e}")
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            time_diff = current_time - self.last_fps_time
            if time_diff > 0.1:  # Add small buffer to avoid division by zero
                self.current_fps = self.fps_counter / time_diff
            else:
                self.current_fps = 0
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _draw_pose_on_frame(self, frame, pose_data):
        """Draw pose landmarks on frame for OSC-only mode - using same method as video player"""
        try:
            if not pose_data or not hasattr(pose_data, 'landmarks') or pose_data.landmarks is None or len(pose_data.landmarks) == 0:
                logger.debug("No pose data or landmarks available")
                return frame
            
            # Debug logging
            logger.debug(f"Drawing pose: landmarks shape: {pose_data.landmarks.shape}, has confidence: {hasattr(pose_data, 'confidence')}")
            if hasattr(pose_data, 'confidence'):
                logger.debug(f"Confidence shape: {pose_data.confidence.shape}, sample values: {pose_data.confidence[:5]}")
            
            # Create a copy of the frame to draw on
            display_frame = frame.copy()
            
            # Check if we have confidence data
            if hasattr(pose_data, 'confidence') and pose_data.confidence is not None:
                # Draw pose landmarks (same as video player)
                for i, (landmark, confidence) in enumerate(zip(pose_data.landmarks, pose_data.confidence)):
                    if float(confidence) > 0.5:  # Convert numpy value to float for comparison
                        x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                        cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)  # Red for live pose (same as video player)
                
                # Draw skeleton connections (exactly same as video player)
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
                        float(pose_data.confidence[start_idx]) > 0.5 and float(pose_data.confidence[end_idx]) > 0.5):
                        start_pos = pose_data.landmarks[start_idx]
                        end_pos = pose_data.landmarks[end_idx]
                        start_x, start_y = int(start_pos[0] * frame.shape[1]), int(start_pos[1] * frame.shape[0])
                        end_x, end_y = int(end_pos[0] * frame.shape[1]), int(end_pos[1] * frame.shape[0])
                        cv2.line(display_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)  # Red for live pose (same as video player)
            else:
                # Fallback: draw all landmarks without confidence filtering
                logger.debug("No confidence data, drawing all landmarks")
                for i, landmark in enumerate(pose_data.landmarks):
                    x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                    cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)  # Red for live pose
                
                # Draw basic skeleton connections
                basic_connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                ]
                
                for start_idx, end_idx in basic_connections:
                    if (start_idx < len(pose_data.landmarks) and end_idx < len(pose_data.landmarks)):
                        start_pos = pose_data.landmarks[start_idx]
                        end_pos = pose_data.landmarks[end_idx]
                        start_x, start_y = int(start_pos[0] * frame.shape[1]), int(start_pos[1] * frame.shape[0])
                        end_x, end_y = int(end_pos[0] * frame.shape[1]), int(end_pos[1] * frame.shape[0])
                        cv2.line(display_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            
            return display_frame
            
        except Exception as e:
            logger.warning(f"Error drawing pose on frame: {e}")
            logger.debug(f"Pose data type: {type(pose_data)}")
            if hasattr(pose_data, '__dict__'):
                logger.debug(f"Pose data attributes: {pose_data.__dict__.keys()}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return frame
    
    def _show_basic_metrics(self):
        """Show basic metrics for OSC-only mode"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / elapsed if elapsed > 0.1 else 0  # Add small buffer to avoid division by zero
        
        # Get recording status
        if self.video_recorder and self.video_recorder.is_recording():
            warmup_status = self.video_recorder.get_warmup_status()
            
            if warmup_status["status"] == "warming_up":
                progress = warmup_status["progress"]
                total = warmup_status["total"]
                percent = warmup_status["percent"]
                recording_status = f"🟠 WARMING UP ({percent}%)"
                recording_stats = {}
            else:
                recording_status = "🔴 RECORDING"
                recording_stats = self.video_recorder.get_recording_stats()
        else:
            recording_status = "⚪ STOPPED"
            recording_stats = {}
        
        print(f"\033[2J\033[H")  # Clear screen and move cursor to top
        print(f"🚀 OSC-Only Mode - Frame {self.frame_count}")
        print(f"⏱️  Elapsed: {elapsed:.1f}s | FPS: {self.current_fps:.1f} | Avg: {avg_fps:.1f}")
        print(f"📡 OSC Streaming: {'✅' if self.osc_streamer else '❌'}")
        print(f"📹 Camera: ✅ | Pose Detection: ✅")
        print(f"🎬 Recording: {recording_status}")
        
        if warmup_status and warmup_status["status"] == "warming_up":
            print(f"   🔥 Progress: {progress}/{total} frames ({percent}%)")
            print(f"   📁 Output: {self.video_recorder.current_filename}")
        elif recording_stats:
            print(f"   📊 Frames: {recording_stats.get('frame_count', 0)} | Time: {recording_stats.get('elapsed_time', 0):.1f}s")
            print(f"   📁 Output: {recording_stats.get('filename', 'N/A')}")
        
        if self.config.record_video:
            print(f"💡 Press 'q' to quit | 'v' to stop recording | 'h' for help")
        else:
            print(f"💡 Press 'q' to quit | 'v' to start/stop recording | 'h' for help")
        print(f"{'='*50}")
    

    
    def _show_recording_help(self):
        """Show recording controls help"""
        print("\n" + "=" * 50)
        print("🎬 VIDEO RECORDING CONTROLS")
        print("=" * 50)
        
        if self.config.record_video:
            print("🎥 AUTO-RECORDING MODE (--record-video flag passed)")
            print("V - Stop video recording (recording started automatically)")
        else:
            print("🎥 MANUAL RECORDING MODE")
            print("V - Start/Stop video recording")
        
        print("Q - Quit the application")
        print("H - Show this help")
        print("")
        print("📹 RECORDING FEATURES:")
        print("• Video + stick figure overlay + microphone audio")
        print("• Mac-optimized H.264 compression (hardware accelerated)")
        print("• Automatic saving to prevent data loss")
        print("• Organized storage with timestamps")
        print(f"• Output directory: {self.config.record_dir}/")
        print("=" * 50)
    
    def _show_metrics(self):
        """Show real-time metrics"""
        if self.frame_count % 30 == 0:  # Every 30 frames
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.info(f"Frame: {self.frame_count}, FPS: {self.current_fps:.1f}, Time: {elapsed:.1f}s")
    
    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up recall system...")
        self.running = False
        
        if self.pose_tracker:
            self.pose_tracker.release()
        
        if self.video_player:
            self.video_player.cleanup()
        
        # Clean up OSC streamer
        if self.osc_streamer:
            self.osc_streamer.close()
            logger.info("OSC streamer closed")
        
        # Stop video recording if active
        if self.video_recorder and self.video_recorder.is_recording():
            logger.info("Stopping active video recording...")
            self.video_recorder.stop_recording()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Show final statistics
        self._show_final_stats()
        
        logger.info("Recall system cleanup complete")
    
    def _show_final_stats(self):
        """Show final performance statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / elapsed if elapsed > 0.1 else 0  # Add small buffer to avoid division by zero
        
        logger.info("=" * 50)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Total matches found: {len(self.matches_history)}")
        
        # Show pose matcher stats
        if self.pose_matcher:
            matcher_stats = self.pose_matcher.get_performance_stats()
            if "avg_match_time" in matcher_stats:
                logger.info(f"Average match time: {matcher_stats['avg_match_time']:.3f}s")
                logger.info(f"Total matches performed: {matcher_stats['total_matches']}")
        else:
            logger.info("Mode: OSC-only (no pose matching)")
        
        logger.info("=" * 50)
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        logger.info(f"System {status}")
    
    def reset_players(self):
        """Reset video players"""
        if self.video_player:
            self.video_player.reset()
        logger.info("Video players reset")
    
    def set_top_n(self, top_n: int):
        """Set top-N matches"""
        self.config.top_n = max(1, min(10, top_n))
        logger.info(f"Top-N matches set to {self.config.top_n}")
    
    def quit(self):
        """Quit the system"""
        logger.info("Quitting recall system...")
        self.running = False
    
    def get_statistics(self) -> dict:
        """Get current system statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / elapsed if elapsed > 0.1 else 0  # Add small buffer to avoid division by zero
        
        stats = {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "current_fps": self.current_fps,
            "average_fps": avg_fps,
            "total_matches": len(self.matches_history),
            "current_pose": self.current_pose is not None,
            "paused": self.paused,
        }
        
        # Add matcher stats only if pose matcher exists
        if self.pose_matcher:
            stats["matcher_stats"] = self.pose_matcher.get_performance_stats()
        else:
            stats["matcher_stats"] = {"mode": "osc_only"}
        
        return stats
    
    def get_match_history(self) -> List[List[Match]]:
        """Get match history"""
        return self.matches_history.copy()
    
    def clear_history(self):
        """Clear match history"""
        self.matches_history.clear()
        logger.info("Match history cleared")
    
    def get_video_files_for_matching(self) -> List[Path]:
        """Get filtered video files for matching based on JSON config"""
        if hasattr(self, 'config_loader') and self.config_loader:
            return self.config_loader.get_video_files_for_matching(self.config.video_dir)
        else:
            # Fallback to default behavior
            return self.config.get_video_files()
    
    def get_config_summary(self) -> str:
        """Get configuration summary"""
        if hasattr(self, 'config_loader') and self.config_loader:
            return self.config_loader.get_config_summary()
        return "JSON config loader not available"
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._cleanup()


class RecallSystemWithKeyboard(RecallSystem):
    """Recall system with keyboard controls"""
    
    def __init__(self, config: RecallConfig, osc_only: bool = False):
        super().__init__(config, osc_only=osc_only)
        self._setup_keyboard_controls()
    
    def _setup_keyboard_controls(self):
        """Setup keyboard controls"""
        def keyboard_listener():
            import keyboard
            
            def on_key_press(event):
                if event.name == 'q':
                    logger.info("Q pressed - quitting")
                    self.quit()
                elif event.name == 'p':
                    logger.info("P pressed - toggling pause")
                    self.toggle_pause()
                elif event.name == 'r':
                    logger.info("R pressed - resetting players")
                    self.reset_players()
                elif event.name in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    top_n = int(event.name)
                    logger.info(f"{top_n} pressed - setting top-N to {top_n}")
                    self.set_top_n(top_n)
            
            keyboard.on_press(on_key_press)
        
        # Start keyboard listener in background thread
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        keyboard_thread.start()


def create_recall_system(config: RecallConfig, with_keyboard: bool = True, osc_only: bool = False) -> RecallSystem:
    """Create a recall system instance"""
    if osc_only:
        # OSC-only mode - lightweight system without keyboard controls
        return RecallSystem(config, osc_only=True)
    elif with_keyboard:
        return RecallSystemWithKeyboard(config, osc_only=False)
    else:
        return RecallSystem(config, osc_only=False) 