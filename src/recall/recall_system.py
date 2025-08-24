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
from .osc_streamer import create_osc_streamer
from .json_config_loader import create_config_loader

logger = logging.getLogger(__name__)


class RecallSystem:
    """Main system orchestrator for live pose matching and video playback with dual-window display"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.running = True
        self.paused = False
        
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
        
        # Initialize OSC streamer if enabled
        self.osc_streamer = None
        if config.osc_enabled:
            self.osc_streamer = create_osc_streamer(
                host=config.osc_host,
                port=config.osc_port,
                stream_rate=config.osc_stream_rate,
                enabled=config.osc_enabled
            )
            logger.info(f"✅ OSC streaming enabled: {config.osc_host}:{config.osc_port}")
        
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
        
        logger.info("Recall system initialized with LanceDB-based pose matching")
    
    def run_live(self):
        """Main live processing loop with dual-window display"""
        logger.info("Starting live camera mode with LanceDB pose matching")
        
        # Start camera
        if not self.pose_tracker.start_camera():
            logger.error("Failed to start camera")
            return
        
        self.start_time = time.time()
        logger.info("✅ Camera started successfully")
        logger.info("Press 'q' in any video window to quit")
        logger.info(f"🎯 Matching every {self.config.match_interval} seconds")
        logger.info(f"🎬 Playing matched videos for {self.config.match_playback_duration} seconds")
        
        # Create initial live camera window
        logger.info("Creating live camera window...")
        try:
            # Get initial frame to create window
            result = self.pose_tracker.get_next_pose()
            if result is not None:
                pose_data, frame = result
                logger.info(f"Got initial frame: {frame.shape}")
                self.video_player.display_live_frame(frame, pose_data, None)
                logger.info("✅ Live camera window created successfully")
                
                # Force window to appear
                cv2.waitKey(100)
                logger.info("Forced window display")
            else:
                logger.warning("No initial frame available")
        except Exception as e:
            logger.error(f"Error creating live camera window: {e}")
        
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
                    self.osc_streamer.stream_pose(pose_data)
                
                # Check if it's time to match (every 2 seconds)
                current_time = time.time()
                if current_time - self.last_match_time >= self.config.match_interval:
                    logger.info(f"🎯 Performing match at {current_time:.1f}s")
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
                
                # Sleep to maintain frame rate
                time.sleep(0.033)  # ~30 FPS
                
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
                    self.osc_streamer.stream_pose(pose_data)
                
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
                
                # Sleep to maintain frame rate
                time.sleep(0.033)  # ~30 FPS
                
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
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
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
        
        # Show final statistics
        self._show_final_stats()
        
        logger.info("Recall system cleanup complete")
    
    def _show_final_stats(self):
        """Show final performance statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        logger.info("=" * 50)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Total matches found: {len(self.matches_history)}")
        
        # Show pose matcher stats
        matcher_stats = self.pose_matcher.get_performance_stats()
        if "avg_match_time" in matcher_stats:
            logger.info(f"Average match time: {matcher_stats['avg_match_time']:.3f}s")
            logger.info(f"Total matches performed: {matcher_stats['total_matches']}")
        
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
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "current_fps": self.current_fps,
            "average_fps": avg_fps,
            "total_matches": len(self.matches_history),
            "current_pose": self.current_pose is not None,
            "paused": self.paused,
            "matcher_stats": self.pose_matcher.get_performance_stats()
        }
    
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
    
    def __init__(self, config: RecallConfig):
        super().__init__(config)
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


def create_recall_system(config: RecallConfig, with_keyboard: bool = True) -> RecallSystem:
    """Create a recall system instance"""
    if with_keyboard:
        return RecallSystemWithKeyboard(config)
    else:
        return RecallSystem(config) 