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
from .pose_matcher import create_pose_matcher
from .video_player import create_video_player

logger = logging.getLogger(__name__)


class RecallSystem:
    """Main system orchestrator for live pose matching and video playback with dual-window display"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.running = True
        self.paused = False
        
        # Initialize components
        self.pose_tracker = PoseTracker(config)
        self.pose_matcher = create_pose_matcher(config, use_cache=True)
        self.video_player = create_video_player(config, with_controls=True)
        
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
        
        logger.info("Recall system initialized with dual-window display")
    
    def run_live(self):
        """Main live processing loop with dual-window display"""
        logger.info("Starting live camera mode with dual-window display")
        
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
                    self.video_player.display_live_frame(frame)
                    continue
                
                self.current_pose = pose_data
                self.frame_count += 1
                
                # Check if it's time to match (every 2 seconds)
                current_time = time.time()
                if current_time - self.last_match_time >= self.config.match_interval:
                    logger.info(f"🎯 Performing match at {current_time:.1f}s")
                    self._perform_matching(pose_data)
                    self.last_match_time = current_time
                
                # Display live frame with current match info
                current_match = self.video_player.current_match
                self.video_player.display_live_frame(frame, pose_data, current_match)
                
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
        logger.info(f"Starting video mode with dual-window display: {video_path}")
        
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
                    self.video_player.display_live_frame(frame)
                    continue
                
                self.current_pose = pose_data
                self.frame_count += 1
                
                # Check if it's time to match (every 2 seconds)
                current_time = time.time()
                if current_time - self.last_match_time >= self.config.match_interval:
                    self._perform_matching(pose_data)
                    self.last_match_time = current_time
                
                # Display live frame with current match info
                current_match = self.video_player.current_match
                self.video_player.display_live_frame(frame, pose_data, current_match)
                
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
            # Find matches
            matches = self.pose_matcher.find_matches(pose, self.config.top_n)
            
            if not matches:
                logger.warning("No matches found")
                return
            
            logger.info(f"Found {len(matches)} matches before randomization")
            for i, match in enumerate(matches[:3]):  # Show top 3
                video_name = Path(match.video_file).stem
                logger.info(f"  Top match {i+1}: {video_name} at {match.timestamp:.2f}s (score: {match.similarity_score:.3f})")
            
            # Randomly select from top matches
            selected_matches = self.pose_matcher.random_select(matches, 1)  # Play one match at a time
            self.current_matches = selected_matches
            
            # Store in history
            self.matches_history.append(selected_matches)
            
            # Play matched video
            for match in selected_matches:
                self.video_player.play_match(match)
                # Display matched pose in separate window
                self.video_player.display_matched_pose(match, pose)
            
            # Log matches to terminal
            logger.info(f"🎯 Selected {len(selected_matches)} matches:")
            for i, match in enumerate(selected_matches):
                video_name = Path(match.video_file).stem
                logger.info(f"  Selected match {i+1}: {video_name} at {match.timestamp:.2f}s (score: {match.similarity_score:.3f})")
            
        except Exception as e:
            logger.error(f"Error performing matching: {e}")
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _show_metrics(self):
        """Show real-time metrics in terminal"""
        if self.frame_count % 60 == 0:  # Show every 60 frames (2 seconds at 30 FPS)
            similarity_scores = [m.similarity_score for m in self.current_matches]
            current_match = self.video_player.current_match
            match_status = "Playing" if current_match else "Idle"
            
            logger.info(f"📊 Frame: {self.frame_count}, FPS: {self.current_fps:.1f}, "
                       f"Status: {match_status}, "
                       f"Matches: {len(self.current_matches)}, "
                       f"Scores: {[f'{s:.3f}' for s in similarity_scores]}")
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources")
        
        self.running = False
        self.pose_tracker.release()
        self.video_player.stop_all()
        
        # Log final statistics
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"📈 Processing complete: {self.frame_count} frames in {total_time:.2f}s")
            logger.info(f"📈 Average FPS: {self.frame_count / total_time:.1f}")
            logger.info(f"📈 Total matches: {len(self.matches_history)}")
    
    def toggle_pause(self):
        """Toggle pause/resume"""
        self.paused = not self.paused
        status = "PAUSED" if self.paused else "RUNNING"
        logger.info(f"⏸️ System {status.lower()}")
    
    def reset_players(self):
        """Reset video players"""
        self.video_player.stop_all()
        logger.info("🔄 Reset video players")
    
    def set_top_n(self, top_n: int):
        """Set top-N matches"""
        if 1 <= top_n <= 9:
            self.config.top_n = top_n
            logger.info(f"🎯 Set top-N to {top_n}")
    
    def quit(self):
        """Quit the system"""
        logger.info("👋 Quitting system")
        self.running = False
    
    def get_statistics(self) -> dict:
        """Get system statistics"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'frame_count': self.frame_count,
            'total_time': total_time,
            'average_fps': self.frame_count / total_time if total_time > 0 else 0,
            'current_fps': self.current_fps,
            'total_matches': len(self.matches_history),
            'current_matches': len(self.current_matches),
            'playing_videos': len(self.video_player.get_playing_videos()),
            'paused': self.paused
        }
    
    def get_match_history(self) -> List[List[Match]]:
        """Get match history"""
        return self.matches_history.copy()
    
    def clear_history(self):
        """Clear match history"""
        self.matches_history.clear()
        logger.info("🗑️ Cleared match history")
    
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
        self.keyboard_thread = None
        self._setup_keyboard_controls()
    
    def _setup_keyboard_controls(self):
        """Setup keyboard controls in a separate thread"""
        def keyboard_listener():
            try:
                import keyboard
                
                def on_key_press(event):
                    if event.name == 'space':
                        self.toggle_pause()
                    elif event.name == 'r':
                        self.reset_players()
                    elif event.name in '123456789':
                        top_n = int(event.name)
                        self.set_top_n(top_n)
                    elif event.name == 'q':
                        self.quit()
                
                keyboard.on_press(on_key_press)
                keyboard.wait('q')  # Wait for quit key
                
            except ImportError:
                logger.warning("keyboard module not available, keyboard controls disabled")
            except Exception as e:
                logger.error(f"Error setting up keyboard controls: {e}")
        
        self.keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        logger.info("⌨️ Keyboard controls enabled: Space=Pause, R=Reset, 1-9=Top-N, Q=Quit")


def create_recall_system(config: RecallConfig, with_keyboard: bool = True) -> RecallSystem:
    """Create recall system with optional keyboard controls"""
    if with_keyboard:
        return RecallSystemWithKeyboard(config)
    else:
        return RecallSystem(config) 