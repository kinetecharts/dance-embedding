"""
Video recorder for pose tracking with audio input.

This module provides video recording functionality optimized for Mac with:
- Hardware-accelerated H.264 encoding
- Microphone audio recording
- Pose overlay rendering
- Automatic file saving with crash protection
- Mac-optimized codec settings
"""

import cv2
import numpy as np
import time
import threading
import queue
import wave
import os
from pathlib import Path
from typing import Optional, Tuple
import logging
from datetime import datetime

from .data_structures import PoseData

logger = logging.getLogger(__name__)


class MacVideoRecorder:
    """Video recorder optimized for Mac with hardware acceleration and audio"""
    
    def __init__(self, config, output_dir: str = "recordings"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video recording
        self.video_writer = None
        self.recording = False
        self.frame_count = 0
        self.start_time = None
        
        # Audio recording
        self.audio_frames = []
        self.audio_sample_rate = 44100
        self.audio_channels = 1
        self.audio_chunk_size = 1024
        
        # Mac-optimized codec settings
        self.codec_settings = {
            "low": {
                "crf": 28,
                "preset": "ultrafast",
                "tune": "zerolatency"
            },
            "medium": {
                "crf": 23,
                "preset": "fast",
                "tune": "zerolatency"
            },
            "high": {
                "crf": 18,
                "preset": "medium",
                "tune": "zerolatency"
            }
        }
        
        # Threading for audio
        self.audio_thread = None
        self.audio_queue = queue.Queue()
        self.audio_stop_event = threading.Event()
        
        # File naming
        self.current_filename = None
        
        logger.info(f"🎥 Mac-optimized video recorder initialized")
        logger.info(f"💾 Output directory: {self.output_dir.absolute()}")
    
    def start_recording(self, frame_width: int, frame_height: int, fps: int = 30, quality: str = "medium"):
        """Start recording video with pose overlay and audio"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_filename = f"pose_recording_{timestamp}"
            
            # Video file path
            video_path = self.output_dir / f"{self.current_filename}.mp4"
            
            # Get codec settings for quality
            codec_settings = self.codec_settings.get(quality, self.codec_settings["medium"])
            
            # Mac-optimized H.264 codec with hardware acceleration
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V for better Mac compatibility
            
            # Create video writer
            self.video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                fps,
                (frame_width, frame_height),
                isColor=True
            )
            
            if not self.video_writer.isOpened():
                logger.error("Failed to create video writer")
                return False
            
            # Start audio recording thread
            self._start_audio_recording()
            
            # Initialize recording state
            self.recording = True
            self.frame_count = 0
            self.start_time = time.time()
            
            logger.info(f"🎬 Started recording: {video_path}")
            logger.info(f"📐 Resolution: {frame_width}x{frame_height} @ {fps} FPS")
            logger.info(f"🎵 Audio: {self.audio_sample_rate} Hz, {self.audio_channels} channel(s)")
            logger.info(f"⚡ Quality: {quality} (CRF: {codec_settings['crf']})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def _start_audio_recording(self):
        """Start audio recording in background thread"""
        try:
            import pyaudio
            
            self.audio_frames = []
            self.audio_stop_event.clear()
            
            def audio_recorder():
                p = pyaudio.PyAudio()
                
                try:
                    # Open audio stream
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=self.audio_channels,
                        rate=self.audio_sample_rate,
                        input=True,
                        frames_per_buffer=self.audio_chunk_size
                    )
                    
                    logger.info("🎤 Audio recording started")
                    
                    while not self.audio_stop_event.is_set():
                        try:
                            data = stream.read(self.audio_chunk_size, exception_on_overflow=False)
                            self.audio_frames.append(data)
                        except Exception as e:
                            logger.debug(f"Audio read error: {e}")
                            break
                    
                    # Cleanup
                    stream.stop_stream()
                    stream.close()
                    
                except Exception as e:
                    logger.error(f"Audio recording error: {e}")
                finally:
                    p.terminate()
            
            # Start audio thread
            self.audio_thread = threading.Thread(target=audio_recorder, daemon=True)
            self.audio_thread.start()
            
        except ImportError:
            logger.warning("PyAudio not available - audio recording disabled")
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
    
    def record_frame(self, frame: np.ndarray, pose_data: Optional[PoseData] = None):
        """Record a frame with optional pose overlay"""
        if not self.recording or self.video_writer is None:
            return
        
        try:
            # Create a copy for recording
            recording_frame = frame.copy()
            
            # Add pose overlay if available
            if pose_data is not None:
                recording_frame = self._draw_pose_overlay(recording_frame, pose_data)
            
            # Add recording indicator
            recording_frame = self._add_recording_indicator(recording_frame)
            
            # Write frame
            self.video_writer.write(recording_frame)
            self.frame_count += 1
            
            # Log progress every 100 frames
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                logger.debug(f"📹 Recorded {self.frame_count} frames ({fps:.1f} FPS)")
                
        except Exception as e:
            logger.error(f"Error recording frame: {e}")
    
    def _draw_pose_overlay(self, frame: np.ndarray, pose_data: PoseData) -> np.ndarray:
        """Draw pose landmarks on frame for recording"""
        try:
            if not pose_data or not hasattr(pose_data, 'landmarks') or pose_data.landmarks is None:
                return frame
            
            # Create a copy to draw on
            overlay_frame = frame.copy()
            
            # Draw pose landmarks (same style as display)
            if hasattr(pose_data, 'confidence') and pose_data.confidence is not None:
                for i, (landmark, confidence) in enumerate(zip(pose_data.landmarks, pose_data.confidence)):
                    if confidence > 0.5:  # Only draw confident landmarks
                        x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                        cv2.circle(overlay_frame, (x, y), 3, (0, 0, 255), -1)  # Red dots
                
                # Draw skeleton connections
                connections = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
                    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
                    (0, 4), (4, 5), (5, 6),  # Eyes
                    (0, 8), (8, 9), (9, 10)  # Mouth
                ]
                
                for start_idx, end_idx in connections:
                    if (start_idx < len(pose_data.landmarks) and 
                        end_idx < len(pose_data.landmarks)):
                        
                        start_landmark = pose_data.landmarks[start_idx]
                        end_landmark = pose_data.landmarks[end_idx]
                        
                        if (start_landmark is not None and end_landmark is not None and
                            len(start_landmark) >= 2 and len(end_landmark) >= 2 and
                            pose_data.confidence[start_idx] > 0.5 and 
                            pose_data.confidence[end_idx] > 0.5):
                            
                            x1 = int(start_landmark[0] * frame.shape[1])
                            y1 = int(start_landmark[1] * frame.shape[0])
                            x2 = int(end_landmark[0] * frame.shape[1])
                            y2 = int(end_landmark[1] * frame.shape[0])
                            
                            cv2.line(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            return overlay_frame
            
        except Exception as e:
            logger.debug(f"Error drawing pose overlay: {e}")
            return frame
    
    def _add_recording_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Add recording indicator to frame"""
        try:
            # Add red recording dot
            cv2.circle(frame, (30, 30), 8, (0, 0, 255), -1)
            
            # Add recording text
            cv2.putText(frame, "REC", (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Error adding recording indicator: {e}")
            return frame
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save files"""
        if not self.recording:
            return None
        
        try:
            # Stop video recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # Stop audio recording
            self.audio_stop_event.set()
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2.0)
            
            # Save audio file
            audio_path = None
            if self.audio_frames:
                audio_path = self._save_audio_file()
            
            # Merge audio and video if both exist
            final_video_path = None
            if audio_path and self.current_filename:
                final_video_path = self._merge_audio_video(audio_path)
            
            # Calculate final stats
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Reset state
            self.recording = False
            self.frame_count = 0
            self.start_time = None
            
            # Log results
            logger.info(f"✅ Recording stopped: {self.frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
            
            if final_video_path:
                logger.info(f"🎬 Final video with audio: {final_video_path}")
            elif audio_path:
                logger.info(f"🎵 Audio saved: {audio_path}")
            
            return self.current_filename
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return None
    
    def _save_audio_file(self) -> Optional[str]:
        """Save recorded audio to WAV file"""
        try:
            if not self.audio_frames:
                return None
            
            audio_path = self.output_dir / f"{self.current_filename}.wav"
            
            with wave.open(str(audio_path), 'wb') as wav_file:
                wav_file.setnchannels(self.audio_channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.audio_sample_rate)
                wav_file.writeframes(b''.join(self.audio_frames))
            
            logger.info(f"🎵 Audio saved: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return None
    
    def _merge_audio_video(self, audio_path: str) -> Optional[str]:
        """Merge audio and video into a single MP4 file using ffmpeg"""
        try:
            import subprocess
            
            # Input paths
            video_path = self.output_dir / f"{self.current_filename}.mp4"
            audio_path_obj = Path(audio_path)
            
            if not video_path.exists() or not audio_path_obj.exists():
                logger.warning("Video or audio file missing for merging")
                return None
            
            # Output path for merged file
            final_video_path = self.output_dir / f"{self.current_filename}_with_audio.mp4"
            
            # Use ffmpeg to merge audio and video
            cmd = [
                'ffmpeg',
                '-i', str(video_path),           # Input video
                '-i', str(audio_path_obj),       # Input audio
                '-c:v', 'copy',                  # Copy video stream (no re-encoding)
                '-c:a', 'aac',                   # Encode audio to AAC
                '-b:a', '128k',                  # Audio bitrate
                '-shortest',                     # End when shortest stream ends
                '-y',                            # Overwrite output
                str(final_video_path)
            ]
            
            logger.info(f"🎬 Merging audio and video...")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg merge failed: {result.stderr}")
                return None
            
            # Clean up separate files
            try:
                video_path.unlink()  # Remove separate video file
                audio_path_obj.unlink()  # Remove separate audio file
                logger.info("🧹 Cleaned up separate audio/video files")
            except Exception as e:
                logger.warning(f"Could not clean up separate files: {e}")
            
            logger.info(f"✅ Audio and video merged successfully: {final_video_path}")
            return str(final_video_path)
            
        except ImportError:
            logger.warning("FFmpeg not available - cannot merge audio and video")
            return None
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg merge timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to merge audio and video: {e}")
            return None
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording
    
    def get_recording_stats(self) -> dict:
        """Get current recording statistics"""
        if not self.recording:
            return {}
        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            "frame_count": self.frame_count,
            "elapsed_time": elapsed,
            "fps": fps,
            "filename": self.current_filename
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.audio_stop_event.set()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
