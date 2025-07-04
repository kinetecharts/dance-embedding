"""Pose extraction module using Rerun and MediaPipe."""

from __future__ import annotations

import csv
import logging
import os
import time
import urllib.request
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
import numpy.typing as npt
import rerun as rr
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
POSE_KEYPOINTS: Final = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# MediaPipe pose model URL
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# Pose connections for drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # Hands
    (27, 31), (28, 32),  # Feet
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),  # Mouth
]


@dataclass
class VideoFrame:
    """Represents a video frame with metadata."""
    data: cv2.typing.MatLike
    time: float
    idx: int


@dataclass
class PoseData:
    """Represents pose data for a single frame."""
    timestamp: float
    frame_number: int
    landmarks_2d: npt.NDArray[np.float32] | None
    landmarks_3d: npt.NDArray[np.float32] | None
    confidence_scores: list[float] | None


class VideoSource:
    """Handles video input from files or camera."""
    
    def __init__(self, path: str, use_camera: bool = False) -> None:
        if use_camera:
            self.capture = cv2.VideoCapture(0)
            self.is_camera = True
        else:
            self.capture = cv2.VideoCapture(path)
            self.is_camera = False

        self.start_time = time.time() if self.is_camera else None

        if not self.capture.isOpened():
            raise RuntimeError(f"Couldn't open video at {path}")

    def close(self) -> None:
        """Release the video capture."""
        self.capture.release()

    def stream_bgr(self) -> Iterator[VideoFrame]:
        """Stream video frames in BGR format."""
        while self.capture.isOpened():
            idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            is_open, bgr = self.capture.read()

            # Use proper timestamps for camera, fallback to video timestamps for files
            if self.is_camera:
                time_ms = (time.time() - self.start_time) * 1000
            else:
                time_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)

            if not is_open:
                break

            yield VideoFrame(data=bgr, time=time_ms * 1e-3, idx=idx)


class PoseExtractor:
    """Extracts pose data from videos using MediaPipe and Rerun."""
    
    def __init__(self, model_path: str | None = None, use_rerun: bool = False, generate_overlay_video: bool = True):
        """Initialize the pose extractor.
        
        Args:
            model_path: Path to MediaPipe model file. If None, uses default.
            use_rerun: Whether to use Rerun for visualization.
            generate_overlay_video: Whether to generate a video with pose overlays.
        """
        self.use_rerun = use_rerun
        self.generate_overlay_video = generate_overlay_video
        
        # Get model path
        if model_path is None:
            model_path = self._get_default_model_path()
        
        # Initialize MediaPipe pose landmarker
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=False,  # We don't need segmentation for CSV export
        )
        
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        
        # Initialize Rerun if requested
        if self.use_rerun:
            self._setup_rerun()
    
    def _get_default_model_path(self) -> str:
        """Get the default MediaPipe pose model path, downloading if necessary."""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "pose_landmarker_heavy.task"
        
        if not model_path.exists():
            logger.info("Downloading MediaPipe pose model...")
            try:
                urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
                logger.info(f"Model downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise RuntimeError("Could not download MediaPipe pose model")
        
        return str(model_path)
    
    def _setup_rerun(self) -> None:
        """Setup Rerun for visualization."""
        rr.init("pose_extraction", spawn=True)
        
        # Log annotation context
        rr.log(
            "/",
            rr.AnnotationContext(
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="Person"),
                    keypoint_annotations=[rr.AnnotationInfo(id=lm.value, label=lm.name) for lm in mp_pose.PoseLandmark],
                    keypoint_connections=mp_pose.POSE_CONNECTIONS,
                ),
            ),
            static=True,
        )
        rr.log("person", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    def _draw_pose_overlay(self, frame: np.ndarray, landmarks_2d: npt.NDArray[np.float32] | None, 
                          confidence_scores: list[float] | None) -> np.ndarray:
        """Draw pose landmarks and connections on the frame."""
        if landmarks_2d is None:
            return frame
        
        overlay = frame.copy()
        
        # Draw connections
        for connection in POSE_CONNECTIONS:
            if (connection[0] < len(landmarks_2d) and connection[1] < len(landmarks_2d) and
                confidence_scores and confidence_scores[connection[0]] > 0.5 and confidence_scores[connection[1]] > 0.5):
                
                pt1 = (int(landmarks_2d[connection[0]][0]), int(landmarks_2d[connection[0]][1]))
                pt2 = (int(landmarks_2d[connection[1]][0]), int(landmarks_2d[connection[1]][1]))
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks_2d):
            if confidence_scores and confidence_scores[i] > 0.5:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)  # smaller filled
                cv2.circle(overlay, (x, y), 3, (255, 255, 255), 1)  # smaller outline
        
        # Add frame info
        cv2.putText(overlay, f"Frame: {len(landmarks_2d)} landmarks", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay
    
    def extract_pose_from_video(self, video_path: str, output_path: str | None = None, 
                               overlay_video_path: str | None = None) -> list[PoseData]:
        """Extract pose data from a video file.
        
        Args:
            video_path: Path to the input video file.
            output_path: Path to save the CSV file. If None, auto-generates based on video path.
            overlay_video_path: Path to save the overlay video. If None, auto-generates.
            
        Returns:
            List of pose data for each frame.
        """
        if output_path is None:
            # Generate output path based on video path
            video_name = Path(video_path).stem
            output_path = f"data/poses/{video_name}.csv"
        
        if overlay_video_path is None and self.generate_overlay_video:
            # Generate overlay video path
            video_name = Path(video_path).stem
            overlay_video_path = f"data/video_with_pose/{video_name}_with_pose.mp4"
        
        # Ensure output directories exist
        os.makedirs(Path(output_path).parent, exist_ok=True)
        if overlay_video_path:
            os.makedirs(Path(overlay_video_path).parent, exist_ok=True)
        
        pose_data_list = []
        
        # Re-initialize pose_landmarker for each video
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self._get_default_model_path(),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        
        # Setup video writer for overlay video
        video_writer = None
        if overlay_video_path:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
        
        with closing(VideoSource(video_path)) as video_source:
            # Count total frames and get fps for progress bar
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_iter = video_source.stream_bgr()
            last_time = -1
            
            for frame in tqdm(frame_iter, total=total_frames, desc=f"Frames ({Path(video_path).name})", leave=False):
                # Use embedded timecode, but force monotonicity if needed
                if frame.time <= last_time:
                    frame.time = last_time + 1e-6
                last_time = frame.time
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.data)
                if self.use_rerun:
                    rr.set_time("time", duration=frame.time)
                    rr.set_time("frame_idx", sequence=frame.idx)
                
                results = pose_landmarker.detect_for_video(mp_image, int(frame.time * 1000))
                h, w, _ = frame.data.shape
                landmarks_2d = self._read_landmark_positions_2d(results, w, h)
                landmarks_3d = self._read_landmark_positions_3d(results)
                confidence_scores = self._extract_confidence_scores(results)
                
                pose_data = PoseData(
                    timestamp=frame.time,
                    frame_number=frame.idx,
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    confidence_scores=confidence_scores
                )
                pose_data_list.append(pose_data)
                
                # Generate overlay video frame
                if video_writer and landmarks_2d is not None:
                    overlay_frame = self._draw_pose_overlay(frame.data, landmarks_2d, confidence_scores)
                    video_writer.write(overlay_frame)
                
                if self.use_rerun:
                    self._log_to_rerun(frame, landmarks_2d, landmarks_3d)
        
        # Close video writer
        if video_writer:
            video_writer.release()
            logger.info(f"Generated overlay video: {overlay_video_path}")
        
        # Export to CSV
        self._export_to_csv(pose_data_list, output_path)
        
        logger.info(f"Extracted pose data from {len(pose_data_list)} frames")
        logger.info(f"Saved pose data to {output_path}")
        
        return pose_data_list
    
    def _read_landmark_positions_2d(
        self,
        results: Any,
        image_width: int,
        image_height: int,
    ) -> npt.NDArray[np.float32] | None:
        """Extract 2D landmark positions from MediaPipe results."""
        if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
            return None
        
        pose_landmarks = results.pose_landmarks[0]
        normalized_landmarks = [pose_landmarks[lm] for lm in mp_pose.PoseLandmark]
        return np.array([(image_width * lm.x, image_height * lm.y) for lm in normalized_landmarks])
    
    def _read_landmark_positions_3d(self, results: Any) -> npt.NDArray[np.float32] | None:
        """Extract 3D landmark positions from MediaPipe results."""
        if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
            return None
        
        pose_landmarks = results.pose_landmarks[0]
        landmarks = [pose_landmarks[lm] for lm in mp_pose.PoseLandmark]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    
    def _extract_confidence_scores(self, results: Any) -> list[float] | None:
        """Extract confidence scores from MediaPipe results."""
        if results.pose_landmarks is None or len(results.pose_landmarks) == 0:
            return None
        
        pose_landmarks = results.pose_landmarks[0]
        return [lm.visibility for lm in pose_landmarks]
    
    def _log_to_rerun(self, frame: VideoFrame, landmarks_2d: npt.NDArray[np.float32] | None, 
                     landmarks_3d: npt.NDArray[np.float32] | None) -> None:
        """Log pose data to Rerun for visualization."""
        if not self.use_rerun:
            return
        
        # Log video frame
        rr.log("video/bgr", rr.Image(frame.data, color_model="BGR").compress(jpeg_quality=75))
        
        # Log 2D pose
        if landmarks_2d is not None:
            rr.log(
                "video/pose/points",
                rr.Points2D(landmarks_2d, class_ids=1, keypoint_ids=mp_pose.PoseLandmark),
            )
        
        # Log 3D pose
        if landmarks_3d is not None:
            rr.log(
                "person/pose/points",
                rr.Points3D(landmarks_3d, class_ids=1, keypoint_ids=mp_pose.PoseLandmark),
            )
    
    def _export_to_csv(self, pose_data_list: list[PoseData], output_path: str) -> None:
        """Export pose data to CSV format."""
        if not pose_data_list:
            logger.warning("No pose data to export")
            return
        
        # Define CSV columns
        columns = ['timestamp', 'frame_number']
        
        # Add 2D coordinates
        for keypoint in POSE_KEYPOINTS:
            columns.extend([f"{keypoint}_x", f"{keypoint}_y"])
        
        # Add 3D coordinates
        for keypoint in POSE_KEYPOINTS:
            columns.extend([f"{keypoint}_z"])
        
        # Add confidence scores
        for keypoint in POSE_KEYPOINTS:
            columns.append(f"{keypoint}_confidence")
        
        # Write CSV file
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for pose_data in pose_data_list:
                row = {
                    'timestamp': pose_data.timestamp,
                    'frame_number': pose_data.frame_number,
                }
                
                # Add 2D coordinates
                if pose_data.landmarks_2d is not None:
                    for i, keypoint in enumerate(POSE_KEYPOINTS):
                        row[f"{keypoint}_x"] = pose_data.landmarks_2d[i, 0]
                        row[f"{keypoint}_y"] = pose_data.landmarks_2d[i, 1]
                else:
                    for keypoint in POSE_KEYPOINTS:
                        row[f"{keypoint}_x"] = None
                        row[f"{keypoint}_y"] = None
                
                # Add 3D coordinates
                if pose_data.landmarks_3d is not None:
                    for i, keypoint in enumerate(POSE_KEYPOINTS):
                        row[f"{keypoint}_z"] = pose_data.landmarks_3d[i, 2]
                else:
                    for keypoint in POSE_KEYPOINTS:
                        row[f"{keypoint}_z"] = None
                
                # Add confidence scores
                if pose_data.confidence_scores is not None:
                    for i, keypoint in enumerate(POSE_KEYPOINTS):
                        row[f"{keypoint}_confidence"] = pose_data.confidence_scores[i]
                else:
                    for keypoint in POSE_KEYPOINTS:
                        row[f"{keypoint}_confidence"] = None
                
                writer.writerow(row)
    
    def process_video_directory(self, input_dir: str = "data/video", output_dir: str = "data/poses") -> None:
        """Process all videos in a directory.
        
        Args:
            input_dir: Directory containing video files.
            output_dir: Directory to save pose CSV files.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory {input_dir} does not exist")
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported video formats
        video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
        
        video_files = [f for f in input_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        for video_file in video_files:
            logger.info(f"Processing {video_file.name}")
            try:
                output_file = output_path / f"{video_file.stem}.csv"
                self.extract_pose_from_video(str(video_file), str(output_file))
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
                continue
        
        logger.info("Finished processing all videos")


# Remove the main function and CLI - this should be a pure module
# The CLI is handled in main.py 