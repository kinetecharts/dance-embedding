"""Comprehensive 3D visualization using Rerun."""

import numpy as np
import time
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .data_structures import PoseData, NormalizedPose, Match, get_pose_connections
from .config import RecallConfig

logger = logging.getLogger(__name__)


class RerunVisualizer:
    """Comprehensive 3D visualization using Rerun"""
    
    def __init__(self, config: RecallConfig):
        self.config = config
        self.rr = None
        self.last_update = 0
        self.update_interval = 1.0 / config.rerun_max_fps
        
        if config.use_rerun:
            try:
                import rerun as rr
                self.rr = rr
                # Initialize Rerun without port parameter (it's not supported in this version)
                self.rr.init("recall-system", spawn=config.rerun_spawn)
                self.setup_visualization()
                logger.info("Initialized Rerun visualization")
            except ImportError:
                logger.warning("Rerun not available, visualization disabled")
        else:
            logger.info("Rerun visualization disabled")
    
    def setup_visualization(self):
        """Setup Rerun visualization layout and components"""
        if self.rr is None:
            return
        
        try:
            # Set coordinate system
            self.rr.log("view", self.rr.ViewCoordinates.RIGHT_HAND_Y_UP)
            
            # Setup layout based on configuration
            if self.config.visualization_layout == "multi_view":
                self._setup_multi_view_layout()
            elif self.config.visualization_layout == "side_by_side":
                self._setup_side_by_side_layout()
            else:  # single_view
                self._setup_single_view_layout()
            
            # Setup common components
            self._setup_common_components()
            
        except Exception as e:
            logger.error(f"Error setting up Rerun visualization: {e}")
    
    def _setup_multi_view_layout(self):
        """Setup multi-view layout with separate spaces"""
        # Live pose view
        self.rr.log("live_pose", self.rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[255, 0, 0]]
        ))
        
        # Matches view
        self.rr.log("matches", self.rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[0, 255, 0]]
        ))
        
        # Playback view
        self.rr.log("playback", self.rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[0, 0, 255]]
        ))
    
    def _setup_side_by_side_layout(self):
        """Setup side-by-side comparison layout"""
        # Live pose (left side)
        self.rr.log("comparison/live", self.rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[-1.5, 0, 0]],
            colors=[[255, 0, 0]]
        ))
        
        # Matched pose (right side)
        self.rr.log("comparison/matched", self.rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[1.5, 0, 0]],
            colors=[[0, 0, 255]]
        ))
    
    def _setup_single_view_layout(self):
        """Setup single view layout"""
        self.rr.log("single_view", self.rr.Boxes3D(
            half_sizes=[[0.1, 0.1, 0.1]],
            centers=[[0, 0, 0]],
            colors=[[255, 255, 255]]
        ))
    
    def _setup_common_components(self):
        """Setup common visualization components"""
        # Metrics display
        self.rr.log("metrics", self.rr.TextLog("System starting..."))
        
        # Status display
        self.rr.log("status", self.rr.TextLog("Ready"))
        
        # Similarity scores
        self.rr.log("similarity_scores", self.rr.TextLog("No matches yet"))
    
    def visualize_live_pose(self, pose: PoseData):
        """Visualize live pose in 3D"""
        if self.rr is None or not self._should_update():
            return
        
        try:
            entity_path = self._get_entity_path("live_pose")
            
            # Visualize landmarks
            self.rr.log(f"{entity_path}/landmarks", self.rr.Points3D(
                positions=pose.landmarks,
                colors=[[255, 0, 0]] * len(pose.landmarks),
                radii=[0.02] * len(pose.landmarks)
            ))
            
            # Visualize skeleton
            self._visualize_skeleton(pose.landmarks, f"{entity_path}/skeleton", [255, 0, 0])
            
        except Exception as e:
            logger.error(f"Error visualizing live pose: {e}")
    
    def visualize_matches(self, live_pose: PoseData, matches: List[Match]):
        """Visualize live pose and matched poses"""
        if self.rr is None or not self._should_update():
            return
        
        try:
            # Clear previous matches
            entity_path = self._get_entity_path("matches")
            self.rr.log(entity_path, self.rr.Clear())
            
            # Visualize live pose
            self.visualize_live_pose(live_pose)
            
            # Visualize matched poses with different colors
            colors = [[0, 255, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255]]
            for i, match in enumerate(matches):
                color = colors[i % len(colors)]
                match_path = f"{entity_path}/match_{i}"
                
                # Visualize matched pose
                self.rr.log(f"{match_path}/landmarks", self.rr.Points3D(
                    positions=match.normalized_pose.coordinates,
                    colors=[color] * len(match.normalized_pose.coordinates),
                    radii=[0.015] * len(match.normalized_pose.coordinates)
                ))
                
                # Visualize skeleton for matched pose
                self._visualize_skeleton(match.normalized_pose.coordinates, f"{match_path}/skeleton", color)
            
            # Show similarity scores
            self.show_similarity_scores([m.similarity_score for m in matches])
            
        except Exception as e:
            logger.error(f"Error visualizing matches: {e}")
    
    def visualize_playback_pose(self, pose: PoseData, video_name: str):
        """Visualize video playback pose"""
        if self.rr is None or not self._should_update():
            return
        
        try:
            entity_path = self._get_entity_path("playback")
            playback_path = f"{entity_path}/{video_name}"
            
            # Visualize playback pose
            self.rr.log(f"{playback_path}/landmarks", self.rr.Points3D(
                positions=pose.landmarks,
                colors=[[0, 0, 255]] * len(pose.landmarks),
                radii=[0.02] * len(pose.landmarks)
            ))
            
            # Visualize skeleton for playback pose
            self._visualize_skeleton(pose.landmarks, f"{playback_path}/skeleton", [0, 0, 255])
            
        except Exception as e:
            logger.error(f"Error visualizing playback pose: {e}")
    
    def _visualize_skeleton(self, landmarks: np.ndarray, entity_path: str, color: List[int]):
        """Visualize pose skeleton with connections"""
        if self.rr is None:
            return
        
        try:
            connections = get_pose_connections()
            
            for connection in connections:
                start_idx = connection.start_idx
                end_idx = connection.end_idx
                
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_pos = landmarks[start_idx]
                    end_pos = landmarks[end_idx]
                    
                    self.rr.log(f"{entity_path}/{connection.name}", self.rr.LineStrips3D(
                        positions=[[start_pos, end_pos]],
                        colors=[color]
                    ))
                    
        except Exception as e:
            logger.error(f"Error visualizing skeleton: {e}")
    
    def show_similarity_scores(self, scores: List[float]):
        """Display similarity scores as text"""
        if self.rr is None:
            return
        
        try:
            score_text = " | ".join([f"Match {i+1}: {score:.3f}" for i, score in enumerate(scores)])
            self.rr.log("similarity_scores", self.rr.TextLog(score_text))
        except Exception as e:
            logger.error(f"Error showing similarity scores: {e}")
    
    def show_system_status(self, status: str):
        """Display system status"""
        if self.rr is None:
            return
        
        try:
            self.rr.log("status", self.rr.TextLog(status))
        except Exception as e:
            logger.error(f"Error showing system status: {e}")
    
    def show_metrics(self, similarity_scores: List[float], fps: float, match_count: int):
        """Display real-time metrics"""
        if self.rr is None:
            return
        
        try:
            metrics_text = f"FPS: {fps:.1f} | Matches: {match_count} | Scores: {[f'{s:.3f}' for s in similarity_scores]}"
            self.rr.log("metrics", self.rr.TextLog(metrics_text))
        except Exception as e:
            logger.error(f"Error showing metrics: {e}")
    
    def clear_playback(self):
        """Clear playback visualizations"""
        if self.rr is None:
            return
        
        try:
            entity_path = self._get_entity_path("playback")
            self.rr.log(entity_path, self.rr.Clear())
        except Exception as e:
            logger.error(f"Error clearing playback: {e}")
    
    def clear_all(self):
        """Clear all visualizations"""
        if self.rr is None:
            return
        
        try:
            self.rr.log("", self.rr.Clear())
        except Exception as e:
            logger.error(f"Error clearing all visualizations: {e}")
    
    def _get_entity_path(self, component: str) -> str:
        """Get entity path based on visualization layout"""
        if self.config.visualization_layout == "single_view":
            return f"single_view/{component}"
        elif self.config.visualization_layout == "side_by_side":
            return f"comparison/{component}"
        else:  # multi_view
            return component
    
    def _should_update(self) -> bool:
        """Check if we should update visualization based on FPS limit"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False
    
    def toggle_view_mode(self):
        """Toggle between visualization layouts"""
        if self.rr is None:
            return
        
        layouts = ["single_view", "multi_view", "side_by_side"]
        current_idx = layouts.index(self.config.visualization_layout)
        next_idx = (current_idx + 1) % len(layouts)
        
        self.config.visualization_layout = layouts[next_idx]
        self.setup_visualization()
        logger.info(f"Switched to {self.config.visualization_layout} layout")


class OptimizedRerunVisualizer(RerunVisualizer):
    """Optimized Rerun visualization for real-time performance"""
    
    def __init__(self, config: RecallConfig):
        super().__init__(config)
        self.pose_buffer = []
        self.buffer_size = 5
    
    def buffer_pose(self, pose: PoseData, entity_path: str):
        """Buffer pose for batch update"""
        self.pose_buffer.append((pose, entity_path))
        
        if len(self.pose_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Flush buffered poses to Rerun"""
        if not self.pose_buffer or self.rr is None:
            return
        
        try:
            # Batch update all poses
            for pose, entity_path in self.pose_buffer:
                self.rr.log(f"{entity_path}/landmarks", self.rr.Points3D(
                    positions=pose.landmarks,
                    colors=[[255, 0, 0]] * len(pose.landmarks),
                    radii=[0.02] * len(pose.landmarks)
                ))
            
            self.pose_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing pose buffer: {e}")


def create_rerun_visualizer(config: RecallConfig, optimized: bool = False) -> RerunVisualizer:
    """Create Rerun visualizer with optional optimization"""
    if optimized:
        return OptimizedRerunVisualizer(config)
    else:
        return RerunVisualizer(config) 