"""Interactive visualization for dimension reduction results with video synchronization."""

from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .reduction_methods import ReductionMethods

logger = logging.getLogger(__name__)


class DimensionReductionVisualizer:
    """Interactive visualizer for dimension reduction with video synchronization."""
    
    def __init__(self, output_dir: str = "data/analysis/dimension_reduction"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reduction_methods = ReductionMethods()
        self.pose_data = None
        self.video_path = None
        self.reduced_data = None
        self.model = None
        self.timestamps = None
        self.frame_numbers = None
        
    def load_data(self, video_path: str, pose_csv_path: str) -> None:
        """Load video and pose data.
        
        Args:
            video_path: Path to video file
            pose_csv_path: Path to pose CSV file
        """
        # Load pose data
        if not Path(pose_csv_path).exists():
            raise FileNotFoundError(f"Pose CSV file not found: {pose_csv_path}")
        
        self.pose_data = pd.read_csv(pose_csv_path)
        logger.info(f"Loaded pose data: {len(self.pose_data)} frames")
        
        # Validate pose data
        required_columns = ['timestamp', 'frame_number']
        missing_columns = [col for col in required_columns if col not in self.pose_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in pose CSV: {missing_columns}")
        
        # Extract timestamps and frame numbers
        self.timestamps = self.pose_data['timestamp'].values
        self.frame_numbers = self.pose_data['frame_number'].values
        
        # Load video info
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.video_frames / self.video_fps
        cap.release()
        
        logger.info(f"Video info: {self.video_frames} frames, {self.video_fps:.2f} FPS, "
                   f"{self.video_duration:.2f} seconds")
    
    def create_visualization(self, method: str = 'umap', dimensions: str = '2d', 
                           use_3d_coords: bool = False, confidence_threshold: float = 0.5,
                           **kwargs) -> None:
        """Create dimension reduction visualization.
        
        Args:
            method: Reduction method ('pca', 'tsne', 'umap')
            dimensions: Output dimensions ('2d' or '3d')
            use_3d_coords: Whether to use 3D coordinates from pose data
            confidence_threshold: Minimum confidence for keypoints
            **kwargs: Additional method-specific parameters
        """
        if self.pose_data is None:
            raise ValueError("No pose data loaded. Call load_data() first.")
        
        # Determine number of components
        n_components = 3 if dimensions == '3d' else 2
        
        # Preprocess pose data
        logger.info(f"Preprocessing pose data for {method} reduction...")
        features = self.reduction_methods.preprocess_pose_data(
            self.pose_data, 
            use_3d=use_3d_coords, 
            confidence_threshold=confidence_threshold
        )
        
        # Apply dimension reduction
        logger.info(f"Applying {method.upper()} reduction to {n_components}D...")
        self.reduced_data, self.model = self.reduction_methods.reduce_dimensions(
            features, method=method, n_components=n_components, **kwargs
        )
        
        logger.info(f"Reduction complete. Data shape: {self.reduced_data.shape}")
        self.method = method  # Store for output file naming
    
    def create_interactive_plot(self, save_html: bool = True, 
                              show_trajectory: bool = True) -> go.Figure:
        """Create interactive Plotly visualization with video synchronization.
        
        Args:
            save_html: Whether to save as HTML file
            show_trajectory: Whether to show movement trajectory
            
        Returns:
            Plotly figure object
        """
        if self.reduced_data is None:
            raise ValueError("No reduced data available. Call create_visualization() first.")
        
        # Determine if 2D or 3D
        is_3d = self.reduced_data.shape[1] == 3
        
        # Create subplot layout with video frame
        if is_3d:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('3D Pose Movement', 'Video Frame', 'Timeline', ''),
                specs=[[{"type": "scene"}, {"type": "image"}],
                       [{"type": "scatter", "colspan": 2}, None]],
                row_heights=[0.7, 0.3],
                vertical_spacing=0.1
            )
        else:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('2D Pose Movement', 'Video Frame', 'Timeline', ''),
                specs=[[{"type": "scatter"}, {"type": "image"}],
                       [{"type": "scatter", "colspan": 2}, None]],
                row_heights=[0.7, 0.3],
                vertical_spacing=0.1
            )
        
        # Add main pose movement plot
        if is_3d:
            # 3D scatter plot
            fig.add_trace(go.Scatter3d(
                x=self.reduced_data[:, 0],
                y=self.reduced_data[:, 1],
                z=self.reduced_data[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.timestamps,
                    colorscale='viridis',
                    colorbar=dict(title="Time (s)")
                ),
                text=[f"Frame {i}<br>Time: {t:.2f}s" for i, t in zip(self.frame_numbers, self.timestamps)],
                hovertemplate='%{text}<extra></extra>',
                name="Pose Points"
            ), row=1, col=1)
            
            # Add trajectory line
            if show_trajectory:
                fig.add_trace(go.Scatter3d(
                    x=self.reduced_data[:, 0],
                    y=self.reduced_data[:, 1],
                    z=self.reduced_data[:, 2],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name="Movement Trajectory"
                ), row=1, col=1)
        else:
            # 2D scatter plot
            fig.add_trace(go.Scatter(
                x=self.reduced_data[:, 0],
                y=self.reduced_data[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.timestamps,
                    colorscale='viridis',
                    colorbar=dict(title="Time (s)")
                ),
                text=[f"Frame {i}<br>Time: {t:.2f}s" for i, t in zip(self.frame_numbers, self.timestamps)],
                hovertemplate='%{text}<extra></extra>',
                name="Pose Points"
            ), row=1, col=1)
            
            # Add trajectory line
            if show_trajectory:
                fig.add_trace(go.Scatter(
                    x=self.reduced_data[:, 0],
                    y=self.reduced_data[:, 1],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name="Movement Trajectory"
                ), row=1, col=1)
        
        # Add timeline slider
        fig.add_trace(go.Scatter(
            x=self.timestamps,
            y=[0] * len(self.timestamps),
            mode='markers',
            marker=dict(
                size=10,
                color=self.timestamps,
                colorscale='viridis',
                showscale=False
            ),
            text=[f"Frame {i}<br>Time: {t:.2f}s" for i, t in zip(self.frame_numbers, self.timestamps)],
            hovertemplate='%{text}<extra></extra>',
            name="Timeline"
        ), row=2, col=1)
        
        # Add current time indicator
        fig.add_trace(go.Scatter(
            x=[self.timestamps[0]],
            y=[0],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond'
            ),
            name="Current Time",
            showlegend=False
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Pose Movement Visualization - {Path(self.video_path).name}",
            width=1200,
            height=800,
            showlegend=True
        )
        
        # Update subplot layouts
        if is_3d:
            fig.update_scenes(
                row=1, col=1,
                xaxis_title="Component 1",
                yaxis_title="Component 2", 
                zaxis_title="Component 3"
            )
        else:
            fig.update_xaxes(title_text="Component 1", row=1, col=1)
            fig.update_yaxes(title_text="Component 2", row=1, col=1)
        
        # Timeline layout
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        
        # Add video frame placeholder
        fig.add_annotation(
            text="Video Frame<br>(Click timeline to update)",
            xref="x2", yref="y2",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray"),
            bgcolor="lightgray",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Add interactive JavaScript for video synchronization
        fig.add_annotation(
            text="",
            xref="paper", yref="paper",
            x=0, y=0,
            showarrow=False,
            textangle=0,
            font=dict(size=1, color="white"),
            bgcolor="white",
            bordercolor="white",
            borderwidth=0
        )
        
        # Add custom JavaScript for video synchronization
        custom_js = """
        <script>
        // Video synchronization functionality
        function updateVideoFrame(time) {
            // This would be implemented with actual video frame extraction
            console.log('Updating to time:', time);
            
            // Update current time indicator
            var timeline = document.querySelector('.js-plotly-plot');
            if (timeline) {
                // Update the red diamond position
                var traces = timeline.data;
                if (traces.length > 0) {
                    // Find the closest time index
                    var times = traces[0].x;
                    var closestIndex = 0;
                    var minDiff = Math.abs(times[0] - time);
                    
                    for (var i = 1; i < times.length; i++) {
                        var diff = Math.abs(times[i] - time);
                        if (diff < minDiff) {
                            minDiff = diff;
                            closestIndex = i;
                        }
                    }
                    
                    // Update current time indicator
                    if (traces.length > 2) {
                        traces[2].x = [times[closestIndex]];
                        traces[2].y = [0];
                        Plotly.redraw(timeline);
                    }
                }
            }
        }
        
        // Add click handler to timeline
        document.addEventListener('DOMContentLoaded', function() {
            var plot = document.querySelector('.js-plotly-plot');
            if (plot) {
                plot.on('plotly_click', function(data) {
                    if (data.points && data.points.length > 0) {
                        var point = data.points[0];
                        if (point.data.name === 'Timeline') {
                            updateVideoFrame(point.x);
                        }
                    }
                });
            }
        });
        </script>
        """
        
        if save_html:
            video_name = Path(self.video_path).stem
            method = getattr(self, 'method', 'reduction')
            html_path = self.output_dir / f"{video_name}_{method}_interactive_{is_3d and '3d' or '2d'}.html"
            
            # Save with custom JavaScript
            fig.write_html(str(html_path), include_plotlyjs=True)
            
            # Add custom JavaScript to the HTML file
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            # Insert custom JavaScript before closing body tag
            html_content = html_content.replace('</body>', f'{custom_js}\n</body>')
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Interactive plot with video sync saved to {html_path}")
        
        return fig
    
    def create_static_plot(self, save_png: bool = True) -> plt.Figure:
        """Create static matplotlib visualization.
        
        Args:
            save_png: Whether to save as PNG file
            
        Returns:
            Matplotlib figure object
        """
        if self.reduced_data is None:
            raise ValueError("No reduced data available. Call create_visualization() first.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(
            self.reduced_data[:, 0],
            self.reduced_data[:, 1],
            c=self.timestamps,
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (s)')
        
        # Add trajectory line
        ax.plot(self.reduced_data[:, 0], self.reduced_data[:, 1], 
               'r-', linewidth=2, alpha=0.5, label='Movement Trajectory')
        
        # Add frame annotations for key points
        step = max(1, len(self.frame_numbers) // 10)  # Show every 10th frame
        for i in range(0, len(self.frame_numbers), step):
            ax.annotate(f'{self.frame_numbers[i]}', 
                       (self.reduced_data[i, 0], self.reduced_data[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(f'Pose Movement Visualization - {Path(self.video_path).name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_png:
            video_name = Path(self.video_path).stem
            method = getattr(self, 'method', 'reduction')
            png_path = self.output_dir / f"{video_name}_{method}_static_2d.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            logger.info(f"Static plot saved to {png_path}")
        
        return fig
    
    def save_reduced_data(self, save_json: bool = True) -> None:
        """Save reduced data to file.
        
        Args:
            save_json: Whether to save as JSON file
        """
        if self.reduced_data is None:
            raise ValueError("No reduced data available. Call create_visualization() first.")
        
        if save_json:
            video_name = Path(self.video_path).stem
            method = getattr(self, 'method', 'reduction')
            json_path = self.output_dir / f"{video_name}_{method}_reduced_data.json"
            
            data_dict = {
                'reduced_data': self.reduced_data.tolist(),
                'timestamps': self.timestamps.tolist(),
                'frame_numbers': self.frame_numbers.tolist(),
                'video_path': self.video_path,
                'video_fps': self.video_fps,
                'video_duration': self.video_duration
            }
            
            with open(json_path, 'w') as f:
                json.dump(data_dict, f, indent=2)
            
            logger.info(f"Reduced data saved to {json_path}")
    
    def show(self) -> None:
        """Display the interactive visualization."""
        fig = self.create_interactive_plot()
        fig.show()
    
    def create_video_synchronized_plot(self) -> go.Figure:
        """Create a plot with video synchronization controls.
        
        Returns:
            Plotly figure with video controls
        """
        if self.reduced_data is None:
            raise ValueError("No reduced data available. Call create_visualization() first.")
        
        # Create subplot with video frame
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Pose Movement', 'Video Frame'),
            specs=[[{"type": "scatter"}, {"type": "image"}]]
        )
        
        # Add pose movement plot
        fig.add_trace(
            go.Scatter(
                x=self.reduced_data[:, 0],
                y=self.reduced_data[:, 1],
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=self.timestamps,
                    colorscale='viridis',
                    colorbar=dict(title="Time (s)")
                ),
                name="Pose Movement"
            ),
            row=1, col=1
        )
        
        # Add slider for time control
        fig.update_layout(
            sliders=[{
                'steps': [
                    {
                        'args': [[f'frame{i}'], {
                            'frame': {'duration': 100, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f'{t:.1f}s',
                        'method': 'animate'
                    }
                    for i, t in enumerate(self.timestamps)
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Time: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        return fig
    
    def get_frame_at_time(self, time: float) -> Optional[int]:
        """Get frame number closest to the given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Frame number or None if not found
        """
        if self.timestamps is None:
            return None
        
        # Find closest timestamp
        time_diff = np.abs(self.timestamps - time)
        closest_idx = np.argmin(time_diff)
        
        return self.frame_numbers[closest_idx]
    
    def get_pose_at_time(self, time: float) -> Optional[np.ndarray]:
        """Get pose data at the given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Pose data or None if not found
        """
        if self.reduced_data is None or self.timestamps is None:
            return None
        
        # Find closest timestamp
        time_diff = np.abs(self.timestamps - time)
        closest_idx = np.argmin(time_diff)
        
        return self.reduced_data[closest_idx]
    
    def extract_video_frames(self, output_dir: str = "data/analysis/video_frames") -> None:
        """Extract video frames for synchronization.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        logger.info(f"Extracting video frames to {output_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame if it corresponds to a pose frame
            if frame_count in self.frame_numbers:
                frame_path = output_path / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(self.frame_numbers)} frames")
    
    def create_video_player_html(self, save_html: bool = True) -> str:
        """Create a standalone HTML video player with timeline synchronization.
        
        Args:
            save_html: Whether to save as HTML file
            
        Returns:
            HTML content as string
        """
        video_name = Path(self.video_path).stem
        
        # Create HTML with video player and timeline
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Video Player - {video_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .video-section {{
            text-align: center;
            margin-bottom: 20px;
        }}
        video {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .timeline-section {{
            margin: 20px 0;
        }}
        .timeline {{
            width: 100%;
            height: 60px;
            background: #f0f0f0;
            border-radius: 8px;
            position: relative;
            cursor: pointer;
        }}
        .timeline-marker {{
            position: absolute;
            width: 4px;
            height: 100%;
            background: #007bff;
            border-radius: 2px;
            top: 0;
            left: 0%;
            transition: left 0.1s ease;
        }}
        .time-display {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
            color: #333;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .btn:hover {{
            background: #0056b3;
        }}
        .btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        .pose-info {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .pose-info h3 {{
            margin-top: 0;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Player - {video_name}</h1>
        
        <div class="video-section">
            <video id="videoPlayer" controls>
                <source src="../video/{Path(self.video_path).name}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <div class="time-display">
            <span id="currentTime">0.00</span> / <span id="totalTime">{self.video_duration:.2f}</span> seconds
        </div>
        
        <div class="timeline-section">
            <div class="timeline" id="timeline">
                <div class="timeline-marker" id="timelineMarker"></div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="playPause()">Play/Pause</button>
            <button class="btn" onclick="seekToStart()">Start</button>
            <button class="btn" onclick="seekToEnd()">End</button>
            <button class="btn" onclick="stepForward()">Step Forward</button>
            <button class="btn" onclick="stepBackward()">Step Backward</button>
        </div>
        
        <div class="pose-info">
            <h3>Pose Information</h3>
            <p><strong>Total Frames:</strong> {len(self.frame_numbers)}</p>
            <p><strong>Video Duration:</strong> {self.video_duration:.2f} seconds</p>
            <p><strong>Frame Rate:</strong> {self.video_fps:.2f} FPS</p>
            <p><strong>Current Frame:</strong> <span id="currentFrame">0</span></p>
            <p><strong>Pose Data Available:</strong> <span id="poseAvailable">Yes</span></p>
        </div>
    </div>

    <script>
        const video = document.getElementById('videoPlayer');
        const timeline = document.getElementById('timeline');
        const timelineMarker = document.getElementById('timelineMarker');
        const currentTimeDisplay = document.getElementById('currentTime');
        const currentFrameDisplay = document.getElementById('currentFrame');
        
        // Pose data timestamps
        const poseTimestamps = {self.timestamps.tolist()};
        const poseFrameNumbers = {self.frame_numbers.tolist()};
        const videoDuration = {self.video_duration};
        
        // Update timeline marker position
        function updateTimelineMarker() {{
            const progress = (video.currentTime / videoDuration) * 100;
            timelineMarker.style.left = progress + '%';
            currentTimeDisplay.textContent = video.currentTime.toFixed(2);
            
            // Find closest pose frame
            const currentTime = video.currentTime;
            let closestFrame = 0;
            let minDiff = Math.abs(poseTimestamps[0] - currentTime);
            
            for (let i = 1; i < poseTimestamps.length; i++) {{
                const diff = Math.abs(poseTimestamps[i] - currentTime);
                if (diff < minDiff) {{
                    minDiff = diff;
                    closestFrame = poseFrameNumbers[i];
                }}
            }}
            
            currentFrameDisplay.textContent = closestFrame;
        }}
        
        // Timeline click handler
        timeline.addEventListener('click', function(e) {{
            const rect = timeline.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const progress = clickX / rect.width;
            const newTime = progress * videoDuration;
            video.currentTime = newTime;
        }});
        
        // Video event listeners
        video.addEventListener('timeupdate', updateTimelineMarker);
        video.addEventListener('loadedmetadata', function() {{
            updateTimelineMarker();
        }});
        
        // Control functions
        function playPause() {{
            if (video.paused) {{
                video.play();
            }} else {{
                video.pause();
            }}
        }}
        
        function seekToStart() {{
            video.currentTime = 0;
        }}
        
        function seekToEnd() {{
            video.currentTime = videoDuration;
        }}
        
        function stepForward() {{
            const currentTime = video.currentTime;
            let nextTime = currentTime + 0.1; // Step by 0.1 seconds
            
            // Find next pose frame
            for (let i = 0; i < poseTimestamps.length; i++) {{
                if (poseTimestamps[i] > currentTime) {{
                    nextTime = poseTimestamps[i];
                    break;
                }}
            }}
            
            video.currentTime = Math.min(nextTime, videoDuration);
        }}
        
        function stepBackward() {{
            const currentTime = video.currentTime;
            let prevTime = currentTime - 0.1; // Step by 0.1 seconds
            
            // Find previous pose frame
            for (let i = poseTimestamps.length - 1; i >= 0; i--) {{
                if (poseTimestamps[i] < currentTime) {{
                    prevTime = poseTimestamps[i];
                    break;
                }}
            }}
            
            video.currentTime = Math.max(prevTime, 0);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            switch(e.key) {{
                case ' ':
                    e.preventDefault();
                    playPause();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    stepForward();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    stepBackward();
                    break;
                case 'Home':
                    e.preventDefault();
                    seekToStart();
                    break;
                case 'End':
                    e.preventDefault();
                    seekToEnd();
                    break;
            }}
        }});
    </script>
</body>
</html>
        """
        
        if save_html:
            html_path = self.output_dir / f"{video_name}_video_player.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            logger.info(f"Video player HTML saved to {html_path}")
        
        return html_content
    
    def create_combined_visualization(self, save_html: bool = True) -> str:
        """Create a combined visualization with dimension reduction plot and video player.
        
        Args:
            save_html: Whether to save as HTML file
            
        Returns:
            HTML content as string
        """
        video_name = Path(self.video_path).stem
        
        # Create HTML with combined visualization
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dance Motion Analysis - {video_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        .main-content {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .plot-section {{
            flex: 1;
            min-height: 500px;
        }}
        .video-section {{
            flex: 1;
            min-height: 500px;
        }}
        .video-container {{
            text-align: center;
            margin-bottom: 20px;
        }}
        video {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .timeline-section {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .timeline {{
            width: 100%;
            height: 60px;
            background: #e9ecef;
            border-radius: 8px;
            position: relative;
            cursor: pointer;
            margin: 10px 0;
        }}
        .timeline-marker {{
            position: absolute;
            width: 4px;
            height: 100%;
            background: #007bff;
            border-radius: 2px;
            top: 0;
            left: 0%;
            transition: left 0.1s ease;
        }}
        .time-display {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
            color: #333;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        .btn:hover {{
            background: #0056b3;
        }}
        .info-panel {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .info-panel h3 {{
            margin-top: 0;
            color: #333;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .stat-label {{
            font-weight: bold;
            color: #666;
            font-size: 12px;
        }}
        .stat-value {{
            font-size: 18px;
            color: #333;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Dance Motion Analysis - {video_name}</h1>
            <p>Interactive visualization of pose data in reduced-dimensional space with synchronized video playback</p>
        </div>
        
        <div class="main-content">
            <div class="plot-section">
                <h3>Pose Movement in Reduced Space</h3>
                <div id="plotContainer"></div>
            </div>
            
            <div class="video-section">
                <h3>Video Playback</h3>
                <div class="video-container">
                    <video id="videoPlayer" controls>
                        <source src="../video/{Path(self.video_path).name}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
        </div>
        
        <div class="timeline-section">
            <h3>Synchronized Timeline</h3>
            <div class="time-display">
                <span id="currentTime">0.00</span> / <span id="totalTime">{self.video_duration:.2f}</span> seconds
            </div>
            <div class="timeline" id="timeline">
                <div class="timeline-marker" id="timelineMarker"></div>
            </div>
            <div class="controls">
                <button class="btn" onclick="playPause()">Play/Pause</button>
                <button class="btn" onclick="seekToStart()">Start</button>
                <button class="btn" onclick="seekToEnd()">End</button>
                <button class="btn" onclick="stepForward()">Step Forward</button>
                <button class="btn" onclick="stepBackward()">Step Backward</button>
            </div>
        </div>
        
        <div class="info-panel">
            <h3>Analysis Information</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">Total Frames</div>
                    <div class="stat-value">{len(self.frame_numbers)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Video Duration</div>
                    <div class="stat-value">{self.video_duration:.2f}s</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Frame Rate</div>
                    <div class="stat-value">{self.video_fps:.2f} FPS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Current Frame</div>
                    <div class="stat-value" id="currentFrame">0</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Reduced Dimensions</div>
                    <div class="stat-value">{self.reduced_data.shape[1]}D</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Pose Keypoints</div>
                    <div class="stat-value">33</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('videoPlayer');
        const timeline = document.getElementById('timeline');
        const timelineMarker = document.getElementById('timelineMarker');
        const currentTimeDisplay = document.getElementById('currentTime');
        const currentFrameDisplay = document.getElementById('currentFrame');
        
        // Pose data
        const poseTimestamps = {self.timestamps.tolist()};
        const poseFrameNumbers = {self.frame_numbers.tolist()};
        const reducedData = {self.reduced_data.tolist()};
        const videoDuration = {self.video_duration};
        
        // Create the plot
        const plotData = [
            {{
                x: reducedData.map(d => d[0]),
                y: reducedData.map(d => d[1]),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: 8,
                    color: poseTimestamps,
                    colorscale: 'viridis',
                    colorbar: {{ title: "Time (s)" }}
                }},
                text: poseFrameNumbers.map((f, i) => `Frame ${{f}}<br>Time: ${{poseTimestamps[i].toFixed(2)}}s`),
                hovertemplate: '%{{text}}<extra></extra>',
                name: 'Pose Points'
            }},
            {{
                x: reducedData.map(d => d[0]),
                y: reducedData.map(d => d[1]),
                mode: 'lines',
                type: 'scatter',
                line: {{ color: 'red', width: 3 }},
                name: 'Movement Trajectory',
                showlegend: true
            }}
        ];
        
        const plotLayout = {{
            title: 'Pose Movement in Reduced Space',
            xaxis: {{ title: 'Component 1' }},
            yaxis: {{ title: 'Component 2' }},
            width: 600,
            height: 400,
            showlegend: true
        }};
        
        Plotly.newPlot('plotContainer', plotData, plotLayout);
        
        // Update timeline marker position
        function updateTimelineMarker() {{
            const progress = (video.currentTime / videoDuration) * 100;
            timelineMarker.style.left = progress + '%';
            currentTimeDisplay.textContent = video.currentTime.toFixed(2);
            
            // Find closest pose frame
            const currentTime = video.currentTime;
            let closestFrame = 0;
            let minDiff = Math.abs(poseTimestamps[0] - currentTime);
            
            for (let i = 1; i < poseTimestamps.length; i++) {{
                const diff = Math.abs(poseTimestamps[i] - currentTime);
                if (diff < minDiff) {{
                    minDiff = diff;
                    closestFrame = poseFrameNumbers[i];
                }}
            }}
            
            currentFrameDisplay.textContent = closestFrame;
            
            // Highlight current point in plot
            updatePlotHighlight(currentTime);
        }}
        
        // Update plot highlight
        function updatePlotHighlight(currentTime) {{
            // Find closest timestamp index
            let closestIndex = 0;
            let minDiff = Math.abs(poseTimestamps[0] - currentTime);
            
            for (let i = 1; i < poseTimestamps.length; i++) {{
                const diff = Math.abs(poseTimestamps[i] - currentTime);
                if (diff < minDiff) {{
                    minDiff = diff;
                    closestIndex = i;
                }}
            }}
            
            // Update plot with highlighted point
            const update = {{
                'marker.size': Array(poseTimestamps.length).fill(8)
            }};
            update['marker.size'][closestIndex] = 15;
            
            Plotly.restyle('plotContainer', update, [0]);
        }}
        
        // Timeline click handler
        timeline.addEventListener('click', function(e) {{
            const rect = timeline.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const progress = clickX / rect.width;
            const newTime = progress * videoDuration;
            video.currentTime = newTime;
        }});
        
        // Video event listeners
        video.addEventListener('timeupdate', updateTimelineMarker);
        video.addEventListener('loadedmetadata', function() {{
            updateTimelineMarker();
        }});
        
        // Control functions
        function playPause() {{
            if (video.paused) {{
                video.play();
            }} else {{
                video.pause();
            }}
        }}
        
        function seekToStart() {{
            video.currentTime = 0;
        }}
        
        function seekToEnd() {{
            video.currentTime = videoDuration;
        }}
        
        function stepForward() {{
            const currentTime = video.currentTime;
            let nextTime = currentTime + 0.1;
            
            for (let i = 0; i < poseTimestamps.length; i++) {{
                if (poseTimestamps[i] > currentTime) {{
                    nextTime = poseTimestamps[i];
                    break;
                }}
            }}
            
            video.currentTime = Math.min(nextTime, videoDuration);
        }}
        
        function stepBackward() {{
            const currentTime = video.currentTime;
            let prevTime = currentTime - 0.1;
            
            for (let i = poseTimestamps.length - 1; i >= 0; i--) {{
                if (poseTimestamps[i] < currentTime) {{
                    prevTime = poseTimestamps[i];
                    break;
                }}
            }}
            
            video.currentTime = Math.max(prevTime, 0);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            switch(e.key) {{
                case ' ':
                    e.preventDefault();
                    playPause();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    stepForward();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    stepBackward();
                    break;
                case 'Home':
                    e.preventDefault();
                    seekToStart();
                    break;
                case 'End':
                    e.preventDefault();
                    seekToEnd();
                    break;
            }}
        }});
    </script>
</body>
</html>
        """
        
        if save_html:
            html_path = self.output_dir / f"{video_name}_combined_visualization.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            logger.info(f"Combined visualization saved to {html_path}")
        
        return html_content 