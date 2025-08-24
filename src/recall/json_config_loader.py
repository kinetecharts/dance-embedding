"""JSON configuration loader for the recall system with video file filtering."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from fnmatch import fnmatch

from .config import RecallConfig

logger = logging.getLogger(__name__)


class JSONConfigLoader:
    """Loads configuration from JSON files with video filtering capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("src/recall/config.json")
        self.config_data = {}
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from JSON file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
            
            logger.info(f"✅ Configuration loaded from: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return False
    
    def get_video_files_for_matching(self, video_dir: str) -> List[Path]:
        """Get list of video files to use for matching based on config"""
        video_path = Path(video_dir)
        if not video_path.exists():
            logger.warning(f"Video directory does not exist: {video_dir}")
            return []
        
        # Get video matching configuration
        video_config = self.config_data.get("video_matching", {})
        specific_videos = video_config.get("specific_videos", [])
        load_specific_videos = video_config.get("load_specific_videos", False)
        exclude_patterns = video_config.get("exclude_patterns", [])
        include_patterns = video_config.get("include_patterns", [])
        video_extensions = video_config.get("video_extensions", [".mp4", ".avi", ".mov", ".mkv", ".webm"])
        
        # If load_specific_videos is true and specific videos are specified, use only those
        if load_specific_videos and specific_videos:
            logger.info(f"Loading specific videos: {specific_videos}")
            video_files = []
            for video_name in specific_videos:
                video_file = video_path / video_name
                if video_file.exists():
                    video_files.append(video_file)
                else:
                    logger.warning(f"Specified video not found: {video_name}")
            return video_files
        
        # Otherwise, load all videos from directory (when load_specific_videos is false)
        logger.info(f"Loading all videos from: {video_dir}")
        all_videos = []
        
        # Get all video files with specified extensions
        for ext in video_extensions:
            all_videos.extend(video_path.glob(f"*{ext}"))
        
        # Apply include/exclude patterns
        filtered_videos = []
        for video_file in all_videos:
            video_name = video_file.name
            
            # Check exclude patterns
            excluded = False
            for pattern in exclude_patterns:
                if fnmatch(video_name, pattern):
                    logger.debug(f"Excluding video (pattern: {pattern}): {video_name}")
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Check include patterns (if any are specified)
            if include_patterns:
                included = False
                for pattern in include_patterns:
                    if fnmatch(video_name, pattern):
                        included = True
                        break
                
                if not included:
                    logger.debug(f"Video not in include patterns: {video_name}")
                    continue
            
            filtered_videos.append(video_file)
        
        logger.info(f"Found {len(filtered_videos)} videos for matching")
        return filtered_videos
    
    def create_recall_config(self) -> RecallConfig:
        """Create RecallConfig from JSON configuration"""
        recall_config = self.config_data.get("recall_system", {})
        paths_config = self.config_data.get("paths", {})
        osc_config = self.config_data.get("osc_streaming", {})
        
        # Create config with JSON values
        config = RecallConfig(
            mode=recall_config.get("mode", "camera"),
            input_video=recall_config.get("input_video"),
            top_n=recall_config.get("top_n", 5),
            match_every=recall_config.get("match_every", 60),
            match_interval=recall_config.get("match_interval", 2.0),
            match_playback_duration=recall_config.get("match_playback_duration", 2.0),
            similarity_metric=recall_config.get("similarity_metric", "euclidean"),
            pose_dir=paths_config.get("pose_dir", "data/poses"),
            video_dir=paths_config.get("video_dir", "data/video"),
            video_with_pose_dir=paths_config.get("video_with_pose_dir", "data/video_with_pose"),
            confidence_threshold=recall_config.get("confidence_threshold", 0.5),
            normalize_rotation=recall_config.get("normalize_rotation", False),
            max_cache_size=recall_config.get("max_cache_size", 1000),
            parallel_workers=recall_config.get("parallel_workers", 4),
            video_player=recall_config.get("video_player", "opencv"),
            
            # OSC configuration
            osc_enabled=osc_config.get("enabled", False),
            osc_host=osc_config.get("host", "127.0.0.1"),
            osc_port=osc_config.get("port", 6448),
            osc_stream_rate=osc_config.get("stream_rate", 30.0),
            osc_hand_joints_only=osc_config.get("hand_joints_only", True)
        )
        
        # Override joint weights if specified in JSON
        joint_weights = self.config_data.get("joint_weights")
        if joint_weights:
            config.joint_weights = joint_weights
        
        logger.info("✅ RecallConfig created from JSON configuration")
        return config
    
    def get_osc_config(self) -> Dict[str, Any]:
        """Get OSC configuration from JSON"""
        return self.config_data.get("osc_streaming", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration from JSON"""
        return self.config_data.get("performance", {})
    
    def reload_config(self) -> bool:
        """Reload configuration from file"""
        return self.load_config()
    
    def get_config_summary(self) -> str:
        """Get a summary of the current configuration"""
        if not self.config_data:
            return "No configuration loaded"
        
        summary = []
        summary.append("Configuration Summary:")
        summary.append("=" * 30)
        
        # Recall system config
        recall = self.config_data.get("recall_system", {})
        summary.append(f"Mode: {recall.get('mode', 'camera')}")
        summary.append(f"Top N: {recall.get('top_n', 5)}")
        summary.append(f"Match Interval: {recall.get('match_interval', 2.0)}s")
        
        # Video matching config
        video_config = self.config_data.get("video_matching", {})
        specific_videos = video_config.get("specific_videos", [])
        if specific_videos:
            summary.append(f"Specific Videos: {len(specific_videos)}")
            for video in specific_videos[:3]:  # Show first 3
                summary.append(f"  - {video}")
            if len(specific_videos) > 3:
                summary.append(f"  ... and {len(specific_videos) - 3} more")
        else:
            summary.append("Video Loading: All videos from directory")
        
        # OSC config
        osc = self.config_data.get("osc_streaming", {})
        if osc.get("enabled"):
            summary.append(f"OSC Streaming: {osc.get('host')}:{osc.get('port')}")
        else:
            summary.append("OSC Streaming: Disabled")
        
        return "\n".join(summary)


# Convenience function to create config loader
def create_config_loader(config_path: Optional[str] = None) -> JSONConfigLoader:
    """Create a JSON config loader instance"""
    return JSONConfigLoader(config_path)


# Convenience function to create RecallConfig from JSON
def create_recall_config_from_json(config_path: Optional[str] = None) -> RecallConfig:
    """Create RecallConfig directly from JSON file"""
    loader = create_config_loader(config_path)
    return loader.create_recall_config()
