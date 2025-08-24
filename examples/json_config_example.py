#!/usr/bin/env python3
"""Example of using JSON configuration for video file selection in the recall system."""

import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recall.json_config_loader import create_config_loader, create_recall_config_from_json
from recall.recall_system import create_recall_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate JSON configuration loading and video file selection"""
    logger.info("JSON Configuration Example")
    logger.info("=" * 40)
    
    # Create config loader
    config_loader = create_config_loader()
    
    # Show configuration summary
    logger.info("\n" + config_loader.get_config_summary())
    
    # Get video files for matching
    video_files = config_loader.get_video_files_for_matching("data/video")
    logger.info(f"\nVideo files for matching: {len(video_files)}")
    for video_file in video_files:
        logger.info(f"  - {video_file.name}")
    
    # Create RecallConfig from JSON
    logger.info("\nCreating RecallConfig from JSON...")
    config = create_recall_config_from_json()
    
    # Show some config values
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Top N: {config.top_n}")
    logger.info(f"Match Interval: {config.match_interval}s")
    logger.info(f"OSC Enabled: {config.osc_enabled}")
    if config.osc_enabled:
        logger.info(f"OSC Host: {config.osc_host}:{config.osc_port}")
    
    # Create recall system with JSON config
    logger.info("\nCreating recall system with JSON configuration...")
    with create_recall_system(config, with_keyboard=False) as system:
        # Get video files through the system
        system_videos = system.get_video_files_for_matching()
        logger.info(f"System video files: {len(system_videos)}")
        
        # Show config summary
        logger.info("\n" + system.get_config_summary())
    
    logger.info("\nJSON configuration example completed!")


def demonstrate_video_filtering():
    """Demonstrate different video filtering scenarios"""
    logger.info("\nVideo Filtering Examples")
    logger.info("=" * 30)
    
    # Create config loader
    config_loader = create_config_loader()
    
    # Example 1: Load all videos
    logger.info("\n1. Loading all videos from directory:")
    all_videos = config_loader.get_video_files_for_matching("data/video")
    logger.info(f"   Found {len(all_videos)} videos")
    
    # Example 2: Show specific video configuration
    video_config = config_loader.config_data.get("video_matching", {})
    logger.info("\n2. Video matching configuration:")
    logger.info(f"   Load specific videos: {video_config.get('load_specific_videos', False)}")
    logger.info(f"   Specific videos: {video_config.get('specific_videos', [])}")
    logger.info(f"   Exclude patterns: {video_config.get('exclude_patterns', [])}")
    logger.info(f"   Include patterns: {video_config.get('include_patterns', [])}")


if __name__ == "__main__":
    main()
    demonstrate_video_filtering()
