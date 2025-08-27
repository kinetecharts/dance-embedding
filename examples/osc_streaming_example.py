#!/usr/bin/env python3
"""Example of enabling OSC streaming in the recall system with single-stream data."""
# to run: uv run examples/osc_streaming_example.py
# or: source .venv/bin/activate and run: python examples/osc_streaming_example.py
# address: /pose/data [21 values]

import argparse
import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recall.config import RecallConfig
from recall.recall_system import create_recall_system

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    """Run recall system with OSC streaming enabled"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run recall system with OSC streaming")
    parser.add_argument(
        "--skip-matching", 
        action="store_true",
        help="Skip pose matching and only stream OSC data"
    )
    parser.add_argument(
        "--match-interval", 
        type=float, 
        default=2.0,
        help="Interval between pose matches in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--match-duration", 
        type=float, 
        default=2.0,
        help="Duration to play matched poses in seconds (default: 2.0)"
    )
    args = parser.parse_args()
    
    logger.info("Starting recall system with OSC streaming...")
    
    if args.skip_matching:
        logger.info("🚀 POSE MATCHING DISABLED - Using lightweight OSC-only mode")
        logger.info("   - No video loading or vector database initialization")
        logger.info("   - Only camera input and OSC streaming")
        # Use default intervals since they won't be used in OSC-only mode
        match_interval = 2.0
        match_duration = 2.0
    else:
        logger.info(f"🎬 Pose matching enabled - matching every {args.match_interval}s for {args.match_duration}s")
        match_interval = args.match_interval
        match_duration = args.match_duration
    
    # Create configuration with OSC enabled
    config = RecallConfig(
        mode="camera",  # Use camera input
        match_interval=match_interval,  # Match interval (or very long if disabled)
        match_playback_duration=match_duration,  # Play matches duration (or 0 if disabled)
        
        # Enable OSC streaming (configuration now handled by JSON config)
        osc_enabled=True
    )
    
    logger.info("OSC streaming enabled - configuration loaded from config.json")
    logger.info("Single stream will be sent to:")
    logger.info("  - /pose/data [21 values] - all pose data in one message")
    logger.info("  - Port 6448 - main stream")
    logger.info("  - Port 1234 - duplicate stream")
    
    # Create and run recall system
    if args.skip_matching:
        # Use lightweight OSC-only mode - no video loading or matching
        logger.info("🚀 Creating lightweight OSC-only system...")
        with create_recall_system(config, with_keyboard=False, osc_only=True) as system:
            try:
                system.run_live()
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            except Exception as e:
                logger.error(f"Error running OSC-only system: {e}")
    else:
        # Use full recall system with video matching
        logger.info("🎬 Creating full recall system with video matching...")
        with create_recall_system(config, with_keyboard=True, osc_only=False) as system:
            try:
                system.run_live()
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            except Exception as e:
                logger.error(f"Error running recall system: {e}")
    
    logger.info("Recall system stopped")


if __name__ == "__main__":
    main()
