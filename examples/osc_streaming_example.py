#!/usr/bin/env python3
"""Example of enabling OSC streaming in the recall system with single-stream data."""
# to run: uv run examples/osc_streaming_example.py
# address: /pose/data [15 values]

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
    logger.info("Starting recall system with OSC streaming...")
    
    # Create configuration with OSC enabled
    config = RecallConfig(
        mode="camera",  # Use camera input
        match_interval=2.0,  # Match every 2 seconds
        match_playback_duration=2.0,  # Play matches for 2 seconds
        
        # Enable OSC streaming (configuration now handled by JSON config)
        osc_enabled=True
    )
    
    logger.info("OSC streaming enabled - configuration loaded from config.json")
    logger.info("Single stream will be sent to:")
    logger.info("  - /pose/data [15 values] - all pose data in one message")
    logger.info("  - Port 6448 - single port for all data")
    
    # Create and run recall system
    with create_recall_system(config, with_keyboard=True) as system:
        try:
            system.run_live()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error running recall system: {e}")
    
    logger.info("Recall system stopped")


if __name__ == "__main__":
    main()
