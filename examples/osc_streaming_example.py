#!/usr/bin/env python3
"""Example of enabling OSC streaming in the recall system for hand joints."""
# to run: uv run examples/osc_streaming_example.py
# address: /pose/left_hand/wrist

import logging
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recall.config import RecallConfig
from recall.recall_system import create_recall_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run recall system with OSC streaming enabled"""
    logger.info("Starting recall system with OSC streaming...")
    
    # Create configuration with OSC enabled
    config = RecallConfig(
        mode="camera",  # Use camera input
        match_interval=2.0,  # Match every 2 seconds
        match_playback_duration=2.0,  # Play matches for 2 seconds
        
        # Enable OSC streaming
        osc_enabled=True,
        osc_host="127.0.0.1",  # Localhost
        osc_port=6448,          # Port 6448 Wekinator default port
        # osc_port=1234,          # Isadora default port
        osc_stream_rate=15.0,   # 30 Hz streaming
        osc_hand_joints_only=False  # Only stream hand joints
    )
    
    logger.info(f"OSC streaming configured: {config.osc_host}:{config.osc_port}")
    logger.info(f"Streaming rate: {config.osc_stream_rate} Hz")
    logger.info("Hand joints will be streamed to:")
    logger.info("  - /pose/left_hand/* (left hand landmarks)")
    logger.info("  - /pose/right_hand/* (right hand landmarks)")
    
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
