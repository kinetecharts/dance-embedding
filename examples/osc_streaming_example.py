#!/usr/bin/env python3
"""
OSC Streaming Example for Dance Embedding Recall System

This example demonstrates how to enable OSC streaming in the main recall system.
It streams pose data to configured OSC endpoints for real-time motion tracking.

Usage:
    # Basic OSC streaming with matching
    uv run examples/osc_streaming_example.py
    
    # OSC streaming without matching (lightweight mode)
    uv run examples/osc_streaming_example.py --skip-matching
    
    # List available cameras
    uv run examples/osc_streaming_example.py --list-cameras
    
    # Use specific camera
    uv run examples/osc_streaming_example.py --camera-id 1
    
    # Custom matching intervals
    uv run examples/osc_streaming_example.py --match-interval 2.0 --match-duration 1.0
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recall.recall_system import create_recall_system
from recall.json_config_loader import create_recall_config_from_json

def list_available_cameras():
    """List all available camera devices on the system"""
    import cv2
    
    print("🔍 Scanning for available cameras...")
    print("=" * 50)
    
    available_cameras = []
    max_cameras_to_check = 10  # Check first 10 camera indices
    
    for camera_id in range(max_cameras_to_check):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"📹 Camera {camera_id}: {width}x{height} @ {fps:.1f} FPS")
                available_cameras.append(camera_id)
            else:
                print(f"⚠️  Camera {camera_id}: Opened but cannot read frames")
            
            cap.release()
        else:
            # Camera not available at this index
            pass
    
    if not available_cameras:
        print("❌ No cameras found!")
        print("\nTroubleshooting tips:")
        print("- Check camera permissions in System Preferences")
        print("- Ensure camera is not being used by another application")
        print("- Try disconnecting and reconnecting the camera")
        print("- On macOS, check Privacy & Security > Camera settings")
    else:
        print(f"\n✅ Found {len(available_cameras)} camera(s): {available_cameras}")
        print(f"💡 Use --camera-id <number> to select a specific camera")
        print(f"💡 Default camera is 0 (first available)")
    
    return available_cameras

def main():
    """Main function for OSC streaming example"""
    parser = argparse.ArgumentParser(
        description="OSC Streaming Example for Dance Embedding Recall System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic OSC streaming
    uv run examples/osc_streaming_example.py
    
    # Lightweight OSC-only mode (no matching)
    uv run examples/osc_streaming_example.py --skip-matching
    
    # Use specific camera
    uv run examples/osc_streaming_example.py --camera-id 1
    
    # List available cameras
    uv run examples/osc_streaming_example.py --list-cameras
        """
    )
    
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip video matching and run in lightweight OSC-only mode"
    )
    
    parser.add_argument(
        "--match-interval",
        type=float,
        default=5.0,
        help="Interval between pose matching attempts (seconds, default: 5.0)"
    )
    
    parser.add_argument(
        "--match-duration",
        type=float,
        default=2.0,
        help="Duration of pose matching window (seconds, default: 2.0)"
    )
    
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit"
    )
    
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID to use (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Handle camera listing
    if args.list_cameras:
        list_available_cameras()
        return
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "src" / "recall" / "config.json"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Please ensure config.json exists in src/recall/")
        return
    
    try:
        config = create_recall_config_from_json(str(config_path))
        logger.info(f"✅ Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Update camera ID in config if specified
    if args.camera_id != 0:
        config.camera_id = args.camera_id
        logger.info(f"📹 Using camera ID: {args.camera_id}")
    
    # Create and run recall system
    try:
        if args.skip_matching:
            logger.info("🚀 Starting OSC-only mode (no video matching)")
            logger.info("📡 Streaming pose data to OSC ports 6448 and 1234")
            logger.info("🎥 Video display with pose visualization enabled")
            logger.info("💡 Press 'q' in video window to quit")
            
            # In OSC-only mode, these values aren't used but set defaults
            match_interval = 5.0
            match_duration = 2.0
        else:
            logger.info("🚀 Starting full recall system with OSC streaming")
            logger.info(f"📡 Streaming pose data to OSC ports 6448 and 1234")
            logger.info(f"🎯 Pose matching every {args.match_interval}s for {args.match_duration}s")
            logger.info("💡 Press 'q' in video window to quit")
            
            match_interval = args.match_interval
            match_duration = args.match_duration
        
        # Create recall system
        recall_system = create_recall_system(
            config=config,
            with_keyboard=True,
            osc_only=args.skip_matching
        )
        
        # Run the system
        recall_system.run_live()
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error running recall system: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
