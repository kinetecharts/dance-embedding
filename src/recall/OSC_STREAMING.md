# OSC Streaming for Hand Joints

This module provides OSC (Open Sound Control) streaming capabilities for pose landmarks, specifically focused on hand joint data from MediaPipe pose estimation.

## Overview

The OSC streaming system allows you to stream hand joint positions in real-time to other applications that can receive OSC messages. This is useful for:

- Interactive installations
- Real-time music/audio applications
- Motion-controlled systems
- Cross-platform communication between different software

## Features

- **Hand-focused streaming**: Streams only hand joint data (left and right hands)
- **Configurable rate**: Adjustable streaming frequency (default: 30 Hz)
- **3D coordinates**: Supports both 2D and 3D coordinate systems
- **Confidence scores**: Optional inclusion of MediaPipe confidence values
- **UDP transport**: Uses UDP for low-latency communication

## OSC Message Format

### Address Patterns

Hand joint data is streamed using the following OSC address patterns:

- **Left Hand**: `/pose/left_hand/{joint_name}`
- **Right Hand**: `/pose/right_hand/{joint_name}`

### Joint Names

The following hand joints are streamed:

- `wrist` - Wrist position
- `thumb_tip`, `thumb_ip`, `thumb_mcp`, `thumb_cmc` - Thumb joints
- `index_tip`, `index_dip`, `index_pip`, `index_mcp` - Index finger joints
- `middle_tip`, `middle_dip`, `middle_pip`, `middle_mcp` - Middle finger joints
- `ring_tip`, `ring_dip`, `ring_pip`, `ring_mcp` - Ring finger joints
- `pinky_tip`, `pinky_dip`, `pinky_pip`, `pinky_mcp` - Pinky finger joints

### Message Arguments

Each OSC message contains:

1. **X coordinate** (float): Horizontal position (0.0 = left, 1.0 = right)
2. **Y coordinate** (float): Vertical position (0.0 = top, 1.0 = bottom)
3. **Z coordinate** (float): Depth position (if 3D enabled)
4. **Confidence** (float): MediaPipe confidence score (0.0-1.0, if enabled)

## Configuration

### Basic Configuration

```python
from recall.config import RecallConfig

config = RecallConfig(
    # Enable OSC streaming
    osc_enabled=True,
    osc_host="127.0.0.1",      # Target host
    osc_port=6448,              # Target port
    osc_stream_rate=30.0,       # Hz
    osc_hand_joints_only=True   # Only stream hands
)
```

### Advanced Configuration

```python
from recall.osc_streamer import OSCConfig

osc_config = OSCConfig(
    host="192.168.1.100",       # Remote host
    port=9000,                  # Custom port
    enabled=True,
    stream_rate=60.0,           # 60 Hz streaming
    hand_joints_only=True,
    include_confidence=True,     # Include confidence scores
    include_3d=True,            # Include 3D coordinates
    left_hand_prefix="/hands/left",    # Custom address prefix
    right_hand_prefix="/hands/right"   # Custom address prefix
)
```

## Usage Examples

### 1. Basic OSC Streaming in Recall System

```python
from recall.recall_system import create_recall_system
from recall.config import RecallConfig

# Create config with OSC enabled
config = RecallConfig(
    osc_enabled=True,
    osc_host="127.0.0.1",
    osc_port=8000
)

# Run system with OSC streaming
with create_recall_system(config) as system:
    system.run_live()
```

### 2. Standalone OSC Streamer

```python
from recall.osc_streamer import create_osc_streamer
from recall.data_structures import PoseData

# Create OSC streamer
streamer = create_osc_streamer(
    host="127.0.0.1",
    port=6448,
    stream_rate=30.0
)

# Stream pose data
pose_data = get_pose_data()  # Your pose data source
streamer.stream_pose(pose_data)

# Clean up
streamer.close()
```

### 3. Test OSC Streaming

```bash
# Run the test script
python test_osc_streaming.py

# This will stream simulated hand movements to localhost:6448
```

## Receiving OSC Messages

### Max/MSP Example

```maxmsp
[udpreceive 8000]
|
[route /pose/left_hand/wrist]
|
[unpack f f f f]  # x, y, z, confidence
|
[print left_wrist]
```

### TouchOSC

1. Set TouchOSC to receive on port 6448
2. Add faders/buttons with addresses like `/pose/left_hand/wrist`
3. The values will update in real-time with hand positions

### Python OSC Receiver

```python
from pythonosc import dispatcher
from pythonosc import osc_server

def handle_hand_joint(address, *args):
    print(f"{address}: {args}")

# Create dispatcher
disp = dispatcher.Dispatcher()
disp.map("/pose/left_hand/*", handle_hand_joint)
disp.map("/pose/right_hand/*", handle_hand_joint)

# Start server
server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 8001), disp)
server.serve_forever()
```

## Performance Considerations

- **Streaming rate**: Higher rates (60+ Hz) may impact performance
- **Network latency**: UDP provides low latency but no guaranteed delivery
- **Coordinate precision**: Coordinates are normalized (0.0-1.0) for consistency
- **Memory usage**: Minimal overhead, streams only when pose data is available

## Troubleshooting

### Common Issues

1. **No OSC messages received**
   - Check firewall settings
   - Verify host/port configuration
   - Ensure OSC is enabled in config

2. **High latency**
   - Reduce streaming rate
   - Check network configuration
   - Monitor system performance

3. **Missing joints**
   - Verify MediaPipe pose detection is working
   - Check confidence thresholds
   - Ensure pose data contains all 33 landmarks

### Debug Mode

Enable debug logging to see OSC message details:

```python
import logging
logging.getLogger('recall.osc_streamer').setLevel(logging.DEBUG)
```

## Dependencies

- `python-osc>=1.8.0` - OSC protocol implementation
- `numpy` - Numerical operations
- `mediapipe` - Pose estimation (for landmark indices)

## Installation

```bash
# Install OSC dependency
uv add python-osc

# Or install all project dependencies
uv sync
```

## Future Enhancements

- [ ] Full body joint streaming
- [ ] Custom joint selection
- [ ] Multiple OSC destinations
- [ ] OSC message bundling
- [ ] WebSocket fallback
- [ ] JSON message format support
