# Advanced OSC Streaming for Pose Data

This module provides advanced OSC (Open Sound Control) streaming capabilities for comprehensive pose data, including hands, feet, body orientation, and movement analysis from MediaPipe pose estimation.

## Overview

The advanced OSC streaming system allows you to stream comprehensive pose data in real-time to other applications that can receive OSC messages. This includes hands, feet, body orientation, and movement analysis. This is useful for:

- Interactive installations
- Real-time music/audio applications
- Motion-controlled systems
- Dance and movement analysis
- Cross-platform communication between different software

## Features

- **Comprehensive pose streaming**: Streams hands, feet, body orientation, and movement data
- **Body-relative coordinates**: Normalized by torso length for consistent measurements
- **Location-independent**: Same pose produces same values regardless of camera distance
- **Z-filtered movement**: Fast rise, slow decay filtering for velocity and acceleration
- **Configurable rate**: Adjustable streaming frequency (default: 30 Hz)
- **3D coordinates**: Supports both 2D and 3D coordinate systems
- **UDP transport**: Uses UDP for low-latency communication

## OSC Message Format

### Single Stream Architecture

The system streams a single OSC message containing **21 values** packed into one stream:

**OSC Address:** `/pose/data`  
**Port:** 6448 (configurable)  
**Data Format:** Single message with 21 float values

### Data Breakdown (21 values):

1. **Left Hand X, Y, Z** (body-relative, normalized by torso length)
2. **Right Hand X, Y, Z** (body-relative, normalized by torso length)  
3. **Left Foot X, Y, Z** (body-relative, normalized by torso length)
4. **Right Foot X, Y, Z** (body-relative, normalized by torso length)
5. **Torso Rotation Yaw, Pitch** (degrees)
6. **Head Rotation Yaw, Pitch** (relative to torso, degrees)
7. **Torso Position X, Y, Z** (frame coordinates, 0.0-1.0)
8. **Velocity Magnitude** (Z-filtered, fast rise, slow decay)
9. **Acceleration Magnitude** (Z-filtered, fast rise, slow decay)

### Example OSC Message:
```
/pose/data 0.5 -0.2 1.2 0.6 0.1 1.1 -0.3 0.8 0.0 -0.2 0.9 0.1 -45.2 12.8 -2.1 15.3 0.48 0.52 0.1 0.8 0.6
```

### Coordinate System

**Body-Relative Coordinates (Values 1-12):**
- **Origin**: Chest center point (midpoint between shoulders and hips)
- **Scale**: Normalized by torso length (1.0 = one torso length)
- **Benefits**: Same pose produces same values regardless of distance from camera

**Frame-Relative Coordinates (Values 17-19):**
- **X-axis**: 0.0 (left) to 1.0 (right)
- **Y-axis**: 0.0 (top) to 1.0 (bottom)
- **Z-axis**: 0.0 (closer) to 1.0 (farther)

**Rotation Data (Values 13-16):**
- **Torso Rotation**: 
  - **Yaw**: 0° when facing camera, positive when turning right, negative when turning left
  - **Pitch**: 0° when level, positive when leaning forward, negative when leaning back
- **Head Rotation**: Relative to torso orientation
  - **Yaw**: 0° when aligned with body, positive when turning right relative to body, negative when turning left relative to body
  - **Pitch**: 0° when level with body, positive when nodding up, negative when nodding down
- **Units**: Degrees (-180° to +180°)

## Configuration

### JSON Configuration

The OSC streaming is configured through `config.json`:

```json
{
  "osc_streaming": {
    "enabled": true,
    "stream_rate": 30.0,
    "streams": {
      "pose_data": {
        "enabled": true,
        "host": "127.0.0.1",
        "port": 6448,
        "address": "/pose/data",
        "z_filter": {
          "velocity_fast_rise": 0.8,
          "velocity_slow_decay": 0.95,
          "acceleration_fast_rise": 0.9,
          "acceleration_slow_decay": 0.98
        }
      }
    }
  }
}
```

### Z-Filter Parameters

- **velocity_fast_rise**: How quickly velocity responds to increases (0.8 = fast)
- **velocity_slow_decay**: How slowly velocity decreases (0.95 = slow)
- **acceleration_fast_rise**: How quickly acceleration responds to changes (0.9 = fast)
- **acceleration_slow_decay**: How slowly acceleration decreases (0.98 = slow)

## Usage Examples

### 1. Basic OSC Streaming in Recall System

```python
from recall.recall_system import create_recall_system
from recall.config import RecallConfig

# Create config with OSC enabled
config = RecallConfig(
    osc_enabled=True
)

# The system will automatically load OSC configuration from config.json
```

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
