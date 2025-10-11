# OSC to WebSocket Bridge

A real-time bridge service that listens for OSC (Open Sound Control) data streams and forwards them to web clients via WebSocket. Designed specifically for the dance embedding system to stream pose data to web applications.

## Features

- **Real-time OSC to WebSocket forwarding** - Listens for OSC messages and broadcasts them to connected web clients
- **Web interface** - Beautiful, responsive web dashboard for monitoring OSC data streams
- **Pose data support** - Optimized for MediaPipe pose landmarks and dance embedding data
- **Filtering and search** - Filter messages by OSC address and data type
- **Statistics** - Real-time server statistics and connection monitoring
- **Auto-reconnection** - WebSocket clients automatically reconnect on connection loss

## Quick Start

### 1. Install Dependencies

```bash
# Install OSC bridge dependencies (already included in pyproject.toml dev dependencies)
uv sync --dev
```

### 2. Start the Bridge Server

```bash
uv run python -m uvicorn src.osc_to_websocket.server:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open the Web Interface

Navigate to: http://localhost:8000

### 4. Start Sending OSC Data

The bridge listens for OSC messages on **UDP port 6448** (default dance embedding port).

#### Option A: Use the Dance Embedding System
```bash
uv run examples/osc_streaming_example.py --skip-matching
```

#### Option B: Use the Test Client
```bash
uv run src/osc_to_websocket/test_client.py --duration 60
```

## Configuration

### OSC Server Settings

The OSC server listens on:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 6448 (configurable in server.py)

### WebSocket Server Settings

The WebSocket server runs on:
- **Host**: 0.0.0.0
- **Port**: 8000
- **Endpoint**: `/ws`

## API Endpoints

### WebSocket
- `ws://localhost:8000/ws` - Real-time OSC data stream

### HTTP
- `GET /` - Web dashboard interface
- `GET /api/stats` - Server statistics (JSON)

## Message Format

OSC messages are converted to JSON format for WebSocket transmission:

```json
{
  "type": "osc_message",
  "address": "/pose/landmarks",
  "args": [0.5, 0.3, 0.1, 0.9, 0.6, 0.2, 0.8, ...],
  "timestamp": "2024-01-15T10:30:45.123456",
  "arg_count": 132,
  "data_type": "pose",
  "pose_info": {
    "landmark_count": 33,
    "has_confidence": true
  }
}
```

### Data Types

The bridge automatically categorizes OSC messages:

- **Pose Data** (`/pose/*`) - Body pose landmarks and pose-related data
- **Body Data** (`/body/*`) - Body orientation, center position, scale
- **Hand Data** (`/hand/*`) - Hand landmarks and gestures
- **Face Data** (`/face/*`) - Face landmarks and emotions
- **System Data** (`/system/*`) - FPS, memory usage, processing time
- **Unknown** - Other OSC messages

## Web Interface Features

### Dashboard
- **Connection Status** - Real-time WebSocket connection indicator
- **Statistics** - Messages received/sent, active clients, uptime
- **Server Info** - Port configuration and protocol details

### Live Message Log
- **Real-time Display** - See OSC messages as they arrive
- **Filtering** - Filter by OSC address or data type
- **Search** - Search through message history
- **Controls** - Pause/resume, clear log, auto-scroll
- **Timestamps** - Optional timestamp display
- **Line Limits** - Configurable maximum log lines

## Integration with Dance Embedding System

The bridge is designed to work seamlessly with the dance embedding system:

1. **Start the bridge server**:
   ```bash
   uvicorn src.osc_to_websocket.server:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the dance embedding system with OSC streaming**:
   ```bash
   uv run examples/osc_streaming_example.py --skip-matching
   ```

3. **View real-time pose data** in the web interface at http://localhost:8000

## Development

### Project Structure
```
src/osc_to_websocket/
├── server.py              # Main FastAPI server
├── test_client.py         # OSC test client
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── static/
    └── index.html        # Web dashboard interface
```

### Adding Custom OSC Handlers

To add custom OSC message handlers, modify the `create_osc_dispatcher()` function in `server.py`:

```python
def create_osc_dispatcher():
    dispatcher = Dispatcher()
    
    def custom_handler(address: str, *args):
        # Custom processing logic
        message = {
            "type": "custom_message",
            "address": address,
            "args": list(args),
            "timestamp": datetime.now().isoformat()
        }
        asyncio.create_task(broadcast_to_clients(message))
    
    # Add custom handlers
    dispatcher.map("/custom/address", custom_handler)
    
    return dispatcher
```

### WebSocket Client Integration

To integrate with your own web application:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'osc_message') {
        // Handle OSC message
        console.log('OSC:', data.address, data.args);
        
        // Process pose data
        if (data.data_type === 'pose') {
            // Update pose visualization
        }
    }
};
```

## Troubleshooting

### Common Issues

1. **OSC messages not appearing**:
   - Check that the dance embedding system is sending to port 6448
   - Verify the bridge server is running
   - Check firewall settings

2. **WebSocket connection fails**:
   - Ensure the bridge server is running on port 8000
   - Check browser console for errors
   - Verify CORS settings if accessing from different domain

3. **High CPU usage**:
   - Reduce message frequency in test client
   - Increase log line limits
   - Filter messages to reduce processing

4. **Dependencies not found**:
   - Make sure you've run `uv sync --dev` to install all dependencies
   - Use `uv run` instead of `python` for all commands

### Logs

The server logs are displayed in the terminal where you started the bridge. Look for:
- OSC server startup messages
- WebSocket connection events
- Error messages

## Performance

### Recommended Settings

- **Pose data rate**: 30 Hz (0.033s interval)
- **Other data rate**: 10 Hz (0.1s interval)
- **Max log lines**: 200-500
- **WebSocket clients**: < 50 for optimal performance

### Scaling

For high-volume applications:
- Consider using Redis for message queuing
- Implement client-specific message filtering
- Use binary protocols (MessagePack) for large payloads
- Deploy behind a reverse proxy (nginx)

## License

Part of the dance embedding system project.
