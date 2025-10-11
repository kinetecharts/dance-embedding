"""
OSC to WebSocket Bridge Server

This service listens for OSC data streams and forwards them to web clients via WebSocket.
Designed for the dance embedding system to stream pose data to web applications.

Usage:
    uvicorn src.osc_to_websocket.server:app --reload --host 0.0.0.0 --port 8000

The server will:
- Listen for OSC on UDP port 6448 (default dance embedding OSC port)
- Forward all OSC messages to connected WebSocket clients
- Provide a simple web interface for testing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from starlette.websockets import WebSocketState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="OSC to WebSocket Bridge",
    description="Forwards OSC data streams to web clients via WebSocket",
    version="1.0.0"
)

# Track connected WebSocket clients
clients: Set[WebSocket] = set()

# Message storage for OSC to WebSocket forwarding
recent_messages = []
max_recent_messages = 1000

# Statistics
stats = {
    "messages_received": 0,
    "messages_sent": 0,
    "clients_connected": 0,
    "start_time": datetime.now(),
    "last_message_time": None
}


@app.get("/")
async def index():
    """Serve the main web interface"""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/stats")
async def get_stats():
    """Get server statistics"""
    return {
        **stats,
        "active_clients": len(clients),
        "uptime_seconds": (datetime.now() - stats["start_time"]).total_seconds()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time OSC data streaming"""
    await websocket.accept()
    clients.add(websocket)
    stats["clients_connected"] += 1
    
    logger.info(f"WebSocket client connected. Total clients: {len(clients)}")
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "message": "Connected to OSC bridge",
            "timestamp": datetime.now().isoformat(),
            "server_stats": get_stats()
        }))
        
        # Send recent messages to the client
        last_message_count = 0
        
        while True:
            if websocket.application_state == WebSocketState.DISCONNECTED:
                break
                
            try:
                # Check if there are new messages
                current_message_count = len(recent_messages)
                if current_message_count > last_message_count:
                    # Send new messages
                    for i in range(last_message_count, current_message_count):
                        if i < len(recent_messages):
                            message = recent_messages[i]
                            await websocket.send_text(json.dumps(message))
                            stats["messages_sent"] += 1
                    
                    last_message_count = current_message_count
                
                # Wait a bit before checking again
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                break
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        clients.discard(websocket)
        stats["clients_connected"] = max(0, stats["clients_connected"] - 1)
        logger.info(f"WebSocket client removed. Total clients: {len(clients)}")


# Note: broadcast_to_clients function removed - now using message queue approach


def create_osc_dispatcher():
    """Create OSC dispatcher with handlers for pose data"""
    dispatcher = Dispatcher()
    
    def osc_message_handler(address: str, *args):
        """Handle incoming OSC messages"""
        try:
            # Update statistics
            stats["messages_received"] += 1
            stats["last_message_time"] = datetime.now().isoformat()
            
            # Create message for WebSocket clients
            message = {
                "type": "osc_message",
                "address": address,
                "args": list(args),
                "timestamp": datetime.now().isoformat(),
                "arg_count": len(args)
            }
            
            # Add type-specific processing for pose data
            if address.startswith("/pose"):
                message["data_type"] = "pose"
                # Extract pose-specific information
                if len(args) >= 3:
                    message["pose_info"] = {
                        "landmark_count": len(args) // 3,  # Assuming x, y, z per landmark
                        "has_confidence": len(args) % 3 == 0
                    }
            elif address.startswith("/body"):
                message["data_type"] = "body"
            elif address.startswith("/hand"):
                message["data_type"] = "hand"
            else:
                message["data_type"] = "unknown"
            
            # Store message for WebSocket clients
            recent_messages.append(message)
            if len(recent_messages) > max_recent_messages:
                recent_messages.pop(0)  # Remove oldest message
            
            # Log high-frequency messages less frequently
            if stats["messages_received"] % 100 == 0:
                logger.info(f"Processed {stats['messages_received']} OSC messages. Active clients: {len(clients)}")
                
        except Exception as e:
            logger.error(f"Error processing OSC message: {e}")
    
    # Set up handlers for different OSC address patterns
    dispatcher.set_default_handler(osc_message_handler)
    
    # Specific handlers for pose data (optional - default handler catches all)
    dispatcher.map("/pose/*", osc_message_handler)
    dispatcher.map("/body/*", osc_message_handler)
    dispatcher.map("/hand/*", osc_message_handler)
    dispatcher.map("/face/*", osc_message_handler)
    
    return dispatcher


async def start_osc_server(host: str = "0.0.0.0", port: int = 6448):
    """Start the OSC server"""
    try:
        dispatcher = create_osc_dispatcher()
        server = AsyncIOOSCUDPServer((host, port), dispatcher, asyncio.get_running_loop())
        transport, protocol = await server.create_serve_endpoint()
        
        logger.info(f"OSC server listening on udp://{host}:{port}")
        return transport, protocol
    except Exception as e:
        logger.error(f"Failed to start OSC server: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup"""
    # Start OSC server
    await start_osc_server("0.0.0.0", 6448)
    logger.info("OSC to WebSocket bridge started successfully")
    logger.info("Web interface available at: http://localhost:8000")
    logger.info("WebSocket endpoint: ws://localhost:8000/ws")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down OSC to WebSocket bridge")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.osc_to_websocket.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
