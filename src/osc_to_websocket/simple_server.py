#!/usr/bin/env python3
"""
Simple OSC to WebSocket Bridge Server

A minimal implementation with extensive logging for debugging.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_osc_bridge")

app = FastAPI(title="Simple OSC Bridge", version="1.0.0")

# Track connected WebSocket clients
clients: Set[WebSocket] = set()

# Statistics
stats = {
    "messages_received": 0,
    "messages_sent": 0,
    "clients_connected": 0,
    "start_time": datetime.now()
}


@app.get("/")
async def index():
    """Simple web interface"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple OSC Bridge</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            #log { background: #f0f0f0; padding: 20px; height: 400px; overflow-y: scroll; }
            .message { margin: 5px 0; padding: 5px; border-left: 3px solid #007acc; }
            .osc { border-left-color: #28a745; }
            .ws { border-left-color: #dc3545; }
            .connection { border-left-color: #ffc107; }
        </style>
    </head>
    <body>
        <h1>Simple OSC Bridge</h1>
        <p>WebSocket connected: <span id="status">Disconnected</span></p>
        <p>Messages received: <span id="count">0</span></p>
        <div id="log"></div>
        
        <script>
            const log = document.getElementById('log');
            const status = document.getElementById('status');
            const count = document.getElementById('count');
            let messageCount = 0;
            
            function addLog(message, type = 'message') {
                const div = document.createElement('div');
                div.className = `message ${type}`;
                div.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                log.appendChild(div);
                log.scrollTop = log.scrollHeight;
            }
            
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                status.textContent = 'Connected';
                status.style.color = 'green';
                addLog('WebSocket connected', 'connection');
            };
            
            ws.onclose = () => {
                status.textContent = 'Disconnected';
                status.style.color = 'red';
                addLog('WebSocket disconnected', 'connection');
            };
            
            ws.onmessage = (event) => {
                messageCount++;
                count.textContent = messageCount;
                
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'osc_message') {
                        addLog(`OSC: ${data.address} (${data.arg_count} args)`, 'osc');
                    } else {
                        addLog(`WS: ${data.type} - ${data.message || JSON.stringify(data)}`, 'ws');
                    }
                } catch (e) {
                    addLog(`Raw: ${event.data}`, 'ws');
                }
            };
            
            ws.onerror = (error) => {
                addLog(`Error: ${error}`, 'ws');
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


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
    """WebSocket endpoint"""
    print(f"🔌 New WebSocket connection attempt from {websocket.client}")
    
    await websocket.accept()
    clients.add(websocket)
    stats["clients_connected"] += 1
    
    print(f"✅ WebSocket connected! Total clients: {len(clients)}")
    print(f"📊 Current stats: {stats}")
    
    try:
        # Send welcome message
        welcome_msg = {
            "type": "welcome",
            "message": "Connected to Simple OSC Bridge",
            "timestamp": datetime.now().isoformat(),
            "client_count": len(clients)
        }
        
        await websocket.send_text(json.dumps(welcome_msg))
        print(f"📤 Sent welcome message to client")
        
        # Keep connection alive and send periodic updates
        while True:
            if websocket.application_state != WebSocketState.CONNECTED:
                print("❌ WebSocket disconnected")
                break
                
            # Send periodic stats update
            stats_msg = {
                "type": "stats",
                "messages_received": stats["messages_received"],
                "messages_sent": stats["messages_sent"],
                "clients_connected": len(clients),
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(stats_msg))
            print(f"📊 Sent stats update: {stats_msg}")
            
            await asyncio.sleep(5)  # Send stats every 5 seconds
            
    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected normally")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        clients.discard(websocket)
        stats["clients_connected"] = max(0, stats["clients_connected"] - 1)
        print(f"🧹 WebSocket client removed. Total clients: {len(clients)}")


def osc_message_handler(address: str, *args):
    """Handle incoming OSC messages"""
    stats["messages_received"] += 1
    
    print(f"📨 OSC MESSAGE RECEIVED:")
    print(f"   Address: {address}")
    print(f"   Args: {len(args)} values")
    print(f"   First few args: {args[:5] if args else 'None'}")
    print(f"   Total messages received: {stats['messages_received']}")
    print(f"   Active WebSocket clients: {len(clients)}")
    
    # Create message for WebSocket clients
    message = {
        "type": "osc_message",
        "address": address,
        "args": list(args),
        "arg_count": len(args),
        "timestamp": datetime.now().isoformat(),
        "message_id": stats["messages_received"]
    }
    
    # Send to all connected WebSocket clients
    if clients:
        print(f"📤 Forwarding to {len(clients)} WebSocket client(s)...")
        
        # Use asyncio.create_task to send to all clients
        async def send_to_all():
            failed_clients = []
            for client in list(clients):
                try:
                    await client.send_text(json.dumps(message))
                    stats["messages_sent"] += 1
                    print(f"   ✅ Sent to client {id(client)}")
                except Exception as e:
                    print(f"   ❌ Failed to send to client {id(client)}: {e}")
                    failed_clients.append(client)
            
            # Remove failed clients
            for client in failed_clients:
                clients.discard(client)
                stats["clients_connected"] = max(0, stats["clients_connected"] - 1)
                print(f"   🧹 Removed failed client {id(client)}")
        
        # Schedule the send operation
        asyncio.create_task(send_to_all())
    else:
        print("⚠️  No WebSocket clients connected - message not forwarded")


def create_osc_dispatcher():
    """Create OSC dispatcher"""
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(osc_message_handler)
    print("🎛️  OSC dispatcher created with default handler")
    return dispatcher


async def start_osc_server(host: str = "0.0.0.0", port: int = 6448):
    """Start the OSC server"""
    try:
        print(f"🚀 Starting OSC server on {host}:{port}")
        dispatcher = create_osc_dispatcher()
        server = AsyncIOOSCUDPServer((host, port), dispatcher, asyncio.get_running_loop())
        transport, protocol = await server.create_serve_endpoint()
        
        print(f"✅ OSC server listening on udp://{host}:{port}")
        return transport, protocol
    except Exception as e:
        print(f"❌ Failed to start OSC server: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup"""
    print("🎭 Starting Simple OSC Bridge Server")
    print("=" * 50)
    
    # Start OSC server
    await start_osc_server("0.0.0.0", 6448)
    
    print("✅ Simple OSC Bridge started successfully")
    print("🌐 Web interface: http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("📡 OSC endpoint: udp://0.0.0.0:6448")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down Simple OSC Bridge Server")
    print(f"📊 Final stats: {stats}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
