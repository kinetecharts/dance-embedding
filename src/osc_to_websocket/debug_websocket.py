#!/usr/bin/env python3
"""
Debug WebSocket and OSC Integration

This script connects to WebSocket and sends OSC messages to test the flow.
"""

import asyncio
import websockets
import json
import time
import threading
from pythonosc.udp_client import SimpleUDPClient


async def websocket_listener():
    """Listen for WebSocket messages"""
    uri = "ws://localhost:8000/ws"
    
    print("🔌 Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✅ WebSocket connected!")
        
        # Listen for messages
        message_count = 0
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                message_count += 1
                
                try:
                    data = json.loads(message)
                    if data.get('type') == 'osc_message':
                        print(f"📨 OSC Message {message_count}: {data.get('address')} ({len(data.get('args', []))} args)")
                    elif data.get('type') == 'welcome':
                        print(f"👋 Welcome: {data.get('message')}")
                    else:
                        print(f"📨 Other Message {message_count}: {data.get('type')}")
                except json.JSONDecodeError:
                    print(f"📨 Raw Message {message_count}: {message[:100]}...")
                    
            except asyncio.TimeoutError:
                print(".", end="", flush=True)
                continue


def osc_sender():
    """Send OSC messages"""
    print("📡 Starting OSC sender...")
    time.sleep(2)  # Wait for WebSocket to connect
    
    client = SimpleUDPClient("127.0.0.1", 6448)
    
    for i in range(20):
        # Send test message
        client.send_message("/test/debug", [i, f"message_{i}", i * 0.1])
        print(f"📤 Sent OSC message {i}")
        time.sleep(0.5)
    
    print("📡 OSC sender finished")


async def main():
    print("🎭 Debug WebSocket and OSC Integration")
    print("=" * 50)
    
    # Start OSC sender in a separate thread
    osc_thread = threading.Thread(target=osc_sender, daemon=True)
    osc_thread.start()
    
    # Start WebSocket listener
    await websocket_listener()


if __name__ == "__main__":
    asyncio.run(main())
