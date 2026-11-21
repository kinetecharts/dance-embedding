#!/usr/bin/env python3
"""
Debug Message Queue

This script tests the message queue approach by sending OSC messages
and checking if they appear in the WebSocket.
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
        
        # Listen for welcome message
        welcome = await websocket.recv()
        welcome_data = json.loads(welcome)
        print(f"👋 Welcome: {welcome_data.get('message')}")
        
        # Listen for OSC messages
        message_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 15:  # Listen for 15 seconds
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                message_count += 1
                
                data = json.loads(message)
                if data.get('type') == 'osc_message':
                    print(f"📨 OSC Message {message_count}: {data.get('address')} ({len(data.get('args', []))} args)")
                else:
                    print(f"📨 Other Message: {data.get('type')}")
                    
            except asyncio.TimeoutError:
                print(".", end="", flush=True)
                continue
        
        print(f"\n📊 Received {message_count} messages in {time.time() - start_time:.1f} seconds")


def send_osc_messages():
    """Send OSC messages"""
    print("📡 Starting OSC sender in 2 seconds...")
    time.sleep(2)
    
    client = SimpleUDPClient("127.0.0.1", 6448)
    
    for i in range(10):
        client.send_message("/test/queue", [i, f"test_{i}", i * 0.1])
        print(f"📤 Sent OSC message {i}")
        time.sleep(1)
    
    print("📡 OSC sender finished")


async def main():
    print("🎭 Debug Message Queue")
    print("=" * 30)
    
    # Start OSC sender in a separate thread
    osc_thread = threading.Thread(target=send_osc_messages, daemon=True)
    osc_thread.start()
    
    # Start WebSocket listener
    await websocket_listener()


if __name__ == "__main__":
    asyncio.run(main())
