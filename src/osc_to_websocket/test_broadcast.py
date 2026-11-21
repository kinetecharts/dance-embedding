#!/usr/bin/env python3
"""
Test OSC Message Broadcasting

This script tests if OSC messages are being properly broadcast to WebSocket clients.
"""

import asyncio
import websockets
import json
import time
import threading
from pythonosc.udp_client import SimpleUDPClient


async def websocket_client():
    """Connect as a WebSocket client and listen for messages"""
    uri = "ws://localhost:8000/ws"
    
    print("🔌 Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✅ WebSocket connected!")
        
        # Listen for the welcome message
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
                    if message_count >= 5:  # Stop after 5 messages
                        break
                else:
                    print(f"📨 Other Message: {data.get('type')}")
                    
            except asyncio.TimeoutError:
                print(".", end="", flush=True)
                continue
        
        print(f"\n📊 Received {message_count} messages in {time.time() - start_time:.1f} seconds")


def send_osc_messages():
    """Send OSC messages in a separate thread"""
    print("📡 Starting OSC sender in 3 seconds...")
    time.sleep(3)
    
    client = SimpleUDPClient("127.0.0.1", 6448)
    
    for i in range(10):
        client.send_message("/test/broadcast", [i, f"test_{i}", i * 0.1])
        print(f"📤 Sent OSC message {i}")
        time.sleep(1)
    
    print("📡 OSC sender finished")


async def main():
    print("🎭 Test OSC Message Broadcasting")
    print("=" * 40)
    
    # Start OSC sender in a separate thread
    osc_thread = threading.Thread(target=send_osc_messages, daemon=True)
    osc_thread.start()
    
    # Start WebSocket client
    await websocket_client()


if __name__ == "__main__":
    asyncio.run(main())
