#!/usr/bin/env python3
"""
Test Simple WebSocket Connection

This script connects to the simple server and shows all messages received.
"""

import asyncio
import websockets
import json
import time
from pythonosc.udp_client import SimpleUDPClient


async def websocket_test():
    """Test WebSocket connection"""
    uri = "ws://localhost:8000/ws"
    
    print("🔌 Connecting to simple server...")
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to simple server!")
        
        # Listen for messages
        message_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 10:  # Listen for 10 seconds
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                message_count += 1
                
                data = json.loads(message)
                if data.get('type') == 'osc_message':
                    print(f"📨 OSC Message {message_count}: {data.get('address')} ({data.get('arg_count')} args)")
                    print(f"   Args: {data.get('args', [])[:5]}...")
                elif data.get('type') == 'welcome':
                    print(f"👋 Welcome: {data.get('message')}")
                elif data.get('type') == 'stats':
                    print(f"📊 Stats: {data.get('messages_received')} received, {data.get('messages_sent')} sent")
                else:
                    print(f"📨 Other: {data}")
                    
            except asyncio.TimeoutError:
                print(".", end="", flush=True)
                continue
        
        print(f"\n📊 Received {message_count} messages in {time.time() - start_time:.1f} seconds")


def send_test_osc():
    """Send test OSC messages"""
    print("📡 Sending test OSC messages...")
    time.sleep(2)  # Wait for WebSocket to connect
    
    client = SimpleUDPClient("127.0.0.1", 6448)
    
    for i in range(5):
        client.send_message("/test/simple", [i, f"hello_{i}", i * 0.1])
        print(f"📤 Sent OSC message {i}")
        time.sleep(1)
    
    print("📡 OSC sender finished")


async def main():
    print("🎭 Test Simple WebSocket Connection")
    print("=" * 40)
    
    # Start OSC sender in background
    import threading
    osc_thread = threading.Thread(target=send_test_osc, daemon=True)
    osc_thread.start()
    
    # Test WebSocket
    await websocket_test()


if __name__ == "__main__":
    asyncio.run(main())
