#!/usr/bin/env python3
"""
Test Raw Pose Landmarks Streaming

This script tests the raw pose landmarks streaming functionality.
"""

import asyncio
import websockets
import json
import time
import threading
from pythonosc.udp_client import SimpleUDPClient


async def websocket_listener():
    """Listen for WebSocket messages and filter for raw pose landmarks"""
    uri = "ws://localhost:8000/ws"
    
    print("🔌 Connecting to WebSocket...")
    async with websockets.connect(uri) as websocket:
        print("✅ WebSocket connected!")
        
        # Listen for messages
        message_count = 0
        raw_pose_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 15:  # Listen for 15 seconds
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                message_count += 1
                
                data = json.loads(message)
                if data.get('type') == 'osc_message':
                    address = data.get('address', '')
                    args = data.get('args', [])
                    
                    if '/pose/raw_landmarks' in address:
                        raw_pose_count += 1
                        print(f"📨 Raw Pose Landmarks #{raw_pose_count}:")
                        print(f"   Address: {address}")
                        print(f"   Args count: {len(args)}")
                        print(f"   Expected: 132 values (33 landmarks × 4 values)")
                        
                        # Reconstruct as array of arrays
                        if len(args) == 132:
                            landmarks_array = []
                            for i in range(0, len(args), 4):
                                landmark = [args[i], args[i+1], args[i+2], args[i+3]]
                                landmarks_array.append(landmark)
                            
                            print(f"   ✅ Reconstructed as array of {len(landmarks_array)} subarrays:")
                            print(f"   First landmark: {landmarks_array[0]}")
                            print(f"   Last landmark: {landmarks_array[-1]}")
                            print(f"   Sample landmarks: {landmarks_array[:3]}...")
                        else:
                            print(f"   ❌ Unexpected data length: {len(args)}")
                        print()
                    elif '/pose/data' in address:
                        print(f"📊 Processed Pose Data: {len(args)} values")
                    else:
                        print(f"📨 Other OSC: {address} ({len(args)} values)")
                        
            except asyncio.TimeoutError:
                print(".", end="", flush=True)
                continue
        
        print(f"\n📊 Summary:")
        print(f"   Total messages: {message_count}")
        print(f"   Raw pose landmarks: {raw_pose_count}")
        print(f"   Duration: {time.time() - start_time:.1f}s")


def send_test_osc():
    """Send test OSC messages to trigger raw pose streaming"""
    print("📡 Sending test OSC messages...")
    time.sleep(3)  # Wait for WebSocket to connect
    
    client = SimpleUDPClient("127.0.0.1", 6448)
    
    # Send some test messages
    for i in range(5):
        client.send_message("/test/raw_pose", [i, f"test_{i}", i * 0.1])
        print(f"📤 Sent test OSC message {i}")
        time.sleep(1)
    
    print("📡 Test OSC sender finished")


async def main():
    print("🎭 Test Raw Pose Landmarks Streaming")
    print("=" * 50)
    print("Make sure the dance embedding system is running with --raw-pose flag:")
    print("uv run examples/osc_streaming_example.py --raw-pose --skip-matching")
    print("=" * 50)
    
    # Start OSC sender in background
    osc_thread = threading.Thread(target=send_test_osc, daemon=True)
    osc_thread.start()
    
    # Test WebSocket
    await websocket_listener()


if __name__ == "__main__":
    asyncio.run(main())
