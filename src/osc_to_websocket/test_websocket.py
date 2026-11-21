#!/usr/bin/env python3
"""
Test WebSocket Connection to OSC Bridge

This script tests the WebSocket connection and shows what messages are received.
"""

import asyncio
import websockets
import json
import time


async def test_websocket_connection():
    """Test WebSocket connection and display messages"""
    uri = "ws://localhost:8000/ws"
    
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Send a test message to verify the connection
            await websocket.send(json.dumps({"type": "test", "message": "Hello from test client"}))
            
            # Listen for messages for 10 seconds
            print("📡 Listening for messages for 10 seconds...")
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < 10:
                try:
                    # Wait for a message with a timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message_count += 1
                    
                    try:
                        data = json.loads(message)
                        print(f"📨 Message {message_count}: {data.get('type', 'unknown')} - {data.get('address', 'no address')}")
                        if data.get('type') == 'osc_message':
                            print(f"   Args: {len(data.get('args', []))} values")
                    except json.JSONDecodeError:
                        print(f"📨 Message {message_count}: {message[:100]}...")
                        
                except asyncio.TimeoutError:
                    # No message received in this second
                    print(".", end="", flush=True)
                    continue
                    
            print(f"\n📊 Received {message_count} messages in 10 seconds")
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False
        
    return True


async def main():
    print("🎭 WebSocket Connection Test")
    print("=" * 40)
    
    # Test WebSocket connection
    success = await test_websocket_connection()
    
    if success:
        print("\n🎉 WebSocket test completed!")
        print("💡 If you see messages, the WebSocket is working correctly.")
        print("💡 If you don't see messages, make sure OSC data is being sent to the bridge.")
    else:
        print("\n❌ WebSocket test failed!")
        print("💡 Make sure the OSC bridge server is running on port 8000")


if __name__ == "__main__":
    asyncio.run(main())
