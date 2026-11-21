#!/usr/bin/env python3
"""
Test UV Integration for OSC Bridge

Quick test to verify that the OSC bridge works with uv dependency management.
"""

import subprocess
import sys
import time
import threading
import json


def test_dependencies():
    """Test that all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import websockets
        from pythonosc.udp_client import SimpleUDPClient
        print("✅ All dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: uv sync --dev")
        return False


def test_osc_bridge_startup():
    """Test that the OSC bridge can start up"""
    print("🚀 Testing OSC bridge startup...")
    
    try:
        # Start the bridge server
        process = subprocess.Popen([
            "uv", "run", "uvicorn",
            "src.osc_to_websocket.server:app",
            "--host", "127.0.0.1",
            "--port", "8001",  # Use different port for testing
            "--log-level", "error"  # Reduce log noise
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if server is running by trying to connect
        try:
            import urllib.request
            import urllib.error
            
            response = urllib.request.urlopen("http://127.0.0.1:8001/api/stats", timeout=5)
            if response.getcode() == 200:
                print("✅ OSC bridge started successfully")
                stats = json.loads(response.read().decode())
                print(f"   Server uptime: {stats.get('uptime_seconds', 0):.1f}s")
                return True, process
            else:
                print(f"❌ Server responded with status {response.getcode()}")
                return False, process
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"❌ Could not connect to server: {e}")
            return False, process
            
    except Exception as e:
        print(f"❌ Failed to start OSC bridge: {e}")
        return False, None


def test_osc_message_flow():
    """Test OSC message flow through the bridge"""
    print("📡 Testing OSC message flow...")
    
    try:
        from pythonosc.udp_client import SimpleUDPClient
        
        # Send test OSC message
        client = SimpleUDPClient("127.0.0.1", 6448)
        client.send_message("/test/message", [1, 2, 3, "hello"])
        
        print("✅ OSC message sent successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send OSC message: {e}")
        return False


def test_websocket_connection():
    """Test WebSocket connection to the bridge"""
    print("🔌 Testing WebSocket connection...")
    
    try:
        import websockets
        import asyncio
        
        async def test_ws():
            uri = "ws://127.0.0.1:8001/ws"
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                message = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(message)
                
                if data.get("type") == "welcome":
                    print("✅ WebSocket connection successful")
                    return True
                else:
                    print(f"❌ Unexpected message: {data}")
                    return False
        
        result = asyncio.run(test_ws())
        return result
        
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False


def main():
    print("🎭 Testing OSC Bridge UV Integration")
    print("=" * 50)
    
    # Test 1: Dependencies
    if not test_dependencies():
        return False
    
    # Test 2: Bridge startup
    success, process = test_osc_bridge_startup()
    if not success:
        return False
    
    try:
        # Test 3: OSC message flow
        test_osc_message_flow()
        
        # Test 4: WebSocket connection
        test_websocket_connection()
        
        print("\n🎉 All tests passed! OSC bridge is working with uv")
        return True
        
    finally:
        # Cleanup
        if process:
            print("🧹 Cleaning up...")
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
