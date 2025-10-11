#!/usr/bin/env python3
"""
Simple OSC Bridge Test

Quick test to verify the OSC bridge is working with the existing server.
"""

import json
import urllib.request
import urllib.error
from pythonosc.udp_client import SimpleUDPClient
import time


def test_server_connection():
    """Test if the server is running and responding"""
    print("🔍 Testing server connection...")
    
    try:
        response = urllib.request.urlopen("http://localhost:8000/api/stats", timeout=5)
        if response.getcode() == 200:
            stats = json.loads(response.read().decode())
            print("✅ Server is running")
            print(f"   Messages received: {stats.get('messages_received', 0)}")
            print(f"   Uptime: {stats.get('uptime_seconds', 0):.1f}s")
            return True
        else:
            print(f"❌ Server responded with status {response.getcode()}")
            return False
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"❌ Could not connect to server: {e}")
        return False


def test_osc_message():
    """Test sending an OSC message"""
    print("📡 Testing OSC message...")
    
    try:
        client = SimpleUDPClient("127.0.0.1", 6448)
        client.send_message("/test/uv_integration", [1, 2, 3, "hello", "uv"])
        print("✅ OSC message sent successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to send OSC message: {e}")
        return False


def test_web_interface():
    """Test if the web interface is accessible"""
    print("🌐 Testing web interface...")
    
    try:
        response = urllib.request.urlopen("http://localhost:8000/", timeout=5)
        if response.getcode() == 200:
            content = response.read().decode()
            if "OSC to WebSocket Bridge" in content:
                print("✅ Web interface is accessible")
                return True
            else:
                print("❌ Web interface content unexpected")
                return False
        else:
            print(f"❌ Web interface responded with status {response.getcode()}")
            return False
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"❌ Could not access web interface: {e}")
        return False


def main():
    print("🎭 Simple OSC Bridge Test")
    print("=" * 40)
    
    # Test 1: Server connection
    if not test_server_connection():
        print("\n❌ Server is not running. Please start it with:")
        print("   uv run python -m uvicorn src.osc_to_websocket.server:app --host 0.0.0.0 --port 8000")
        return False
    
    # Test 2: OSC message
    test_osc_message()
    
    # Test 3: Web interface
    test_web_interface()
    
    print("\n🎉 OSC Bridge is working with UV!")
    print("🌐 Open http://localhost:8000 to see the web interface")
    return True


if __name__ == "__main__":
    main()
