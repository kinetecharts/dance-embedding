#!/usr/bin/env python3
"""
OSC Bridge Demo Script

Demonstrates the complete OSC to WebSocket bridge system.
Starts both the bridge server and a test client.
"""

import subprocess
import time
import threading
import sys
from pathlib import Path


def start_bridge():
    """Start the OSC bridge server"""
    print("🚀 Starting OSC bridge server...")
    try:
        subprocess.run([
            "uv", "run", "python", "-m", "uvicorn",
            "src.osc_to_websocket.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start bridge server: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 Bridge server stopped")
        return False


def start_test_client():
    """Start the OSC test client"""
    print("📡 Starting OSC test client...")
    time.sleep(2)  # Wait for bridge to start
    
    try:
        subprocess.run([
            "uv", "run", "src/osc_to_websocket/test_client.py",
            "--duration", "30",
            "--pose-rate", "30",
            "--other-rate", "10"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start test client: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 Test client stopped")
        return False


def main():
    print("🎭 OSC to WebSocket Bridge Demo")
    print("=" * 50)
    print("This demo will:")
    print("1. Start the OSC bridge server on http://localhost:8000")
    print("2. Start a test client to send OSC messages")
    print("3. Open your browser to see the live data stream")
    print()
    print("Press Ctrl+C to stop everything")
    print("=" * 50)
    
    # Check if required packages are installed
    try:
        import fastapi
        import uvicorn
        import websockets
        from pythonosc.udp_client import SimpleUDPClient
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements:")
        print("uv sync --dev")
        return
    
    # Start bridge server in a separate thread
    bridge_thread = threading.Thread(target=start_bridge, daemon=True)
    bridge_thread.start()
    
    # Wait a moment for server to start
    time.sleep(3)
    
    # Start test client
    try:
        start_test_client()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    finally:
        print("🔄 Cleaning up...")


if __name__ == "__main__":
    main()
