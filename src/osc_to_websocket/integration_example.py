#!/usr/bin/env python3
"""
Integration Example: Dance Embedding + OSC Bridge

This example shows how to integrate the dance embedding system with the OSC bridge
for real-time pose data streaming to web applications.
"""

import subprocess
import time
import threading
import sys
import webbrowser
from pathlib import Path


def start_osc_bridge():
    """Start the OSC bridge server"""
    print("🌉 Starting OSC bridge server...")
    try:
        subprocess.run([
            "uv", "run", "python", "-m", "uvicorn",
            "src.osc_to_websocket.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start OSC bridge: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 OSC bridge stopped")
        return False


def start_dance_embedding():
    """Start the dance embedding system with OSC streaming"""
    print("💃 Starting dance embedding system...")
    time.sleep(3)  # Wait for bridge to start
    
    try:
        subprocess.run([
            "uv", "run", "examples/osc_streaming_example.py",
            "--skip-matching",
            "--record-video"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dance embedding: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 Dance embedding stopped")
        return False


def open_web_interface():
    """Open the web interface in the browser"""
    print("🌐 Opening web interface...")
    time.sleep(5)  # Wait for services to start
    
    try:
        webbrowser.open("http://localhost:8000")
        print("✅ Web interface opened in browser")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("Please open http://localhost:8000 manually")


def main():
    print("🎭 Dance Embedding + OSC Bridge Integration")
    print("=" * 60)
    print("This example will:")
    print("1. Start the OSC bridge server (http://localhost:8000)")
    print("2. Start the dance embedding system with OSC streaming")
    print("3. Open the web interface to view live pose data")
    print("4. Record video with pose overlay and audio")
    print()
    print("Requirements:")
    print("- Camera access for pose detection")
    print("- OSC bridge dependencies installed")
    print("- Dance embedding system configured")
    print()
    print("Press Ctrl+C to stop everything")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("examples/osc_streaming_example.py").exists():
        print("❌ Please run this script from the dance_embedding project root")
        return
    
    # Start OSC bridge in a separate thread
    bridge_thread = threading.Thread(target=start_osc_bridge, daemon=True)
    bridge_thread.start()
    
    # Open web interface in a separate thread
    web_thread = threading.Thread(target=open_web_interface, daemon=True)
    web_thread.start()
    
    # Start dance embedding system (this will block)
    try:
        start_dance_embedding()
    except KeyboardInterrupt:
        print("\n👋 Integration example stopped by user")
    finally:
        print("🔄 Cleaning up...")


if __name__ == "__main__":
    main()
