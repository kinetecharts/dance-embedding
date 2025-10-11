#!/usr/bin/env python3
"""
OSC Bridge Startup Script

Convenient script to start the OSC to WebSocket bridge with proper configuration.
"""

import uvicorn
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Start OSC to WebSocket Bridge")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--osc-port", type=int, default=6448, help="OSC listening port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    
    args = parser.parse_args()
    
    print("🎭 Starting OSC to WebSocket Bridge")
    print(f"📡 OSC listening on: udp://0.0.0.0:{args.osc_port}")
    print(f"🌐 WebSocket server: http://{args.host}:{args.port}")
    print(f"📊 Web dashboard: http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "src.osc_to_websocket.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Bridge stopped by user")
    except Exception as e:
        print(f"❌ Error starting bridge: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
