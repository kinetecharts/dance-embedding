#!/usr/bin/env python3
"""Listen on an OSC port and print what arrives, once per second.

Use it to check that the pose app is sending before you open Wekinator.
Close Wekinator first: only one program can listen on a port at a time.

    uv run scripts/osc_check.py            # port 6448 (what Wekinator hears)
    uv run scripts/osc_check.py 12000      # port 12000 (what Max hears from Wekinator)

Example output:
    /pose/data: 18 msgs/s, 21 values, first 3: [0.12, -0.05, 1.64]
"""

import functools
import sys
import threading
import time
from collections import defaultdict

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

print = functools.partial(print, flush=True)

port = int(sys.argv[1]) if len(sys.argv) > 1 else 6448
counts = defaultdict(int)
last = {}
lock = threading.Lock()


def on_message(address, *values):
    with lock:
        counts[address] += 1
        last[address] = values


dispatcher = Dispatcher()
dispatcher.set_default_handler(on_message)

try:
    server = ThreadingOSCUDPServer(("0.0.0.0", port), dispatcher)
except OSError:
    sys.exit(f"Port {port} is already in use. Quit Wekinator (or Max) and try again.")

threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"Listening on port {port}. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
        with lock:
            if not counts:
                print("nothing yet")
            for address, n in sorted(counts.items()):
                values = last[address]
                first = [round(v, 2) if isinstance(v, float) else v for v in values[:3]]
                print(f"{address}: {n} msgs/s, {len(values)} values, first 3: {first}")
            counts.clear()
except KeyboardInterrupt:
    server.shutdown()
