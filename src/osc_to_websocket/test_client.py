#!/usr/bin/env python3
"""
OSC Test Client

Sends test OSC messages to the bridge server for testing purposes.
Simulates pose data from the dance embedding system.

Usage:
    python src/osc_to_websocket/test_client.py
"""

import time
import random
import math
from pythonosc.udp_client import SimpleUDPClient
import argparse


class PoseDataSimulator:
    """Simulates pose data for testing the OSC bridge"""
    
    def __init__(self, host="127.0.0.1", port=6448):
        self.client = SimpleUDPClient(host, port)
        self.frame_count = 0
        
    def send_pose_data(self, num_landmarks=33):
        """Send pose landmark data (MediaPipe format)"""
        landmarks = []
        
        # Generate realistic pose landmarks
        for i in range(num_landmarks):
            # Simulate body movement with sine waves
            t = self.frame_count * 0.1
            x = 0.5 + 0.3 * math.sin(t + i * 0.1)
            y = 0.5 + 0.2 * math.cos(t + i * 0.15)
            z = 0.1 * math.sin(t * 2 + i * 0.05)
            confidence = 0.8 + 0.2 * random.random()
            
            landmarks.extend([x, y, z, confidence])
        
        # Send pose data
        self.client.send_message("/pose/landmarks", landmarks)
        
        # Send pose summary
        self.client.send_message("/pose/summary", [
            len(landmarks) // 4,  # landmark count
            self.frame_count,     # frame number
            time.time()           # timestamp
        ])
        
    def send_body_data(self):
        """Send body-specific data"""
        # Body orientation
        yaw = math.sin(self.frame_count * 0.05) * 30  # -30 to +30 degrees
        pitch = math.cos(self.frame_count * 0.03) * 15  # -15 to +15 degrees
        roll = math.sin(self.frame_count * 0.07) * 10   # -10 to +10 degrees
        
        self.client.send_message("/body/orientation", [yaw, pitch, roll])
        
        # Body center position
        center_x = 0.5 + 0.1 * math.sin(self.frame_count * 0.02)
        center_y = 0.5 + 0.1 * math.cos(self.frame_count * 0.025)
        
        self.client.send_message("/body/center", [center_x, center_y])
        
        # Body scale (torso length)
        scale = 1.0 + 0.1 * math.sin(self.frame_count * 0.01)
        self.client.send_message("/body/scale", [scale])
        
    def send_hand_data(self, hand_id="left"):
        """Send hand-specific data"""
        # Hand landmarks (21 per hand)
        landmarks = []
        for i in range(21):
            t = self.frame_count * 0.1
            x = 0.3 + 0.2 * math.sin(t + i * 0.1)
            y = 0.7 + 0.2 * math.cos(t + i * 0.15)
            z = 0.1 * math.sin(t * 2 + i * 0.05)
            confidence = 0.7 + 0.3 * random.random()
            
            landmarks.extend([x, y, z, confidence])
        
        self.client.send_message(f"/hand/{hand_id}/landmarks", landmarks)
        
        # Hand gesture classification
        gestures = ["fist", "open", "point", "peace", "thumbs_up"]
        gesture = random.choice(gestures)
        confidence = random.random()
        
        self.client.send_message(f"/hand/{hand_id}/gesture", [gesture, confidence])
        
    def send_face_data(self):
        """Send face-specific data"""
        # Face landmarks (reduced to 68 for UDP message size limits)
        landmarks = []
        for i in range(68):  # Reduced from 468 to 68
            t = self.frame_count * 0.05
            x = 0.5 + 0.1 * math.sin(t + i * 0.01)
            y = 0.3 + 0.1 * math.cos(t + i * 0.01)
            z = 0.05 * math.sin(t * 3 + i * 0.005)
            confidence = 0.6 + 0.4 * random.random()
            
            landmarks.extend([x, y, z, confidence])
        
        self.client.send_message("/face/landmarks", landmarks)
        
        # Face emotions
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        emotion = random.choice(emotions)
        confidence = random.random()
        
        self.client.send_message("/face/emotion", [emotion, confidence])
        
    def send_system_data(self):
        """Send system-level data"""
        # FPS
        fps = 30 + random.uniform(-2, 2)
        self.client.send_message("/system/fps", [fps])
        
        # Memory usage (simulated)
        memory_mb = 100 + random.uniform(-10, 10)
        self.client.send_message("/system/memory", [memory_mb])
        
        # Processing time
        processing_ms = 5 + random.uniform(-1, 1)
        self.client.send_message("/system/processing_time", [processing_ms])
        
    def run_simulation(self, duration=60, pose_interval=0.033, other_interval=0.1):
        """Run the complete simulation"""
        print(f"Starting OSC simulation for {duration} seconds...")
        print(f"Pose data every {pose_interval:.3f}s, other data every {other_interval:.3f}s")
        print("Press Ctrl+C to stop")
        
        start_time = time.time()
        last_pose_time = 0
        last_other_time = 0
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Send pose data at high frequency
                if current_time - last_pose_time >= pose_interval:
                    self.send_pose_data()
                    last_pose_time = current_time
                
                # Send other data at lower frequency
                if current_time - last_other_time >= other_interval:
                    self.send_body_data()
                    self.send_hand_data("left")
                    self.send_hand_data("right")
                    self.send_face_data()
                    self.send_system_data()
                    last_other_time = current_time
                
                self.frame_count += 1
                time.sleep(0.001)  # Small delay to prevent overwhelming
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        
        print(f"Simulation complete. Sent {self.frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="OSC Test Client for Dance Embedding Bridge")
    parser.add_argument("--host", default="127.0.0.1", help="OSC server host")
    parser.add_argument("--port", type=int, default=6448, help="OSC server port")
    parser.add_argument("--duration", type=int, default=60, help="Simulation duration in seconds")
    parser.add_argument("--pose-rate", type=float, default=30.0, help="Pose data rate (Hz)")
    parser.add_argument("--other-rate", type=float, default=10.0, help="Other data rate (Hz)")
    
    args = parser.parse_args()
    
    simulator = PoseDataSimulator(args.host, args.port)
    pose_interval = 1.0 / args.pose_rate
    other_interval = 1.0 / args.other_rate
    
    simulator.run_simulation(
        duration=args.duration,
        pose_interval=pose_interval,
        other_interval=other_interval
    )


if __name__ == "__main__":
    main()
