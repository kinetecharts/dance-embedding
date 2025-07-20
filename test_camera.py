#!/usr/bin/env python3
"""Simple camera test to diagnose access issues."""

import cv2
import time
import sys

def test_camera():
    """Test camera access"""
    print("Testing camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open camera 0")
        print("Trying camera 1...")
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("❌ Failed to open camera 1")
            print("Camera access denied or no camera available")
            print("\nPossible solutions:")
            print("1. Check camera permissions in System Preferences > Security & Privacy > Camera")
            print("2. Try video mode instead: python -m recall.main --mode video --input data/video/dai2.mov")
            return False
    
    print("✅ Camera opened successfully!")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera")
        cap.release()
        return False
    
    print(f"✅ Frame read successfully! Shape: {frame.shape}")
    
    # Show frame for 3 seconds
    print("Showing camera feed for 3 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Camera Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            print("❌ Failed to read frame")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed successfully!")
    return True

if __name__ == "__main__":
    test_camera() 