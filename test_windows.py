#!/usr/bin/env python3
"""Simple test to verify OpenCV windows are working."""

import cv2
import numpy as np
import time

def test_opencv_windows():
    """Test if OpenCV windows can be created and displayed."""
    print("Testing OpenCV windows...")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "OpenCV Test Window", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_image, "Press 'q' to close", (50, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Try to create window
    try:
        cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Window", test_image)
        print("✅ Test window created successfully")
        print("You should see a black window with white text")
        print("Press 'q' in the window to close it")
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            time.sleep(0.1)
        
        cv2.destroyAllWindows()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error creating test window: {e}")
        print("This might indicate an issue with OpenCV or display")

def test_camera():
    """Test if camera can be accessed."""
    print("\nTesting camera access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read camera frame")
            cap.release()
            return False
        
        print(f"✅ Camera working - frame shape: {frame.shape}")
        
        # Show camera frame
        cv2.imshow("Camera Test", frame)
        print("You should see your camera feed")
        print("Press 'q' to close")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera test completed")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

if __name__ == "__main__":
    print("OpenCV Window Test")
    print("=" * 50)
    
    # Test basic windows
    test_opencv_windows()
    
    # Test camera
    test_camera()
    
    print("\nTest completed!") 