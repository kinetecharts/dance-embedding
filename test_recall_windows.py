#!/usr/bin/env python3
"""Minimal test to debug recall system window display."""

import cv2
import numpy as np
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_recall_windows():
    """Test the exact window creation logic from recall system."""
    print("Testing Recall System Windows...")
    print("=" * 50)
    
    # Test 1: Basic camera capture and display
    print("1. Testing camera capture...")
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
        
        print(f"✅ Camera frame shape: {frame.shape}")
        
        # Test 2: Create live camera window (like recall system)
        print("\n2. Creating Live Camera window...")
        try:
            # Try different window creation methods for macOS
            cv2.namedWindow("Live Camera", cv2.WINDOW_AUTOSIZE)  # Changed to AUTOSIZE
            cv2.moveWindow("Live Camera", 100, 100)
            
            # Add text to frame like recall system
            display_frame = frame.copy()
            cv2.putText(display_frame, "Live Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Live Camera", display_frame)
            print("✅ Live Camera window created")
            print("   Position: (100, 100)")
            print("   Type: WINDOW_AUTOSIZE")
            
        except Exception as e:
            print(f"❌ Error creating Live Camera window: {e}")
            return False
        
        # Test 3: Create matched video window
        print("\n3. Creating Matched Video window...")
        try:
            # Create a test video frame
            test_video_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_video_frame, "Matched Video", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(test_video_frame, "Test Match", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(test_video_frame, "Score: 0.850", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(test_video_frame, "Press 'q' to close", (10, test_video_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Create and position window exactly like recall system
            cv2.namedWindow("Matched Video", cv2.WINDOW_AUTOSIZE)  # Changed to AUTOSIZE
            cv2.moveWindow("Matched Video", 800, 100)
            
            # Display frame
            cv2.imshow("Matched Video", test_video_frame)
            print("✅ Matched Video window created")
            print("   Position: (800, 100)")
            print("   Type: WINDOW_AUTOSIZE")
            
        except Exception as e:
            print(f"❌ Error creating Matched Video window: {e}")
            return False
        
        # Test 4: Run display loop
        print("\n4. Running display loop...")
        print("You should see TWO windows:")
        print("  - Live Camera (left at 100,100)")
        print("  - Matched Video (right at 800,100)")
        print("If you don't see them:")
        print("  1. Check if they're behind other windows")
        print("  2. Look for windows with these exact names")
        print("  3. Try moving your mouse to the positions")
        print("  4. Check your Dock for OpenCV windows")
        print("  5. Try Cmd+Tab to cycle through windows")
        print("Press 'q' in either window to quit")
        
        # Force window to front and wait a moment
        cv2.waitKey(100)  # Wait for windows to appear
        
        start_time = time.time()
        while True:
            # Update live camera frame
            ret, frame = cap.read()
            if ret:
                display_frame = frame.copy()
                cv2.putText(display_frame, "Live Camera", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Time: {time.time() - start_time:.1f}s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Live Camera", display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            time.sleep(0.033)  # ~30 FPS
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_recall_windows() 