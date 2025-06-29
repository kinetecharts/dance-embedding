# Pose Extraction Component - Requirements

## Overview

The Pose Extraction Component is responsible for extracting pose sequences from dance videos using the Rerun framework, following their human pose tracking example. The extracted pose data will be saved as CSV files with timestamps for synchronized playback with the original video.

## Goals

### Primary Goals
1. **Video Processing**: Extract pose data from videos located in `data/video/` folder
2. **Rerun Integration**: Use Rerun framework's human pose tracking capabilities
3. **Data Export**: Save pose sequences as CSV files with timestamps
4. **Synchronization**: Ensure pose data can be played back in sync with original video

### Secondary Goals
- Batch processing of multiple video files
- Quality validation of extracted pose data
- Performance optimization for large video files
- Error handling for corrupted or unsupported video formats

## Technical Requirements

### Core Technologies

#### Rerun Framework Integration
- **Rerun SDK**: Use Rerun's Python SDK for pose tracking
- **Human Pose Tracking**: Follow Rerun's human pose tracking example
- **Real-time Processing**: Leverage Rerun's efficient pose detection pipeline
- **Visualization**: Use Rerun's built-in visualization capabilities for debugging

#### Video Processing
- **Supported Formats**: MP4, AVI, MOV, WebM (same as Rerun supports)
- **Frame Extraction**: Extract frames at configurable FPS (24-60)
- **Resolution Handling**: Support various video resolutions
- **Batch Processing**: Process multiple videos in sequence

#### Data Storage
- **CSV Format**: Save pose data in comma-separated values format
- **Timestamp Column**: Include timestamp for each pose frame
- **Pose Keypoints**: Store all 33 MediaPipe pose keypoints
- **Confidence Scores**: Include confidence scores for each keypoint

### File Structure Requirements

#### Input Structure
```
data/
└── video/
    ├── dance_001.mp4
    ├── dance_002.avi
    ├── performance_001.mov
    └── ...
```

#### Output Structure
```
data/
├── video/
│   ├── dance_001.mp4
│   ├── dance_002.avi
│   └── ...
└── poses/
    ├── dance_001.csv
    ├── dance_002.csv
    └── ...
```

#### CSV File Format
```csv
timestamp,frame_number,nose_x,nose_y,nose_confidence,left_eye_x,left_eye_y,left_eye_confidence,...,right_ankle_x,right_ankle_y,right_ankle_confidence
0.0,0,100.5,200.3,0.95,95.2,195.1,0.92,...,150.8,450.2,0.88
0.033,1,101.2,201.1,0.94,96.1,196.3,0.91,...,151.5,451.8,0.87
...
```

## Functional Requirements

### Video Processing
- [ ] **Video Discovery**: Automatically detect videos in `data/video/` folder
- [ ] **Format Validation**: Validate video format compatibility
- [ ] **Metadata Extraction**: Extract video metadata (FPS, duration, resolution)
- [ ] **Frame Extraction**: Extract frames at consistent FPS for pose detection

### Pose Extraction
- [ ] **Rerun Integration**: Implement Rerun's human pose tracking pipeline
- [ ] **Keypoint Detection**: Extract all 33 MediaPipe pose keypoints
- [ ] **Confidence Scoring**: Capture confidence scores for each keypoint
- [ ] **Real-time Processing**: Process frames in real-time or near real-time

### Data Export
- [ ] **CSV Generation**: Create CSV files with pose data and timestamps
- [ ] **File Naming**: Match CSV filename to video filename (e.g., `dance_001.mp4` → `dance_001.csv`)
- [ ] **Timestamp Accuracy**: Ensure precise timestamp alignment with video frames
- [ ] **Data Validation**: Validate exported data for completeness and accuracy

### Synchronization
- [ ] **Frame Alignment**: Ensure pose data aligns with video frames
- [ ] **Timestamp Precision**: Maintain sub-frame timestamp precision
- [ ] **Playback Compatibility**: Enable synchronized playback with original video
- [ ] **Frame Rate Consistency**: Handle different input/output frame rates

## Implementation Requirements

### Rerun Framework Setup
```python
import rerun as rr
from rerun import RecordingStreamBuilder

# Initialize Rerun recording
rec = RecordingStreamBuilder("pose_extraction").connect()

# Set up pose tracking
rr.init("pose_extraction", spawn=True)
```

### Pose Extraction Pipeline
```python
def extract_pose_from_video(video_path: str, output_path: str):
    """
    Extract pose data from video using Rerun framework
    
    Args:
        video_path: Path to input video file
        output_path: Path to output CSV file
    """
    # 1. Load video using Rerun
    # 2. Extract frames at consistent FPS
    # 3. Apply pose detection to each frame
    # 4. Collect pose data with timestamps
    # 5. Export to CSV format
    pass
```

### CSV Export Format
```python
def export_pose_to_csv(pose_data: List[Dict], output_path: str):
    """
    Export pose data to CSV format with timestamps
    
    Args:
        pose_data: List of pose frames with timestamps
        output_path: Path to output CSV file
    """
    # Define CSV columns
    columns = ['timestamp', 'frame_number'] + [
        f"{keypoint}_{coord}" 
        for keypoint in POSE_KEYPOINTS 
        for coord in ['x', 'y', 'confidence']
    ]
    
    # Write CSV file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(pose_data)
```

## Data Schema

### Pose Keypoints (33 MediaPipe Keypoints)
```python
POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]
```

### CSV Column Structure
- **timestamp**: Frame timestamp in seconds (float)
- **frame_number**: Sequential frame number (integer)
- **{keypoint}_x**: X coordinate for each keypoint (float)
- **{keypoint}_y**: Y coordinate for each keypoint (float)
- **{keypoint}_confidence**: Confidence score for each keypoint (float, 0-1)

## Performance Requirements

### Processing Speed
- **Real-time Processing**: Process videos at 30 FPS or higher
- **Batch Processing**: Support processing multiple videos sequentially
- **Memory Efficiency**: Handle large video files without memory issues
- **GPU Acceleration**: Leverage GPU acceleration when available

### Quality Requirements
- **Pose Detection Accuracy**: >90% accuracy for clear dance movements
- **Timestamp Precision**: <1ms timestamp precision
- **Data Completeness**: <5% missing pose detections
- **Confidence Threshold**: Minimum confidence score of 0.5 for keypoints

## Error Handling

### Video Processing Errors
- **Unsupported Formats**: Graceful handling of unsupported video formats
- **Corrupted Files**: Detection and reporting of corrupted video files
- **Missing Files**: Handle missing or inaccessible video files
- **Permission Issues**: Handle file permission errors

### Pose Detection Errors
- **Low Confidence**: Handle frames with low pose detection confidence
- **Missing Keypoints**: Interpolate or handle missing keypoint detections
- **Tracking Loss**: Handle temporary loss of pose tracking
- **Multiple Persons**: Handle videos with multiple persons (focus on primary dancer)

## Testing Requirements

### Unit Testing
- [ ] **Video Loading**: Test video file loading and validation
- [ ] **Pose Detection**: Test pose detection accuracy and consistency
- [ ] **CSV Export**: Test CSV file generation and format correctness
- [ ] **Timestamp Accuracy**: Test timestamp precision and alignment

### Integration Testing
- [ ] **End-to-End Pipeline**: Test complete pose extraction pipeline
- [ ] **Batch Processing**: Test processing multiple videos
- [ ] **Error Handling**: Test error scenarios and recovery
- [ ] **Performance Testing**: Test processing speed and memory usage

### Validation Testing
- [ ] **Data Quality**: Validate extracted pose data quality
- [ ] **Synchronization**: Test pose-video synchronization
- [ ] **Playback Testing**: Test synchronized playback with original video
- [ ] **Cross-Platform**: Test on different operating systems

## Success Criteria

### Functional Success Criteria
- [ ] Successfully extract pose data from all supported video formats
- [ ] Generate CSV files with correct timestamps and pose data
- [ ] Achieve >90% pose detection accuracy for clear dance movements
- [ ] Maintain <1ms timestamp precision for synchronization

### Performance Success Criteria
- [ ] Process videos at 30 FPS or higher
- [ ] Handle video files up to 1GB without memory issues
- [ ] Complete batch processing of 10+ videos without errors
- [ ] Generate CSV files within 2x real-time duration

### Quality Success Criteria
- [ ] <5% missing pose detections across all processed videos
- [ ] >0.5 confidence score for 90% of detected keypoints
- [ ] Successful synchronized playback with original videos
- [ ] CSV files pass data validation and format checks

## Future Enhancements

### Advanced Features
- **3D Pose Extraction**: Support for 3D pose coordinates
- **Multiple Person Tracking**: Handle multiple dancers in single video
- **Pose Smoothing**: Apply smoothing algorithms to reduce jitter
- **Custom Keypoints**: Support for additional custom keypoints

### Performance Improvements
- **Parallel Processing**: Process multiple videos in parallel
- **Streaming Processing**: Process videos without loading entire file
- **Compression**: Compress pose data for storage efficiency
- **Caching**: Cache processed results for repeated access

This requirements document provides a comprehensive specification for the pose extraction component using the Rerun framework, ensuring high-quality pose data extraction with proper synchronization for dance motion analysis. 