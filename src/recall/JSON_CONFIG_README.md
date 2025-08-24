# JSON Configuration System

The recall system now supports JSON-based configuration for flexible video file selection and system settings.

## Overview

The JSON configuration system allows you to:
- Specify which video files to use for matching
- Configure all recall system parameters
- Set OSC streaming options
- Define performance and caching settings
- Use pattern-based video filtering

## Configuration File

The system looks for `src/recall/config.json` by default, or you can specify a custom path.

### Basic Structure

```json
{
  "recall_system": { ... },
  "paths": { ... },
  "osc_streaming": { ... },
  "video_matching": { ... },
  "joint_weights": { ... },
  "performance": { ... }
}
```

## Video Matching Configuration

### Key Features

1. **Specific Video Selection**: List exact video files to use
2. **Pattern-based Filtering**: Include/exclude videos using glob patterns
3. **Fallback Behavior**: Load all videos if no specific ones specified
4. **Extension Filtering**: Support for multiple video formats

### Configuration Options

```json
{
  "video_matching": {
    "load_specific_videos": true,
    "specific_videos": [
      "dance_sequence_1.mp4",
      "dance_sequence_2.mp4",
      "performance_highlight.mp4"
    ],
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "exclude_patterns": [
      "*_test.mp4",
      "*_draft.mp4",
      "temp_*"
    ],
    "include_patterns": [
      "*_final.mp4",
      "*_performance.mp4"
    ]
  }
}
```

### Video Loading Behavior

1. **If `load_specific_videos` is true AND `specific_videos` is provided**:
   - Only load the specified videos
   - Skip videos that don't exist

2. **If `load_specific_videos` is false**:
   - Load all videos from the video directory
   - Apply include/exclude patterns if specified

3. **Pattern Filtering**:
   - `exclude_patterns`: Videos matching these patterns are skipped
   - `include_patterns`: Only videos matching these patterns are loaded (if specified)

## Usage Examples

### 1. Load Specific Videos Only

```json
{
  "video_matching": {
    "load_specific_videos": true,
    "specific_videos": ["performance1.mp4", "performance2.mp4"]
  }
}
```

### 2. Load All Videos with Exclusions

```json
{
  "video_matching": {
    "load_specific_videos": false,
    "exclude_patterns": ["*_test.mp4", "*_draft.mp4"]
  }
}
```

### 3. Pattern-based Selection

```json
{
  "video_matching": {
    "load_specific_videos": false,
    "include_patterns": ["*_final.mp4", "*_performance.mp4"],
    "exclude_patterns": ["*_test.mp4", "temp_*"]
  }
}
```

### 4. Mixed Approach (Specific + All)

```json
{
  "video_matching": {
    "load_specific_videos": false,
    "specific_videos": ["highlight.mp4"],
    "exclude_patterns": ["*_draft.mp4"]
  }
}
```

## Python API

### Basic Usage

```python
from recall.json_config_loader import create_recall_config_from_json

# Create config from JSON
config = create_recall_config_from_json()

# Use in recall system
from recall.recall_system import create_recall_system
system = create_recall_system(config)
```

### Advanced Usage

```python
from recall.json_config_loader import create_config_loader

# Create config loader
loader = create_config_loader("custom_config.json")

# Get video files for matching
video_files = loader.get_video_files_for_matching("data/video")

# Create recall config
config = loader.create_recall_config()

# Get configuration summary
summary = loader.get_config_summary()
print(summary)
```

### Integration with Recall System

```python
from recall.recall_system import create_recall_system
from recall.json_config_loader import create_recall_config_from_json

# Create system with JSON config
config = create_recall_config_from_json()
with create_recall_system(config) as system:
    # Get filtered video files
    videos = system.get_video_files_for_matching()
    
    # Show config summary
    print(system.get_config_summary())
    
    # Run the system
    system.run_live()
```

## Configuration Sections

### Recall System

```json
{
  "recall_system": {
    "mode": "camera",
    "top_n": 5,
    "match_interval": 2.0,
    "similarity_metric": "euclidean"
  }
}
```

### Paths

```json
{
  "paths": {
    "pose_dir": "data/poses",
    "video_dir": "data/video",
    "video_with_pose_dir": "data/video_with_pose"
  }
}
```

### OSC Streaming

```json
{
  "osc_streaming": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 6448,
    "stream_rate": 30.0
  }
}
```

### Performance

```json
{
  "performance": {
    "enable_caching": true,
    "max_memory_usage_mb": 1024,
    "enable_profiling": false
  }
}
```

## File Naming Conventions

### Recommended Patterns

- **Performance videos**: `*_performance.mp4`, `*_final.mp4`
- **Test videos**: `*_test.mp4`, `*_draft.mp4`
- **Temporary files**: `temp_*`, `*_tmp.mp4`

### Example Directory Structure

```
data/video/
├── dance_performance_final.mp4
├── contemporary_final.mp4
├── ballet_performance.mp4
├── dance_test_draft.mp4
├── temp_processing.mp4
└── archive_old.mp4
```

## Error Handling

The system gracefully handles:
- Missing configuration files
- Non-existent video files
- Invalid pattern syntax
- Missing directories

## Performance Considerations

- **Large video collections**: Use specific video lists for faster loading
- **Pattern matching**: Complex patterns may impact performance
- **Memory usage**: Monitor with `max_memory_usage_mb` setting

## Troubleshooting

### Common Issues

1. **No videos loaded**
   - Check `video_matching.enabled` is true
   - Verify video directory path
   - Check file extensions in `video_extensions`

2. **Wrong videos loaded**
   - Review `specific_videos` list
   - Check `include_patterns` and `exclude_patterns`
   - Verify `load_all_if_empty` setting

3. **Configuration not loaded**
   - Check file path and permissions
   - Validate JSON syntax
   - Check log messages for errors

### Debug Mode

Enable debug logging to see detailed video filtering:

```python
import logging
logging.getLogger('recall.json_config_loader').setLevel(logging.DEBUG)
```

## Future Enhancements

- [ ] Dynamic configuration reloading
- [ ] Configuration validation schemas
- [ ] Multiple configuration profiles
- [ ] Environment variable overrides
- [ ] Configuration inheritance
- [ ] Web-based configuration editor
