# Quick Start Guide

Get the Dance Recall System running in 3 simple steps!

## Prerequisites

Make sure you have the system installed and set up:
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not done already)
uv pip install -e .

# Create data directories
mkdir -p data/video data/poses
```

## Step 1: Start Video Monitor

Load your dance videos for processing:
```bash
# Start the monitor to automatically process videos
python monitor_videos.py
```

**What it does:**
- Watches `data/video/` for new video files
- Automatically extracts pose data when videos are added
- Processes videos in the background

**How to use:**
1. Add your dance videos to `data/video/` folder
2. The monitor will automatically detect and process them
3. Pose CSV files will be created in `data/poses/`

## Step 2: Start Web Server

View pose analysis and visualizations:
```bash
# Start the web application server
cd src/viewer/webapp
python server.py
// or
uv run server.py
```

**What it does:**
- Serves interactive web interface at http://127.0.0.1:50680/
- Shows synchronized video and pose visualizations
- Provides dimension reduction analysis (PCA, t-SNE, UMAP)

**How to use:**
1. Open browser to http://127.0.0.1:50680/
2. View processed videos with pose overlays
3. Explore motion analysis and embeddings

## Step 3: Start Live Recall

Run real-time pose matching with camera:
```bash
# Build LanceDB database for fast matching
python rebuild_database.py

# Start live camera mode
python -m recall.main --mode camera --top-n 1 --match-interval 2.0 --playback-duration 3.0
```

## Step 4: osc streaming
```
uv run examples/osc_streaming_example.py --record-video --skip-matching
uv run examples/osc_streaming_example.py --record-video --raw-pose --skip-matching
```

## Step 5: Forward osc to websocket, port 8000 ws://localhost:8000/ws
```
uv run src/osc_to_websocket/simple_server.py
```

**What it does:**
- Captures live camera feed
- Matches your poses against the database
- Shows side-by-side: live pose vs matched reference pose

**How to use:**
1. Stand in front of camera
2. Perform dance movements
3. Watch matched reference poses appear on the right
4. Press 'q' to quit

## Quick Commands Summary

```bash
# Terminal 1: Video processing
python monitor_videos.py

# Terminal 2: Web interface  
cd src/viewer/webapp && python server.py

# Terminal 3: Live recall
python rebuild_database.py
python -m recall.main --mode camera --top-n 1 --match-interval 2.0
```

## What You'll See

1. **Video Monitor**: Processing status and pose extraction progress
2. **Web Interface**: Interactive visualizations at http://127.0.0.1:50680/
3. **Live Recall**: Dual-window display with live camera and matched poses

## Troubleshooting

**No videos processed:**
- Check that videos are in `data/video/` folder
- Ensure video files are supported (.mp4, .mov, .avi)

**Web server not working:**
- Check if port 50680 is available
- Try different port: `python server.py --port 8080`

**Live recall not working:**
- Ensure camera permissions are granted
- Check that LanceDB database was built: `python rebuild_database.py`
- Try with video file first: `python -m recall.main --mode video --input data/video/test.mp4`

## Performance Tips

- Use `--top-n 1` for fastest matching
- Use `--match-interval 2.0` or higher for better performance
- Ensure good lighting for camera mode
- Works best with 3-10 reference videos in database

---

**Need help?** Check the main README.md for detailed documentation. 