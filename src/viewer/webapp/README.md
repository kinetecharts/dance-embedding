# Dance Motion Web Playback App

This web app provides synchronized playback and interactive visualization of dance motion data, including:
- Video playback with pose overlays
- Interactive dimension reduction plot (e.g., UMAP, t-SNE, PCA)
- Timeline navigation
- Bidirectional sync between video and plot

## Features
- Stream video with pose overlays from your dataset
- Load pose and reduced data (CSV format)
- Interactive Plotly plot of reduced motion
- Click timeline or plot to seek video
- Responsive, modern UI
- Automatic file discovery and filtering

## Setup

1. **Install dependencies**
   ```bash
   uv pip install .
   ```

   This will install all required dependencies (including Flask and pandas) as specified in your `pyproject.toml`.

2. **Prepare your data**
   - Place your video files with pose overlays in `data/video_with_pose/`
   - Place your pose CSVs in `data/poses/`
   - Place your reduced data CSVs in `data/dimension_reduction/`

3. **Run the server**
   ```bash
   cd src/viewer/webapp
   python server.py
   ```

4. **Open your browser**
   - Go to [http://127.0.0.1:50680/](http://127.0.0.1:50680/)

## Usage

- Select the video, pose, and reduced data files from the dropdowns
- The app automatically discovers available files and filters out system files (like .DS_Store)
- Click **Load**
- Use the video player, timeline, and plot interactively:
  - Play/pause/seek video
  - Click on the timeline to jump to a time
  - Click on the plot to jump to a frame in the video
  - The plot highlight follows the video in real time

## Data Format

### Video Files
- **Location**: `data/video_with_pose/`
- **Format**: MP4, AVI, MOV, WebM, MKV
- **Content**: Videos with pose landmark overlays for review

### Pose Data
- **Location**: `data/poses/`
- **Format**: CSV with columns: `timestamp`, `frame_number`, `{keypoint}_x`, `{keypoint}_y`, `{keypoint}_z`, `{keypoint}_confidence`

### Reduced Data
- **Location**: `data/dimension_reduction/`
- **Format**: CSV with columns: `timestamp`, `frame_number`, `x`, `y`
- **Naming**: `{video_name}_{method}_reduced.csv` (e.g., `kurt_umap_reduced.csv`)

## API Endpoints

- `GET /list_videos` - List available video files (filters out .DS_Store)
- `GET /list_poses` - List available pose CSV files
- `GET /list_reduced` - List available reduced data CSV files
- `GET /list_reductions_for_video/{video_basename}` - List reduction methods for a video
- `GET /video/{filename}` - Stream video file with range request support
- `GET /pose/{filename}` - Serve pose data as JSON
- `GET /reduced/{filename}` - Serve reduced data as JSON

## Customization
- To support more file types, update the video extensions in `server.py`
- To use your own data, ensure filenames match and are placed in the correct folders
- The app automatically handles CSV to JSON conversion for the frontend

## Troubleshooting
- If video does not play, check browser console for errors and ensure the video file exists in `data/video_with_pose/`
- If plot or timeline does not update, check that the reduced data CSV is present and valid
- For CORS or network errors, ensure you are accessing via `localhost` and not a remote IP
- If files don't appear in dropdowns, check that they're in the correct directories and have the right extensions

## Development
- Frontend: `static/index.html`, `static/app.js`, `static/style.css`
- Backend: `server.py` (Flask)

## License
MIT (or as per your project) 