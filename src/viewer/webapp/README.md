# Dance Motion Web Playback App

This web app provides synchronized playback and interactive visualization of dance motion data, including:
- Video playback
- Interactive dimension reduction plot (e.g., UMAP, t-SNE, PCA)
- Timeline navigation
- Bidirectional sync between video and plot

## Features
- Stream video from your dataset
- Load pose and reduced data (CSV/JSON)
- Interactive Plotly plot of reduced motion
- Click timeline or plot to seek video
- Responsive, modern UI

## Setup

1. **Install dependencies**
   ```bash
   uv pip install .
   ```

   This will install all required dependencies (including Flask and pandas) as specified in your `pyproject.toml`.

2. **Prepare your data**
   - Place your video files in `data/video/`
   - Place your pose CSVs in `data/poses/`
   - Place your reduced data JSONs in `data/analysis/dimension_reduction/`

3. **Run the server**
   ```bash
   cd src/dimension_reduction/webapp
   python server.py
   ```

4. **Open your browser**
   - Go to [http://127.0.0.1:50680/](http://127.0.0.1:50680/)

## Usage

- Select the video, pose, and reduced data files from the dropdowns (currently hardcoded to `kurt.*` for demo)
- Click **Load**
- Use the video player, timeline, and plot interactively:
  - Play/pause/seek video
  - Click on the timeline to jump to a time
  - Click on the plot to jump to a frame in the video
  - The plot highlight follows the video in real time

## Customization
- To support more files, add endpoints to list available files or update the dropdowns in `app.js`
- To use your own data, ensure filenames match and are placed in the correct folders

## Troubleshooting
- If video does not play, check browser console for errors and ensure the video file exists in `data/video/`
- If plot or timeline does not update, check that the reduced data JSON is present and valid
- For CORS or network errors, ensure you are accessing via `localhost` and not a remote IP

## Development
- Frontend: `static/index.html`, `static/app.js`, `static/style.css`
- Backend: `server.py` (Flask)

## License
MIT (or as per your project) 