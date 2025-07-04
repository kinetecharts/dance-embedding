import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_DIR = PROJECT_ROOT / "data/video"  # List originals for dropdown
VIDEO_WITH_POSE_DIR = PROJECT_ROOT / "data/video_with_pose"  # For overlay playback
POSE_DIR = PROJECT_ROOT / "data/poses"
REDUCED_DIR = PROJECT_ROOT / "data/dimension_reduction"

from flask import Flask, send_from_directory, jsonify, request, render_template, Response
import pandas as pd
import json
import mimetypes

app = Flask(__name__, static_folder="static", template_folder="static")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/video/<filename>")
def serve_video(filename):
    # Try to serve overlay video first (robust for any extension)
    from pathlib import Path
    stem = Path(filename).stem
    # Search for any overlay video with the same stem
    overlay_candidates = list(VIDEO_WITH_POSE_DIR.glob(f"{stem}_with_pose.*"))
    print("Overlay candidates:", overlay_candidates)
    overlay_path = None
    if overlay_candidates:
        # Prefer .mp4 if available
        mp4s = [f for f in overlay_candidates if f.suffix.lower() == '.mp4']
        overlay_path = mp4s[0] if mp4s else overlay_candidates[0]
    if overlay_path and overlay_path.exists():
        video_path = overlay_path
    else:
        video_path = VIDEO_DIR / filename
    if not video_path.exists():
        return "Video not found", 404
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_from_directory(video_path.parent, video_path.name)
    # Handle range requests for video seeking
    size = video_path.stat().st_size
    byte1, byte2 = 0, None
    m = None
    import re
    m = re.search(r'bytes=(\d+)-(\d*)', range_header)
    if m:
        g = m.groups()
        byte1 = int(g[0])
        if g[1]:
            byte2 = int(g[1])
    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1 + 1
    data = None
    with open(video_path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)
    rv = Response(data, 206, mimetype=mimetypes.guess_type(str(video_path))[0],
                  content_type=mimetypes.guess_type(str(video_path))[0],
                  direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    return rv

@app.route("/pose/<filename>")
def serve_pose(filename):
    pose_path = POSE_DIR / filename
    if not pose_path.exists():
        return jsonify({"error": "Pose file not found"}), 404
    df = pd.read_csv(pose_path)
    return df.to_json(orient="records")

@app.route("/reduced/<filename>")
def serve_reduced(filename):
    reduced_path = REDUCED_DIR / filename
    if not reduced_path.exists():
        return jsonify({"error": "Reduced data not found"}), 404
    # Read CSV file and convert to JSON format
    df = pd.read_csv(reduced_path)
    return df.to_json(orient="records")

@app.route("/list_videos")
def list_videos():
    # List only original videos for dropdown
    video_extensions = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    files = [f.name for f in VIDEO_DIR.glob("*") 
             if f.is_file() and f.suffix.lower() in video_extensions 
             and not f.name.startswith('.')]
    return jsonify(sorted(files, key=lambda x: x.lower()))

@app.route("/list_poses")
def list_poses():
    files = [f.name for f in POSE_DIR.glob("*.csv") if f.is_file()]
    return jsonify(files)

@app.route("/list_reduced")
def list_reduced():
    files = [f.name for f in REDUCED_DIR.glob("*.csv") if f.is_file()]
    return jsonify(files)

@app.route("/list_reductions_for_video/<video_basename>")
def list_reductions_for_video(video_basename):
    # Find all reduced files for this video and extract method names
    files = [f.name for f in REDUCED_DIR.glob(f"{video_basename}_*_reduced.csv") if f.is_file()]
    methods = []
    prefix = f"{video_basename}_"
    suffix = "_reduced.csv"
    for fname in files:
        if fname.startswith(prefix) and fname.endswith(suffix):
            method = fname[len(prefix):-len(suffix)]
            methods.append(method)
    return jsonify(sorted(set(methods)))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

def start_server(port=50680, debug=True):
    """Start the viewer server."""
    app.run(debug=debug, port=port)

if __name__ == "__main__":
    start_server() 