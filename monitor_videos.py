import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            _, ext = os.path.splitext(event.src_path)
            if ext.lower() in VIDEO_EXTENSIONS:
                print(f"New video detected: {event.src_path}")
                # Run pose extraction for this video
                subprocess.run([
                    "python", "-m", "pose_extraction.main",
                    "--video", event.src_path
                ], check=True)
                # Find the output pose CSV
                base = os.path.splitext(os.path.basename(event.src_path))[0]
                pose_csv = f"data/poses/{base}.csv"
                # Run dimension reduction for this pose CSV for all methods
                methods = ["umap", "pca", "tsne"]
                for method in methods:
                    subprocess.run([
                        "python", "-m", "dimension_reduction.main",
                        "--video", event.src_path,
                        "--pose-csv", pose_csv,
                        "--method", method
                    ], check=True)
                print(f"Processing complete for {event.src_path}")

if __name__ == "__main__":
    path = "data/video"
    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    print(f"Monitoring {path} for new video files...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join() 