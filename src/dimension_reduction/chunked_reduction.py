import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

CHUNK_SIZE = 30
STRIDE = 15
N_KEYPOINTS = 33
N_DIMS = 3


def chunk_pose_data(pose_df, chunk_size=CHUNK_SIZE, stride=STRIDE):
    pose_cols = [col for col in pose_df.columns if any(col.endswith(f'_{d}') for d in ['x', 'y', 'z'])]
    pose_data = pose_df[pose_cols].values
    n_frames = pose_data.shape[0]
    chunks = []
    chunk_timestamps = []
    chunk_frame_numbers = []
    for start in range(0, n_frames - chunk_size + 1, stride):
        chunk = pose_data[start:start+chunk_size]
        if chunk.shape[0] < chunk_size:
            continue  # skip incomplete chunk
        flat = chunk.flatten()
        chunks.append(flat)
        # Use the timestamp and frame number of the first frame in the chunk
        chunk_timestamps.append(pose_df.iloc[start]['timestamp'])
        chunk_frame_numbers.append(pose_df.iloc[start]['frame_number'])
    return np.array(chunks), chunk_timestamps, chunk_frame_numbers


def run_reduction(X, method, n_components=2):
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=42)
        X_red = model.fit_transform(X)
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=42)
        X_red = model.fit_transform(X)
    elif method == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=42)
        X_red = model.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    return X_red


def main():
    parser = argparse.ArgumentParser(description="Chunked dimension reduction for pose data")
    parser.add_argument('--pose-csv', type=str, required=True, help='Input pose CSV file')
    parser.add_argument('--output-dir', type=str, default='data/dimension_reduction', help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=['pca', 'tsne', 'umap'], help='Reduction methods')
    args = parser.parse_args()

    pose_df = pd.read_csv(args.pose_csv)
    X, chunk_timestamps, chunk_frame_numbers = chunk_pose_data(pose_df)
    X = np.nan_to_num(X, nan=0.0)
    print(f"Total chunks: {X.shape[0]}, vector size per chunk: {X.shape[1]}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    base = Path(args.pose_csv).stem
    for method in args.methods:
        X_red = run_reduction(X, method, n_components=2)
        out_path = Path(args.output_dir) / f"{base}_c-{method}_reduced.csv"
        df = pd.DataFrame(X_red, columns=[f"c{i+1}" for i in range(X_red.shape[1])])
        df.insert(0, 'timestamp', chunk_timestamps)
        df.insert(1, 'frame_number', chunk_frame_numbers)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main() 