import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

CHUNK_SIZE = 30
STRIDE = 15
N_KEYPOINTS = 33
N_DIMS = 3
ROOT_JOINT = 23  # left_hip (or 24 for right_hip, or average)
TORSO_JOINTS = [11, 12, 23, 24]  # shoulders and hips


def normalize_and_flatten(chunk):
    # chunk: (frames, joints*3)
    chunk = chunk.reshape(CHUNK_SIZE, N_KEYPOINTS, N_DIMS)
    # Translation: subtract root joint (left_hip)
    root = chunk[:, ROOT_JOINT, :]
    chunk -= root[:, None, :]
    # Scale: use mean torso length (shoulder-hip average)
    torso_lens = []
    for f in range(CHUNK_SIZE):
        l_shoulder = chunk[f, 11, :]
        r_shoulder = chunk[f, 12, :]
        l_hip = chunk[f, 23, :]
        r_hip = chunk[f, 24, :]
        torso = np.linalg.norm((l_shoulder + r_shoulder)/2 - (l_hip + r_hip)/2)
        torso_lens.append(torso)
    scale = np.mean(torso_lens) if np.mean(torso_lens) > 1e-6 else 1.0
    chunk /= scale
    return chunk.reshape(-1)

def compute_velocity(chunk):
    # chunk: (frames, joints, 3)
    return np.diff(chunk, axis=0)  # (frames-1, joints, 3)

def compute_acceleration(vel):
    return np.diff(vel, axis=0)  # (frames-2, joints, 3)

def chunk_pose_data(pose_df, chunk_size=CHUNK_SIZE, stride=STRIDE, add_vel=False, add_acc=False):
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
        norm_chunk = normalize_and_flatten(chunk.copy())
        features = [norm_chunk]
        if add_vel or add_acc:
            chunk3d = chunk.reshape(CHUNK_SIZE, N_KEYPOINTS, N_DIMS)
            vel = compute_velocity(chunk3d)
            if add_vel:
                features.append(vel.reshape(-1))
            if add_acc:
                acc = compute_acceleration(vel)
                features.append(acc.reshape(-1))
        flat = np.concatenate(features)
        chunks.append(flat)
        chunk_timestamps.append(pose_df.iloc[start]['timestamp'])
        chunk_frame_numbers.append(pose_df.iloc[start]['frame_number'])
    return np.array(chunks), chunk_timestamps, chunk_frame_numbers

def run_pca(X, var_threshold=0.95):
    pca = PCA(n_components=min(X.shape[0], X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X)
    # Find number of components to reach variance threshold
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = np.searchsorted(cumsum, var_threshold) + 1
    X_pca = X_pca[:, :n_comp]
    return X_pca, n_comp, pca

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
    parser = argparse.ArgumentParser(description="Chunked dimension reduction for pose data (normalized, scaled, PCA-preprocessed)")
    parser.add_argument('--pose-csv', type=str, required=True, help='Input pose CSV file')
    parser.add_argument('--output-dir', type=str, default='data/dimension_reduction', help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=['pca', 'tsne', 'umap'], help='Reduction methods')
    parser.add_argument('--add-velocity', action='store_true', help='Append velocity features')
    parser.add_argument('--add-acceleration', action='store_true', help='Append acceleration features')
    parser.add_argument('--pca-var', type=float, default=0.95, help='Variance threshold for PCA pre-reduction (for t-SNE/UMAP)')
    args = parser.parse_args()

    pose_df = pd.read_csv(args.pose_csv)
    X, chunk_timestamps, chunk_frame_numbers = chunk_pose_data(
        pose_df,
        add_vel=args.add_velocity or args.add_acceleration,
        add_acc=args.add_acceleration
    )
    X = np.nan_to_num(X, nan=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    base = Path(args.pose_csv).stem

    # PCA pre-reduction for t-SNE/UMAP
    X_pca, n_pca, pca_model = run_pca(X_scaled, var_threshold=args.pca_var)
    print(f"PCA pre-reduction: {n_pca} components explain {args.pca_var*100:.1f}% variance")

    # Save PCA output (2D for plot, all for pre-reduction)
    X_pca2d = run_reduction(X_scaled, 'pca', n_components=2)
    out_path = Path(args.output_dir) / f"{base}_c-pca_reduced.csv"
    df = pd.DataFrame(X_pca2d, columns=[f"c{i+1}" for i in range(X_pca2d.shape[1])])
    df.insert(0, 'timestamp', chunk_timestamps)
    df.insert(1, 'frame_number', chunk_frame_numbers)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    for method in args.methods:
        if method == 'pca':
            continue  # already saved above
        X_red = run_reduction(X_pca, method, n_components=2)
        out_path = Path(args.output_dir) / f"{base}_c-{method}_reduced.csv"
        df = pd.DataFrame(X_red, columns=[f"c{i+1}" for i in range(X_red.shape[1])])
        df.insert(0, 'timestamp', chunk_timestamps)
        df.insert(1, 'frame_number', chunk_frame_numbers)
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main() 