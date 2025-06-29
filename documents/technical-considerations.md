# Dance Motion Embedding System - Technical Considerations

## Embedding Generation Strategies

### 1. Pose Representation (Single Frame Embedding)

Each pose frame from MediaPipe provides 33 keypoints with 2D or 3D coordinates:
- **Raw Format**: 33 keypoints × (x, y) or (x, y, z) coordinates
- **Total Dimensions**: 66 (2D) or 99 (3D) dimensions per frame

#### Pose Embedding Options

**1. Raw Flattened Vector**
```python
# Simple flattening approach
pose_vec = [x1, y1, x2, y2, ..., x33, y33]  # 66 dimensions
```
- **Pros**: Simple, preserves all spatial information
- **Cons**: Sensitive to translation, scale, rotation; no semantic structure

**2. Normalized Embedding**
```python
def normalize_pose(pose_landmarks):
    # Center around hip center (translation invariant)
    hip_center = calculate_hip_center(pose_landmarks)
    centered = pose_landmarks - hip_center
    
    # Scale by shoulder-to-hip distance
    shoulder_hip_distance = calculate_shoulder_hip_distance(centered)
    scaled = centered / shoulder_hip_distance
    
    # Optional: Align facing direction
    aligned = align_facing_direction(scaled)
    
    return aligned
```
- **Pros**: Translation and scale invariant, more robust
- **Cons**: May lose some spatial context

**3. Bone Orientation Features**
```python
def extract_bone_features(pose_landmarks):
    # Define bone connections (joint pairs)
    bone_connections = [
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
        (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
    ]
    
    bone_vectors = []
    for start_joint, end_joint in bone_connections:
        vector = pose_landmarks[end_joint] - pose_landmarks[start_joint]
        bone_vectors.append(vector)
    
    # Calculate angles between bones
    bone_angles = calculate_bone_angles(bone_vectors)
    
    return np.concatenate([bone_vectors, bone_angles])
```
- **Pros**: Captures skeletal structure, rotation invariant
- **Cons**: More complex, may lose absolute positioning

**4. Pre-trained Feature Extraction**
```python
# Use pre-trained models like PoseNet or ActionNet
class PretrainedPoseEncoder(nn.Module):
    def __init__(self, pretrained_model='posenet'):
        super().__init__()
        self.backbone = load_pretrained_model(pretrained_model)
        self.feature_dim = 512
        
    def forward(self, pose_data):
        features = self.backbone(pose_data)
        return self.normalize_features(features)
```
- **Pros**: Leverages pre-trained knowledge, better generalization
- **Cons**: Dependency on external models, may not be dance-specific

### 2. Movement Segment Embedding (5-second Windows)

For capturing motion patterns over time, consider these approaches:

#### Temporal Modeling Approaches

**1. RNN/LSTM/GRU Encoder**
```python
class TemporalRNNEncoder(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=512, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.output_projection = nn.Linear(hidden_dim * 2, 1024)
        
    def forward(self, pose_sequence):
        # pose_sequence: (batch_size, 150, 66) for 5s at 30fps
        lstm_out, (hidden, cell) = self.lstm(pose_sequence)
        
        # Use final hidden state or global pooling
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Bidirectional
        return self.output_projection(final_hidden)
```

**2. Transformer Encoder**
```python
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(hidden_dim, 1024)
        
    def forward(self, pose_sequence):
        x = self.input_projection(pose_sequence)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        
        encoded = self.transformer(x)
        encoded = encoded.transpose(0, 1)
        
        # Global pooling
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        return self.output_projection(pooled)
```

**3. Temporal Convolutional Networks (TCN)**
```python
class TemporalCNN(nn.Module):
    def __init__(self, input_dim=66, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            self.conv_layers.append(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
            )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(hidden_dims[-1], 1024)
        
    def forward(self, pose_sequence):
        # pose_sequence: (batch_size, 150, 66)
        x = pose_sequence.transpose(1, 2)  # (batch_size, 66, 150)
        
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool1d(x, kernel_size=2)
        
        pooled = self.global_pool(x).squeeze(-1)
        return self.output_projection(pooled)
```

**4. Autoencoder/Variational Autoencoder**
```python
class MotionVAE(nn.Module):
    def __init__(self, input_dim=66, seq_len=150, latent_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * seq_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim * seq_len)
        )
        
    def encode(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder(z)
        return h.view(h.size(0), -1, 66)  # Reshape back
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
```

**5. Motiongram/Spectrogram Approach**
```python
def create_motiongram(pose_sequence):
    """Convert pose sequence to 2D image-like representation"""
    # pose_sequence: (seq_len, num_joints, 2)
    
    # Calculate joint velocities
    velocities = np.diff(pose_sequence, axis=0)
    
    # Create motiongram by stacking joint trajectories
    motiongram = np.zeros((pose_sequence.shape[1], pose_sequence.shape[0]))
    
    for joint_idx in range(pose_sequence.shape[1]):
        # Use magnitude of velocity for each joint
        joint_velocity = np.linalg.norm(velocities[:, joint_idx, :], axis=1)
        motiongram[joint_idx, 1:] = joint_velocity
    
    return motiongram

class MotiongramCNN(nn.Module):
    def __init__(self, input_channels=33, embedding_dim=1024):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.output_projection = nn.Linear(256, embedding_dim)
        
    def forward(self, motiongram):
        # motiongram: (batch_size, 33, height, width)
        features = self.conv_layers(motiongram)
        features = features.view(features.size(0), -1)
        return self.output_projection(features)
```

### 3. Projection and Visualization

#### Dimensionality Reduction Techniques

**1. UMAP (Recommended)**
```python
import umap

def project_embeddings_umap(embeddings, n_components=3):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    return reducer.fit_transform(embeddings)
```

**2. t-SNE**
```python
from sklearn.manifold import TSNE

def project_embeddings_tsne(embeddings, n_components=3):
    tsne = TSNE(
        n_components=n_components,
        perplexity=30,
        random_state=42
    )
    return tsne.fit_transform(embeddings)
```

**3. PCA**
```python
from sklearn.decomposition import PCA

def project_embeddings_pca(embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)
```

#### Visualization Strategies

**1. 3D Scatter Plot with Color Coding**
```python
import plotly.express as px
import plotly.graph_objects as go

def visualize_embeddings_3d(embeddings_3d, labels=None, colors=None):
    fig = go.Figure()
    
    if labels is not None:
        # Color by dance style or movement type
        for label in np.unique(labels):
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                name=label,
                marker=dict(size=5)
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors)
        ))
    
    fig.update_layout(
        title="Dance Motion Embeddings",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3"
        )
    )
    return fig
```

### 4. Future Prediction (Live Tracking)

#### Prediction Model Architectures

**1. Sequence-to-Sequence Models**
```python
class MotionPredictor(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=512, prediction_horizon=30):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self.prediction_horizon = prediction_horizon
        
    def forward(self, embedding_sequence):
        # Encode the sequence
        encoder_output, (hidden, cell) = self.encoder(embedding_sequence)
        
        # Initialize decoder with last embedding
        decoder_input = embedding_sequence[:, -1:, :]
        
        predictions = []
        for _ in range(self.prediction_horizon):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.output_projection(decoder_output)
            predictions.append(prediction)
            decoder_input = prediction
        
        return torch.cat(predictions, dim=1)
```

**2. Diffusion Models**
```python
class MotionDiffusionPredictor(nn.Module):
    def __init__(self, embedding_dim=1024, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.embedding_dim = embedding_dim
        
        # Noise scheduler
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Denoising network
        self.denoising_net = nn.Sequential(
            nn.Linear(embedding_dim + 1, 512),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x, t):
        # x: (batch_size, embedding_dim)
        # t: (batch_size,)
        t_emb = t.unsqueeze(-1).float() / self.timesteps
        x_t = torch.cat([x, t_emb], dim=-1)
        return self.denoising_net(x_t)
```

## Critical Considerations

### 1. Data Preprocessing

**Noise Handling**
```python
def smooth_keypoints(keypoints, window_size=5):
    """Apply moving average smoothing to keypoints"""
    smoothed = np.zeros_like(keypoints)
    for i in range(len(keypoints)):
        start = max(0, i - window_size // 2)
        end = min(len(keypoints), i + window_size // 2 + 1)
        smoothed[i] = np.mean(keypoints[start:end], axis=0)
    return smoothed
```

**Sampling Rate Consistency**
```python
def resample_pose_sequence(pose_sequence, target_fps=30, original_fps=60):
    """Resample pose sequence to consistent FPS"""
    original_timestamps = np.arange(len(pose_sequence)) / original_fps
    target_timestamps = np.arange(0, len(pose_sequence) / original_fps, 1/target_fps)
    
    resampled = []
    for t in target_timestamps:
        # Find nearest neighbor or interpolate
        idx = np.argmin(np.abs(original_timestamps - t))
        resampled.append(pose_sequence[idx])
    
    return np.array(resampled)
```

### 2. Performance Optimization

**Real-time Latency**
- **Model Quantization**: Use INT8 quantization for inference
- **Model Pruning**: Remove unnecessary weights
- **GPU Acceleration**: Use CUDA for parallel processing
- **Batch Processing**: Process multiple frames simultaneously

**Memory Management**
```python
class MemoryEfficientEmbedding:
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        
    def generate_embeddings(self, pose_sequences):
        embeddings = []
        for i in range(0, len(pose_sequences), self.max_batch_size):
            batch = pose_sequences[i:i+self.max_batch_size]
            with torch.no_grad():
                batch_embeddings = self.model(batch)
                embeddings.append(batch_embeddings.cpu())
        return torch.cat(embeddings, dim=0)
```

### 3. Embedding Quality

**Similarity Metrics**
```python
def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def evaluate_embedding_quality(embeddings, labels):
    """Evaluate embedding quality using clustering metrics"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Clustering
    kmeans = KMeans(n_clusters=len(np.unique(labels)))
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Silhouette score
    silhouette = silhouette_score(embeddings, cluster_labels)
    
    return {
        'silhouette_score': silhouette,
        'cluster_labels': cluster_labels
    }
```

### 4. Variability Handling

**Style Normalization**
```python
def normalize_by_style(embeddings, style_labels):
    """Normalize embeddings by dance style"""
    normalized_embeddings = np.zeros_like(embeddings)
    
    for style in np.unique(style_labels):
        style_mask = style_labels == style
        style_embeddings = embeddings[style_mask]
        
        # Z-score normalization within style
        mean = np.mean(style_embeddings, axis=0)
        std = np.std(style_embeddings, axis=0)
        
        normalized_embeddings[style_mask] = (style_embeddings - mean) / (std + 1e-8)
    
    return normalized_embeddings
```

### 5. Labeling Strategy

**Movement Annotation**
```python
# Define movement categories
MOVEMENT_CATEGORIES = {
    'jump': ['leap', 'hop', 'bound'],
    'turn': ['pirouette', 'spot_turn', 'traveling_turn'],
    'gesture': ['arm_wave', 'point', 'reach'],
    'step': ['walk', 'run', 'skip', 'gallop'],
    'balance': ['arabesque', 'attitude', 'scale']
}

def annotate_movements(pose_sequences, movement_labels):
    """Create movement annotations for training"""
    annotations = []
    for sequence, labels in zip(pose_sequences, movement_labels):
        annotation = {
            'sequence': sequence,
            'movement_type': labels,
            'confidence': calculate_movement_confidence(sequence, labels)
        }
        annotations.append(annotation)
    return annotations
```

## Implementation Recommendations

### 1. Start Simple
- Begin with raw flattened pose vectors
- Implement basic normalization (translation, scale)
- Use simple LSTM for segment embedding

### 2. Iterate and Improve
- Add more sophisticated preprocessing
- Experiment with different model architectures
- Implement attention mechanisms for better performance

### 3. Focus on Real-time Performance
- Optimize models for inference speed
- Use model compression techniques
- Implement efficient similarity search

### 4. Validate Continuously
- Test with diverse dance styles
- Validate embedding quality metrics
- Ensure real-time performance requirements

This technical approach provides a comprehensive foundation for building a robust dance motion embedding system that can handle both offline analysis and real-time prediction. 