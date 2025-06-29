# Dance Motion Embedding System - Architecture

## System Overview

The Dance Motion Embedding System is designed as a modular, scalable architecture that processes dance videos through pose estimation, embedding generation, and real-time prediction capabilities.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Pose Pipeline  │───▶│ Embedding Gen.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Data Storage  │    │  Analysis &     │
                       │                 │    │ Visualization   │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Real-time      │    │  Future         │
                       │  Tracking       │    │  Prediction     │
                       └─────────────────┘    └─────────────────┘
```

## Data Pipeline Architecture

### 1. Video Input Layer
- **Supported Formats**: MP4, AVI, MOV, WebM
- **Frame Extraction**: Configurable FPS (24-60)
- **Batch Processing**: Support for multiple video files
- **Quality Control**: Resolution validation and preprocessing

### 2. Pose Detection Pipeline
```
Video Frame → MediaPipe Pose → Keypoint Extraction → Normalization → Storage
```

#### MediaPipe Integration
- **Pose Model**: MediaPipe Pose (33 keypoints)
- **Output Format**: 2D/3D coordinates per keypoint
- **Confidence Scores**: Per-keypoint detection confidence
- **Landmark Structure**: Head, torso, arms, legs, hands, feet

#### Data Normalization
- **Translation**: Center pose around hip center
- **Scale**: Normalize by shoulder-to-hip distance
- **Rotation**: Align facing direction (optional)
- **Missing Data**: Interpolation for occluded keypoints

### 3. Embedding Generation Layer

#### Pose Embedding Models
```
Raw Pose Data → Preprocessing → Embedding Model → Pose Vector (128-512d)
```

**Model Options:**
- **Transformer Encoder**: Self-attention for joint relationships
- **Graph Neural Network**: Skeletal structure modeling
- **Temporal CNN**: Motion pattern extraction
- **Autoencoder**: Unsupervised pose representation

#### Segment Embedding Models
```
Pose Sequence → Temporal Model → Segment Vector (256-1024d)
```

**Model Options:**
- **LSTM/GRU**: Sequential motion modeling
- **Transformer**: Long-range temporal dependencies
- **Temporal CNN**: Fixed-window processing
- **Variational Autoencoder**: Probabilistic motion encoding

### 4. Data Storage Architecture

#### Storage Layers
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Video     │    │  Pose Data      │    │  Embeddings     │
│   Storage       │    │  (HDF5/Parquet) │    │  (Vector DB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Metadata       │    │  Cache Layer    │
                       │  (SQLite/PostgreSQL) │  (Redis)        │
                       └─────────────────┘    └─────────────────┘
```

#### Data Schema
- **Video Metadata**: File info, duration, FPS, resolution
- **Pose Data**: Timestamp, keypoints, confidence scores
- **Embeddings**: Pose vectors, segment vectors, model metadata
- **Annotations**: Dance style, choreography, performance metrics

### 5. Real-time System Architecture

#### Live Tracking Pipeline
```
Camera Input → Frame Buffer → Pose Detection → Embedding Gen → Prediction → Output
```

#### Performance Optimization
- **GPU Acceleration**: CUDA/OpenCL for pose detection
- **Model Quantization**: Reduced precision for speed
- **Streaming Processing**: Pipeline parallelism
- **Caching**: Embedding and prediction caching

## Embedding Model Architecture

### Pose Embedding Design

#### Input Representation
```python
# 33 keypoints × 2D coordinates
pose_input = [x1, y1, x2, y2, ..., x33, y33]  # 66 dimensions

# Normalized representation
normalized_pose = normalize_pose(pose_input)
```

#### Model Architectures

**1. Transformer-based Pose Encoder**
```python
class PoseTransformer(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=256, num_heads=8):
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(...)
        self.output = nn.Linear(hidden_dim, embedding_dim)
```

**2. Graph Neural Network**
```python
class PoseGNN(nn.Module):
    def __init__(self, num_joints=33, hidden_dim=256):
        self.joint_encoder = nn.Linear(2, hidden_dim)
        self.gnn_layers = nn.ModuleList([GNNLayer(...) for _ in range(3)])
        self.global_pool = GlobalAttentionPool()
```

### Segment Embedding Design

#### Input Representation
```python
# 5 seconds × 30 FPS × 66 dimensions
segment_input = [pose1, pose2, ..., pose150]  # 150 × 66
```

#### Model Architectures

**1. LSTM-based Segment Encoder**
```python
class SegmentLSTM(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=512, num_layers=3):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.output = nn.Linear(hidden_dim, embedding_dim)
```

**2. Temporal Convolutional Network**
```python
class SegmentTCN(nn.Module):
    def __init__(self, input_dim=66, hidden_dims=[128, 256, 512]):
        self.tcn_layers = nn.ModuleList([
            TemporalConv1d(input_dim if i == 0 else hidden_dims[i-1], 
                          hidden_dims[i]) for i in range(len(hidden_dims))
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
```

## Prediction System Architecture

### Future Movement Prediction

#### Model Architecture
```python
class MotionPredictor(nn.Module):
    def __init__(self, embedding_dim=512, prediction_horizon=30):
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)
        self.decoder = nn.LSTM(hidden_dim, embedding_dim, num_layers=2)
        self.output = nn.Linear(embedding_dim, prediction_horizon * embedding_dim)
```

#### Prediction Pipeline
1. **Current State**: Extract current pose/segment embedding
2. **Context Window**: Use recent history (last 1-3 seconds)
3. **Future Prediction**: Generate next 1-3 seconds of motion
4. **Confidence Scoring**: Estimate prediction uncertainty

### Similarity Search Architecture

#### FAISS Integration
```python
class MotionSimilaritySearch:
    def __init__(self, embedding_dim=512):
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product
        self.embeddings = []
    
    def add_embeddings(self, embeddings):
        self.index.add(embeddings)
        self.embeddings.extend(embeddings)
    
    def search(self, query_embedding, k=10):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
```

## Visualization Architecture

### 3D Motion Visualization

#### WebGL-based Renderer
```javascript
class MotionVisualizer {
    constructor(canvas) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, width/height, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({canvas: canvas});
    }
    
    renderPose(pose_data) {
        // Render skeletal structure
        this.drawSkeleton(pose_data);
        // Render motion trajectory
        this.drawTrajectory(pose_data);
    }
}
```

### Embedding Space Visualization

#### Dimensionality Reduction Pipeline
```python
class EmbeddingVisualizer:
    def __init__(self, method='umap'):
        self.reducer = UMAP(n_components=3, random_state=42)
        self.embeddings_3d = None
    
    def fit_transform(self, embeddings):
        self.embeddings_3d = self.reducer.fit_transform(embeddings)
        return self.embeddings_3d
    
    def visualize_clusters(self, labels):
        # 3D scatter plot with color coding
        fig = px.scatter_3d(self.embeddings_3d, color=labels)
        return fig
```

## Deployment Architecture

### Development Environment
- **Local Development**: Docker containers for consistency
- **GPU Support**: CUDA-enabled containers for training
- **Data Versioning**: DVC for large file management
- **Experiment Tracking**: MLflow for model experiments

### Production Deployment
- **Microservices**: Separate services for pose detection, embedding, prediction
- **Load Balancing**: Multiple instances for high availability
- **Monitoring**: Prometheus + Grafana for system metrics
- **Logging**: Structured logging with ELK stack

### Scalability Considerations
- **Horizontal Scaling**: Stateless services for easy scaling
- **Data Partitioning**: Shard embeddings by dance style/time
- **Caching Strategy**: Multi-level caching (Redis + CDN)
- **Async Processing**: Queue-based processing for batch operations 