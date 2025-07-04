import * as THREE from 'https://esm.sh/three@0.153.0';
import { OrbitControls } from 'https://esm.sh/three@0.153.0/examples/jsm/controls/OrbitControls.js';
import { PANEL_SIZE } from './config.js';

// POSE_KEYPOINTS and POSE_CONNECTIONS must match your data
const POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
];
const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [27, 29], [29, 31], [24, 26], [26, 28], [28, 30], [30, 32],
    [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22], [27, 31], [28, 32],
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]
];

let pose3DScene, pose3DCamera, pose3DRenderer, pose3DControls, pose3DJoints = [], pose3DBones = [];
let videoPlayer = null;
let lastFrameIdx = -1;

// Add a 2D grid for the floor in the 3D pose view
function setup3DPoseViewer() {
    const container = document.getElementById('pose3d');
    pose3DScene = new THREE.Scene();
    pose3DScene.background = new THREE.Color(0x222222);
    pose3DCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
    pose3DCamera.position.set(0, 0, 500);
    pose3DRenderer = new THREE.WebGLRenderer({ antialias: true });
    pose3DRenderer.setSize(PANEL_SIZE, PANEL_SIZE);
    container.appendChild(pose3DRenderer.domElement);
    pose3DControls = new OrbitControls(pose3DCamera, pose3DRenderer.domElement);
    pose3DControls.enableDamping = true;
    pose3DControls.dampingFactor = 0.1;
    pose3DControls.screenSpacePanning = false;
    pose3DControls.minDistance = 100;
    pose3DControls.maxDistance = 1000;
    // Add lights
    pose3DScene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const light = new THREE.DirectionalLight(0xffffff, 0.6);
    light.position.set(0, 0, 500);
    pose3DScene.add(light);
    // Add a 2D grid for the floor
    const grid = new THREE.GridHelper(600, 20, 0x888888, 0x444444);
    grid.position.y = -150;
    pose3DScene.add(grid);
    // Create joints
    for (let i = 0; i < 33; i++) {
        const geom = new THREE.SphereGeometry(3, 16, 16);
        const mat = new THREE.MeshStandardMaterial({ color: 0x00aaff });
        const joint = new THREE.Mesh(geom, mat);
        pose3DScene.add(joint);
        pose3DJoints.push(joint);
    }
    // Create bones
    for (let i = 0; i < POSE_CONNECTIONS.length; i++) {
        const mat = new THREE.LineBasicMaterial({ color: 0x00ff00 });
        const geom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(), new THREE.Vector3()
        ]);
        const bone = new THREE.Line(geom, mat);
        pose3DScene.add(bone);
        pose3DBones.push(bone);
    }
    animate3DPose();
}

function animate3DPose() {
    requestAnimationFrame(animate3DPose);
    if (pose3DControls) pose3DControls.update();
    pose3DRenderer && pose3DRenderer.render(pose3DScene, pose3DCamera);
}

// Update 3D pose on video time update
function update3DPose(frameIdx) {
    if (!window.poseData || !window.poseData[frameIdx]) return;
    const pose = window.poseData[frameIdx];
    const points = [];
    for (let i = 0; i < 33; i++) {
        const x = pose[`${POSE_KEYPOINTS[i]}_x`];
        const y = pose[`${POSE_KEYPOINTS[i]}_y`];
        const z = pose[`${POSE_KEYPOINTS[i]}_z`];
        points.push([x, -y, -z * 500]);
    }
    // Center
    const meanX = points.reduce((sum, p) => sum + p[0], 0) / points.length;
    const meanY = points.reduce((sum, p) => sum + p[1], 0) / points.length;
    const meanZ = points.reduce((sum, p) => sum + p[2], 0) / points.length;
    for (let i = 0; i < points.length; i++) {
        points[i][0] -= meanX;
        points[i][1] -= meanY;
        points[i][2] -= meanZ;
    }
    const scale = 0.5;
    const scaledPoints = points.map(p => [p[0] * scale, p[1] * scale, p[2] * scale]);
    for (let i = 0; i < 33; i++) {
        if (pose3DJoints && pose3DJoints[i]) {
            pose3DJoints[i].position.set(scaledPoints[i][0], scaledPoints[i][1], scaledPoints[i][2]);
        }
    }
    for (let i = 0; i < POSE_CONNECTIONS.length; i++) {
        const [a, b] = POSE_CONNECTIONS[i];
        if (pose3DBones && pose3DBones[i]) {
            const geom = pose3DBones[i].geometry;
            geom.setFromPoints([
                new THREE.Vector3(scaledPoints[a][0], scaledPoints[a][1], scaledPoints[a][2]),
                new THREE.Vector3(scaledPoints[b][0], scaledPoints[b][1], scaledPoints[b][2])
            ]);
            geom.attributes.position.needsUpdate = true;
        }
    }
}

function continuousPoseUpdate() {
    if (!window.poseData || !videoPlayer) {
        requestAnimationFrame(continuousPoseUpdate);
        return;
    }
    // Find the closest frame to the current video time
    const timestamps = window.timestamps || window.poseData.map(p => p.timestamp);
    const t = videoPlayer.currentTime;
    let idx = 0;
    let minDiff = Math.abs(timestamps[0] - t);
    for (let i = 1; i < timestamps.length; i++) {
        const diff = Math.abs(timestamps[i] - t);
        if (diff < minDiff) {
            minDiff = diff;
            idx = i;
        }
    }
    if (idx !== lastFrameIdx) {
        update3DPose(idx);
        lastFrameIdx = idx;
    }
    requestAnimationFrame(continuousPoseUpdate);
}

// Listen for video time update from app.js
document.addEventListener('update3DPose', (e) => {
    update3DPose(e.detail.frameIdx);
});

// Listen for poseData loaded (from app.js)
document.addEventListener('poseDataLoaded', () => {
    update3DPose(0);
});

// Initialize viewer on load
window.addEventListener('DOMContentLoaded', () => {
    videoPlayer = document.getElementById('videoPlayer');
    setup3DPoseViewer();
    continuousPoseUpdate();
}); 