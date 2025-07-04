import { DIMENSION_PLOT_PANEL_SIZE } from './config.js';

let poseData = [];
let reducedData = [];
let timestamps = [];
let frameNumbers = [];
let videoDuration = 0;
let plotTraceIdx = 0;

const videoPlayer = document.getElementById('videoPlayer');
const plotDiv = document.getElementById('plot');
const timelineDiv = document.getElementById('timeline');
const videoSelect = document.getElementById('videoSelect');
const methodSelect = document.getElementById('methodSelect');
const poseSelect = document.getElementById('poseSelect');
const reducedSelect = document.getElementById('reducedSelect');
const loadBtn = document.getElementById('loadBtn');

let poseFiles = [];
let reducedFiles = [];

async function fetchFileList(endpoint) {
    const resp = await fetch(endpoint);
    if (!resp.ok) return [];
    return await resp.json();
}

async function populateDropdowns() {
    const videos = await fetchFileList('/list_videos');
    poseFiles = await fetchFileList('/list_poses');
    reducedFiles = await fetchFileList('/list_reduced');
    videoSelect.innerHTML = videos.map(f => `<option value="${f}">${f}</option>`).join('');
    // Trigger change to auto-select pose/reduced
    videoSelect.onchange();
}

videoSelect.onchange = async function() {
    const videoFile = videoSelect.value;
    // Remove _with_pose suffix if present
    let base = videoFile.replace(/\.[^/.]+$/, "");
    if (base.endsWith('_with_pose')) {
        base = base.slice(0, -10); // Remove '_with_pose'
    }
    // Populate reduction methods for this video
    const methods = await fetchFileList(`/list_reductions_for_video/${base}`);
    methodSelect.innerHTML = methods.length ? methods.map(m => `<option value="${m}">${m.toUpperCase()}</option>`).join('') : '<option value="">(not found)</option>';
    methodSelect.onchange();
};

methodSelect.onchange = function() {
    const videoFile = videoSelect.value;
    // Remove _with_pose suffix if present
    let base = videoFile.replace(/\.[^/.]+$/, "");
    if (base.endsWith('_with_pose')) {
        base = base.slice(0, -10); // Remove '_with_pose'
    }
    const method = methodSelect.value;
    // Find reduced file for this video and method
    const reducedMatch = reducedFiles.find(f => f.startsWith(`${base}_${method}_`));
    reducedSelect.innerHTML = reducedMatch ? `<option value="${reducedMatch}">${reducedMatch}</option>` : '<option value="">(not found)</option>';
    // Pose file
    const poseMatch = poseFiles.find(f => f.replace(/\.[^/.]+$/, "") === base);
    poseSelect.innerHTML = poseMatch ? `<option value="${poseMatch}">${poseMatch}</option>` : '<option value="">(not found)</option>';
    loadBtn.disabled = !(poseMatch && reducedMatch);
};

poseSelect.onchange = reducedSelect.onchange = function() {
    loadBtn.disabled = !(poseSelect.value && reducedSelect.value);
};

loadBtn.onclick = async function() {
    const videoFile = videoSelect.value;
    const poseFile = poseSelect.value;
    const reducedFile = reducedSelect.value;
    if (!videoFile || !poseFile || !reducedFile) return;
    videoPlayer.src = `/video/${videoFile}`;
    videoPlayer.load();
    poseData = await fetch(`/pose/${poseFile}`).then(r => r.json());
    window.poseData = poseData;
    reducedData = await fetch(`/reduced/${reducedFile}`).then(r => r.json());
    
    // Extract data from CSV format
    timestamps = reducedData.map(d => d.timestamp);
    frameNumbers = reducedData.map(d => d.frame_number);
    videoDuration = timestamps[timestamps.length - 1] || 0;
    
    renderPlot();
    setupSync();
    // Notify 3D viewer that poseData is loaded
    document.dispatchEvent(new Event('poseDataLoaded'));
};

function renderPlot() {
    // Extract x, y coordinates from CSV data
    const coords = reducedData.map(d => [d.x, d.y]);
    const color = timestamps;
    const trace = {
        x: coords.map(d => d[0]),
        y: coords.map(d => d[1]),
        mode: 'markers+lines',
        marker: {
            size: 8,
            color: color,
            colorscale: 'Viridis',
            colorbar: { title: 'Time (s)' }
        },
        line: { color: 'red', width: 1 },
        text: frameNumbers.map((f, i) => `Frame ${f}<br>Time: ${timestamps[i].toFixed(2)}s`),
        hovertemplate: '%{text}<extra></extra>',
        name: 'Trajectory'
    };
    const highlight = {
        x: [coords[0][0]],
        y: [coords[0][1]],
        mode: 'markers',
        marker: { size: 16, color: 'red', symbol: 'diamond' },
        name: 'Current'
    };
    plotTraceIdx = 1;
    Plotly.newPlot(plotDiv, [trace, highlight], {
        title: 'Dimension Reduction Trajectory',
        width: DIMENSION_PLOT_PANEL_SIZE,
        height: DIMENSION_PLOT_PANEL_SIZE,
        xaxis: { title: 'Component 1' },
        yaxis: { title: 'Component 2' },
        showlegend: false
    });
}

// Timeline rendering and sync
function setupTimeline() {
    // Simple color bar timeline
    timelineDiv.innerHTML = '';
    const canvas = document.createElement('canvas');
    canvas.width = timelineDiv.offsetWidth || PANEL_SIZE;
    canvas.height = 30;
    timelineDiv.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
    grad.addColorStop(0, '#440154');
    grad.addColorStop(0.5, '#21908d');
    grad.addColorStop(1, '#fde725');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 10, canvas.width, 10);

    // Click to seek
    canvas.onclick = function(e) {
        const x = e.offsetX;
        const t = (x / canvas.width) * videoDuration;
        videoPlayer.currentTime = t;
    };
}

// Synchronize plot highlight with video
function setupSync() {
    videoPlayer.ontimeupdate = function() {
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
        // Update highlight
        const coords = reducedData.map(d => [d.x, d.y]);
        Plotly.restyle(plotDiv, {
            x: [[coords[idx][0]]],
            y: [[coords[idx][1]]]
        }, [plotTraceIdx]);
        // Notify 3D viewer to update pose
        document.dispatchEvent(new CustomEvent('update3DPose', { detail: { frameIdx: idx } }));
    };

    // Click on plot to seek video
    plotDiv.on('plotly_click', function(data) {
        if (data.points && data.points.length > 0) {
            const idx = data.points[0].pointIndex;
            if (typeof idx === 'number') {
                videoPlayer.currentTime = timestamps[idx];
            }
        }
    });
} 

// (3D viewer code removed; now handled in pose3d.js)

window.onload = populateDropdowns;