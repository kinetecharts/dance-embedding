import { DIMENSION_PLOT_PANEL_SIZE } from './config.js';

let poseData = [];
let reducedData = [];
let timestamps = [];
let frameNumbers = [];
let videoDuration = 0;
let plotTraceIdx = 0;

let rangeStart = 0;
let rangeEnd = 0;
let filteredTimestamps = [];
let filteredReducedData = [];
let filteredPoseData = [];
let isChunked = false;

let chunkedTimestamps = [];
let chunkedReducedData = [];

let plotXKey = 'x';
let plotYKey = 'y';

const videoPlayer = document.getElementById('videoPlayer');
const plotDiv = document.getElementById('plot');
const timelineDiv = document.getElementById('timeline');
const videoSelect = document.getElementById('videoSelect');
const methodSelect = document.getElementById('methodSelect');
const poseSelect = document.getElementById('poseSelect');
const reducedSelect = document.getElementById('reducedSelect');
const loadBtn = document.getElementById('loadBtn');
const rangeSlider = document.getElementById('rangeSlider');
const rangeLabel = document.getElementById('rangeLabel');

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
    // Auto-load if possible
    if (methodSelect.value && videoSelect.value) {
        loadBtn.onclick();
    }
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
    // Auto-load if possible
    if (poseMatch && reducedMatch) {
        loadBtn.onclick();
    }
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
    // Always load original pose data and timestamps
    poseData = await fetch(`/pose/${poseFile}`).then(r => r.json());
    window.poseData = poseData;
    timestamps = poseData.map(d => d.timestamp); // <-- always from poseData
    // Load reduced data
    reducedData = await fetch(`/reduced/${reducedFile}`).then(r => r.json());
    isChunked = /_c-/.test(reducedFile);
    // Auto-detect plot keys
    if (reducedData.length > 0) {
        const keys = Object.keys(reducedData[0]);
        if (keys.includes('x') && keys.includes('y')) {
            plotXKey = 'x';
            plotYKey = 'y';
        } else {
            // Use first two numeric columns after timestamp/frame_number
            const skip = new Set(['timestamp', 'frame_number']);
            const numericKeys = keys.filter(k => !skip.has(k));
            plotXKey = numericKeys[0];
            plotYKey = numericKeys[1];
        }
    }
    if (isChunked) {
        chunkedTimestamps = reducedData.map(d => d.timestamp);
        chunkedReducedData = reducedData;
        // DO NOT overwrite timestamps or poseData here!
    }
    videoDuration = timestamps[timestamps.length - 1] || 0;
    // Initialize noUiSlider for timeline range
    if (rangeSlider.noUiSlider) {
        rangeSlider.noUiSlider.destroy();
    }
    noUiSlider.create(rangeSlider, {
        start: [0, videoDuration],
        connect: true,
        range: {
            min: 0,
            max: videoDuration
        },
        step: 0.01,
        tooltips: [true, true],
        format: {
            to: v => v.toFixed(2),
            from: v => parseFloat(v)
        }
    });
    rangeLabel.textContent = `Showing: 0.00s - ${videoDuration.toFixed(2)}s`;
    rangeStart = 0;
    rangeEnd = videoDuration;
    rangeSlider.noUiSlider.on('update', function(values) {
        rangeStart = parseFloat(values[0]);
        rangeEnd = parseFloat(values[1]);
        rangeLabel.textContent = `Showing: ${values[0]}s - ${values[1]}s`;
        filterDataByRange();
        renderPlot();
        if (videoPlayer.currentTime < rangeStart) videoPlayer.currentTime = rangeStart;
        if (videoPlayer.currentTime > rangeEnd) videoPlayer.currentTime = rangeEnd;
        document.dispatchEvent(new Event('poseDataLoaded'));
    });
    filterDataByRange();
    renderPlot();
    setupSync();
    document.dispatchEvent(new Event('poseDataLoaded'));
};

function filterDataByRange() {
    if (isChunked) {
        // For chunked, filter chunked data for plot, but keep original poseData for 3D
        filteredTimestamps = [];
        filteredReducedData = [];
        for (let i = 0; i < chunkedTimestamps.length; i++) {
            if (chunkedTimestamps[i] >= rangeStart && chunkedTimestamps[i] <= rangeEnd) {
                filteredTimestamps.push(chunkedTimestamps[i]);
                filteredReducedData.push(chunkedReducedData[i]);
            }
        }
        // filteredPoseData remains as original poseData for 3D
        filteredPoseData = poseData;
        window.filteredPoseData = filteredPoseData;
        window.filteredTimestamps = timestamps;
    } else {
        // Filter reducedData, timestamps, poseData by range
        filteredTimestamps = [];
        filteredReducedData = [];
        filteredPoseData = [];
        for (let i = 0; i < timestamps.length; i++) {
            if (timestamps[i] >= rangeStart && timestamps[i] <= rangeEnd) {
                filteredTimestamps.push(timestamps[i]);
                filteredReducedData.push(reducedData[i]);
                filteredPoseData.push(poseData[i]);
            }
        }
        window.filteredPoseData = filteredPoseData;
        window.filteredTimestamps = filteredTimestamps;
    }
}

function renderPlot() {
    // Use filtered data
    const coords = filteredReducedData.map(d => [d[plotXKey], d[plotYKey]]);
    const color = filteredTimestamps;
    if (coords.length === 0) return;
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
        text: filteredTimestamps.map((t, i) => `Chunk ${i}<br>Time: ${t.toFixed(2)}s`),
        hovertemplate: '%{text}<extra></extra>',
        name: 'Trajectory'
    };
    // Highlight: for chunked, highlight the current chunk; for normal, highlight current frame
    let highlightIdx = 0;
    if (isChunked) {
        // Find last chunk index with timestamp <= current video time
        let t = videoPlayer.currentTime;
        for (let i = 0; i < filteredTimestamps.length; i++) {
            if (filteredTimestamps[i] <= t) {
                highlightIdx = i;
            } else {
                break;
            }
        }
    }
    const highlight = {
        x: [coords[highlightIdx][0]],
        y: [coords[highlightIdx][1]],
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
        let t = videoPlayer.currentTime;
        // Clamp to range
        if (t < rangeStart) {
            videoPlayer.currentTime = rangeStart;
            t = rangeStart;
        }
        if (t > rangeEnd) {
            videoPlayer.currentTime = rangeEnd;
            videoPlayer.pause();
            t = rangeEnd;
        }
        // For plot highlight (chunked: advance at chunk boundaries)
        if (isChunked) {
            renderPlot(); // re-render to update highlight
        } else {
            // Update highlight for normal reduction
            let idx = 0;
            for (let i = 0; i < filteredTimestamps.length; i++) {
                if (filteredTimestamps[i] <= t) {
                    idx = i;
                } else {
                    break;
                }
            }
            const coords = filteredReducedData.map(d => [d[plotXKey], d[plotYKey]]);
            if (coords.length > 0) {
                Plotly.restyle(plotDiv, {
                    x: [[coords[idx][0]]],
                    y: [[coords[idx][1]]]
                }, [plotTraceIdx]);
            }
        }
        // For 3D pose: always sync to closest pose frame in original poseData
        let poseIdx = 0;
        let minDiff = Math.abs(timestamps[0] - t);
        for (let i = 1; i < timestamps.length; i++) {
            const diff = Math.abs(timestamps[i] - t);
            if (diff < minDiff) {
                minDiff = diff;
                poseIdx = i;
            }
        }
        // console.log(`[3D Sync] videoTime: ${t.toFixed(3)}, poseIdx: ${poseIdx}, poseTimestamp: ${timestamps[poseIdx].toFixed(3)}, isChunked: ${isChunked}`);
        document.dispatchEvent(new CustomEvent('update3DPose', { detail: { frameIdx: poseIdx } }));
        // Debug overlay: show video time and pose timestamp
        debugOverlay.textContent = `Video: ${t.toFixed(2)}s | Pose: ${timestamps[poseIdx].toFixed(2)}s`;
    };
    // Clamp seeking as well
    videoPlayer.onseeking = function() {
        if (videoPlayer.currentTime < rangeStart) videoPlayer.currentTime = rangeStart;
        if (videoPlayer.currentTime > rangeEnd) videoPlayer.currentTime = rangeEnd;
    };
    // Click on plot to seek video
    plotDiv.on('plotly_click', function(data) {
        if (data.points && data.points.length > 0) {
            const idx = data.points[0].pointIndex;
            if (typeof idx === 'number') {
                const t = filteredTimestamps[idx];
                videoPlayer.currentTime = t;
            }
        }
    });
} 

// (3D viewer code removed; now handled in pose3d.js)

// Add debug overlay for video time and pose timestamp
let debugOverlay = document.getElementById('debugOverlay');
if (!debugOverlay) {
    debugOverlay = document.createElement('div');
    debugOverlay.id = 'debugOverlay';
    debugOverlay.style.position = 'absolute';
    debugOverlay.style.top = '8px';
    debugOverlay.style.right = '16px';
    debugOverlay.style.background = 'rgba(0,0,0,0.7)';
    debugOverlay.style.color = '#fff';
    debugOverlay.style.padding = '4px 12px';
    debugOverlay.style.borderRadius = '6px';
    debugOverlay.style.fontSize = '1rem';
    debugOverlay.style.zIndex = 1000;
    document.body.appendChild(debugOverlay);
}

window.onload = populateDropdowns;