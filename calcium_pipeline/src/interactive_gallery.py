"""
Interactive ROI Gallery Generator
=================================

Creates an interactive HTML viewer for evaluating contour detection results.

Features:
- Toggle between background layers (max, correlation, std, mean projections)
- Toggle overlay options (contours, seed circles, ROI labels)
- Click on any ROI to see detailed diagnostics
- Zoom and pan controls
- Filter ROIs by confidence, size, circularity

Author: Calcium Pipeline
"""

import numpy as np
import json
import base64
import os
import logging
from typing import Optional, List, Dict, Any
from io import BytesIO

logger = logging.getLogger(__name__)


def array_to_base64_png(arr: np.ndarray, cmap: str = 'gray', vmin: float = None, vmax: float = None) -> str:
    """Convert numpy array to base64-encoded PNG string."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Normalize array
    if vmin is None:
        vmin = np.percentile(arr, 1)
    if vmax is None:
        vmax = np.percentile(arr, 99)
    
    arr_norm = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1).astype(np.float32)
    
    # Apply colormap — request float32 output to avoid a full float64 RGBA copy
    cm = plt.get_cmap(cmap)
    arr_colored = (cm(arr_norm, bytes=True)[:, :, :3]).astype(np.uint8)
    
    # Convert to PNG
    img = Image.fromarray(arr_colored)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_roi_diagnostic_data(
    seeds,  # ContourSeedResult
    movie: np.ndarray,
    projections,  # ProjectionSet
    max_rois: int = 500,
    traces_denoised: Optional[np.ndarray] = None,
    spike_trains: Optional[np.ndarray] = None,
    pipeline_traces_dff: Optional[np.ndarray] = None,
    pipeline_traces_raw: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Generate diagnostic data for each ROI.
    
    When pipeline traces are provided (pipeline_traces_dff, pipeline_traces_raw),
    the gallery displays these directly — giving an honest view of exactly what
    the pipeline produced and what OASIS received.  When not available, falls
    back to extracting traces from the movie using a bounding box.
    
    Parameters
    ----------
    pipeline_traces_dff : array (N, T), optional
        Pipeline's corrected ΔF/F₀ traces (what OASIS received).
    pipeline_traces_raw : array (N, T), optional
        Raw fluorescence traces from the pipeline's weighted extraction.
    traces_denoised : array (N, T), optional
        OASIS-denoised traces.
    spike_trains : array (N, T), optional
        Inferred spike trains.
    """
    
    T, d1, d2 = movie.shape
    has_pipeline_traces = (pipeline_traces_dff is not None and 
                           pipeline_traces_raw is not None)
    
    if has_pipeline_traces:
        logger.info(f"  Using pipeline traces ({pipeline_traces_dff.shape[0]} ROIs)")
    else:
        logger.info(f"  No pipeline traces — extracting from movie (bounding box)")
    
    roi_data = []
    
    for i in range(min(seeds.n_seeds, max_rois)):
        y, x = seeds.centers[i]
        r = seeds.radii[i]
        
        # Extract local ROI region (for bounding box info)
        margin = int(r * 3)
        y_min = max(0, int(y) - margin)
        y_max = min(d1, int(y) + margin)
        x_min = max(0, int(x) - margin)
        x_max = min(d2, int(x) + margin)
        
        if has_pipeline_traces and i < pipeline_traces_dff.shape[0]:
            # Use the pipeline's actual traces — this is what OASIS saw
            raw_trace = pipeline_traces_raw[i]
            dff = pipeline_traces_dff[i] * 100  # convert to percent for display
        else:
            # Fallback: extract from movie with bounding box
            raw_trace = movie[:, max(0, int(y)-int(r)):min(d1, int(y)+int(r)+1),
                                 max(0, int(x)-int(r)):min(d2, int(x)+int(r)+1)].mean(axis=(1, 2))
            trace_mean = np.mean(raw_trace)
            if trace_mean > 1.0:
                baseline = np.percentile(raw_trace, 10)
                if baseline > 0:
                    dff = (raw_trace - baseline) / baseline * 100
                else:
                    dff = raw_trace - np.mean(raw_trace)
            else:
                dff = raw_trace * 100
        
        # Denoised trace (from OASIS deconvolution)
        denoised_dff = None
        if traces_denoised is not None and i < traces_denoised.shape[0]:
            den = traces_denoised[i]
            # Convert to percent for display, consistent with dff trace
            denoised_dff = (den * 100).tolist()
        
        # Spike times (from deconvolution)
        spike_frames = None
        if spike_trains is not None and i < spike_trains.shape[0]:
            spk = spike_trains[i]
            spike_frames = np.where(spk > 0)[0].tolist()
        
        # Get contour points if available
        contour_points = None
        circularity = None
        solidity = None
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            contour = seeds.contours[i].contour.squeeze()
            if len(contour.shape) == 2:
                contour_points = contour.tolist()
            circularity = seeds.contours[i].circularity
            solidity = seeds.contours[i].solidity
        
        roi_info = {
            'id': i,
            'center': [float(y), float(x)],
            'radius': float(r),
            'confidence': float(seeds.confidence[i]),
            'intensity': float(seeds.intensities[i]),
            'has_contour': bool(seeds.contour_success[i]),
            'contour_points': contour_points,
            'circularity': float(circularity) if circularity else None,
            'solidity': float(solidity) if solidity else None,
            'source': str(seeds.source_projection[i]) if hasattr(seeds, 'source_projection') else 'unknown',
            'boundary_touching': bool(seeds.boundary_touching[i]) if (hasattr(seeds, 'boundary_touching') and len(seeds.boundary_touching)) else False,
            'bbox': [int(y_min), int(y_max), int(x_min), int(x_max)],
            'trace': dff.tolist() if hasattr(dff, 'tolist') else list(dff),
            'trace_raw': raw_trace.tolist() if hasattr(raw_trace, 'tolist') else list(raw_trace),
            'trace_denoised': denoised_dff,
            'spike_frames': spike_frames,
            'uses_pipeline_traces': has_pipeline_traces and i < pipeline_traces_dff.shape[0],
        }
        
        roi_data.append(roi_info)
    
    return roi_data


def generate_interactive_gallery(
    seeds,  # ContourSeedResult
    projections,  # ProjectionSet
    movie: np.ndarray,
    output_path: str,
    title: str = "ROI Detection Gallery",
    max_rois: int = 500,
    movie_processed: Optional[np.ndarray] = None,
    traces_denoised: Optional[np.ndarray] = None,
    spike_trains: Optional[np.ndarray] = None,
    pipeline_traces_dff: Optional[np.ndarray] = None,
    pipeline_traces_raw: Optional[np.ndarray] = None,
) -> str:
    """
    Generate an interactive HTML gallery for ROI visualization.
    
    Parameters
    ----------
    seeds : ContourSeedResult
        Detection results with contours
    projections : ProjectionSet
        Projection images
    movie : np.ndarray
        Raw movie (T, Y, X) — used for background images (baseline frame,
        peak frame)
    output_path : str
        Path to save HTML file
    title : str
        Title for the gallery
    max_rois : int
        Maximum number of ROIs to include
    movie_processed : np.ndarray, optional
        Preprocessed movie (motion-corrected + bleach-corrected).  Used for
        trace extraction.  If not provided, falls back to ``movie``.
        
    Returns
    -------
    str
        Path to generated HTML file
    """
    # Use processed movie for traces, raw movie for background images
    movie_for_traces = movie_processed if movie_processed is not None else movie
    logger.info(f"Generating interactive gallery with {min(seeds.n_seeds, max_rois)} ROIs...")
    
    # Generate base64 images for each projection
    logger.info("  Encoding projection images...")
    
    max_img = array_to_base64_png(projections.max_proj, 'gray')
    corr_img = array_to_base64_png(projections.correlation, 'gray')
    std_img = array_to_base64_png(projections.std_proj, 'gray')
    mean_img = array_to_base64_png(projections.mean_proj, 'gray')
    
    # Baseline frame: average of first N frames (low-activity reference)
    n_baseline = min(50, movie.shape[0])
    baseline_frame = np.mean(movie[:n_baseline], axis=0)
    baseline_img = array_to_base64_png(baseline_frame, 'gray')
    
    # Max-projection frame: the single frame whose mean intensity is highest
    # (i.e. the frame contributing most to the max projection)
    frame_means = np.mean(movie, axis=(1, 2))
    max_frame_idx = int(np.argmax(frame_means))
    max_frame = movie[max_frame_idx].astype(np.float32)
    max_frame_img = array_to_base64_png(max_frame, 'gray')
    logger.info(f"  Max-intensity frame: #{max_frame_idx} "
                f"(mean={frame_means[max_frame_idx]:.1f})")
    
    # Generate ROI data (traces from processed movie, thumbnails from raw)
    logger.info("  Computing ROI diagnostics...")
    roi_data = generate_roi_diagnostic_data(
        seeds, movie_for_traces, projections, max_rois,
        traces_denoised=traces_denoised,
        spike_trains=spike_trains,
        pipeline_traces_dff=pipeline_traces_dff,
        pipeline_traces_raw=pipeline_traces_raw,
    )
    
    # Image dimensions
    d1, d2 = projections.max_proj.shape
    
    # Statistics
    stats = {
        'total_seeds': seeds.n_seeds,
        'with_contours': int(seeds.contour_success.sum()),
        'fallback': int((~seeds.contour_success).sum()),
        'median_radius': float(np.median(seeds.radii)),
        'median_confidence': float(np.median(seeds.confidence)),
        'radius_range': [float(seeds.radii.min()), float(seeds.radii.max())],
    }
    
    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        /* Left panel - Controls */
        .controls-panel {{
            width: 280px;
            background: #16213e;
            padding: 15px;
            overflow-y: auto;
            border-right: 1px solid #0f3460;
        }}
        
        .controls-panel h2 {{
            color: #e94560;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        
        .control-group {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #0f3460;
        }}
        
        .control-group h3 {{
            color: #00d9ff;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        
        .control-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            cursor: pointer;
        }}
        
        .control-item input[type="radio"],
        .control-item input[type="checkbox"] {{
            margin-right: 10px;
            cursor: pointer;
        }}
        
        .control-item label {{
            cursor: pointer;
            font-size: 13px;
        }}
        
        .slider-container {{
            margin: 10px 0;
        }}
        
        .slider-container label {{
            display: block;
            font-size: 12px;
            margin-bottom: 5px;
            color: #aaa;
        }}
        
        .slider-container input[type="range"] {{
            width: 100%;
        }}
        
        .slider-value {{
            font-size: 11px;
            color: #00d9ff;
        }}
        
        .stats-box {{
            background: #0f3460;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
        }}
        
        .stats-box .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }}
        
        .stats-box .stat-value {{
            color: #00d9ff;
            font-weight: bold;
        }}
        
        /* Main viewer */
        .viewer-panel {{
            flex: 1;
            position: relative;
            overflow: hidden;
            background: #0a0a0a;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .canvas-container {{
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            cursor: grab;
        }}
        
        .canvas-container:active {{
            cursor: grabbing;
        }}
        
        #mainCanvas {{
            /* Canvas will be positioned via JS */
            position: absolute;
            image-rendering: pixelated;
        }}
        
        /* Zoom controls */
        .zoom-controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            background: rgba(22, 33, 62, 0.9);
            padding: 10px 20px;
            border-radius: 25px;
        }}
        
        .zoom-btn {{
            background: #0f3460;
            border: none;
            color: white;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 18px;
            transition: background 0.2s;
        }}
        
        .zoom-btn:hover {{
            background: #e94560;
        }}
        
        .zoom-level {{
            display: flex;
            align-items: center;
            font-size: 14px;
            min-width: 60px;
            justify-content: center;
        }}
        
        /* Right panel - ROI details */
        .details-panel {{
            width: 350px;
            background: #16213e;
            padding: 15px;
            overflow-y: auto;
            border-left: 1px solid #0f3460;
        }}
        
        .details-panel h2 {{
            color: #e94560;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        
        .roi-details {{
            display: none;
        }}
        
        .roi-details.active {{
            display: block;
        }}
        
        .roi-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .roi-id {{
            font-size: 24px;
            font-weight: bold;
            color: #00d9ff;
        }}
        
        .roi-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .roi-badge.contour {{
            background: #00c853;
            color: #000;
        }}
        
        .roi-badge.fallback {{
            background: #ff5252;
            color: #fff;
        }}
        
        .roi-badge.boundary {{
            background: #ff9900;
            color: #000;
        }}
        
        .detail-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .detail-item {{
            background: #0f3460;
            padding: 10px;
            border-radius: 6px;
        }}
        
        .detail-item .label {{
            font-size: 11px;
            color: #888;
            margin-bottom: 4px;
        }}
        
        .detail-item .value {{
            font-size: 16px;
            font-weight: bold;
            color: #00d9ff;
        }}
        
        .trace-container {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        
        .trace-container h4 {{
            color: #00d9ff;
            margin-bottom: 10px;
            font-size: 13px;
        }}
        
        #traceCanvas {{
            width: 100%;
            height: 120px;
            background: #0a0a1a;
            border-radius: 4px;
        }}
        
        .no-selection {{
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }}
        
        .no-selection .icon {{
            font-size: 48px;
            margin-bottom: 15px;
        }}
        
        /* ROI list */
        .roi-list {{
            max-height: 300px;
            overflow-y: auto;
            margin-top: 15px;
        }}
        
        .roi-list-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #0f3460;
            margin: 4px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        
        .roi-list-item:hover {{
            background: #1a4a7a;
        }}
        
        .roi-list-item.selected {{
            background: #e94560;
        }}
        
        .roi-list-item .roi-num {{
            font-weight: bold;
        }}
        
        .roi-list-item .roi-conf {{
            font-size: 12px;
            color: #aaa;
        }}
        
        /* Tooltip */
        .tooltip {{
            position: absolute;
            background: rgba(22, 33, 62, 0.95);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            border: 1px solid #0f3460;
            display: none;
        }}
        
        .tooltip.visible {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Controls Panel -->
        <div class="controls-panel">
            <h2>{title}</h2>
            
            <div class="control-group">
                <h3>Background Layer</h3>
                <div class="control-item">
                    <input type="radio" name="background" id="bg-max" value="max" checked>
                    <label for="bg-max">Max Projection</label>
                </div>
                <div class="control-item">
                    <input type="radio" name="background" id="bg-corr" value="corr">
                    <label for="bg-corr">Correlation</label>
                </div>
                <div class="control-item">
                    <input type="radio" name="background" id="bg-std" value="std">
                    <label for="bg-std">Std Projection</label>
                </div>
                <div class="control-item">
                    <input type="radio" name="background" id="bg-mean" value="mean">
                    <label for="bg-mean">Mean Projection</label>
                </div>
                <div class="control-item">
                    <input type="radio" name="background" id="bg-baseline" value="baseline">
                    <label for="bg-baseline">Baseline Frame</label>
                </div>
                <div class="control-item">
                    <input type="radio" name="background" id="bg-maxframe" value="maxframe">
                    <label for="bg-maxframe">Peak Frame (#{max_frame_idx})</label>
                </div>
            </div>
            
            <div class="control-group">
                <h3>Overlays</h3>
                <div class="control-item">
                    <input type="checkbox" id="show-contours" checked>
                    <label for="show-contours">Show Contours</label>
                </div>
                <div class="control-item">
                    <input type="checkbox" id="show-circles" checked>
                    <label for="show-circles">Show Seed Circles</label>
                </div>
                <div class="control-item">
                    <input type="checkbox" id="show-labels">
                    <label for="show-labels">Show ROI Labels</label>
                </div>
                <div class="control-item">
                    <input type="checkbox" id="show-fallback" checked>
                    <label for="show-fallback">Show Fallback ROIs</label>
                </div>
                <div class="control-item">
                    <input type="checkbox" id="show-edge-rois" checked>
                    <label for="show-edge-rois" style="color: #ff9900;">Show Edge ROIs</label>
                </div>
            </div>
            
            <div class="control-group">
                <h3>Filters</h3>
                <div class="slider-container">
                    <label>Min Confidence: <span class="slider-value" id="conf-value">0.00</span></label>
                    <input type="range" id="min-confidence" min="0" max="1" step="0.05" value="0">
                </div>
                <div class="slider-container">
                    <label>Min Radius: <span class="slider-value" id="radius-value">0</span>px</label>
                    <input type="range" id="min-radius" min="0" max="50" step="1" value="0">
                </div>
            </div>
            
            <div class="control-group">
                <h3>Statistics</h3>
                <div class="stats-box">
                    <div class="stat-row">
                        <span>Total ROIs:</span>
                        <span class="stat-value">{stats['total_seeds']}</span>
                    </div>
                    <div class="stat-row">
                        <span>With Contours:</span>
                        <span class="stat-value">{stats['with_contours']}</span>
                    </div>
                    <div class="stat-row">
                        <span>Fallback:</span>
                        <span class="stat-value">{stats['fallback']}</span>
                    </div>
                    <div class="stat-row">
                        <span>Median Radius:</span>
                        <span class="stat-value">{stats['median_radius']:.1f}px</span>
                    </div>
                    <div class="stat-row">
                        <span>Median Confidence:</span>
                        <span class="stat-value">{stats['median_confidence']:.2f}</span>
                    </div>
                    <div class="stat-row">
                        <span>Visible ROIs:</span>
                        <span class="stat-value" id="visible-count">{len(roi_data)}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Viewer -->
        <div class="viewer-panel">
            <div class="canvas-container" id="canvasContainer">
                <canvas id="mainCanvas"></canvas>
            </div>
            
            <div class="zoom-controls">
                <button class="zoom-btn" id="zoom-out">−</button>
                <span class="zoom-level"><span id="zoom-percent">100</span>%</span>
                <button class="zoom-btn" id="zoom-in">+</button>
                <button class="zoom-btn" id="zoom-fit">⊡</button>
                <button class="zoom-btn" id="zoom-reset">1:1</button>
            </div>
            
            <div class="tooltip" id="tooltip"></div>
        </div>
        
        <!-- Right Details Panel -->
        <div class="details-panel">
            <h2>ROI Details</h2>
            
            <div class="no-selection" id="noSelection">
                <p>Click on an ROI to view details</p>
            </div>
            
            <div class="roi-details" id="roiDetails">
                <div class="roi-header">
                    <span class="roi-id">ROI #<span id="roi-num">0</span></span>
                    <span class="roi-badge" id="roi-badge">Contour</span>
                </div>
                
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="label">Confidence</div>
                        <div class="value" id="detail-confidence">0.00</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Radius</div>
                        <div class="value" id="detail-radius">0px</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Circularity</div>
                        <div class="value" id="detail-circularity">N/A</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Source</div>
                        <div class="value" id="detail-source">max</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Center (Y, X)</div>
                        <div class="value" id="detail-center">0, 0</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Intensity</div>
                        <div class="value" id="detail-intensity">0.00</div>
                    </div>
                </div>
                
                <div class="trace-container">
                    <h4>Calcium Trace (dF/F %)</h4>
                    <canvas id="traceCanvas"></canvas>
                </div>
            </div>
            
            <h3 style="margin-top: 20px; color: #00d9ff; font-size: 14px;">ROI List</h3>
            <div style="margin: 8px 0;">
                <input type="text" id="roiSearch" placeholder="Search ROI # (e.g. 42, 100-200)"
                    style="width:100%; padding:8px 10px; border-radius:6px; border:1px solid #0f3460;
                    background:#0a0a1a; color:#eee; font-size:13px; outline:none;"
                    onfocus="this.style.borderColor='#00d9ff'" onblur="this.style.borderColor='#0f3460'">
            </div>
            <div class="roi-list" id="roiList"></div>
        </div>
    </div>
    
    <script>
        // Data
        const imageWidth = {d2};
        const imageHeight = {d1};
        const roiData = {json.dumps(roi_data)};
        
        const images = {{
            max: new Image(),
            corr: new Image(),
            std: new Image(),
            mean: new Image(),
            baseline: new Image(),
            maxframe: new Image()
        }};
        
        images.max.src = 'data:image/png;base64,{max_img}';
        images.corr.src = 'data:image/png;base64,{corr_img}';
        images.std.src = 'data:image/png;base64,{std_img}';
        images.mean.src = 'data:image/png;base64,{mean_img}';
        images.baseline.src = 'data:image/png;base64,{baseline_img}';
        images.maxframe.src = 'data:image/png;base64,{max_frame_img}';
        
        // State
        let currentBackground = 'max';
        let zoom = 1;
        let panX = 0, panY = 0;
        let isDragging = false;
        let dragStart = {{ x: 0, y: 0 }};
        let selectedRoi = null;
        let showContours = true;
        let showCircles = true;
        let showLabels = false;
        let showFallback = true;
        let showEdgeRois = true;
        let minConfidence = 0;
        let minRadius = 0;
        
        // Canvas setup
        const canvas = document.getElementById('mainCanvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('canvasContainer');
        
        canvas.width = imageWidth;
        canvas.height = imageHeight;
        
        // Wait for images to load
        let loadedCount = 0;
        let imagesLoaded = false;
        
        Object.values(images).forEach(img => {{
            img.onload = () => {{
                loadedCount++;
                if (loadedCount === 6) {{
                    imagesLoaded = true;
                    initializeDisplay();
                }}
            }};
        }});
        
        function initializeDisplay() {{
            // Multiple attempts to ensure container is properly sized
            let attempts = 0;
            function tryInit() {{
                const rect = container.getBoundingClientRect();
                if (rect.width > 100 && rect.height > 100) {{
                    fitToView();
                    render();
                    populateRoiList();
                }} else if (attempts < 10) {{
                    attempts++;
                    requestAnimationFrame(tryInit);
                }}
            }}
            requestAnimationFrame(tryInit);
        }}
        
        // Re-fit on window resize
        window.addEventListener('resize', () => {{
            if (imagesLoaded) {{
                fitToView();
                render();
            }}
        }});
        
        function getVisibleRois() {{
            return roiData.filter(roi => {{
                if (roi.confidence < minConfidence) return false;
                if (roi.radius < minRadius) return false;
                if (!showFallback && !roi.has_contour) return false;
                if (!showEdgeRois && roi.boundary_touching) return false;
                return true;
            }});
        }}
        
        function render() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw background
            ctx.drawImage(images[currentBackground], 0, 0);
            
            const visibleRois = getVisibleRois();
            document.getElementById('visible-count').textContent = visibleRois.length;
            
            // Draw ROIs
            visibleRois.forEach(roi => {{
                const [y, x] = roi.center;
                const r = roi.radius;
                
                // Determine colors
                const isSelected = selectedRoi === roi.id;
                let strokeColor = roi.has_contour ? '#00ff88' : '#ff4444';
                if (roi.boundary_touching) strokeColor = '#ff9900';
                let lineWidth = isSelected ? 3 : 1.5;
                
                if (isSelected) {{
                    strokeColor = '#ffff00';
                }}
                
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth;
                
                // Draw contour if available and enabled
                if (showContours && roi.has_contour && roi.contour_points) {{
                    ctx.beginPath();
                    const pts = roi.contour_points;
                    ctx.moveTo(pts[0][0], pts[0][1]);
                    for (let i = 1; i < pts.length; i++) {{
                        ctx.lineTo(pts[i][0], pts[i][1]);
                    }}
                    ctx.closePath();
                    ctx.stroke();
                }}
                
                // Draw circle if enabled
                if (showCircles) {{
                    ctx.beginPath();
                    ctx.arc(x, y, r, 0, 2 * Math.PI);
                    if (roi.has_contour) {{
                        ctx.setLineDash([]);
                    }} else {{
                        ctx.setLineDash([4, 4]);
                    }}
                    ctx.stroke();
                    ctx.setLineDash([]);
                }}
                
                // Draw label if enabled
                if (showLabels) {{
                    ctx.fillStyle = strokeColor;
                    ctx.font = '10px sans-serif';
                    ctx.fillText(roi.id.toString(), x + r + 2, y - r);
                }}
            }});
            
            // Update canvas transform
            updateTransform();
        }}
        
        function updateTransform() {{
            const containerRect = container.getBoundingClientRect();
            const scaledWidth = imageWidth * zoom;
            const scaledHeight = imageHeight * zoom;
            
            // Center the canvas in the container, plus any pan offset
            const offsetX = (containerRect.width - scaledWidth) / 2 + panX;
            const offsetY = (containerRect.height - scaledHeight) / 2 + panY;
            
            canvas.style.left = offsetX + 'px';
            canvas.style.top = offsetY + 'px';
            canvas.style.width = scaledWidth + 'px';
            canvas.style.height = scaledHeight + 'px';
            
            document.getElementById('zoom-percent').textContent = Math.round(zoom * 100);
        }}
        
        function getMinZoom() {{
            // Minimum zoom is when image fills the container
            const containerRect = container.getBoundingClientRect();
            const availableWidth = containerRect.width;
            const availableHeight = containerRect.height - 70;
            const scaleX = availableWidth / imageWidth;
            const scaleY = availableHeight / imageHeight;
            return Math.max(scaleX, scaleY);  // Use max to ensure image FILLS (not fits)
        }}
        
        function fitToView() {{
            const containerRect = container.getBoundingClientRect();
            const availableWidth = containerRect.width;
            const availableHeight = containerRect.height - 70;  // Space for zoom controls
            
            if (availableWidth <= 0 || availableHeight <= 0) {{
                return;  // Container not ready
            }}
            
            // Fill the container (use max scale so image covers the area)
            const scaleX = availableWidth / imageWidth;
            const scaleY = availableHeight / imageHeight;
            zoom = Math.max(scaleX, scaleY);  // Fill, not fit
            
            panX = 0;
            panY = 0;
            updateTransform();
        }}
        
        function findRoiAtPoint(canvasX, canvasY) {{
            const visibleRois = getVisibleRois();
            for (let i = visibleRois.length - 1; i >= 0; i--) {{
                const roi = visibleRois[i];
                const [y, x] = roi.center;
                const r = roi.radius;
                const dist = Math.sqrt((canvasX - x) ** 2 + (canvasY - y) ** 2);
                if (dist <= r * 1.5) {{
                    return roi;
                }}
            }}
            return null;
        }}
        
        function selectRoi(roi) {{
            selectedRoi = roi ? roi.id : null;
            
            if (roi) {{
                document.getElementById('noSelection').style.display = 'none';
                document.getElementById('roiDetails').classList.add('active');
                
                document.getElementById('roi-num').textContent = roi.id;
                
                const badge = document.getElementById('roi-badge');
                badge.textContent = roi.has_contour ? 'Contour' : 'Fallback';
                badge.className = 'roi-badge ' + (roi.has_contour ? 'contour' : 'fallback');
                if (roi.boundary_touching) {{
                    badge.textContent = 'Edge ROI';
                    badge.className = 'roi-badge boundary';
                }}
                
                document.getElementById('detail-confidence').textContent = roi.confidence.toFixed(3);
                document.getElementById('detail-radius').textContent = roi.radius.toFixed(1) + 'px';
                document.getElementById('detail-circularity').textContent = 
                    roi.circularity ? roi.circularity.toFixed(3) : 'N/A';
                document.getElementById('detail-source').textContent = roi.source;
                document.getElementById('detail-center').textContent = 
                    roi.center[0].toFixed(1) + ', ' + roi.center[1].toFixed(1);
                document.getElementById('detail-intensity').textContent = roi.intensity.toFixed(3);
                
                drawTrace(roi.trace, roi.trace_denoised, roi.spike_frames);
                
                // Highlight in list
                document.querySelectorAll('.roi-list-item').forEach(item => {{
                    item.classList.toggle('selected', parseInt(item.dataset.id) === roi.id);
                }});
            }} else {{
                document.getElementById('noSelection').style.display = 'block';
                document.getElementById('roiDetails').classList.remove('active');
            }}
            
            render();
        }}
        
        function drawTrace(trace, traceDenoised, spikeFrames) {{
            const traceCanvas = document.getElementById('traceCanvas');
            const tctx = traceCanvas.getContext('2d');
            
            // Set actual size
            traceCanvas.width = traceCanvas.offsetWidth * 2;
            traceCanvas.height = traceCanvas.offsetHeight * 2;
            tctx.scale(2, 2);
            
            const w = traceCanvas.offsetWidth;
            const h = traceCanvas.offsetHeight;
            
            tctx.fillStyle = '#0a0a1a';
            tctx.fillRect(0, 0, w, h);
            
            if (!trace || trace.length === 0) return;
            
            // Compute range across both raw and denoised
            let allVals = trace.slice();
            if (traceDenoised && traceDenoised.length > 0) {{
                allVals = allVals.concat(traceDenoised);
            }}
            const min = Math.min(...allVals);
            const max = Math.max(...allVals);
            const range = max - min || 1;
            
            function toY(val) {{ return h - ((val - min) / range) * (h - 20) - 10; }}
            function toX(i, len) {{ return (i / (len - 1)) * w; }}
            
            // Draw grid
            tctx.strokeStyle = '#1a1a3a';
            tctx.lineWidth = 0.5;
            for (let i = 0; i <= 4; i++) {{
                const y = h * i / 4;
                tctx.beginPath();
                tctx.moveTo(0, y);
                tctx.lineTo(w, y);
                tctx.stroke();
            }}
            
            // Draw zero line
            tctx.strokeStyle = '#444';
            tctx.beginPath();
            tctx.moveTo(0, toY(0));
            tctx.lineTo(w, toY(0));
            tctx.stroke();
            
            // Draw raw trace (dim)
            tctx.strokeStyle = 'rgba(0, 217, 255, 0.25)';
            tctx.lineWidth = 1;
            tctx.beginPath();
            for (let i = 0; i < trace.length; i++) {{
                const x = toX(i, trace.length);
                const y = toY(trace[i]);
                if (i === 0) tctx.moveTo(x, y);
                else tctx.lineTo(x, y);
            }}
            tctx.stroke();
            
            // Draw denoised trace (bright, primary)
            if (traceDenoised && traceDenoised.length > 0) {{
                tctx.strokeStyle = '#00e676';
                tctx.lineWidth = 1.8;
                tctx.beginPath();
                for (let i = 0; i < traceDenoised.length; i++) {{
                    const x = toX(i, traceDenoised.length);
                    const y = toY(traceDenoised[i]);
                    if (i === 0) tctx.moveTo(x, y);
                    else tctx.lineTo(x, y);
                }}
                tctx.stroke();
            }} else {{
                // No denoised — redraw raw as primary
                tctx.strokeStyle = '#00d9ff';
                tctx.lineWidth = 1.5;
                tctx.beginPath();
                for (let i = 0; i < trace.length; i++) {{
                    const x = toX(i, trace.length);
                    const y = toY(trace[i]);
                    if (i === 0) tctx.moveTo(x, y);
                    else tctx.lineTo(x, y);
                }}
                tctx.stroke();
            }}
            
            // Draw spike markers
            if (spikeFrames && spikeFrames.length > 0) {{
                const refTrace = (traceDenoised && traceDenoised.length > 0) ? traceDenoised : trace;
                tctx.fillStyle = '#ff1744';
                for (let idx of spikeFrames) {{
                    if (idx < refTrace.length) {{
                        const x = toX(idx, refTrace.length);
                        const y = toY(refTrace[idx]);
                        tctx.beginPath();
                        tctx.arc(x, y, 3, 0, Math.PI * 2);
                        tctx.fill();
                    }}
                }}
            }}
            
            // Labels
            tctx.fillStyle = '#666';
            tctx.font = '9px sans-serif';
            tctx.fillText(max.toFixed(0) + '%', 2, 12);
            tctx.fillText(min.toFixed(0) + '%', 2, h - 2);
            
            // Legend
            tctx.font = '10px sans-serif';
            let legendX = w - 180;
            if (traceDenoised && traceDenoised.length > 0) {{
                tctx.fillStyle = 'rgba(0, 217, 255, 0.4)';
                tctx.fillText('— Raw', legendX, 14);
                tctx.fillStyle = '#00e676';
                tctx.fillText('— Denoised', legendX + 50, 14);
            }}
            if (spikeFrames && spikeFrames.length > 0) {{
                tctx.fillStyle = '#ff1744';
                tctx.fillText('● Spikes (' + spikeFrames.length + ')', legendX + 120, 14);
            }}
        }}
        
        function populateRoiList(filter) {{
            const list = document.getElementById('roiList');
            list.innerHTML = '';
            
            let displayRois = roiData;
            if (filter !== undefined && filter !== '') {{
                const term = filter.trim();
                // Support range syntax e.g. "100-200"
                const rangeMatch = term.match(/^(\d+)\s*[-–]\s*(\d+)$/);
                if (rangeMatch) {{
                    const lo = parseInt(rangeMatch[1]);
                    const hi = parseInt(rangeMatch[2]);
                    displayRois = roiData.filter(roi => roi.id >= lo && roi.id <= hi);
                }} else if (/^\d+$/.test(term)) {{
                    // Exact number or prefix match
                    const num = parseInt(term);
                    const exact = roiData.find(roi => roi.id === num);
                    if (exact) {{
                        displayRois = [exact];
                    }} else {{
                        // Prefix match on stringified id
                        displayRois = roiData.filter(roi => String(roi.id).startsWith(term));
                    }}
                }} else {{
                    // Comma-separated list e.g. "1,5,42"
                    const ids = term.split(/[,\s]+/).map(Number).filter(n => !isNaN(n));
                    if (ids.length > 0) {{
                        const idSet = new Set(ids);
                        displayRois = roiData.filter(roi => idSet.has(roi.id));
                    }}
                }}
            }}
            
            // Cap displayed items for DOM performance but show all by default
            const MAX_DISPLAY = 2000;
            const showing = displayRois.slice(0, MAX_DISPLAY);
            
            showing.forEach(roi => {{
                const item = document.createElement('div');
                item.className = 'roi-list-item';
                if (selectedRoi && selectedRoi.id === roi.id) item.className += ' selected';
                item.dataset.id = roi.id;
                item.innerHTML = `
                    <span class="roi-num">#${{roi.id}}</span>
                    <span class="roi-conf">${{roi.confidence.toFixed(2)}} | ${{roi.radius.toFixed(0)}}px</span>
                `;
                item.onclick = () => selectRoi(roi);
                list.appendChild(item);
            }});
            
            if (displayRois.length > MAX_DISPLAY) {{
                const note = document.createElement('div');
                note.style.cssText = 'padding:8px;text-align:center;color:#888;font-size:11px;';
                note.textContent = `Showing ${{MAX_DISPLAY}} of ${{displayRois.length}} — use search to narrow`;
                list.appendChild(note);
            }}
            if (displayRois.length === 0) {{
                const note = document.createElement('div');
                note.style.cssText = 'padding:8px;text-align:center;color:#888;font-size:11px;';
                note.textContent = 'No matching ROIs';
                list.appendChild(note);
            }}
        }}
        
        // Search bar event
        document.getElementById('roiSearch').addEventListener('input', (e) => {{
            populateRoiList(e.target.value);
        }});
        
        // Event listeners
        document.querySelectorAll('input[name="background"]').forEach(input => {{
            input.addEventListener('change', (e) => {{
                currentBackground = e.target.value;
                render();
            }});
        }});
        
        document.getElementById('show-contours').addEventListener('change', (e) => {{
            showContours = e.target.checked;
            render();
        }});
        
        document.getElementById('show-circles').addEventListener('change', (e) => {{
            showCircles = e.target.checked;
            render();
        }});
        
        document.getElementById('show-labels').addEventListener('change', (e) => {{
            showLabels = e.target.checked;
            render();
        }});
        
        document.getElementById('show-fallback').addEventListener('change', (e) => {{
            showFallback = e.target.checked;
            render();
        }});
        
        document.getElementById('show-edge-rois').addEventListener('change', (e) => {{
            showEdgeRois = e.target.checked;
            render();
        }});
        
        document.getElementById('min-confidence').addEventListener('input', (e) => {{
            minConfidence = parseFloat(e.target.value);
            document.getElementById('conf-value').textContent = minConfidence.toFixed(2);
            render();
        }});
        
        document.getElementById('min-radius').addEventListener('input', (e) => {{
            minRadius = parseInt(e.target.value);
            document.getElementById('radius-value').textContent = minRadius;
            render();
        }});
        
        // Zoom controls
        document.getElementById('zoom-in').addEventListener('click', () => {{
            zoom = Math.min(zoom * 1.3, 10);
            updateTransform();
        }});
        
        document.getElementById('zoom-out').addEventListener('click', () => {{
            const minZoom = getMinZoom();
            zoom = Math.max(zoom / 1.3, minZoom);
            updateTransform();
        }});
        
        document.getElementById('zoom-fit').addEventListener('click', fitToView);
        
        document.getElementById('zoom-reset').addEventListener('click', () => {{
            zoom = 1;
            panX = 0;
            panY = 0;
            updateTransform();
        }});
        
        // Mouse wheel zoom
        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const minZoom = getMinZoom();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            zoom = Math.max(minZoom, Math.min(10, zoom * delta));
            updateTransform();
        }});
        
        // Pan
        container.addEventListener('mousedown', (e) => {{
            if (e.button === 0) {{
                isDragging = true;
                dragStart = {{ x: e.clientX - panX, y: e.clientY - panY }};
            }}
        }});
        
        container.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                panX = e.clientX - dragStart.x;
                panY = e.clientY - dragStart.y;
                updateTransform();
            }} else {{
                // Tooltip / hover - convert screen coords to canvas coords
                const rect = canvas.getBoundingClientRect();
                const canvasX = (e.clientX - rect.left) * (imageWidth / rect.width);
                const canvasY = (e.clientY - rect.top) * (imageHeight / rect.height);
                
                const roi = findRoiAtPoint(canvasX, canvasY);
                const tooltip = document.getElementById('tooltip');
                
                if (roi) {{
                    tooltip.innerHTML = `ROI #${{roi.id}}<br>Conf: ${{roi.confidence.toFixed(2)}}<br>R: ${{roi.radius.toFixed(1)}}px`;
                    tooltip.style.left = (e.clientX + 15) + 'px';
                    tooltip.style.top = (e.clientY + 15) + 'px';
                    tooltip.classList.add('visible');
                    container.style.cursor = 'pointer';
                }} else {{
                    tooltip.classList.remove('visible');
                    container.style.cursor = isDragging ? 'grabbing' : 'grab';
                }}
            }}
        }});
        
        container.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        container.addEventListener('mouseleave', () => {{
            isDragging = false;
            document.getElementById('tooltip').classList.remove('visible');
        }});
        
        // Click to select ROI
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const canvasX = (e.clientX - rect.left) * (imageWidth / rect.width);
            const canvasY = (e.clientY - rect.top) * (imageHeight / rect.height);
            
            const roi = findRoiAtPoint(canvasX, canvasY);
            selectRoi(roi);
        }});
        
        // Window resize
        window.addEventListener('resize', () => {{
            updateTransform();
        }});
    </script>
</body>
</html>
'''
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"  Interactive gallery saved to {output_path}")
    return output_path
