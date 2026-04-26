"""
Frame Gallery — Interactive Single-Frame Viewer
================================================

Replaces the MP4-based movie gallery with a self-contained HTML file that
encodes every (subsampled) frame as a base64 PNG and lets the user click
through frames manually.  No video codec, no external files — the HTML
opens standalone in any browser and embeds cleanly in PowerPoint via
Insert > Object > Web Page (or as a linked .html file).

Design rationale
----------------
* Low acquisition frame rates (<=10 Hz) make video playback feel wrong and
  make it hard to inspect individual frames.  Manual stepping is a better fit.
* PowerPoint's video renderer consistently fails on H.264 MP4 files produced
  outside Windows.  Embedding frames as base64 PNG in HTML sidesteps this.
* The full deconvolved spike trace is shown (no subsampling) so every event
  is visible, with a frame-position cursor that updates as you step.

Outputs
-------
    <output_dir>/gallery_frames.html    fully self-contained, ~10-80 MB

Usage
-----
    from src.movie_gallery import generate_movie_gallery

    generate_movie_gallery(
        movie=movie_raw,
        seeds=seeds,
        output_dir='/path/to/output',
        frame_rate=2.0,
        subsample=1,
        traces_denoised=C_denoised,
        spike_trains=S,
    )
"""

import os
import json
import base64
import logging
from io import BytesIO
from typing import Optional, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FRAME ENCODING
# =============================================================================

def _normalize_to_uint8(frame, vmin, vmax):
    f = frame.astype(np.float64)
    f = np.clip((f - vmin) / (vmax - vmin + 1e-10), 0, 1)
    return (f * 255).astype(np.uint8)


def encode_frames_to_b64(movie, subsample=1):
    """Encode movie frames as base64 PNG strings for HTML embedding.

    Returns (frames_b64, frame_indices, vmin, vmax).
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")

    T, H, W = movie.shape
    frame_indices = list(range(0, T, subsample))
    n_out = len(frame_indices)

    sample_idx = np.linspace(0, T - 1, min(200, T), dtype=int)
    sample = movie[sample_idx]
    vmin = float(np.percentile(sample, 1))
    vmax = float(np.percentile(sample, 99.5))
    logger.info(f"  Frame encoding: {n_out} frames ({subsample}x subsample), "
                f"vmin={vmin:.1f}, vmax={vmax:.1f}")

    frames_b64 = []
    for k, fi in enumerate(frame_indices):
        u8 = _normalize_to_uint8(movie[fi], vmin, vmax)
        img = Image.fromarray(u8, mode='L').convert('RGB')
        buf = BytesIO()
        img.save(buf, format='PNG', optimize=False, compress_level=1)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))
        if (k + 1) % 100 == 0:
            logger.info(f"    Encoded {k + 1}/{n_out} frames...")

    total_mb = sum(len(b) * 3 // 4 for b in frames_b64) / 1e6
    logger.info(f"  Encoded {n_out} frames — ~{total_mb:.1f} MB PNG data")
    return frames_b64, frame_indices, vmin, vmax


# =============================================================================
# ROI DATA
# =============================================================================

def extract_roi_data(seeds, movie, frame_rate, max_rois=500,
                     traces_denoised=None, spike_trains=None,
                     deconv_noise=None):
    """Extract ROI metadata and FULL-LENGTH traces (no subsampling).

    Each ROI is assigned a transient SNR score = max(denoised trace) / OASIS_sn.
    After all ROIs are processed the list is sorted by this score descending
    and a 1-based 'rank' field is added, so the gallery list opens with the
    cleanest visually verifiable transients first.

    If deconv_noise is not provided (old results / no deconvolution), the
    peak-to-percentile range of the raw trace is used as a fallback SNR proxy.
    """
    T, d1, d2 = movie.shape
    roi_data = []

    for i in range(min(seeds.n_seeds, max_rois)):
        y, x = seeds.centers[i]
        r = seeds.radii[i]

        y0 = max(0, int(y) - int(r));  y1 = min(d1, int(y) + int(r) + 1)
        x0 = max(0, int(x) - int(r));  x1 = min(d2, int(x) + int(r) + 1)
        raw_trace = movie[:, y0:y1, x0:x1].mean(axis=(1, 2))

        baseline = np.percentile(raw_trace, 10)
        if baseline > 1.0 and baseline > 0:
            dff = ((raw_trace - baseline) / baseline * 100).tolist()
        else:
            dff = (raw_trace * 100).tolist()

        denoised = None
        denoised_arr = None
        if traces_denoised is not None and i < traces_denoised.shape[0]:
            den = traces_denoised[i].astype(np.float64)
            denoised_arr = den
            if abs(np.median(den)) < 5.0:
                denoised = (den * 100).tolist()
            else:
                denoised = den.tolist()

        spikes = []
        if spike_trains is not None and i < spike_trains.shape[0]:
            spikes = np.where(spike_trains[i] > 0)[0].tolist()

        contour_pts = None
        circularity = solidity = None
        if seeds.contour_success[i] and seeds.contours[i] is not None:
            cnt = seeds.contours[i].contour.squeeze()
            if cnt.ndim == 2:
                if cnt.shape[0] > 120:
                    idx = np.round(np.linspace(0, cnt.shape[0]-1, 120)).astype(int)
                    cnt = cnt[idx]
                contour_pts = cnt.tolist()
            circularity = seeds.contours[i].circularity
            solidity = seeds.contours[i].solidity

        # ── Transient SNR ────────────────────────────────────────────────
        # Primary: max(denoised) / OASIS noise estimate (sn).
        # Fallback: robust peak-to-noise of the raw ΔF/F₀ trace.
        snr = 0.0
        if denoised_arr is not None and deconv_noise is not None and i < len(deconv_noise):
            sn = float(deconv_noise[i])
            if sn > 0:
                snr = float(np.max(denoised_arr)) / sn
        elif denoised_arr is not None:
            # Fallback: p95 - p50 / MAD as proxy when sn unavailable
            diff = np.diff(denoised_arr)
            mad = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) / np.sqrt(2)
            if mad > 0:
                snr = float(np.percentile(denoised_arr, 95) - np.percentile(denoised_arr, 50)) / mad
        else:
            # No denoised trace at all — use raw dff range
            raw_arr = np.array(dff)
            diff = np.diff(raw_arr)
            mad = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) / np.sqrt(2)
            if mad > 0:
                snr = float(np.percentile(raw_arr, 95) - np.percentile(raw_arr, 50)) / mad

        roi_data.append({
            'id': i,
            'center': [float(y), float(x)],
            'radius': float(r),
            'confidence': float(seeds.confidence[i]),
            'has_contour': bool(seeds.contour_success[i]),
            'contour_points': contour_pts,
            'circularity': float(circularity) if circularity is not None else None,
            'solidity': float(solidity) if solidity is not None else None,
            'boundary_touching': bool(
                seeds.boundary_touching[i]
                if hasattr(seeds, 'boundary_touching') and len(seeds.boundary_touching) > i
                else False
            ),
            'trace': dff,
            'trace_denoised': denoised,
            'spike_frames': spikes,
            'n_frames': T,
            'snr': round(snr, 2),
            'rank': 0,   # filled in below after sorting
        })

    # ── Rank by transient SNR ─────────────────────────────────────────────
    roi_data.sort(key=lambda r: r['snr'], reverse=True)
    for rank_idx, roi in enumerate(roi_data):
        roi['rank'] = rank_idx + 1

    return roi_data


# =============================================================================
# HTML
# =============================================================================

def _generate_html(roi_data, frames_b64, frame_indices,
                   image_width, image_height, n_frames_original,
                   original_fps, subsample, title):

    roi_json = json.dumps(roi_data)
    frames_js = '[' + ','.join(f'"{b}"' for b in frames_b64) + ']'
    frame_indices_js = json.dumps(frame_indices)
    n_display = len(frames_b64)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --ink:#e8eaf2;--ink2:#8b90aa;--ink3:#4a5068;
  --bg0:#080a10;--bg1:#0f1119;--bg2:#171b27;--bg3:#1e2333;
  --line:#252b3d;--blue:#4f8ef7;--green:#34d472;--red:#f25757;
  --amber:#f5a623;--cyan:#38bdf8;--sel:#f7d44f;
  --r:7px;--mono:'Courier New',Courier,monospace
}}
html,body{{height:100%;overflow:hidden;background:var(--bg0);color:var(--ink);
  font-family:Georgia,'Times New Roman',serif;font-size:13px}}
.shell{{display:grid;grid-template-columns:210px 1fr 295px;height:100vh;gap:1px;background:var(--line)}}
.pane{{background:var(--bg1);overflow-y:auto;padding:13px;display:flex;flex-direction:column;gap:11px}}
.pane-hd{{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--blue);
  letter-spacing:.1em;text-transform:uppercase;padding-bottom:7px;border-bottom:1px solid var(--line)}}
.sect{{border-bottom:1px solid var(--line);padding-bottom:11px}}
.sect-hd{{font-family:var(--mono);font-size:10px;color:var(--ink3);
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:7px}}
.stats{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}
.stat{{background:var(--bg2);border-radius:var(--r);padding:7px 9px;text-align:center}}
.stat .n{{font-family:var(--mono);font-size:16px;font-weight:700;color:var(--ink)}}
.stat .l{{font-size:9px;color:var(--ink3);text-transform:uppercase;letter-spacing:.05em}}
.chk-row{{display:flex;align-items:center;gap:7px;margin:4px 0;cursor:pointer;font-size:12px}}
.chk-row input[type=checkbox]{{accent-color:var(--blue);width:13px;height:13px}}
.sl-row{{display:flex;align-items:center;gap:6px;margin:4px 0;font-size:11px}}
.sl-row span:first-child{{min-width:65px;color:var(--ink2)}}
.sl-row input[type=range]{{flex:1;accent-color:var(--blue);height:3px}}
.sv{{min-width:30px;text-align:right;font-family:var(--mono);font-size:10px;color:var(--ink2)}}
.btn{{background:var(--bg3);border:1px solid var(--line);color:var(--ink);
  border-radius:5px;padding:5px 11px;cursor:pointer;font-size:11px;font-family:var(--mono);
  transition:border-color .12s}}
.btn:hover{{border-color:var(--blue)}}
.btn-row{{display:flex;gap:5px;align-items:center;flex-wrap:wrap}}
/* center */
.pane-center{{background:var(--bg0);display:flex;flex-direction:column;overflow:hidden;padding:0}}
.cv-wrap{{flex:1;position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center}}
.cv-inner{{position:relative;transform-origin:0 0}}
#frameCanvas{{display:block}}
#overlayCanvas,#hitCanvas{{position:absolute;top:0;left:0}}
#overlayCanvas{{pointer-events:none}}
#hitCanvas{{opacity:0;pointer-events:auto;cursor:crosshair}}
.transport{{background:var(--bg1);border-top:1px solid var(--line);padding:9px 13px;
  display:flex;flex-direction:column;gap:6px}}
.scrub-row{{display:flex;align-items:center;gap:7px}}
.scrub-row input[type=range]{{flex:1;accent-color:var(--blue);height:5px;cursor:pointer}}
.tc{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.frame-disp{{font-family:var(--mono);font-size:11px;color:var(--ink);min-width:115px}}
.time-disp{{font-family:var(--mono);font-size:10px;color:var(--ink3)}}
/* right */
.no-sel{{color:var(--ink3);font-size:12px;text-align:center;padding:25px 0;line-height:1.9}}
.roi-hd{{display:flex;align-items:center;gap:8px}}
.roi-num{{font-family:var(--mono);font-size:19px;font-weight:700;color:var(--ink)}}
.badge{{font-size:9px;font-weight:700;padding:2px 7px;border-radius:8px;text-transform:uppercase}}
.badge.c{{background:rgba(52,212,114,.15);color:var(--green)}}
.badge.f{{background:rgba(242,87,87,.15);color:var(--red)}}
.badge.e{{background:rgba(245,166,35,.15);color:var(--amber)}}
.meta{{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:9px}}
.mc{{background:var(--bg2);border-radius:6px;padding:6px 8px}}
.mc .l{{font-size:9px;color:var(--ink3);text-transform:uppercase;letter-spacing:.04em}}
.mc .v{{font-family:var(--mono);font-size:12px;color:var(--ink)}}
.trace-wrap{{background:var(--bg2);border-radius:var(--r);padding:9px;margin-top:3px}}
.trace-hd{{font-size:10px;color:var(--ink3);text-transform:uppercase;
  letter-spacing:.05em;margin-bottom:6px;font-family:var(--mono)}}
#traceCanvas{{width:100%;height:155px;border-radius:4px;display:block}}
.legend{{display:flex;gap:9px;margin-top:5px;flex-wrap:wrap}}
.leg{{font-size:10px;color:var(--ink3);display:flex;align-items:center;gap:4px}}
.leg-dot{{width:7px;height:7px;border-radius:50%}}
.leg-line{{width:13px;height:2px;border-radius:1px}}
.roi-search{{width:100%;padding:5px 8px;background:var(--bg2);border:1px solid var(--line);
  border-radius:5px;color:var(--ink);font-size:11px;outline:none;font-family:var(--mono)}}
.roi-search:focus{{border-color:var(--blue)}}
.roi-list{{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:2px;min-height:0}}
.ri{{display:flex;justify-content:space-between;align-items:center;
  padding:5px 8px;border-radius:5px;cursor:pointer;font-size:11px;transition:background .1s}}
.ri:hover{{background:var(--bg3)}}
.ri.sel{{background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.3)}}
.ri .rn{{font-family:var(--mono);color:var(--ink);font-weight:600}}
.ri .rm{{color:var(--ink3);font-size:10px}}
</style>
</head>
<body>
<div class="shell">

<!-- LEFT -->
<div class="pane">
  <div class="pane-hd">Frame Gallery</div>
  <div class="sect">
    <div class="sect-hd">Recording</div>
    <div class="stats">
      <div class="stat"><div class="n" id="sVis">—</div><div class="l">Visible</div></div>
      <div class="stat"><div class="n">{n_frames_original}</div><div class="l">Frames</div></div>
      <div class="stat"><div class="n">{original_fps}</div><div class="l">Hz</div></div>
      <div class="stat"><div class="n">{n_display}</div><div class="l">Displayed</div></div>
    </div>
  </div>
  <div class="sect">
    <div class="sect-hd">Overlays</div>
    <label class="chk-row"><input type="checkbox" id="chkContours" checked><span>Contours</span></label>
    <label class="chk-row"><input type="checkbox" id="chkCircles" checked><span>Seed circles</span></label>
    <label class="chk-row"><input type="checkbox" id="chkLabels"><span>ROI labels</span></label>
    <label class="chk-row"><input type="checkbox" id="chkFallback" checked><span>Fallback ROIs</span></label>
    <label class="chk-row"><input type="checkbox" id="chkEdge" checked><span>Edge ROIs</span></label>
  </div>
  <div class="sect">
    <div class="sect-hd">Filters</div>
    <div class="sl-row"><span>Min conf</span>
      <input type="range" id="fConf" min="0" max="100" value="0">
      <span class="sv" id="fConfV">0.00</span></div>
    <div class="sl-row"><span>Min radius</span>
      <input type="range" id="fRad" min="0" max="50" value="0">
      <span class="sv" id="fRadV">0</span></div>
  </div>
  <div class="sect">
    <div class="sect-hd">View</div>
    <div class="btn-row">
      <span style="color:var(--ink2);font-size:11px">Zoom&thinsp;<span id="zPct">100</span>%</span>
      <button class="btn" onclick="zIn()">+</button>
      <button class="btn" onclick="zOut()">−</button>
      <button class="btn" onclick="zFit()">Fit</button>
    </div>
  </div>
  <div>
    <div class="sect-hd" style="margin-bottom:5px">Keys</div>
    <div style="font-size:10px;color:var(--ink3);line-height:2.1;font-family:var(--mono)">
      ← →&ensp;step frame<br>Home/End&ensp;first/last<br>
      C&ensp;contours&ensp;L&ensp;labels<br>Esc&ensp;deselect
    </div>
  </div>
</div>

<!-- CENTER -->
<div class="pane-center">
  <div class="cv-wrap" id="cvWrap">
    <div class="cv-inner" id="cvInner">
      <canvas id="frameCanvas"></canvas>
      <canvas id="overlayCanvas"></canvas>
      <canvas id="hitCanvas"></canvas>
    </div>
  </div>
  <div class="transport">
    <div class="scrub-row">
      <input type="range" id="scrubber" min="0" max="{n_display - 1}" value="0" step="1">
    </div>
    <div class="tc">
      <button class="btn" onclick="goFirst()">|◀</button>
      <button class="btn" onclick="stepBack()">◀ −1</button>
      <button class="btn" onclick="stepFwd()">+1 ▶</button>
      <button class="btn" onclick="goLast()">▶|</button>
      <span class="frame-disp">Frame <span id="fDisp">0</span>&thinsp;/&thinsp;{n_frames_original - 1}</span>
      <span class="time-disp"><span id="tDisp">0.00</span> s</span>
    </div>
  </div>
</div>

<!-- RIGHT -->
<div class="pane">
  <div class="pane-hd">ROI Details</div>
  <div class="no-sel" id="noSel">Click an ROI on the<br>frame to inspect it</div>
  <div id="roiDetail" style="display:none;flex-direction:column;gap:9px">
    <div class="roi-hd">
      <span class="roi-num">ROI&thinsp;#<span id="dId">—</span></span>
      <span class="badge" id="dBadge">—</span>
    </div>
    <div class="meta">
      <div class="mc"><div class="l">Rank</div><div class="v" id="dRank">—</div></div>
      <div class="mc"><div class="l">SNR</div><div class="v" id="dSnr">—</div></div>
      <div class="mc"><div class="l">Confidence</div><div class="v" id="dConf">—</div></div>
      <div class="mc"><div class="l">Radius</div><div class="v" id="dRad">—</div></div>
      <div class="mc"><div class="l">Circularity</div><div class="v" id="dCirc">—</div></div>
      <div class="mc"><div class="l">Center Y,X</div><div class="v" id="dCtr">—</div></div>
    </div>
    <div class="trace-wrap">
      <div class="trace-hd">Calcium trace — full recording</div>
      <canvas id="traceCanvas"></canvas>
      <div class="legend">
        <div class="leg"><div class="leg-line" style="background:rgba(56,189,248,.45)"></div>Raw ΔF/F₀</div>
        <div class="leg"><div class="leg-line" style="background:#34d472"></div>Deconvolved</div>
        <div class="leg"><div class="leg-dot" style="background:#f25757"></div>Spike</div>
        <div class="leg"><div class="leg-line" style="background:#f5a623;width:2px;height:10px"></div>Frame</div>
      </div>
    </div>
  </div>
  <div style="margin-top:3px">
    <div class="sect-hd" style="margin-bottom:5px">ROI List</div>
    <input class="roi-search" id="roiSearch" placeholder="42 · 10-50 · 1,5,12">
    <div class="roi-list" id="roiList"></div>
  </div>
</div>

</div>
<script>
const W={image_width},H={image_height};
const nFramesOrig={n_frames_original};
const origFps={original_fps};
const subsample={subsample};
const frameIdx={frame_indices_js};
const nDisp=frameIdx.length;
const roiData={roi_json};
const FRAMES_B64={frames_js};

const fcv=document.getElementById('frameCanvas');
const ocv=document.getElementById('overlayCanvas');
const hcv=document.getElementById('hitCanvas');
const fctx=fcv.getContext('2d');
const octx=ocv.getContext('2d');
[fcv,ocv,hcv].forEach(c=>{{c.width=W;c.height=H;}});

const inner=document.getElementById('cvInner');
const wrap=document.getElementById('cvWrap');

let curDisp=0,zoom=1,panX=0,panY=0;
let dragging=false,dragStart={{x:0,y:0}};
let selRoi=null;
let showContours=true,showCircles=true,showLabels=false;
let showFallback=true,showEdge=true,minConf=0,minRadius=0;

// Image cache
const IMG_CACHE=new Map();
function getImg(d){{
  if(IMG_CACHE.has(d)) return IMG_CACHE.get(d);
  const img=new Image();
  img.src='data:image/png;base64,'+FRAMES_B64[d];
  if(IMG_CACHE.size>=120) IMG_CACHE.delete(IMG_CACHE.keys().next().value);
  IMG_CACHE.set(d,img);
  return img;
}}
function preWarm(d){{
  for(let i=-2;i<=8;i++){{const j=d+i;if(j>=0&&j<nDisp)getImg(j);}}
}}

function renderFrame(){{
  const img=getImg(curDisp);
  const draw=()=>{{fctx.clearRect(0,0,W,H);fctx.drawImage(img,0,0,W,H);}};
  if(img.complete)draw(); else img.onload=draw;
  preWarm(curDisp);
  const origF=frameIdx[curDisp];
  document.getElementById('fDisp').textContent=origF;
  document.getElementById('tDisp').textContent=(origF/origFps).toFixed(2);
  document.getElementById('scrubber').value=curDisp;
  drawOverlay();
  if(selRoi!==null)drawTrace();
}}

const cssv=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function getVisible(){{
  return roiData.filter(r=>
    r.confidence>=minConf && r.radius>=minRadius &&
    (showFallback||r.has_contour) && (showEdge||!r.boundary_touching));
}}
function roiColor(roi){{
  if(roi.id===selRoi)return cssv('--sel');
  if(roi.boundary_touching)return cssv('--amber');
  return roi.has_contour?cssv('--green'):cssv('--red');
}}
function drawOverlay(){{
  octx.clearRect(0,0,W,H);
  const vis=getVisible();
  document.getElementById('sVis').textContent=vis.length;
  vis.forEach(roi=>{{
    const[ry,rx]=roi.center,r=roi.radius,isSel=roi.id===selRoi;
    const col=roiColor(roi);
    octx.strokeStyle=col;octx.lineWidth=isSel?2.5:1.2;
    if(showContours&&roi.has_contour&&roi.contour_points){{
      const pts=roi.contour_points;
      octx.beginPath();octx.moveTo(pts[0][0],pts[0][1]);
      for(let i=1;i<pts.length;i++)octx.lineTo(pts[i][0],pts[i][1]);
      octx.closePath();octx.stroke();
      if(isSel){{octx.fillStyle=col+'20';octx.fill();}}
    }}
    if(showCircles){{
      octx.beginPath();octx.arc(rx,ry,r,0,Math.PI*2);
      if(!roi.has_contour)octx.setLineDash([4,4]);
      octx.stroke();octx.setLineDash([]);
    }}
    if(showLabels){{
      octx.fillStyle=col;octx.font='10px Courier New,monospace';
      octx.fillText(roi.id,rx+r+3,ry-r);
    }}
  }});
}}

function drawTrace(){{
  const roi=roiData.find(r=>r.id===selRoi);
  if(!roi)return;
  const tc=document.getElementById('traceCanvas');
  const tctx=tc.getContext('2d');
  const dpr=window.devicePixelRatio||1;
  tc.width=tc.offsetWidth*dpr;tc.height=tc.offsetHeight*dpr;
  tctx.scale(dpr,dpr);
  const w=tc.offsetWidth,h=tc.offsetHeight;
  tctx.fillStyle=cssv('--bg2')||'#171b27';
  tctx.fillRect(0,0,w,h);

  const raw=roi.trace,den=roi.trace_denoised;
  if(!raw||!raw.length)return;
  const T=raw.length;

  let allV=raw.slice();
  if(den&&den.length===T)allV=allV.concat(den);
  let mn=Math.min(...allV),mx=Math.max(...allV);
  const pad=(mx-mn)*0.05||1;mn-=pad;mx+=pad;
  const rng=mx-mn;
  const toY=v=>h-((v-mn)/rng)*(h-22)-10;
  const toX=i=>(i/(T-1))*w;

  // Grid
  tctx.strokeStyle='rgba(255,255,255,.04)';tctx.lineWidth=0.5;
  for(let g=0;g<=4;g++){{tctx.beginPath();tctx.moveTo(0,h*g/4);tctx.lineTo(w,h*g/4);tctx.stroke();}}
  if(mn<0&&mx>0){{
    tctx.strokeStyle='rgba(255,255,255,.12)';tctx.lineWidth=0.8;
    tctx.beginPath();tctx.moveTo(0,toY(0));tctx.lineTo(w,toY(0));tctx.stroke();
  }}

  // Raw
  tctx.strokeStyle='rgba(56,189,248,.40)';tctx.lineWidth=1.0;
  tctx.beginPath();
  for(let i=0;i<T;i++){{const px=toX(i),py=toY(raw[i]);i===0?tctx.moveTo(px,py):tctx.lineTo(px,py);}}
  tctx.stroke();

  // Denoised
  if(den&&den.length===T){{
    tctx.strokeStyle='#34d472';tctx.lineWidth=1.5;
    tctx.beginPath();
    for(let i=0;i<T;i++){{const px=toX(i),py=toY(den[i]);i===0?tctx.moveTo(px,py):tctx.lineTo(px,py);}}
    tctx.stroke();
  }}

  // Spikes
  if(roi.spike_frames&&roi.spike_frames.length){{
    const ref=(den&&den.length===T)?den:raw;
    tctx.fillStyle='#f25757';
    for(const sf of roi.spike_frames){{
      if(sf<T){{tctx.beginPath();tctx.arc(toX(sf),toY(ref[sf]),3,0,Math.PI*2);tctx.fill();}}
    }}
  }}

  // Frame cursor — map to original frame number
  const origF=frameIdx[curDisp];
  const lineX=toX(Math.min(origF,T-1));
  tctx.strokeStyle='#f5a623';tctx.lineWidth=1.8;tctx.setLineDash([]);
  tctx.beginPath();tctx.moveTo(lineX,0);tctx.lineTo(lineX,h);tctx.stroke();
  tctx.fillStyle='#f5a623';
  tctx.beginPath();tctx.moveTo(lineX-4,0);tctx.lineTo(lineX+4,0);tctx.lineTo(lineX,6);tctx.closePath();tctx.fill();

  // Labels
  tctx.fillStyle='rgba(255,255,255,.22)';tctx.font='9px Courier New,monospace';
  tctx.fillText(mx.toFixed(0)+'%',3,11);tctx.fillText(mn.toFixed(0)+'%',3,h-3);
  tctx.fillStyle='#f5a623';
  const ft='F'+origF,ftw=tctx.measureText(ft).width;
  tctx.fillText(ft,Math.min(Math.max(lineX-ftw/2,2),w-ftw-2),h-3);
}}

// Click to select ROI
hcv.style.pointerEvents='auto';
hcv.addEventListener('click',e=>{{
  const rect=hcv.getBoundingClientRect();
  const cx=(e.clientX-rect.left)*(W/rect.width);
  const cy=(e.clientY-rect.top)*(H/rect.height);
  const vis=getVisible();
  let best=null,bd=Infinity;
  for(const roi of vis){{
    const[ry,rx]=roi.center;
    const d=Math.hypot(cx-rx,cy-ry);
    if(d<=roi.radius*1.5&&d<bd){{best=roi;bd=d;}}
  }}
  selectRoi(best?best.id:null);
}});

function selectRoi(id){{
  selRoi=id;
  const roi=id!==null?roiData.find(r=>r.id===id):null;
  const det=document.getElementById('roiDetail');
  const ns=document.getElementById('noSel');
  if(roi){{
    ns.style.display='none';det.style.display='flex';
    document.getElementById('dId').textContent=roi.id;
    document.getElementById('dRank').textContent='#'+roi.rank;
    document.getElementById('dSnr').textContent=roi.snr.toFixed(1);
    document.getElementById('dConf').textContent=roi.confidence.toFixed(3);
    document.getElementById('dRad').textContent=roi.radius.toFixed(1)+'px';
    document.getElementById('dCirc').textContent=roi.circularity!=null?roi.circularity.toFixed(3):'—';
    document.getElementById('dCtr').textContent=roi.center[0].toFixed(0)+', '+roi.center[1].toFixed(0);
    const badge=document.getElementById('dBadge');
    if(roi.boundary_touching){{badge.textContent='Edge';badge.className='badge e';}}
    else if(roi.has_contour){{badge.textContent='Contour';badge.className='badge c';}}
    else{{badge.textContent='Fallback';badge.className='badge f';}}
    drawTrace();
  }}else{{ns.style.display='block';det.style.display='none';}}
  drawOverlay();
  document.querySelectorAll('.ri').forEach(el=>el.classList.toggle('sel',parseInt(el.dataset.id)===id));
}}

function populateList(filter){{
  const list=document.getElementById('roiList');
  list.innerHTML='';
  let rois=getVisible();
  if(filter){{
    const t=filter.trim();
    const rm=t.match(/^(\\d+)\\s*[-–]\\s*(\\d+)$/);
    if(rm){{const lo=parseInt(rm[1]),hi=parseInt(rm[2]);rois=rois.filter(r=>r.id>=lo&&r.id<=hi);}}
    else if(/^\\d+$/.test(t)){{rois=rois.filter(r=>String(r.id).startsWith(t));}}
    else if(t.includes(',')){{const ids=new Set(t.split(/[,\\s]+/).map(Number));rois=rois.filter(r=>ids.has(r.id));}}
  }}
  rois.slice(0,500).forEach(roi=>{{
    const el=document.createElement('div');
    el.className='ri'+(selRoi===roi.id?' sel':'');
    el.dataset.id=roi.id;
    const sp=roi.spike_frames?roi.spike_frames.length:0;
    el.innerHTML=`<span class="rn">#${{roi.rank}} · ${{roi.id}}</span><span class="rm">SNR ${{roi.snr.toFixed(1)}} · ${{roi.radius.toFixed(0)}}px · ${{sp}}spk</span>`;
    el.onclick=()=>selectRoi(roi.id);
    list.appendChild(el);
  }});
  if(rois.length>500){{
    const d=document.createElement('div');
    d.style.cssText='padding:6px;text-align:center;color:var(--ink3);font-size:10px';
    d.textContent=`Showing 500 of ${{rois.length}} — use search`;
    list.appendChild(d);
  }}
}}

// Navigation
function goTo(d){{curDisp=Math.max(0,Math.min(nDisp-1,d));renderFrame();}}
function stepFwd(){{goTo(curDisp+1);}}
function stepBack(){{goTo(curDisp-1);}}
function goFirst(){{goTo(0);}}
function goLast(){{goTo(nDisp-1);}}
document.getElementById('scrubber').addEventListener('input',e=>goTo(parseInt(e.target.value)));

// Zoom/pan
function applyTransform(){{
  inner.style.transform=`translate(${{panX}}px,${{panY}}px) scale(${{zoom}})`;
  document.getElementById('zPct').textContent=Math.round(zoom*100);
}}
function zFit(){{
  const cr=wrap.getBoundingClientRect();
  zoom=Math.min(cr.width/W,cr.height/H)*0.97;
  panX=(cr.width-W*zoom)/2;panY=(cr.height-H*zoom)/2;applyTransform();
}}
function zIn(){{zoom=Math.min(zoom*1.25,20);applyTransform();}}
function zOut(){{zoom=Math.max(zoom/1.25,0.1);applyTransform();}}
wrap.addEventListener('wheel',e=>{{
  e.preventDefault();zoom=Math.max(0.1,Math.min(20,zoom*(e.deltaY<0?1.1:0.9)));applyTransform();
}});
wrap.addEventListener('mousedown',e=>{{
  if(e.button===1||(e.button===0&&e.shiftKey)){{
    dragging=true;dragStart={{x:e.clientX-panX,y:e.clientY-panY}};e.preventDefault();
  }}
}});
window.addEventListener('mousemove',e=>{{if(dragging){{panX=e.clientX-dragStart.x;panY=e.clientY-dragStart.y;applyTransform();}}}}); 
window.addEventListener('mouseup',()=>dragging=false);

// Controls
document.getElementById('chkContours').onchange=e=>{{showContours=e.target.checked;drawOverlay();}};
document.getElementById('chkCircles').onchange=e=>{{showCircles=e.target.checked;drawOverlay();}};
document.getElementById('chkLabels').onchange=e=>{{showLabels=e.target.checked;drawOverlay();}};
document.getElementById('chkFallback').onchange=e=>{{showFallback=e.target.checked;drawOverlay();populateList();}};
document.getElementById('chkEdge').onchange=e=>{{showEdge=e.target.checked;drawOverlay();populateList();}};
document.getElementById('fConf').oninput=e=>{{minConf=e.target.value/100;document.getElementById('fConfV').textContent=minConf.toFixed(2);drawOverlay();populateList();}};
document.getElementById('fRad').oninput=e=>{{minRadius=parseInt(e.target.value);document.getElementById('fRadV').textContent=minRadius;drawOverlay();populateList();}};
document.getElementById('roiSearch').oninput=e=>populateList(e.target.value);

document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT')return;
  switch(e.key){{
    case'ArrowRight':stepFwd();break;case'ArrowLeft':stepBack();break;
    case'Home':goFirst();break;case'End':goLast();break;
    case'+':case'=':zIn();break;case'-':zOut();break;
    case'f':zFit();break;case'c':document.getElementById('chkContours').click();break;
    case'l':document.getElementById('chkLabels').click();break;
    case'Escape':selectRoi(null);break;
  }}
}});

populateList();zFit();renderFrame();
</script>
</body>
</html>'''


# =============================================================================
# ENTRY POINT
# =============================================================================

def generate_movie_gallery(
    movie,
    seeds,
    output_dir,
    frame_rate=2.0,
    subsample=1,
    max_rois=500,
    title="Frame Gallery",
    traces_denoised=None,
    spike_trains=None,
    movie_processed=None,
    deconv_noise=None,
    # Legacy kwargs silently accepted
    crf=23,
):
    """Generate the self-contained frame gallery HTML.

    Parameters
    ----------
    movie : ndarray (T, H, W)
        Raw movie.  Frames are base64-PNG encoded and embedded in HTML.
    seeds : ContourSeedResult
    output_dir : str
    frame_rate : float
    subsample : int
        Keep every Nth frame (default 1 = all).  subsample=2 halves size.
    max_rois : int
    title : str
    traces_denoised : ndarray (N, T), optional
        Full-length OASIS denoised traces — embedded without subsampling.
    spike_trains : ndarray (N, T), optional

    Returns
    -------
    dict  html_path, n_rois, n_frames_display, file_size_mb (+ legacy keys)
    """
    os.makedirs(output_dir, exist_ok=True)
    T, H_img, W_img = movie.shape

    logger.info("=" * 60)
    logger.info("FRAME GALLERY GENERATION")
    logger.info(f"  Movie: {T} frames, {H_img}x{W_img}, subsample={subsample}")
    logger.info(f"  ROIs: {min(seeds.n_seeds, max_rois)}")
    logger.info("=" * 60)

    logger.info("Encoding frames as PNG...")
    frames_b64, frame_indices, vmin, vmax = encode_frames_to_b64(movie, subsample=subsample)

    movie_for_traces = movie_processed if movie_processed is not None else movie
    logger.info("Extracting ROI data...")
    roi_data = extract_roi_data(
        seeds, movie_for_traces,
        frame_rate=frame_rate, max_rois=max_rois,
        traces_denoised=traces_denoised, spike_trains=spike_trains,
        deconv_noise=deconv_noise,
    )
    logger.info(f"  {len(roi_data)} ROIs extracted")

    logger.info("Building HTML...")
    html = _generate_html(
        roi_data=roi_data, frames_b64=frames_b64, frame_indices=frame_indices,
        image_width=W_img, image_height=H_img,
        n_frames_original=T, original_fps=frame_rate,
        subsample=subsample, title=title,
    )

    html_path = os.path.join(output_dir, 'gallery_frames.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(html_path) / 1e6
    logger.info(f"  Saved: {html_path} ({size_mb:.1f} MB)")
    logger.info("FRAME GALLERY COMPLETE")
    logger.info("  PowerPoint: Insert > Object > Web Page, select gallery_frames.html")

    return {
        'html_path': html_path,
        'n_rois': len(roi_data),
        'n_frames_display': len(frames_b64),
        'file_size_mb': size_mb,
        'mp4_path': None,
        'mp4_size_mb': 0.0,
        'video_fps': frame_rate / max(subsample, 1),
        'subsample': subsample,
    }
