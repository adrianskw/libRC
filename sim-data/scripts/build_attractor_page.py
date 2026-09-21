"""Generate a standalone interactive HTML page visualizing the AE latent attractor.

Embeds the latent trajectory data directly (no runtime fetch). Computes the
breathing-mode period/frequency from the latent trace using real simulation
timestep info parsed from parm.in.
"""
import json
import os
import re

import numpy as np
from scipy.signal import find_peaks

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---- load latent trajectory ----
lat = np.load(f"{BASE}/latent.npz")
z = lat["z"]
frame_indices = lat["frame_indices"].astype(int)
n_train = int(lat["n_train"])
n = len(frame_indices)

# ---- parse real sim params from parm.in ----
with open(f"{BASE}/parm.in") as f:
    parm_text = f.read()


def parm(name):
    m = re.search(rf"^{name}\s+([0-9eE.+-]+)", parm_text, re.MULTILINE)
    return float(m.group(1))


dt0 = parm("DT_0")
v_discharge = parm("V_DISCHARGE")
m_dot = parm("M_DOT")
n_its = int(parm("N_ITS"))

# ---- measure breathing-mode period from z1 ----
z1 = z[:, 0]
z1_detrend = z1 - np.convolve(z1, np.ones(21) / 21, mode="same")
peaks, _ = find_peaks(z1_detrend, distance=15, prominence=0.4 * np.std(z1_detrend))
peak_frames = frame_indices[peaks]
period_iters = float(np.median(np.diff(peak_frames)))
period_s = period_iters * dt0
freq_hz = 1.0 / period_s

print(f"dt0={dt0:.2e}s  peaks found={len(peaks)}  period={period_iters:.1f} iters "
      f"= {period_s*1e6:.2f} us  freq={freq_hz/1e3:.2f} kHz")

# ---- per-field reconstruction error (from evaluate_ae.py run) ----
err_data = [
    {"name": "phi", "pct": 1.360},
    {"name": "T_e", "pct": 2.827},
    {"name": "v_i_z", "pct": 5.748},
    {"name": "v_i_r", "pct": 9.704},
    {"name": "n_n", "pct": 10.968},
    {"name": "n_e", "pct": 12.356},
    {"name": "n_i_dot", "pct": 20.302},
]

data_json = json.dumps({
    "z1": np.round(z[:, 0], 4).tolist(),
    "z2": np.round(z[:, 1], 4).tolist(),
    "z3": np.round(z[:, 2], 4).tolist(),
    "frame": frame_indices.tolist(),
    "n_train": n_train,
}, separators=(",", ":"))
err_json = json.dumps(err_data, separators=(",", ":"))

TEMPLATE = r"""<title>Breathing-mode attractor</title>
<style>
:root {
  --bg:#F2F3F7; --surface:#FFFFFF; --surface-2:#EAECF3;
  --border:#DBDEE8; --border-strong:#C3C7D6;
  --text-primary:#14161F; --text-secondary:#565C70; --text-muted:#8A8FA3;
  --accent:#4F46D6; --accent-2:#0E8F79;
  --font-display: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --font-body: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --radius: 10px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0A0B10; --surface:#14151D; --surface-2:#191B25;
    --border:#262838; --border-strong:#363952;
    --text-primary:#ECEDF5; --text-secondary:#A0A4B8; --text-muted:#6B7086;
    --accent:#8C86FF; --accent-2:#35C9AC;
  }
}
:root[data-theme="dark"] {
  --bg:#0A0B10; --surface:#14151D; --surface-2:#191B25;
  --border:#262838; --border-strong:#363952;
  --text-primary:#ECEDF5; --text-secondary:#A0A4B8; --text-muted:#6B7086;
  --accent:#8C86FF; --accent-2:#35C9AC;
}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--text-primary);font-family:var(--font-body);margin:0;}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;}
.wrap{max-width:1120px;margin:0 auto;padding:2rem 1.25rem 3rem;}
.eyebrow{font-family:var(--font-display);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin:0 0 .6rem;}
h1{font-family:var(--font-display);font-weight:600;font-size:clamp(24px,4vw,34px);letter-spacing:-0.01em;text-wrap:balance;margin:0 0 .7rem;color:var(--text-primary);}
.dek{font-size:16px;line-height:1.65;color:var(--text-secondary);max-width:64ch;margin:0 0 1.5rem;}
.hero{display:flex;align-items:baseline;gap:.9rem;flex-wrap:wrap;margin:0 0 1.75rem;padding:1rem 1.25rem;background:var(--surface);border:1px solid var(--border);border-radius:14px;}
.hero-num{font-family:var(--font-display);font-weight:600;font-size:40px;color:var(--accent);font-variant-numeric:tabular-nums;line-height:1;}
.hero-cap{font-size:13.5px;color:var(--text-secondary);line-height:1.55;max-width:52ch;}
.stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:1.75rem;}
.stat{background:var(--surface);padding:.85rem 1rem;}
.stat-label{font-family:var(--font-display);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted);margin:0 0 .35rem;}
.stat-value{font-family:var(--font-display);font-weight:600;font-size:18px;color:var(--text-primary);font-variant-numeric:tabular-nums;}
.stat-value small{font-weight:400;font-size:11.5px;color:var(--text-secondary);}
.control-rail{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;margin-bottom:.6rem;}
.seg{display:inline-flex;border:1px solid var(--border-strong);border-radius:999px;padding:2px;background:var(--surface-2);}
.seg button{font-family:var(--font-display);font-size:12px;letter-spacing:.02em;padding:.4rem .85rem;border:0;background:transparent;color:var(--text-secondary);border-radius:999px;cursor:pointer;}
.seg button.active{background:var(--accent);color:#fff;}
.hint{margin-left:auto;font-size:12px;color:var(--text-muted);font-family:var(--font-display);}
.legend-row{display:flex;gap:1.25rem;align-items:center;font-size:12px;color:var(--text-secondary);margin:0 0 .75rem;min-height:18px;font-family:var(--font-display);}
.legend-row .sw{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:-1px;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:.75rem;margin-bottom:1.5rem;}
#plot3d{width:100%;height:540px;}
.ts-panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1rem 1rem .25rem;margin-bottom:1.5rem;}
.ts-title{font-family:var(--font-display);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted);margin:0 0 .25rem;}
.ts-chart{width:100%;height:118px;}
.err-panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.25rem;margin-bottom:1.5rem;}
.err-title{font-family:var(--font-display);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted);margin:0 0 .9rem;}
.err-row{display:grid;grid-template-columns:90px 1fr 52px;align-items:center;gap:.6rem;margin-bottom:.55rem;font-size:13px;}
.err-name{font-family:var(--font-display);color:var(--text-secondary);}
.err-track{background:var(--surface-2);border-radius:5px;height:8px;overflow:hidden;}
.err-fill{background:var(--accent);height:100%;border-radius:5px;}
.err-val{font-family:var(--font-display);text-align:right;color:var(--text-primary);font-variant-numeric:tabular-nums;}
.caption{font-size:13px;line-height:1.7;color:var(--text-secondary);max-width:74ch;}
.caption b{color:var(--text-primary);font-weight:500;}
.foot{margin-top:2rem;font-size:11.5px;color:var(--text-muted);font-family:var(--font-display);border-top:1px solid var(--border);padding-top:1rem;}
@media (prefers-reduced-motion: reduce){*{transition:none!important;}}
</style>

<h2 class="sr-only">Interactive 3D plot of a convolutional autoencoder's 3-dimensional latent trajectory for an SPT-100 Hall thruster simulation, revealing a quasi-periodic breathing-mode attractor. Includes time-series views of each latent coordinate and a per-field reconstruction error breakdown.</h2>

<div class="wrap">
  <p class="eyebrow">Latent-space analysis &middot; SPT-100-class Hall thruster, 300&nbsp;V / 5.01&nbsp;mg&middot;s<sup>-1</sup> xenon</p>
  <h1>Breathing-mode attractor</h1>
  <p class="dek">A convolutional autoencoder compresses each 50&times;25 plasma-fluid snapshot &mdash; electron density, potential, temperature, neutral density, ionization rate, and ion velocity &mdash; into just 3 numbers. Plotted against each other across __N__ snapshots, those 3 numbers trace a closed loop: the discharge's breathing-mode limit cycle, recovered without ever telling the network a cycle exists.</p>

  <div class="hero">
    <span class="hero-num">__FREQ_KHZ__ kHz</span>
    <span class="hero-cap">breathing-mode frequency, measured directly from peak spacing in the latent loop (period &asymp; __PERIOD_US__&nbsp;&mu;s, __PERIOD_ITS__ iterations of the 50&nbsp;ns base timestep)</span>
  </div>

  <div class="stat-row">
    <div class="stat"><p class="stat-label">Frames encoded</p><p class="stat-value">__N__ <small>of __N_ITS__</small></p></div>
    <div class="stat"><p class="stat-label">Grid points</p><p class="stat-value">1,250 <small>50&times;25</small></p></div>
    <div class="stat"><p class="stat-label">Fields &rarr; latent</p><p class="stat-value">7 &rarr; 3</p></div>
    <div class="stat"><p class="stat-label">Val. MSE</p><p class="stat-value">0.0220 <small>normalized</small></p></div>
  </div>

  <div class="control-rail">
    <div class="seg" role="group" aria-label="Color the trajectory by">
      <button id="btn-time" class="active">Time</button>
      <button id="btn-split">Train / val</button>
    </div>
    <span class="hint">drag to rotate &middot; scroll to zoom</span>
  </div>
  <div class="legend-row" id="legend"></div>

  <div class="panel"><div id="plot3d"></div></div>

  <div class="ts-panel">
    <p class="ts-title">Latent coordinates vs. time</p>
    <div class="ts-chart" id="ts1"></div>
    <div class="ts-chart" id="ts2"></div>
    <div class="ts-chart" id="ts3"></div>
  </div>

  <div class="err-panel">
    <p class="err-title">Reconstruction error by field (mean relative error, all __N__ frames)</p>
    <div id="err-bars"></div>
  </div>

  <p class="caption">The <b>teal</b> segment is the held-out validation tail &mdash; the last 15% of the run in time, never seen during training. It rides the same loop traced by earlier cycles, which is the generalization check: the network learned the attractor's shape, not the specific frames. The worst-reconstructed field, <b>n_i_dot</b> (ionization rate), is also the most spatially localized &mdash; a 3D bottleneck smooths over its sharp near-channel peak.</p>

  <p class="foot">dt&#8320; = 5&times;10&#8315;&#8312; s &middot; __N_ITS__ iterations (__SIM_MS__ ms simulated) &middot; every 10th frame encoded &middot; conv-AE, 3D bottleneck, 200 epochs</p>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.5.1/plotly.min.js"></script>
<script>
(function(){
  var DATA = __DATA_JSON__;
  var ERR = __ERR_JSON__;
  var root = document.documentElement;
  function isDark(){
    var attr = root.getAttribute('data-theme');
    if(attr==='dark') return true;
    if(attr==='light') return false;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  var dark = isDark();
  var theme = dark ? {
    text:'#A0A4B8', grid:'#262838', accent:'#8C86FF', accent2:'#35C9AC'
  } : {
    text:'#565C70', grid:'#DBDEE8', accent:'#4F46D6', accent2:'#0E8F79'
  };

  var n = DATA.frame.length;
  var nTrain = DATA.n_train;

  function lerpColor(a,b,t){
    var ah=a.match(/\w\w/g).map(function(x){return parseInt(x,16);});
    var bh=b.match(/\w\w/g).map(function(x){return parseInt(x,16);});
    var c=ah.map(function(v,i){return Math.round(v+(bh[i]-v)*t);});
    return 'rgb('+c.join(',')+')';
  }
  var timeColors = DATA.frame.map(function(_,i){return lerpColor(theme.accent, theme.accent2, i/(n-1));});
  var splitColors = DATA.frame.map(function(_,i){return i<nTrain ? theme.accent : theme.accent2;});

  var trace = {
    type:'scatter3d', mode:'lines+markers',
    x:DATA.z1, y:DATA.z2, z:DATA.z3,
    line:{width:2, color: dark ? 'rgba(255,255,255,0.18)' : 'rgba(0,0,0,0.15)'},
    marker: {size:3.5, color: timeColors, line:{width:0}},
    hovertemplate:'frame %{text}<extra></extra>',
    text: DATA.frame
  };

  var layout = {
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    margin:{l:0,r:0,t:10,b:0},
    scene:{
      xaxis:{title:'z1', color:theme.text, gridcolor:theme.grid, zerolinecolor:theme.grid, backgroundcolor:'rgba(0,0,0,0)'},
      yaxis:{title:'z2', color:theme.text, gridcolor:theme.grid, zerolinecolor:theme.grid, backgroundcolor:'rgba(0,0,0,0)'},
      zaxis:{title:'z3', color:theme.text, gridcolor:theme.grid, zerolinecolor:theme.grid, backgroundcolor:'rgba(0,0,0,0)'},
      camera:{eye:{x:1.4,y:1.4,z:0.9}}
    },
    font:{family:'IBM Plex Mono, monospace', color:theme.text, size:11}
  };

  Plotly.newPlot('plot3d', [trace], layout, {displayModeBar:false, responsive:true});

  function setLegend(mode){
    var el = document.getElementById('legend');
    if(mode==='time'){
      el.innerHTML = '<span><span class="sw" style="background:'+theme.accent+'"></span>early</span>' +
                      '<span><span class="sw" style="background:'+theme.accent2+'"></span>late</span>';
    } else {
      el.innerHTML = '<span><span class="sw" style="background:'+theme.accent+'"></span>train (first 1,700)</span>' +
                      '<span><span class="sw" style="background:'+theme.accent2+'"></span>val, held out (last 300)</span>';
    }
  }
  setLegend('time');

  document.getElementById('btn-time').addEventListener('click', function(){
    Plotly.restyle('plot3d', {'marker.color':[timeColors]});
    this.classList.add('active');
    document.getElementById('btn-split').classList.remove('active');
    setLegend('time');
  });
  document.getElementById('btn-split').addEventListener('click', function(){
    Plotly.restyle('plot3d', {'marker.color':[splitColors]});
    this.classList.add('active');
    document.getElementById('btn-time').classList.remove('active');
    setLegend('split');
  });

  function tsChart(divId, arr, label, showX){
    var splitFrame = DATA.frame[nTrain];
    var tr = {
      x:DATA.frame, y:arr, type:'scatter', mode:'lines',
      line:{width:1.4, color:theme.accent}
    };
    var lay = {
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
      margin:{l:34,r:8,t:4,b: showX?24:4},
      xaxis:{showticklabels:showX, color:theme.text, gridcolor:'rgba(0,0,0,0)', zeroline:false, showline:true, linecolor:theme.grid, title: showX? {text:'frame index', font:{size:10}} : undefined},
      yaxis:{title:{text:label,font:{size:10}}, color:theme.text, gridcolor:theme.grid, zeroline:false},
      shapes:[{type:'line', x0:splitFrame, x1:splitFrame, y0:0, y1:1, yref:'paper', line:{color:theme.accent2, width:1, dash:'dot'}}],
      font:{family:'IBM Plex Mono, monospace', color:theme.text, size:10}
    };
    Plotly.newPlot(divId, [tr], lay, {displayModeBar:false, responsive:true});
  }
  tsChart('ts1', DATA.z1, 'z1', false);
  tsChart('ts2', DATA.z2, 'z2', false);
  tsChart('ts3', DATA.z3, 'z3', true);

  var maxErr = Math.max.apply(null, ERR.map(function(e){return e.pct;}));
  var bars = document.getElementById('err-bars');
  bars.innerHTML = ERR.map(function(e){
    var w = (e.pct/maxErr*100).toFixed(1);
    return '<div class="err-row"><span class="err-name">'+e.name+'</span>'+
      '<span class="err-track"><span class="err-fill" style="width:'+w+'%"></span></span>'+
      '<span class="err-val">'+e.pct.toFixed(1)+'%</span></div>';
  }).join('');
})();
</script>
"""

html = TEMPLATE
html = html.replace("__DATA_JSON__", data_json)
html = html.replace("__ERR_JSON__", err_json)
html = html.replace("__N_ITS__", f"{n_its:,}")
html = html.replace("__N__", f"{n:,}")
html = html.replace("__FREQ_KHZ__", f"{freq_hz/1e3:.1f}")
html = html.replace("__PERIOD_US__", f"{period_s*1e6:.1f}")
html = html.replace("__PERIOD_ITS__", f"{period_iters:.0f}")
html = html.replace("__SIM_MS__", f"{n_its*dt0*1e3:.2f}")

out_path = f"{BASE}/latent_attractor.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
