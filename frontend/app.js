/**
 * app.js — Night Navigator Frontend Logic
 * Real API calls, real Leaflet map, real Nominatim geocoding.
 * No hardcoded route data. No fake maps.
 *
 * Bugs fixed v2:
 *  1. pickSug global namespace collision — replaced with per-box data-idx pattern
 *  2. Tailwind dynamic class scanning — replaced with inline style/explicit static classes
 *  3. setModeChip regex replace — replaced with explicit class list management
 *  4. diag-spinner display toggling — uses .hidden class consistently
 *  5. compare-fab disabled state — uses opacity + pointer-events via style attribute
 */

'use strict';

// ─── CONFIG ─────────────────────────────────────────────────────────────────
const DEFAULT_API = localStorage.getItem('nn_api_url') || 'http://localhost:8001';
const BLR_CENTER = [12.9716, 77.5946];
const BLR_BBOX = '77.38,12.77,77.78,13.18';

const LANDMARKS = [
  { name: 'Indiranagar', lat: 12.9784, lon: 77.6408 },
  { name: 'Koramangala', lat: 12.9352, lon: 77.6245 },
  { name: 'Whitefield', lat: 12.9698, lon: 77.7500 },
  { name: 'Electronic City', lat: 12.8458, lon: 77.6603 },
  { name: 'Hebbal', lat: 13.0354, lon: 77.5970 },
  { name: 'MG Road', lat: 12.9756, lon: 77.6101 },
];

// ─── STATE ──────────────────────────────────────────────────────────────────
const S = {
  apiBase: DEFAULT_API,
  origin: null,   // { lat, lon, label }
  dest: null,   // { lat, lon, label }
  waypoints: [],     // [{ lat, lon, label }, ...] max 3
  routeResult: null,
  health: null,
  diagnostics: null,
  pinStep: 'origin',  // 'origin' | 'wp0' | 'wp1' | 'wp2' | 'dest'
  settings: {
    modeOverride: 'auto',
    avoidAlleys: true,
    avoidParks: false,
    avoidIndustrial: true,
    avoidCongestion: false,
    beta: 0.50,
    apiUrl: DEFAULT_API,
  },
};

let map, layerSafest, layerShortest, layerGlow, markerOrigin, markerDest;
const waypointMarkers = [];  // Leaflet markers for waypoints

// ─── MAP INIT ────────────────────────────────────────────────────────────────
function initMap() {
  map = L.map('map-canvas', {
    zoomControl: false,           // we provide our own custom zoom UI
    attributionControl: true,
    scrollWheelZoom: true,        // scroll to zoom enabled
    doubleClickZoom: true,        // double-click zoom enabled
    touchZoom: true,              // pinch-to-zoom on touch
    minZoom: 10,
    maxZoom: 19,
  }).setView(BLR_CENTER, 12);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> · <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  map.on('click', onMapClick);

  // Show current zoom level in custom control
  map.on('zoomend', () => {
    const el = document.getElementById('zoom-level');
    if (el) el.textContent = map.getZoom();
  });

  renderLandmarks();
  renderWaypointSlots();
}

// ─── CUSTOM ZOOM CONTROLS ────────────────────────────────────────────────────
function mapZoomIn() { if (map) map.zoomIn(1); }
function mapZoomOut() { if (map) map.zoomOut(1); }
function mapReset() { if (map) map.setView(BLR_CENTER, 12); }

function onMapClick(e) {
  const { lat, lng: lon } = e.latlng;
  const fallback = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  reverseGeocode(lat, lon)
    .then(label => applyPin(lat, lon, label || fallback))
    .catch(() => applyPin(lat, lon, fallback));
}

function nextPinStep() {
  // Cycle: origin → wp0 → wp1 → wp2 → dest → origin (wrap)
  if (!S.origin) return 'origin';
  if (!S.dest) {
    // fill any empty waypoint slots before dest
    for (let i = 0; i < 3; i++) {
      if (S.waypoints[i] === undefined) return 'dest'; // go to dest directly if no WPs added
    }
    return 'dest';
  }
  return 'origin';  // already full, cycle back
}

function applyPin(lat, lon, label) {
  const step = S.pinStep;
  if (step === 'origin') {
    setOrigin(lat, lon, label);
    S.pinStep = 'dest';
  } else if (step === 'dest') {
    setDest(lat, lon, label);
    S.pinStep = 'origin';
  } else if (step.startsWith('wp')) {
    const idx = parseInt(step[2]);
    setWaypoint(idx, lat, lon, label);
    // Next step → dest (or next empty waypoint)
    S.pinStep = 'dest';
  }
}

function makeIcon(color, symbol) {
  return L.divIcon({
    className: '',
    html: `<div style="width:34px;height:34px;border-radius:50%;background:${color};display:flex;align-items:center;justify-content:center;box-shadow:0 0 14px ${color}88;font-size:18px">${symbol}</div>`,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
  });
}

function setOriginMarker(lat, lon) {
  if (markerOrigin) markerOrigin.remove();
  markerOrigin = L.marker([lat, lon], { icon: makeIcon('#00fcca', '📍'), zIndexOffset: 1000 }).addTo(map);
}
function setDestMarker(lat, lon) {
  if (markerDest) markerDest.remove();
  markerDest = L.marker([lat, lon], { icon: makeIcon('#ff6e81', '🎯'), zIndexOffset: 1000 }).addTo(map);
}
function setWaypointMarker(idx, lat, lon) {
  if (waypointMarkers[idx]) { waypointMarkers[idx].remove(); }
  const colors = ['#c3f400', '#ff916c', '#a78bfa'];
  const symbols = ['⬟', '◆', '★'];
  waypointMarkers[idx] = L.marker([lat, lon], {
    icon: makeIcon(colors[idx], symbols[idx]),
    zIndexOffset: 900,
  }).addTo(map);
}

function drawRoutes(result) {
  clearMapRoutes();
  if (result.path_coords?.length > 1) {
    layerGlow = L.polyline(result.path_coords, { color: '#00FFCC', weight: 14, opacity: 0.14 }).addTo(map);
    layerSafest = L.polyline(result.path_coords, { color: '#00FFCC', weight: 5, opacity: 0.92 }).addTo(map);
    layerSafest.bindPopup(`<b style="color:#00fcca">Guardian Path</b><br>${(result.distance_m / 1000).toFixed(2)} km · Safety: ${(result.avg_safety_score * 100).toFixed(0)}%`);
  }
  if (result.short_coords?.length > 1) {
    layerShortest = L.polyline(result.short_coords, { color: '#64748B', weight: 4, opacity: 0.7, dashArray: '8 6' }).addTo(map);
    layerShortest.bindPopup(`<b>Standard Path</b><br>${(result.comparison.shortest.distance_m / 1000).toFixed(2)} km`);
  }
  if (result.path_coords?.length > 1) {
    map.fitBounds(L.latLngBounds(result.path_coords), { padding: [60, 60] });
  }
}

function clearMapRoutes() {
  [layerGlow, layerSafest, layerShortest].forEach(l => { if (l) map.removeLayer(l); });
  layerGlow = layerSafest = layerShortest = null;
}

// ─── GEOCODING ───────────────────────────────────────────────────────────────
const GEO_HEADERS = { 'Accept-Language': 'en' };  // User-Agent set by browser

async function geocode(query) {
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&viewbox=${BLR_BBOX}&bounded=0&limit=5&addressdetails=1`;
  try {
    const r = await fetch(url, { headers: GEO_HEADERS });
    return await r.json();
  } catch { return []; }
}

async function reverseGeocode(lat, lon) {
  const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`;
  try {
    const r = await fetch(url, { headers: GEO_HEADERS });
    const d = await r.json();
    return d.display_name?.split(',').slice(0, 2).join(', ') || null;
  } catch { return null; }
}

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// ─── AUTOCOMPLETE ────────────────────────────────────────────────────────────
// FIX: each suggestions box stores its own onSelect handler — no global collision
function setupAutocomplete(inputId, sugId, onSelect) {
  const inp = document.getElementById(inputId);
  const box = document.getElementById(sugId);
  box._onSelect = onSelect;   // store handler on the DOM element itself

  const search = debounce(async (q) => {
    if (q.length < 2) { box.classList.add('hidden'); return; }
    const results = await geocode(q);
    if (!results.length) { box.classList.add('hidden'); return; }
    box._results = results;
    box.innerHTML = results.map((r, i) =>
      `<div class="suggest-item" data-idx="${i}">
        <span class="material-symbols-outlined text-outline" style="font-size:15px">location_on</span>
        <span>${r.display_name.split(',').slice(0, 3).join(', ')}</span>
      </div>`
    ).join('');
    // Attach click handlers directly — no window.pickSug collision
    box.querySelectorAll('.suggest-item').forEach(el => {
      el.addEventListener('mousedown', (ev) => {
        ev.preventDefault();
        const idx = parseInt(el.dataset.idx);
        const res = box._results?.[idx];
        if (!res) return;
        const lat = parseFloat(res.lat);
        const lon = parseFloat(res.lon);
        const label = res.display_name.split(',').slice(0, 2).join(', ');
        box._onSelect(lat, lon, label);
        box.classList.add('hidden');
      });
    });
    box.classList.remove('hidden');
  }, 450);

  inp.addEventListener('input', e => search(e.target.value));
  inp.addEventListener('focus', e => { if (e.target.value.length >= 2) search(e.target.value); });
  inp.addEventListener('blur', () => setTimeout(() => box.classList.add('hidden'), 250));
}

// ─── ORIGIN / DEST SETTERS ──────────────────────────────────────────────────
function setOrigin(lat, lon, label) {
  S.origin = { lat, lon, label };
  document.getElementById('origin-input').value = label;
  setOriginMarker(lat, lon);
  map.panTo([lat, lon]);
  checkHint();
}

function setDest(lat, lon, label) {
  S.dest = { lat, lon, label };
  document.getElementById('dest-input').value = label;
  setDestMarker(lat, lon);
  checkHint();
}

// ─── WAYPOINTS ──────────────────────────────────────────────────────────────────
const MAX_WAYPOINTS = 3;
const WP_COLORS = ['#c3f400', '#ff916c', '#a78bfa'];
const WP_LABELS = ['Stop 1', 'Stop 2', 'Stop 3'];

function setWaypoint(idx, lat, lon, label) {
  S.waypoints[idx] = { lat, lon, label };
  setWaypointMarker(idx, lat, lon);
  renderWaypointSlots();
  checkHint();
  toast(`Stop ${idx + 1} set: ${label.split(',')[0]}`, 'ok');
}

function removeWaypoint(idx) {
  S.waypoints.splice(idx, 1);
  if (waypointMarkers[idx]) { waypointMarkers[idx].remove(); waypointMarkers.splice(idx, 1); }
  // Re-index remaining markers visually
  S.waypoints.forEach((wp, i) => {
    if (waypointMarkers[i]) waypointMarkers[i].remove();
    if (wp) setWaypointMarker(i, wp.lat, wp.lon);
  });
  renderWaypointSlots();
}

function renderWaypointSlots() {
  const container = document.getElementById('waypoints-section');
  if (!container) return;

  const canAdd = S.waypoints.length < MAX_WAYPOINTS;

  container.innerHTML = `
    <div style="margin-bottom:8px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <p style="font-size:9px;font-family:'Space Grotesk',sans-serif;font-weight:700;text-transform:uppercase;letter-spacing:.18em;color:#727582">Stops (max ${MAX_WAYPOINTS})</p>
        ${canAdd ? `<button onclick="addWaypointSlot()" style="display:flex;align-items:center;gap:4px;background:rgba(195,244,0,.08);color:#c3f400;border:1px solid rgba(195,244,0,.2);border-radius:9999px;padding:3px 10px;font-size:9px;font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:.06em;text-transform:uppercase;cursor:pointer">
          <span class="material-symbols-outlined" style="font-size:13px">add</span>Add Stop
        </button>` : `<span style="font-size:9px;color:#727582;font-family:'Space Grotesk',sans-serif">Max stops reached</span>`}
      </div>
      <div id="wp-list" style="display:flex;flex-direction:column;gap:6px">
        ${S.waypoints.map((wp, i) => `
          <div style="display:flex;align-items:center;gap:8px;background:rgba(${i === 0 ? '195,244,0' : i === 1 ? '255,145,108' : '167,139,250'},.05);border:1px solid rgba(${i === 0 ? '195,244,0' : i === 1 ? '255,145,108' : '167,139,250'},.18);border-radius:12px;padding:8px 10px">
            <div style="width:22px;height:22px;border-radius:50%;background:${WP_COLORS[i]};display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:10px;font-weight:700;color:#0a0e18">${i + 1}</div>
            <p style="flex:1;font-size:12px;color:#e5e7f6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0">${wp.label.split(',')[0]}</p>
            <button onclick="removeWaypoint(${i})" style="background:none;border:none;cursor:pointer;color:#727582;display:flex;align-items:center;flex-shrink:0;padding:2px">
              <span class="material-symbols-outlined" style="font-size:16px">close</span>
            </button>
          </div>
        `).join('')}
      </div>
    </div>
    ${S.waypoints.length ? `<div style="height:1px;background:rgba(68,72,84,.12);margin-bottom:12px"></div>` : ''}`;
}

function addWaypointSlot() {
  if (S.waypoints.length >= MAX_WAYPOINTS) {
    toast(`Maximum ${MAX_WAYPOINTS} stops allowed`, 'inf');
    return;
  }
  const idx = S.waypoints.length;
  // Set pinStep so next map click places this waypoint
  S.pinStep = `wp${idx}`;
  toast(`Click the map to place Stop ${idx + 1}`, 'inf');
}

function checkHint() {
  const h = document.getElementById('map-hint');
  if (S.origin && S.dest) h.classList.add('hidden');
  else h.classList.remove('hidden');
}

async function useMyLocation() {
  if (!navigator.geolocation) { toast('Geolocation not supported', 'err'); return; }
  toast('Getting location…', 'inf');
  navigator.geolocation.getCurrentPosition(async pos => {
    const { latitude: lat, longitude: lon } = pos.coords;
    const label = await reverseGeocode(lat, lon) || `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    setOrigin(lat, lon, label);
    map.setView([lat, lon], 14);
    toast('Origin set to current location', 'ok');
  }, err => {
    toast(`Location denied: ${err.message}`, 'err');
  });
}

// ─── API ─────────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const r = await fetch(S.apiBase + path, opts);
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`HTTP ${r.status}: ${text.slice(0, 120)}`);
  }
  return r.json();
}

// ─── HEALTH CHECK ────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const h = await apiFetch('/health');
    S.health = h;

    const mode = h.current_mode.toUpperCase();
    setEl('api-dot', null, 'background:#00fcca');
    setEl('api-txt', mode, 'color:#00fcca');
    setEl('mode-pill', mode);
    setEl('h-dot', null, 'background:#00fcca');
    setEl('h-ping', null, 'background:#00fcca');
    setEl('h-mode', `Guardian ${mode}`);
    setEl('h-edges', `${h.nodes.toLocaleString()} nodes · ${h.edges.toLocaleString()} edges`);
    setEl('edge-count', h.edges.toLocaleString());
  } catch (err) {
    setEl('api-dot', null, 'background:#ff716c');
    setEl('api-txt', 'Offline', 'color:#ff716c');
    setEl('h-mode', 'API Offline');
    setEl('h-edges', `Start uvicorn on :8001 — ${err.message}`);
  }
}

// ─── MODE RESOLUTION ─────────────────────────────────────────────────────────
function effectiveMode() {
  if (S.settings.modeOverride !== 'auto') return S.settings.modeOverride;
  if (S.health?.current_mode) return S.health.current_mode;
  const h = new Date().getHours();
  return (h >= 20 || h < 6) ? 'night' : 'day';
}

// ─── FIND ROUTE ──────────────────────────────────────────────────────────────
async function findRoutes() {
  if (!S.origin || !S.dest) {
    toast('Set an origin and destination first', 'inf');
    return;
  }
  S.pinStep = 'origin';
  console.log('[findRoutes] origin:', S.origin, 'dest:', S.dest, 'stops:', S.waypoints.length);
  showLoader(true);
  try {
    const body = {
      origin: [S.origin.lat, S.origin.lon],
      destination: [S.dest.lat, S.dest.lon],
      mode: effectiveMode(),
      avoid_congestion: S.settings.avoidCongestion,
      waypoints: S.waypoints.filter(Boolean).map(w => [w.lat, w.lon]),
    };
    console.log('[findRoutes] POST /route body:', body);
    const result = await apiFetch('/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    console.log('[findRoutes] result:', result);
    S.routeResult = result;

    drawRoutes(result);
    renderStatsBar(result);
    renderComparisonCards(result);
    renderCurrentRouteStats(result);

    // Enable compare FAB
    const fab = document.getElementById('compare-fab');
    fab.removeAttribute('disabled');
    fab.style.opacity = '1';
    fab.style.pointerEvents = 'auto';

    document.getElementById('cur-route-sec').classList.remove('hidden');
    const wc = result.waypoint_count || 0;
    const stopStr = wc ? ` · ${wc} stop${wc > 1 ? 's' : ''}` : '';
    toast(`Route found · ${result.mode_used.toUpperCase()} mode · ${(result.distance_m / 1000).toFixed(2)} km${stopStr}`, 'ok');
  } catch (e) {
    console.error('[findRoutes] error:', e);
    toast(`Route failed: ${e.message}`, 'err');
  } finally {
    showLoader(false);
  }
}

// ─── DIAGNOSTICS ─────────────────────────────────────────────────────────────
async function loadDiagnostics() {
  const spinner = document.getElementById('diag-spinner');
  const content = document.getElementById('diag-content');
  spinner.classList.remove('hidden');
  content.classList.add('hidden');
  try {
    const d = await apiFetch('/diagnostics');
    S.diagnostics = d;
    setEl('diag-ts', `Updated ${new Date().toLocaleTimeString()}`);
    renderDistChart('day-chart', 'day-stats', 'day-mean', d.safety_score, '#00fcca');
    renderDistChart('night-chart', 'night-stats', 'night-mean', d.safety_score_night, '#c3f400');
    checkSkewAlert(d);
    spinner.classList.add('hidden');
    content.classList.remove('hidden');
    loadTrafficScores();
  } catch (e) {
    spinner.classList.add('hidden');
    content.innerHTML = `<p style="color:#ff716c;font-size:13px">Failed to load diagnostics: ${e.message}</p>`;
    content.classList.remove('hidden');
  }
}

async function loadTrafficScores() {
  const el = document.getElementById('area-list');
  try {
    const data = await apiFetch('/traffic-scores');
    const entries = Object.entries(data).slice(0, 14);
    if (!entries.length) { el.innerHTML = '<p style="color:#727582;font-size:12px">No area data</p>'; return; }
    el.innerHTML = entries.map(([area, v]) => {
      const score = (v.traffic_safety_score * 100).toFixed(0);
      const risk = (v.incident_risk * 100).toFixed(0);
      const col = v.traffic_safety_score > 0.6 ? '#00fcca' : v.traffic_safety_score > 0.4 ? '#c3f400' : '#ff6e81';
      return `<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid rgba(68,72,84,.12)">
        <div style="flex:1;min-width:0">
          <p style="font-size:13px;color:#e5e7f6;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${area}</p>
          <p style="font-size:10px;color:#727582">Risk: ${risk}%</p>
        </div>
        <div style="text-align:right;flex-shrink:0">
          <p style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;color:${col}">${score}%</p>
          <p style="font-size:9px;color:#727582;text-transform:uppercase;letter-spacing:.05em">safety</p>
        </div>
        <div style="width:4px;height:28px;border-radius:9999px;background:${col};opacity:.7;flex-shrink:0"></div>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = `<p style="color:#727582;font-size:12px">Traffic data unavailable — ${e.message}</p>`;
  }
}

// ─── DISTRIBUTION CHART ──────────────────────────────────────────────────────
function renderDistChart(chartId, statsId, meanId, data, color) {
  const noData = '<p style="color:#727582;font-size:12px">No data</p>';
  if (!data) { document.getElementById(chartId).innerHTML = noData; return; }

  setEl(meanId, `Mean: ${data.mean?.toFixed(4) ?? '—'}`);

  const buckets = data.buckets || {};
  const total = Object.values(buckets).reduce((a, b) => a + b, 0) || 1;
  const barColors = ['#ff6e81', '#ff916c', '#c3f400', '#a8ffe1', '#00fcca'];

  document.getElementById(chartId).innerHTML = Object.entries(buckets).map(([band, count], i) => {
    const pct = (count / total * 100).toFixed(1);
    const w = Math.max(3, parseFloat(pct));
    return `<div style="display:flex;align-items:center;gap:10px">
      <span style="font-size:10px;color:#727582;font-family:'Space Grotesk',sans-serif;width:60px;flex-shrink:0">${band}</span>
      <div style="flex:1;background:rgba(32,37,52,.8);border-radius:9999px;height:20px;overflow:hidden">
        <div style="height:100%;width:${w}%;min-width:8px;background:${barColors[i]};border-radius:9999px;display:flex;align-items:center;padding-left:6px;transition:width .8s ease">
          <span style="font-size:9px;font-family:'Space Grotesk',sans-serif;font-weight:700;color:#0a0e18">${pct}%</span>
        </div>
      </div>
      <span style="font-size:10px;color:#a7aab9;font-family:'Space Grotesk',sans-serif;width:52px;text-align:right;flex-shrink:0">${count.toLocaleString()}</span>
    </div>`;
  }).join('');

  document.getElementById(statsId).innerHTML = [
    { label: 'Mean', val: data.mean?.toFixed(4) },
    { label: 'Std Dev', val: data.std?.toFixed(4) },
    { label: 'Skewness', val: data.skewness?.toFixed(3) },
  ].map(({ label, val }) =>
    `<div style="background:rgba(26,31,45,.9);border-radius:12px;padding:12px;text-align:center;border:1px solid rgba(68,72,84,.15)">
      <p style="font-size:9px;font-family:'Space Grotesk',sans-serif;text-transform:uppercase;letter-spacing:.08em;color:#727582;margin-bottom:4px">${label}</p>
      <p style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;color:#e5e7f6">${val ?? '—'}</p>
    </div>`
  ).join('');
}

function checkSkewAlert(d) {
  const s = d.safety_score_night?.skewness;
  const el = document.getElementById('skew-alert');
  if (s && Math.abs(s) > 1.5) {
    setEl('skew-txt', `High Skew (${s.toFixed(2)}) — Night scores unevenly distributed`);
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

// ─── RENDER: COMPARISON CARDS ────────────────────────────────────────────────
function renderComparisonCards(result) {
  const c = result.comparison;
  document.getElementById('cmp-empty').classList.add('hidden');
  document.getElementById('cmp-cards').classList.remove('hidden');
  document.getElementById('cmp-subtitle').textContent =
    `${S.origin?.label?.split(',')[0]} → ${S.dest?.label?.split(',')[0]}`;

  const gSafe = (c.safest.avg_safety * 100).toFixed(0);
  const sSafe = (c.shortest.avg_safety * 100).toFixed(0);

  // Safety bars
  setTimeout(() => {
    document.getElementById('g-bar').style.width = `${gSafe}%`;
    document.getElementById('s-bar').style.width = `${sSafe}%`;
  }, 50);

  // Score circles
  document.getElementById('g-score').innerHTML = `<span style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#c3f400">${gSafe}</span>`;
  document.getElementById('s-score').innerHTML = `<span style="font-family:'Space Grotesk',sans-serif;font-weight:700;color:#a7aab9">${sSafe}</span>`;

  const metricHtml = (label, val) =>
    `<div>
      <p style="font-size:9px;font-family:'Space Grotesk',sans-serif;text-transform:uppercase;letter-spacing:.05em;color:#a7aab9;margin-bottom:2px">${label}</p>
      <p style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;color:#e5e7f6">${val}</p>
    </div>`;

  document.getElementById('g-metrics').innerHTML = [
    metricHtml('Distance', `${(c.safest.distance_m / 1000).toFixed(2)} km`),
    metricHtml('Safety', `${(c.safest.avg_safety * 100).toFixed(0)}%`),
    metricHtml('Risk', `${(c.safest.avg_incident * 100).toFixed(0)}%`),
  ].join('');

  document.getElementById('s-metrics').innerHTML = [
    metricHtml('Distance', `${(c.shortest.distance_m / 1000).toFixed(2)} km`),
    metricHtml('Safety', `${(c.shortest.avg_safety * 100).toFixed(0)}%`),
    metricHtml('Risk', `${(c.shortest.avg_incident * 100).toFixed(0)}%`),
  ].join('');

  const gainCol = c.safety_gain_pct >= 0 ? '#a8ffe1' : '#ff716c';
  const ohCol = c.distance_overhead_pct <= 20 ? '#a8ffe1' : c.distance_overhead_pct <= 50 ? '#c3f400' : '#ff6e81';
  const deltaBox = document.getElementById('delta-box');
  deltaBox.classList.remove('hidden');
  deltaBox.innerHTML = `<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div style="text-align:center">
      <p style="font-size:9px;font-family:'Space Grotesk',sans-serif;text-transform:uppercase;letter-spacing:.1em;color:#727582;margin-bottom:4px">Safety Gain</p>
      <p style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:22px;color:${gainCol}">${c.safety_gain_pct >= 0 ? '+' : ''}${c.safety_gain_pct.toFixed(1)}%</p>
    </div>
    <div style="text-align:center">
      <p style="font-size:9px;font-family:'Space Grotesk',sans-serif;text-transform:uppercase;letter-spacing:.1em;color:#727582;margin-bottom:4px">Dist Overhead</p>
      <p style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:22px;color:${ohCol}">${c.distance_overhead_pct >= 0 ? '+' : ''}${c.distance_overhead_pct.toFixed(1)}%</p>
    </div>
  </div>`;
}

// ─── RENDER: STATS BAR ───────────────────────────────────────────────────────
function renderStatsBar(result) {
  const c = result.comparison;
  const mode = result.mode_used?.toUpperCase();
  const bar = document.getElementById('stats-bar');

  const item = (label, val, color = '#e5e7f6') =>
    `<div style="padding:10px 20px;background:rgba(26,31,45,.95);display:flex;flex-direction:column;align-items:center;gap:2px;border-right:1px solid rgba(68,72,84,.2)">
      <span style="font-size:9px;font-family:'Space Grotesk',sans-serif;text-transform:uppercase;letter-spacing:.1em;color:#727582">${label}</span>
      <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;color:${color}">${val}</span>
    </div>`;

  bar.innerHTML =
    item('Mode', mode, '#a8ffe1') +
    item('Distance', `${(result.distance_m / 1000).toFixed(2)} km`) +
    item('Safety', `${(result.avg_safety_score * 100).toFixed(0)}%`, '#00fcca') +
    item('Incident', `${(result.avg_incident_risk * 100).toFixed(0)}%`, result.avg_incident_risk > 0.5 ? '#ff6e81' : '#c3f400') +
    item('Safety Gain', `${c.safety_gain_pct >= 0 ? '+' : ''}${c.safety_gain_pct.toFixed(1)}%`, c.safety_gain_pct > 0 ? '#a8ffe1' : '#ff6e81') +
    item('Dist Overhead', `${c.distance_overhead_pct >= 0 ? '+' : ''}${c.distance_overhead_pct.toFixed(1)}%`, c.distance_overhead_pct <= 30 ? '#c3f400' : '#ff6e81');

  bar.classList.add('on');
}

// ─── RENDER: CURRENT ROUTE STATS (analytics) ─────────────────────────────────
function renderCurrentRouteStats(result) {
  const c = result.comparison;
  const el = document.getElementById('cur-route-grid');
  const row = (label, val) =>
    `<div style="background:rgba(26,31,45,.9);border-radius:12px;padding:12px;border:1px solid rgba(68,72,84,.15)">
      <p style="font-size:9px;font-family:'Space Grotesk',sans-serif;text-transform:uppercase;letter-spacing:.08em;color:#727582;margin-bottom:4px">${label}</p>
      <p style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:13px;color:#e5e7f6">${val}</p>
    </div>`;
  el.innerHTML = [
    row('Guardian Dist', `${(c.safest.distance_m / 1000).toFixed(2)} km`),
    row('Standard Dist', `${(c.shortest.distance_m / 1000).toFixed(2)} km`),
    row('Avg Safety', `${(result.avg_safety_score * 100).toFixed(1)}%`),
    row('Incident Risk', `${(result.avg_incident_risk * 100).toFixed(1)}%`),
    row('Safety Gain', `${c.safety_gain_pct >= 0 ? '+' : ''}${c.safety_gain_pct.toFixed(1)}%`),
    row('Route Mode', result.mode_used?.toUpperCase()),
  ].join('');
}

// ─── LANDMARKS ───────────────────────────────────────────────────────────────
function renderLandmarks() {
  const container = document.getElementById('landmarks');
  container.innerHTML = '';
  LANDMARKS.forEach(lm => {
    const div = document.createElement('div');
    div.style.cssText = 'display:flex;align-items:center;gap:10px;cursor:pointer;padding:8px;border-radius:10px;transition:background .15s';
    div.innerHTML = `
      <div style="width:30px;height:30px;border-radius:50%;background:rgba(32,37,52,.9);display:flex;align-items:center;justify-content:center;flex-shrink:0">
        <span class="material-symbols-outlined" style="font-size:15px;color:#727582">place</span>
      </div>
      <div style="min-width:0">
        <p style="font-size:13px;color:#e5e7f6;font-weight:500">${lm.name}</p>
        <p style="font-size:10px;color:#727582">${lm.lat.toFixed(4)}, ${lm.lon.toFixed(4)}</p>
      </div>`;
    div.addEventListener('mouseover', () => div.style.background = 'rgba(168,255,225,.05)');
    div.addEventListener('mouseout', () => div.style.background = '');
    div.addEventListener('click', () => {
      console.log('[landmark click]', lm.name, lm.lat, lm.lon);
      if (!S.origin) setOrigin(lm.lat, lm.lon, `${lm.name}, Bangalore`);
      else setDest(lm.lat, lm.lon, `${lm.name}, Bangalore`);
      toast(`${S.origin && S.dest ? 'Destination' : 'Origin'} set: ${lm.name}`, 'ok');
    });
    container.appendChild(div);
  });
}

// ─── VIEW SWITCHING ──────────────────────────────────────────────────────────
function switchView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  const target = document.getElementById(`view-${id}`);
  if (target) target.classList.add('active');

  // Sync nav buttons
  document.querySelectorAll('.snav-btn, .nav-lnk').forEach(b => {
    b.classList.toggle('active', b.dataset.view === id);
  });

  // Lazy load diagnostics
  if (id === 'analytics' && !S.diagnostics) loadDiagnostics();
}

// ─── MODE CHIPS ──────────────────────────────────────────────────────────────
// FIX: Use data attribute + explicit styles instead of dynamic Tailwind classes
function setModeChip(mode) {
  ['night', 'day', 'auto'].forEach(m => {
    const el = document.getElementById(`chip-${m}`);
    if (!el) return;
    if (m === mode) {
      el.style.background = 'rgba(0,252,202,.15)';
      el.style.color = '#00fcca';
      el.style.borderColor = 'rgba(0,252,202,.35)';
    } else {
      el.style.background = 'rgba(21,25,38,.8)';
      el.style.color = '#a7aab9';
      el.style.borderColor = 'rgba(68,72,84,.25)';
    }
  });
  // Also update state — respect current chip as mode override for this session
  if (mode !== 'auto') S.settings.modeOverride = mode;
  else S.settings.modeOverride = 'auto';
}

// ─── SETTINGS ────────────────────────────────────────────────────────────────
const TOGGLE_DEFS = [
  { key: 'avoidAlleys', label: 'Avoid Alleys', desc: 'Route around narrow, unmonitored passages' },
  { key: 'avoidParks', label: 'Avoid Unlit Parks', desc: 'Skip green spaces without active lighting' },
  { key: 'avoidIndustrial', label: 'Avoid Industrial', desc: 'Prioritise commercial and residential zones' },
  { key: 'avoidCongestion', label: 'Avoid Congestion', desc: 'Factor in traffic activity scores' },
];

function renderToggles() {
  document.getElementById('toggles').innerHTML = TOGGLE_DEFS.map(d => {
    const on = S.settings[d.key];
    return `<div style="display:flex;align-items:center;justify-content:space-between;padding:14px;border-radius:12px;background:rgba(15,19,30,.8)">
      <div>
        <p style="font-size:14px;color:#e5e7f6;font-weight:500">${d.label}</p>
        <p style="font-size:11px;color:#a7aab9;margin-top:2px">${d.desc}</p>
      </div>
      <button onclick="toggleSetting('${d.key}')"
        style="width:44px;height:24px;border-radius:9999px;border:none;cursor:pointer;position:relative;transition:background .2s;background:${on ? '#00fcca' : 'rgba(32,37,52,.9)'}">
        <span style="position:absolute;top:3px;width:18px;height:18px;background:white;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,.4);transition:left .2s;left:${on ? '23px' : '3px'}"></span>
      </button>
    </div>`;
  }).join('');
}

function toggleSetting(key) {
  S.settings[key] = !S.settings[key];
  renderToggles();
}

function setSettingMode(mode) {
  S.settings.modeOverride = mode;
  document.querySelectorAll('[data-mode-btn]').forEach(b => {
    const active = b.dataset.modeBtn === mode;
    b.style.background = active ? '#00fcca' : 'rgba(21,25,38,.8)';
    b.style.color = active ? '#0a0e18' : '#a7aab9';
    b.style.borderColor = active ? '#00fcca' : 'rgba(68,72,84,.25)';
    b.style.fontWeight = '700';
  });
}

function onBetaChange(val) {
  S.settings.beta = parseFloat(val);
  setEl('beta-val', parseFloat(val).toFixed(2));
}

function saveSettings() {
  S.settings.apiUrl = document.getElementById('api-url-input').value.trim() || DEFAULT_API;
  S.apiBase = S.settings.apiUrl;
  localStorage.setItem('nn_settings', JSON.stringify(S.settings));
  localStorage.setItem('nn_api_url', S.settings.apiUrl);
  toast('Preferences saved', 'ok');
  checkHealth();
  switchView('map');
}

function loadSettings() {
  const raw = localStorage.getItem('nn_settings');
  if (raw) try { Object.assign(S.settings, JSON.parse(raw)); } catch { }
  S.apiBase = S.settings.apiUrl || DEFAULT_API;
  const urlEl = document.getElementById('api-url-input');
  if (urlEl) urlEl.value = S.apiBase;
  const slider = document.getElementById('beta-slider');
  if (slider) slider.value = S.settings.beta;
  setEl('beta-val', S.settings.beta.toFixed(2));
  setSettingMode(S.settings.modeOverride);
  setModeChip(S.settings.modeOverride);
  renderToggles();
}

function resetSettings() {
  localStorage.removeItem('nn_settings');
  S.settings = {
    modeOverride: 'auto', avoidAlleys: true, avoidParks: false,
    avoidIndustrial: true, avoidCongestion: false, beta: 0.50,
    apiUrl: 'http://localhost:8001',
  };
  loadSettings();
  toast('Reset to defaults', 'inf');
}

// ─── CLEAR ALL ───────────────────────────────────────────────────────────────
function clearAll() {
  S.origin = S.dest = S.routeResult = null;
  S.waypoints = [];
  S.pinStep = 'origin';
  if (markerOrigin) { markerOrigin.remove(); markerOrigin = null; }
  if (markerDest) { markerDest.remove(); markerDest = null; }
  waypointMarkers.forEach(m => { if (m) m.remove(); });
  waypointMarkers.length = 0;
  clearMapRoutes();
  document.getElementById('origin-input').value = '';
  document.getElementById('dest-input').value = '';
  document.getElementById('stats-bar').classList.remove('on');
  renderWaypointSlots();

  // Re-disable compare FAB
  const fab = document.getElementById('compare-fab');
  fab.setAttribute('disabled', '');
  fab.style.opacity = '0.3';
  fab.style.pointerEvents = 'none';

  document.getElementById('cmp-empty').classList.remove('hidden');
  document.getElementById('cmp-cards').classList.add('hidden');
  checkHint();
  map.setView(BLR_CENTER, 12);
}

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function setEl(id, text, style) {
  const el = document.getElementById(id);
  if (!el) return;
  if (text !== null && text !== undefined) el.textContent = text;
  if (style) el.setAttribute('style', (el.getAttribute('style') || '') + ';' + style);
}

function toast(msg, type = 'inf') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  const icon = type === 'ok' ? 'check_circle' : type === 'err' ? 'error' : 'info';
  el.innerHTML = `<span class="material-symbols-outlined icon-fill" style="font-size:16px">${icon}</span>${msg}`;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

function showLoader(on) {
  document.getElementById('loader').classList.toggle('on', on);
}

function loadAnalyticsAndSwitch() {
  switchView('analytics');
}

// ─── INIT ────────────────────────────────────────────────────────────────────
async function init() {
  loadSettings();
  initMap();

  // Wire up autocomplete for both inputs — each stores its own handler (no collision)
  setupAutocomplete('origin-input', 'origin-sug', setOrigin);
  setupAutocomplete('dest-input', 'dest-sug', setDest);

  // Disable compare FAB on load
  const fab = document.getElementById('compare-fab');
  fab.style.opacity = '0.3';
  fab.style.pointerEvents = 'none';

  // Default chip state
  setModeChip('auto');

  await checkHealth();
  setInterval(checkHealth, 30000);
  checkHint();

  toast('Night Navigator ready — click map to pin locations', 'ok');
}

// ─── AI AGENT INTEGRATION ──────────────────────────────────────────────────
function appendChat(htmlContent, isUser = false) {
  const box = document.getElementById('ai-chat-box');
  const div = document.createElement('div');
  div.className = isUser
    ? "bg-surface-container-high rounded-2xl rounded-tr-sm p-4 border border-outline-variant/10 shadow-lg relative max-w-[90%] self-end"
    : "bg-surface-container rounded-2xl rounded-tl-sm p-4 border border-primary/20 shadow-lg relative max-w-[90%] self-start";
  div.innerHTML = `<p class="text-sm text-on-surface">${htmlContent}</p>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function queryAIAgent() {
  const inp = document.getElementById('ai-input');
  const query = inp.value.trim();
  if (!query) return;

  inp.value = '';
  appendChat(`<i>${query}</i>`, true);

  const loaderId = 'ai-loader-' + Date.now();
  appendChat(`<div id="${loaderId}" class="flex gap-2 items-center text-primary"><div class="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin"></div><span class="text-xs">Thinking...</span></div>`);

  try {
    const res = await apiFetch('/agent/orchestrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });

    document.getElementById(loaderId)?.parentElement.remove();

    let html = `<p class="font-bold text-primary mb-2">Navigator AI Response:</p>`;

    // Simple markdown to HTML parser
    const mdToHtml = str => (str || '')
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-on-surface">$1</strong>')
      .replace(/_(.*?)_/g, '<i>$1</i>')
      .replace(/\n\s*-\s/g, '<br>&bull; ')
      .replace(/\n\s*•\s/g, '<br>&bull; ')
      .replace(/\n\s*\*\s/g, '<br>&bull; ')
      .replace(/\n/g, '<br>');

    if (res.summary) html += `<div class="space-y-1">${mdToHtml(res.summary)}</div>`;
    else if (res.explanation) html += `<div>${mdToHtml(res.explanation)}</div>`;
    else html += `<p>Done processing your request.</p>`;

    if (res.incident_warnings && res.incident_warnings.includes("WARNING")) {
      html += `<div class="mt-3 p-3 bg-error/10 border border-error/30 rounded-lg text-error text-xs">${res.incident_warnings}</div>`;
    }

    appendChat(html);

    // Draw route if returned
    if (res.route_data && res.route_data.path_coords) {
      S.routeResult = res.route_data;
      drawRoutes(res.route_data);
      renderStatsBar(res.route_data);
    }
  } catch (e) {
    document.getElementById(loaderId)?.parentElement.remove();
    appendChat(`<span class="text-error">Error connecting to AI: ${e.message}</span>`);
  }
}

async function explainRoute() {
  if (!S.routeResult) {
    appendChat(`<span class="text-error">Please generate a route on the map first before asking for an explanation.</span>`);
    return;
  }

  const m = {
    distance_m: S.routeResult.distance_m,
    avg_safety_score: S.routeResult.avg_safety_score,
    avg_incident_risk: S.routeResult.avg_incident_risk
  };

  appendChat(`<i>Can you explain why this route is safe?</i>`, true);

  const loaderId = 'ai-loader-' + Date.now();
  appendChat(`<div id="${loaderId}" class="flex gap-2 items-center text-secondary"><div class="w-3 h-3 border-2 border-secondary border-t-transparent rounded-full animate-spin"></div><span class="text-xs">Analyzing spatial features...</span></div>`);

  try {
    const res = await apiFetch('/agent/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        origin: [S.origin ? S.origin.lat : 0, S.origin ? S.origin.lon : 0],
        destination: [S.dest ? S.dest.lat : 0, S.dest ? S.dest.lon : 0],
        metrics: m
      })
    });

    document.getElementById(loaderId)?.parentElement.remove();

    appendChat(`<p class="font-bold text-secondary mb-2">Safety Explanation:</p><p>${res.explanation}</p>`);
  } catch (e) {
    document.getElementById(loaderId)?.parentElement.remove();
    appendChat(`<span class="text-error">Error connecting to Explanation Agent: ${e.message}</span>`);
  }
}


// ─── SOS EMERGENCY SYSTEM ─────────────────────────────────────────────────────
function openSOS() {
  document.getElementById('sos-modal').classList.add('open');
  // Flash the topnav red briefly for urgency
  const nav = document.getElementById('topnav');
  nav.style.borderBottomColor = 'rgba(255,3,85,.6)';
  setTimeout(() => nav.style.borderBottomColor = '', 1500);
}

function closeSOS() {
  document.getElementById('sos-modal').classList.remove('open');
}

function triggerSOS() {
  // Try to get current location first
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(pos => {
      const { latitude: lat, longitude: lon } = pos.coords;
      const mapsUrl = `https://www.google.com/maps?q=${lat},${lon}`;
      toast(`📍 Location captured: ${lat.toFixed(5)}, ${lon.toFixed(5)}`, 'err');
      // In a real app you'd POST this to a backend / SMS service
      console.log('[SOS] Location for emergency services:', lat, lon, mapsUrl);
    }, () => {
      toast('⚠️ Location unavailable — calling anyway', 'err');
    });
  }
  callEmergency();
}

function callEmergency() {
  closeSOS();
  // On mobile browsers tel: links work; on desktop it shows a toast
  const a = document.createElement('a');
  a.href = 'tel:112';
  a.click();
  toast('🆘 Calling 112 — stay safe!', 'err');
}

function shareLocation() {
  if (!navigator.geolocation) {
    toast('Geolocation not supported on this device', 'err');
    return;
  }
  navigator.geolocation.getCurrentPosition(pos => {
    const { latitude: lat, longitude: lon } = pos.coords;
    const mapsUrl = `https://www.google.com/maps?q=${lat},${lon}`;
    if (navigator.share) {
      navigator.share({
        title: '🆘 Night Navigator SOS',
        text: `I need help! My location: ${lat.toFixed(6)}, ${lon.toFixed(6)}`,
        url: mapsUrl
      }).catch(() => {});
    } else {
      navigator.clipboard.writeText(mapsUrl).then(() => {
        toast('📋 Location link copied to clipboard', 'ok');
      }).catch(() => {
        toast(`Location: ${lat.toFixed(5)}, ${lon.toFixed(5)}`, 'inf');
      });
    }
    closeSOS();
  }, () => {
    toast('Could not get your location', 'err');
  });
}

// Close SOS modal on backdrop click
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('sos-modal').addEventListener('click', e => {
    if (e.target === document.getElementById('sos-modal')) closeSOS();
  });
});

document.addEventListener('DOMContentLoaded', init);
