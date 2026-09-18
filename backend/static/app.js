// Live BTC/USD trig overlay chart.
// Fetches /api/research/trig-state every second and redraws a Canvas chart.

const API_KEY_STORAGE = "btcusd_research_api_key";
const POLL_INTERVAL_MS = 1000;
const WINDOW_SECONDS = 120;

const COLORS = {
  price: "#4dd2c0",
  sin: "#8fb7ff",
  cos: "#c58fff",
  tan: "#ff8f8f",
  cot: "#ffcf8f",
  sec: "#8fffc0",
  csc: "#ff8fd0",
};

// -- API key management -----------------------------------------------------

function getApiKey() {
  // Prefer explicit URL param, then sessionStorage.
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get("key");
  if (fromUrl) {
    sessionStorage.setItem(API_KEY_STORAGE, fromUrl);
    url.searchParams.delete("key");
    window.history.replaceState({}, "", url.toString());
    return fromUrl;
  }
  return sessionStorage.getItem(API_KEY_STORAGE) || "";
}

function promptForKey() {
  const k = window.prompt(
    "Enter your COLLECTOR_API_KEY (this will be stored in this browser session only):"
  );
  if (k) {
    sessionStorage.setItem(API_KEY_STORAGE, k.trim());
  }
  return (k || "").trim();
}

// -- State ------------------------------------------------------------------

const enabled = {
  sin: true, cos: true, tan: true,
  cot: true, sec: true, csc: true,
};

let lastData = null;
let lastError = null;

// -- Chart rendering --------------------------------------------------------

const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth - 32;
  const h = wrap.clientHeight - 16;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", () => { resizeCanvas(); draw(); });

function drawGrid(w, h, pad) {
  ctx.strokeStyle = "#14141a";
  ctx.lineWidth = 1;
  const rows = 6;
  const cols = 8;
  for (let i = 0; i <= rows; i++) {
    const y = pad.top + (h - pad.top - pad.bottom) * (i / rows);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }
  for (let i = 0; i <= cols; i++) {
    const x = pad.left + (w - pad.left - pad.right) * (i / cols);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, h - pad.bottom);
    ctx.stroke();
  }
}

function formatPrice(v) {
  return v.toFixed(2);
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const pad = { top: 16, right: 64, bottom: 34, left: 72 };

  ctx.clearRect(0, 0, w, h);

  if (!lastData || !lastData.points || lastData.points.length < 2) {
    ctx.fillStyle = "#4a4a55";
    ctx.font = "12px monospace";
    ctx.fillText(
      lastError
        ? "error: " + lastError
        : "waiting for data…",
      20, 30
    );
    return;
  }

  const pts = lastData.points;
  const t0 = pts[0].t;
  const t1 = pts[pts.length - 1].t;
  const tSpan = t1 - t0 || 1;

  // Price range for left axis
  let pMin = Infinity, pMax = -Infinity;
  for (const p of pts) {
    if (p.p < pMin) pMin = p.p;
    if (p.p > pMax) pMax = p.p;
  }
  const pPad = (pMax - pMin) * 0.1 || 1;
  pMin -= pPad;
  pMax += pPad;
  const pSpan = pMax - pMin || 1;

  // Trig range for right axis (fixed)
  const trigMin = -3, trigMax = 3;
  const trigSpan = trigMax - trigMin;

  const xFor = (t) => pad.left + (w - pad.left - pad.right) * ((t - t0) / tSpan);
  const yForPrice = (p) => pad.top + (h - pad.top - pad.bottom) * (1 - (p - pMin) / pSpan);
  const yForTrig = (v) => pad.top + (h - pad.top - pad.bottom) * (1 - (v - trigMin) / trigSpan);

  drawGrid(w, h, pad);

  // Axis labels
  ctx.fillStyle = "#4a4a55";
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  for (let i = 0; i <= 6; i++) {
    const v = pMin + (pSpan * i) / 6;
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - i / 6);
    ctx.fillText(formatPrice(v), pad.left - 6, y + 3);
  }
  ctx.textAlign = "left";
  for (let i = 0; i <= 6; i++) {
    const v = trigMin + (trigSpan * i) / 6;
    const y = pad.top + (h - pad.top - pad.bottom) * (1 - i / 6);
    ctx.fillText(v.toFixed(0), w - pad.right + 6, y + 3);
  }

  // Time labels
  ctx.textAlign = "center";
  for (let i = 0; i <= 8; i++) {
    const t = t0 + (tSpan * i) / 8;
    const d = new Date(t);
    const label = d.toTimeString().slice(0, 8);
    const x = pad.left + (w - pad.left - pad.right) * (i / 8);
    ctx.fillText(label, x, h - pad.bottom + 18);
  }

  // Price line
  ctx.strokeStyle = COLORS.price;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const x = xFor(pts[i].t);
    const y = yForPrice(pts[i].p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

    // Trig lines
  const trigFns = ["sin", "cos", "tan", "cot", "sec", "csc"];
  const JUMP_THRESHOLD = 2.0; // break line when value jumps more than this
  for (const fn of trigFns) {
    if (!enabled[fn]) continue;
    ctx.strokeStyle = COLORS[fn];
    ctx.lineWidth = 0.9;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    let pen = false;
    let prevV = null;
    for (const p of pts) {
      const v = p[fn];
      if (v === null || v === undefined || !isFinite(v)) {
        pen = false;
        prevV = null;
        continue;
      }
      // Break the line if the value jumped by more than the threshold
      // (this suppresses singularity spikes without falsifying the data).
      if (prevV !== null && Math.abs(v - prevV) > JUMP_THRESHOLD) {
        pen = false;
      }
      const clamped = Math.max(trigMin, Math.min(trigMax, v));
      const x = xFor(p.t);
      const y = yForTrig(clamped);
      if (!pen) {
        ctx.moveTo(x, y);
        pen = true;
      } else {
        ctx.lineTo(x, y);
      }
      prevV = v;
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Latest price marker
  const lastP = pts[pts.length - 1];
  ctx.fillStyle = COLORS.price;
  ctx.beginPath();
  ctx.arc(xFor(lastP.t), yForPrice(lastP.p), 3, 0, Math.PI * 2);
  ctx.fill();
}

// -- Data polling -----------------------------------------------------------

let apiKey = getApiKey();

async function poll() {
  if (!apiKey) {
    apiKey = promptForKey();
    if (!apiKey) {
      lastError = "no API key";
      draw();
      return;
    }
  }

  try {
    const res = await fetch(
      `/api/research/trig-state?seconds=${WINDOW_SECONDS}`,
      {
        headers: { authorization: `Bearer ${apiKey}` },
      }
    );
    if (res.status === 401) {
      sessionStorage.removeItem(API_KEY_STORAGE);
      apiKey = "";
      lastError = "invalid API key";
    } else if (!res.ok) {
      lastError = `HTTP ${res.status}`;
    } else {
      lastData = await res.json();
      lastError = null;
    }
  } catch (e) {
    lastError = e.message;
  }

  const statusEl = document.getElementById("status-text");
  const extraEl = document.getElementById("status-extra");
  if (lastData) {
    const p = lastData.points[lastData.points.length - 1];
    statusEl.textContent =
      `last price ${p.p.toFixed(2)} · theta ${p.th.toFixed(3)} · ` +
      `${lastData.count} points over ${lastData.window_seconds}s`;
    extraEl.textContent =
      `ref range ${lastData.reference_low.toFixed(2)} – ${lastData.reference_high.toFixed(2)}`;
  } else if (lastError) {
    statusEl.textContent = "error: " + lastError;
    extraEl.textContent = "";
  }

  draw();
}

// -- Toggles ----------------------------------------------------------------

document.querySelectorAll(".toggle").forEach((btn) => {
  btn.addEventListener("click", () => {
    const fn = btn.dataset.fn;
    enabled[fn] = !enabled[fn];
    btn.classList.toggle("on", enabled[fn]);
    draw();
  });
});

// -- Boot -------------------------------------------------------------------

resizeCanvas();
poll();
setInterval(poll, POLL_INTERVAL_MS);
