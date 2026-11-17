/* "General user understanding" section: a made-up country with a simulated soil map.
   Nothing here uses the GEMAS samples except the final scoreboard, which reads the
   published per-method averages (data/gemas_results.json, aggregates only).

   The simulated soil is a smooth random field built from random cosine waves
   (random Fourier features), whose covariance is C(h) = exp(-h^2 / (2 L^2)).
   Measurements add independent noise with variance tau^2. "Smart averaging" is
   simple kriging with exactly that covariance, using the 12 nearest samples. */
"use strict";

(() => {
  const G = {
    W: 1000,            // the made-up country is 1000 km x 1000 km
    CELLS: 90,          // resolution of the drawn soil map
    MU: 6.2,            // average pH of the made-up country
    SD: 1.0,            // spread of pH (standard deviation of the smooth part)
    K: 12,              // neighbours used by smart averaging
    NFEAT: 250,         // cosine waves in the simulated field
    seed: 7,
    L: 220 / 2.45,      // length scale (km); the slider shows the similarity distance 2.45 L
    eta: 0.3,           // share of variation that is random noise; slider
    gap: 0,             // side of the hidden square (km); slider
    showTruth: false,
    click: null,
  };
  const COL = { soil: "#b5562c", water: "#2c6fb5", grey: "#9aa3b2", ok: "#2e7d4f", ink: "#1d2433" };
  const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  const LAY = { margin: { l: 56, r: 16, t: 34, b: 56 }, font: { family: "Segoe UI, system-ui, sans-serif", size: 13, color: COL.ink },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#fff", legend: { orientation: "h", y: -0.28, yanchor: "top" } };
  const $ = (id) => document.getElementById(id);

  /* ---------- random numbers (seeded, so every visitor sees the same country) ---------- */
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  const gauss = (r) => { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

  /* ---------- the simulated world ---------- */
  let world = null;
  function makeWorld() {
    const r = rng(G.seed * 1000 + 1);
    const feats = [];
    for (let m = 0; m < G.NFEAT; m++) feats.push({ wx: gauss(r) / G.L, wy: gauss(r) / G.L, b: 2 * Math.PI * r() });
    const amp = Math.sqrt(2 / G.NFEAT);
    const field = (x, y) => { let s = 0; for (const f of feats) s += Math.cos(f.wx * x + f.wy * y + f.b); return G.MU + G.SD * amp * s; };

    // Survey: 12 x 12 sites on a jittered grid (about one site every 83 km).
    const rs = rng(G.seed * 1000 + 2), n = 12, step = G.W / n, sites = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = (i + 0.2 + 0.6 * rs()) * step, y = (j + 0.2 + 0.6 * rs()) * step;
      sites.push({ x, y, truth: field(x, y), e: gauss(rs) });
    }
    // Evaluation spots: a fine grid with their own measurement noise.
    const re = rng(G.seed * 1000 + 3), ev = [];
    for (let i = 0; i < 40; i++) for (let j = 0; j < 40; j++) {
      const x = (i + 0.5) * G.W / 40, y = (j + 0.5) * G.W / 40;
      ev.push({ x, y, truth: field(x, y), e: gauss(re) });
    }
    const grid = new Float32Array(G.CELLS * G.CELLS);
    for (let i = 0; i < G.CELLS; i++) for (let j = 0; j < G.CELLS; j++) grid[j * G.CELLS + i] = field((i + 0.5) * G.W / G.CELLS, (j + 0.5) * G.W / G.CELLS);
    world = { field, sites, ev, grid };
  }
  const tau = () => Math.sqrt(G.eta / (1 - G.eta)) * G.SD;     // noise SD from the noise share
  const measured = (s) => s.truth + tau() * s.e;                 // what the lab would report

  /* ---------- predictions ---------- */
  const cov = (h) => G.SD * G.SD * Math.exp(-(h * h) / (2 * G.L * G.L));
  function solve(A, b) {           // Gaussian elimination with partial pivoting (small systems)
    const n = b.length, M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }
  function predictor(train) {
    const vals = train.map(measured), mean = vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1);
    const t2 = tau() ** 2 + 1e-4;
    return (x, y) => {
      const d = train.map((s, i) => ({ i, d: Math.hypot(s.x - x, s.y - y) })).sort((a, b) => a.d - b.d);
      const nb = d.slice(0, Math.min(G.K, d.length));
      const A = nb.map((a) => nb.map((b) => cov(Math.hypot(train[a.i].x - train[b.i].x, train[a.i].y - train[b.i].y)) + (a.i === b.i ? t2 : 0)));
      const c0 = nb.map((a) => cov(a.d));
      const lam = solve(A, c0);
      let krig = mean, v = G.SD * G.SD + tau() ** 2;
      nb.forEach((a, k) => { krig += lam[k] * (vals[a.i] - mean); v -= lam[k] * c0[k]; });
      return { mean, nearest: vals[nb[0].i], nearestD: nb[0].d, nearestSite: train[nb[0].i], krig, sd: Math.sqrt(Math.max(v, 0)), weights: nb.map((a, k) => ({ s: train[a.i], w: lam[k] })) };
    };
  }
  const inGap = (p, gap, cx = G.W / 2, cy = G.W / 2) => gap > 0 && Math.abs(p.x - cx) <= gap / 2 && Math.abs(p.y - cy) <= gap / 2;
  const improvement = (rm, rb) => 1 - rm / rb;

  /* Score a hidden square of side `gap`. One square in one place gives a noisy answer
     (it depends on whether that patch happens to be close to the average), so the square
     is moved to 9 positions across the country and the squared errors are pooled. */
  const EV_STEP = 2;   // use every 2nd evaluation spot (a 20 x 20 grid) to keep this fast
  function scoreGap(gap) {
    let sm = 0, sn = 0, sk = 0, nTest = 0;
    const centres = [];
    if (gap === 0) centres.push([G.W / 2, G.W / 2]);
    else for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const lo = gap / 2, span = G.W - gap;
      centres.push([lo + span * (i + 0.5) / 3, lo + span * (j + 0.5) / 3]);
    }
    const ev = world.ev.filter((_, i) => (i % 40) % EV_STEP === 0 && Math.floor(i / 40) % EV_STEP === 0);
    for (const [cx, cy] of centres) {
      const P = predictor(world.sites.filter((s) => !inGap(s, gap, cx, cy)));
      for (const p of gap > 0 ? ev.filter((e) => inGap(e, gap, cx, cy)) : ev) {
        const y = measured(p), q = P(p.x, p.y);
        sm += (y - q.mean) ** 2; sn += (y - q.nearest) ** 2; sk += (y - q.krig) ** 2; nTest++;
      }
    }
    return { krig: improvement(Math.sqrt(sk), Math.sqrt(sm)), nearest: improvement(Math.sqrt(sn), Math.sqrt(sm)), nTest };
  }

  /* ---------- drawing ---------- */
  function phColor(v) {           // acidic = red/orange, neutral = yellow, alkaline = blue
    const t = Math.min(1, Math.max(0, (v - 4) / 4.4));
    const stops = [[178, 24, 43], [239, 138, 98], [253, 219, 160], [209, 229, 240], [103, 169, 207], [33, 102, 172]];
    const s = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(s)), f = s - i;
    return stops[i].map((c, k) => Math.round(c + f * (stops[i + 1][k] - c)));
  }
  function draw() {
    const cv = $("gu-map"), ctx = cv.getContext("2d"), S = cv.width, k = S / G.W, cell = S / G.CELLS;
    ctx.clearRect(0, 0, S, S);
    if (G.showTruth) {
      for (let i = 0; i < G.CELLS; i++) for (let j = 0; j < G.CELLS; j++) {
        const [r, g, b] = phColor(world.grid[j * G.CELLS + i]); ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(i * cell, S - (j + 1) * cell, cell + 0.6, cell + 0.6);
      }
    } else {
      ctx.fillStyle = "#eef0f4"; ctx.fillRect(0, 0, S, S);
      ctx.fillStyle = "#8a93a3"; ctx.font = "14px Segoe UI, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("The real soil between the dots is unknown", S / 2, 22);
    }
    if (G.gap > 0) {
      const g = G.gap * k, x0 = (G.W / 2 - G.gap / 2) * k, y0 = S - (G.W / 2 + G.gap / 2) * k;
      ctx.fillStyle = "rgba(29,36,51,0.18)"; ctx.fillRect(x0, y0, g, g);
      ctx.strokeStyle = COL.ink; ctx.setLineDash([6, 4]); ctx.lineWidth = 2; ctx.strokeRect(x0, y0, g, g); ctx.setLineDash([]);
      ctx.fillStyle = COL.ink; ctx.font = "bold 13px Segoe UI, sans-serif"; ctx.textAlign = "left"; ctx.fillText("no samples here", x0 + 6, y0 + 16);
    }
    if (G.click && G.click.q) {   // lines to the neighbours used, thicker = more weight
      const { x, y, q } = G.click;
      for (const { s, w } of q.weights) {
        if (Math.abs(w) < 0.02) continue;
        ctx.strokeStyle = `rgba(44,111,181,${Math.min(0.9, 0.25 + Math.abs(w))})`; ctx.lineWidth = 1 + 6 * Math.abs(w);
        ctx.beginPath(); ctx.moveTo(x * k, S - y * k); ctx.lineTo(s.x * k, S - s.y * k); ctx.stroke();
      }
    }
    for (const s of world.sites) {
      const hidden = inGap(s, G.gap), [r, g, b] = phColor(measured(s));
      ctx.beginPath(); ctx.arc(s.x * k, S - s.y * k, hidden ? 3 : 5, 0, 2 * Math.PI);
      if (hidden) { ctx.strokeStyle = "#7d8696"; ctx.lineWidth = 1; ctx.stroke(); }
      else { ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fill(); ctx.strokeStyle = COL.ink; ctx.lineWidth = 1; ctx.stroke(); }
    }
    if (G.click) {
      const { x, y } = G.click;
      ctx.fillStyle = "#fff"; ctx.strokeStyle = COL.ink; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x * k, S - y * k, 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = COL.ink; ctx.font = "bold 11px Segoe UI, sans-serif"; ctx.textAlign = "center"; ctx.fillText("?", x * k, S - y * k + 4);
    }
  }

  function guessReadout() {
    const el = $("gu-guess");
    if (!G.click) { el.innerHTML = "Click anywhere on the map to pick a spot that has <b>not</b> been measured."; return; }
    const train = world.sites.filter((s) => !inGap(s, G.gap));
    const q = predictor(train)(G.click.x, G.click.y); G.click.q = q;
    const truth = world.field(G.click.x, G.click.y);
    const err = (v) => Math.abs(v - truth).toFixed(2);
    const best = [["mean", q.mean], ["nearest", q.nearest], ["krig", q.krig]].sort((a, b) => Math.abs(a[1] - truth) - Math.abs(b[1] - truth))[0][0];
    const row = (key, name, how, v) => `<tr class="${best === key ? "gu-best" : ""}"><td>${name}</td><td>${how}</td><td><b>${v.toFixed(2)}</b></td><td>${err(v)}</td></tr>`;
    el.innerHTML = `<p>Nearest measured spot: <b>${q.nearestD.toFixed(0)} km</b> away. Soil here stays similar over about <b>${(2.45 * G.L).toFixed(0)} km</b>.</p>
      <table class="data compact"><tr><th>Guess</th><th>How it guesses</th><th>pH</th><th>off by</th></tr>
      ${row("mean", "1. The average", "same number everywhere", q.mean)}
      ${row("nearest", "2. Copy the nearest", "take the closest dot's value", q.nearest)}
      ${row("krig", "3. Smart averaging (kriging)", "weigh nearby dots by how related they are (blue lines)", q.krig)}
      <tr><td colspan="2"><b>The real soil at this spot</b></td><td><b>${truth.toFixed(2)}</b></td><td></td></tr></table>
      <p class="caption">Smart averaging also says how unsure it is: about ± ${(1.96 * q.sd).toFixed(2)} pH (95%). ${q.nearestD > 2.45 * G.L ? "This spot is farther than the similarity distance, so smart averaging has fallen back to almost exactly the average." : ""}</p>`;
  }

  function gapCurve() {
    const sizes = []; for (let g = 0; g <= 800; g += 50) sizes.push(g);
    const sc = sizes.map(scoreGap), cur = scoreGap(G.gap);
    Plotly.react("gu-curve", [
      { x: sizes, y: sc.map((s) => 100 * s.krig), mode: "lines+markers", name: "3. smart averaging (kriging)", line: { color: COL.water, width: 3 } },
      { x: sizes, y: sc.map((s) => 100 * s.nearest), mode: "lines+markers", name: "2. copy the nearest", line: { color: COL.soil, width: 2, dash: "dot" } },
      { x: [G.gap], y: [100 * cur.krig], mode: "markers", name: "your gap", marker: { size: 14, color: COL.ink, symbol: "diamond" } },
    ], { ...LAY, title: { text: "How much better than just guessing the average?", font: { size: 14 } },
      xaxis: { title: "size of the unsampled region (km across)" }, yaxis: { title: "% better than the average", zeroline: true, zerolinewidth: 2 },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 100 * (1 - Math.sqrt(G.eta)), y1: 100 * (1 - Math.sqrt(G.eta)), line: { dash: "dash", color: COL.grey } }],
      annotations: [{ xref: "paper", x: 1, y: 100 * (1 - Math.sqrt(G.eta)), xanchor: "right", yanchor: "bottom", showarrow: false, text: "best possible, set by the noise", font: { size: 11, color: "#5b6577" } }],
    }, CFG);
    const pct = (v) => `${v >= 0 ? "" : "−"}${Math.abs(100 * v).toFixed(0)}%`;
    $("gu-gap-readout").innerHTML = G.gap === 0
      ? `With no region hidden, smart averaging is <b>${pct(cur.krig)}</b> better than the average, and copying the nearest dot is <b>${pct(cur.nearest)}</b> better. Now make the hidden region bigger.`
      : `Inside a hidden ${G.gap} km region (averaged over 9 positions of the square across the country), smart averaging is <b>${pct(cur.krig)}</b> better than the average, and copying the nearest dot is <b>${Math.abs(100 * cur.nearest).toFixed(0)}%</b> ${cur.nearest >= 0 ? "better" : "<b>worse</b>"}. ` +
        (cur.krig < 0.05 ? "The neighbours are too far away to help, so smart averaging gives back the average." : "Some neighbours are still close enough to help.");
  }

  function fairTest() {
    const r = rng(G.seed * 1000 + 9), sites = world.sites, nHold = Math.round(0.2 * sites.length);
    const evalHold = (holdout) => {
      const P = predictor(sites.filter((s) => !holdout.has(s)));
      let sm = 0, sk = 0; for (const s of holdout) { const y = measured(s), q = P(s.x, s.y); sm += (y - q.mean) ** 2; sk += (y - q.krig) ** 2; }
      return improvement(Math.sqrt(sk), Math.sqrt(sm));
    };
    const rand = [], region = [];
    for (let rep = 0; rep < 5; rep++) {
      const idx = sites.map((_, i) => i);
      for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }   // Fisher-Yates shuffle
      idx.length = nHold;
      rand.push(evalHold(new Set(idx.map((i) => sites[i]))));
    }
    const side = Math.sqrt(0.2) * G.W;   // a square region that holds about 20% of the sites
    for (const [cx, cy] of [[side / 2, side / 2], [G.W - side / 2, side / 2], [side / 2, G.W - side / 2], [G.W - side / 2, G.W - side / 2], [G.W / 2, G.W / 2]])
      region.push(evalHold(new Set(sites.filter((s) => inGap(s, side, cx, cy)))));
    const m = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    Plotly.react("gu-fair", [{ type: "bar", x: ["Hide random dots", "Hide a whole region"], y: [100 * m(rand), 100 * m(region)], marker: { color: [COL.grey, COL.soil] },
      text: [m(rand), m(region)].map((v) => `${(100 * v).toFixed(0)}%`), textposition: "outside", hovertemplate: "%{x}: %{y:.1f}%<extra></extra>" }],
    { ...LAY, title: { text: "Same method, same map, two ways of testing it", font: { size: 14 } }, yaxis: { title: "% better than the average" }, showlegend: false }, CFG);
    $("gu-fair-readout").innerHTML = `Hiding <b>random</b> dots makes smart averaging look <b>${(100 * m(rand)).toFixed(0)}%</b> better than the average, because every hidden dot still has neighbours close by. Hiding a <b>whole region</b> (the situation on a real map with an unsampled area) gives only <b>${(100 * m(region)).toFixed(0)}%</b>. Same method, same data, a different score.`;
  }

  async function scoreboard() {
    try {
      const res = await (await fetch("data/gemas_results.json")).json();
      const name = { "RK (linear)": "Regression Kriging (linear)", UK: "Universal Kriging", OK: "Ordinary Kriging", "Random Forest": "Random Forest", GPR: "Gaussian Process", "RK (RF)": "Regression Kriging (forest)", "LinReg (coords + covariates)": "Straight-line fit (place + climate)", "LinReg (coords)": "Straight-line fit (place only)" };
      const rows = res.summary.filter((r) => r.split === "spatial" && r.method !== "Mean").sort((a, b) => a.skill_mean - b.skill_mean);
      Plotly.react("gu-real", [{ type: "bar", orientation: "h", y: rows.map((r) => name[r.method] || r.method), x: rows.map((r) => 100 * r.skill_mean),
        error_x: { type: "data", array: rows.map((r) => 100 * r.skill_sd), color: COL.ink, thickness: 1 },
        marker: { color: rows.map((r, i) => (i >= rows.length - 5 ? COL.water : COL.grey)) },
        hovertemplate: "%{y}: %{x:.0f}% better than the average<extra></extra>" }],
      { ...LAY, margin: { ...LAY.margin, l: 230 }, title: { text: "Real European soils: % better than guessing the average in unseen regions", font: { size: 14 } },
        xaxis: { title: "% better than the average (bars: mean of 5 tests; whiskers: spread between tests)" }, showlegend: false }, CFG);
      const top = rows.slice(-5), lo = Math.min(...top.map((r) => r.skill_mean)), hi = Math.max(...top.map((r) => r.skill_mean));
      const sd = Math.min(...top.map((r) => r.skill_sd));
      $("gu-real-readout").innerHTML = `On real data, the five best methods (blue) are all about <b>${(100 * lo).toFixed(0)}–${(100 * hi).toFixed(0)}%</b> better than the average. ` +
        `Their differences (${(100 * (hi - lo)).toFixed(1)} points) are smaller than how much each one varies between tests (about ±${(100 * sd).toFixed(0)} points), so we cannot name a single winner. The complicated methods did not clearly beat the simpler ones.`;
    } catch (e) { $("gu-real-readout").textContent = "Could not load the results file."; }
  }

  /* ---------- wiring ---------- */
  function refresh(all = true) {
    $("o-gu-L").textContent = `${(2.45 * G.L).toFixed(0)} km`;
    $("o-gu-eta").textContent = `${Math.round(100 * G.eta)}%`;
    $("o-gu-gap").textContent = G.gap === 0 ? "none" : `${G.gap} km`;
    guessReadout(); draw();
    if (all) { gapCurve(); fairTest(); }
  }
  let timer = null;
  const later = () => { clearTimeout(timer); timer = setTimeout(() => refresh(true), 120); };

  document.addEventListener("DOMContentLoaded", () => {
    if (!$("gu-map")) return;
    makeWorld();
    $("gu-truth").addEventListener("change", (e) => { G.showTruth = e.target.checked; draw(); });
    $("sl-gu-L").addEventListener("input", (e) => { G.L = +e.target.value / 2.45; makeWorld(); refresh(false); later(); });
    $("sl-gu-eta").addEventListener("input", (e) => { G.eta = +e.target.value; refresh(false); later(); });
    $("sl-gu-gap").addEventListener("input", (e) => { G.gap = +e.target.value; refresh(false); later(); });
    $("gu-new").addEventListener("click", () => { G.seed += 1; G.click = null; makeWorld(); refresh(true); });
    $("gu-map").addEventListener("click", (ev) => {
      const b = ev.target.getBoundingClientRect(), k = G.W / b.width;
      G.click = { x: (ev.clientX - b.left) * k, y: G.W - (ev.clientY - b.top) * k };
      guessReadout(); draw();
    });
    refresh(true);
    scoreboard();
  });
})();
