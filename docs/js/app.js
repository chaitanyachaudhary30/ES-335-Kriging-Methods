/* Interactive figures for the kriging project website.
   All data come from precomputed JSON files in docs/data/ (aggregates only). */
"use strict";

const COLORS = { soil: "#b5562c", water: "#2c6fb5", grey: "#9aa3b2", ok: "#2e7d4f", ink: "#1d2433" };
const PLOT_CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
const BASE_LAYOUT = {
  margin: { l: 58, r: 18, t: 36, b: 100 },
  font: { family: "Segoe UI, system-ui, sans-serif", size: 13, color: COLORS.ink },
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#fff",
  legend: { orientation: "h", y: -0.32, yanchor: "top" },
};
const METHOD_LABEL = {
  "Mean": "Mean", "LinReg (coords)": "LinReg (coords)", "LinReg (coords + covariates)": "LinReg (coords+climate)",
  "Random Forest": "Random Forest", "OK": "Ordinary Kriging", "GPR": "GPR", "UK": "Universal Kriging",
  "RK (linear)": "Regression Kriging (linear)", "RK (RF)": "Regression Kriging (RF)",
};

const state = { ds: "gemas", site: null, stats: null, points: null, preds: null, map: null, markers: [] };

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const range = (lo, hi, step) => { const out = []; for (let x = lo; x <= hi + 1e-9; x += step) out.push(+x.toFixed(6)); return out; };

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function loadDataset(ds) {
  state.ds = ds;
  const [stats, points, preds] = await Promise.all([
    getJSON(`data/${ds}_stats.json`), getJSON(`data/${ds}_split_points.json`), getJSON(`data/${ds}_predictions.json`),
  ]);
  Object.assign(state, { stats, points, preds });
  renderData();
  resetExplainer();
  renderSplit();
  renderPredictions();
  renderCheck();
}

/* ---------- Section 2: data ---------- */
function renderData() {
  const s = state.stats.summary;
  $("g-n").textContent = s.n.toLocaleString();
  $("g-nn").textContent = fmt(s.nn_km.median, 0);
  $("g-dens").textContent = fmt(s.density_per_1000km2, 2);

  const body = document.querySelector("#row-table tbody");
  body.innerHTML = state.site.row_counts
    .map((r) => `<tr><td>${r.dataset}</td><td>${r.step}</td><td>${r.rows.toLocaleString()}</td><td>${r.removed || ""}</td><td>${r.note || ""}</td></tr>`)
    .join("");

  const e = state.stats.ph_bin_edges;
  const centers = e.slice(0, -1).map((x, i) => (x + e[i + 1]) / 2);
  Plotly.react("ph-hist", [{ type: "bar", x: centers, y: state.stats.ph_hist, marker: { color: COLORS.soil }, hovertemplate: "pH %{x:.2f}: %{y} samples<extra></extra>" }],
    { ...BASE_LAYOUT, title: { text: "Topsoil pH (CaCl₂)", font: { size: 14 } }, xaxis: { title: "pH" }, yaxis: { title: "samples" }, bargap: 0.05 }, PLOT_CFG);
}

/* ---------- Section 4: variogram explainer ---------- */
const corr = (h, a) => Math.exp((-3 * h) / a);           // exponential model, r(a) = 0.05
const gammaNorm = (h, eta, a) => (h === 0 ? 0 : eta + (1 - eta) * (1 - corr(h, a)));
const rmseRatio = (d, eta, a) => Math.sqrt(1 - (1 - eta) ** 2 * corr(d, a) ** 2);

function phFit() { return state.stats.variograms[0]; }
function bestFit(v) { return v.fits.find((f) => f.model === v.best); }

function resetExplainer() {
  const f = bestFit(phFit());
  $("sl-eta").value = f.nugget_ratio.toFixed(2);
  $("sl-range").value = Math.round(f.practical_range / 10) * 10;
  updateExplainer();
}

function splitDistances() {
  const rows = state.site.split_summary.filter((r) => r.dataset.toLowerCase() === state.ds);
  const m = (split, key) => mean(rows.filter((r) => r.split === split).map((r) => r[key]));
  return { random: m("random", "test_nn_train_km_median"), spatial: m("spatial", "test_nn_train_km_median"), spatialP90: m("spatial", "test_nn_train_km_p90") };
}

function updateExplainer() {
  const eta = +$("sl-eta").value, a = +$("sl-range").value, d = +$("sl-d").value;
  $("o-eta").textContent = eta.toFixed(2); $("o-range").textContent = a; $("o-d").textContent = d;

  const h = range(0, 1000, 5);
  const traces = [{ x: h, y: h.map((x) => gammaNorm(x, eta, a)), mode: "lines", name: "your variogram", line: { color: COLORS.soil, width: 3 } }];
  if ($("cb-emp").checked) {
    const v = phFit(), f = bestFit(v);
    traces.push({ x: v.empirical.h_km, y: v.empirical.gamma.map((g) => g / f.sill), mode: "markers", name: `${state.ds.toUpperCase()} data`,
      marker: { color: COLORS.ink, size: 7 }, customdata: v.empirical.pairs, hovertemplate: "h=%{x:.0f} km<br>γ/sill=%{y:.2f}<br>%{customdata} pairs<extra></extra>" });
    traces.push({ x: v.curves[v.best].h_km, y: v.curves[v.best].gamma.map((g) => g / f.sill), mode: "lines", name: `fitted ${v.best.toLowerCase()}`, line: { color: COLORS.ink, dash: "dot" } });
  }
  Plotly.react("vario-plot", traces, {
    ...BASE_LAYOUT, title: { text: "Variogram: how different are samples h km apart?", font: { size: 14 } },
    xaxis: { title: "distance h (km)", range: [0, 1000] }, yaxis: { title: "semivariance / sill", range: [0, 1.25] },
    shapes: [
      { type: "line", x0: 0, x1: 1000, y0: 1, y1: 1, line: { color: COLORS.grey, dash: "dash", width: 1 } },
      { type: "line", x0: a, x1: a, y0: 0, y1: 1.25, line: { color: COLORS.soil, dash: "dot", width: 1 } },
    ],
    annotations: [
      { x: 8, y: eta, text: "nugget", showarrow: true, ax: 40, ay: 25, font: { size: 11 } },
      { x: a, y: 1.18, text: "range", showarrow: false, font: { size: 11, color: COLORS.soil } },
    ],
  }, PLOT_CFG);

  const dd = range(0, 800, 5), sd = splitDistances();
  const vline = (x) => ({ type: "line", x0: x, x1: x, y0: 0, y1: 1.05, line: { color: COLORS.grey, width: 1.5, dash: "dash" } });
  const vlabel = (x, text, y) => ({ x, y, text, showarrow: false, xanchor: "left", xshift: 4, font: { size: 10, color: COLORS.muted || "#5b6577" }, bgcolor: "rgba(255,255,255,0.85)" });
  Plotly.react("gain-plot", [
    { x: dd, y: dd.map((x) => rmseRatio(x, eta, a)), mode: "lines", name: "best kriging error / error of the average", line: { color: COLORS.water, width: 3 } },
    { x: [d], y: [rmseRatio(d, eta, a)], mode: "markers", name: "your d", marker: { size: 12, color: COLORS.soil } },
    { x: [0, 800], y: [Math.sqrt(eta), Math.sqrt(eta)], mode: "lines", name: "nugget cap √η", line: { color: COLORS.soil, dash: "dot" } },
  ], {
    ...BASE_LAYOUT, title: { text: "How much can one neighbour d km away help?", font: { size: 14 } },
    xaxis: { title: "distance to nearest training sample d (km)" }, yaxis: { title: "RMSE ratio (1 = no gain)", range: [0, 1.05] },
    shapes: [vline(sd.random), vline(sd.spatial), vline(sd.spatialP90)],
    annotations: [
      vlabel(sd.random, `random split: ${sd.random.toFixed(0)} km`, 0.42),
      vlabel(sd.spatial, `spatial, median: ${sd.spatial.toFixed(0)} km`, 0.30),
      vlabel(sd.spatialP90, `spatial, 90th pct: ${sd.spatialP90.toFixed(0)} km`, 0.18),
    ],
  }, PLOT_CFG);

  const r = rmseRatio(d, eta, a);
  $("gain-readout").innerHTML = `With η = <b>${eta.toFixed(2)}</b>, range <b>${a} km</b> and the nearest sample <b>${d} km</b> away, kriging can at best cut the error of the average by <b>${((1 - r) * 100).toFixed(1)}%</b>. ` +
    `Even with a sample right next door, the nugget caps the cut at <b>${((1 - Math.sqrt(eta)) * 100).toFixed(0)}%</b>. ` +
    (d >= a ? "You are beyond the range: kriging just returns the average." : `Correlation at this distance: r(d) = ${corr(d, a).toFixed(2)}.`);
}

/* ---------- Section 5: split map ---------- */
function initMap() {
  state.map = L.map("split-map", { preferCanvas: true, scrollWheelZoom: false }).setView([52, 13], 3);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 8, attribution: "© OpenStreetMap contributors" }).addTo(state.map);
}

function currentSplit() { return document.querySelector("input[name=split]:checked").value; }

function renderSplit() {
  if (!state.map) initMap();
  state.markers.forEach((m) => m.remove());
  state.markers = [];
  const p = state.points, split = currentSplit(), fold = +$("sel-fold").value;
  const folds = p[`${split}_fold`], dist = p[`${split}_dist_km`];
  for (let i = 0; i < p.lat.length; i++) {
    const test = folds[i] === fold;
    const m = L.circleMarker([p.lat[i], p.lon[i]], {
      radius: test ? 4 : 2.5, weight: 0, fillOpacity: test ? 0.9 : 0.55, fillColor: test ? COLORS.soil : COLORS.water,
    });
    if (test) m.bindTooltip(`test sample · nearest training sample ${dist[i].toFixed(0)} km away`);
    m.addTo(state.map);
    state.markers.push(m);
  }

  const toHist = (arr, name, color) => ({ type: "histogram", x: arr, name, opacity: 0.65, marker: { color }, xbins: { start: 0, end: 500, size: 10 }, histnorm: "percent" });
  const v = bestFit(phFit());
  const dd = range(0, 500, 5);
  const testDist = dist.filter((_, i) => folds[i] === fold);
  Plotly.react("dist-plot", [
    toHist(p.random_dist_km, "random split", COLORS.grey),
    toHist(p.spatial_dist_km, "spatial blocks", COLORS.soil),
    { x: dd, y: dd.map((x) => corr(x, v.practical_range)), mode: "lines", name: "correlation r(d), exponential fit (right axis)", yaxis: "y2", line: { color: COLORS.water, width: 2 } },
  ], {
    ...BASE_LAYOUT, barmode: "overlay", margin: { ...BASE_LAYOUT.margin, r: 50 },
    title: { text: "Distance from each test sample to its nearest training sample", font: { size: 14 } },
    xaxis: { title: "distance (km)", range: [0, 500] }, yaxis: { title: "% of test samples" },
    yaxis2: { title: "correlation r(d)", overlaying: "y", side: "right", range: [0, 1.05], showgrid: false },
  }, PLOT_CFG);

  const md = median(testDist), r = corr(md, v.practical_range), eta = v.nugget_ratio;
  $("split-readout").innerHTML = `<b>${split === "random" ? "Random" : "Spatial-block"} split, fold ${fold + 1}:</b> ${testDist.length} test samples (orange), median distance to the nearest training sample (blue) <b>${md.toFixed(0)} km</b>. ` +
    `At that distance the fitted correlation is ${r.toFixed(2)}, so one neighbour could cut the error by at most <b>${((1 - Math.sqrt(1 - (1 - eta) ** 2 * r ** 2)) * 100).toFixed(0)}%</b>.`;
}

/* ---------- Section 6: predicted skill ---------- */
function renderPredictions() {
  const sum = state.preds.summary;
  const order = sum.filter((r) => r.split === "spatial").sort((a, b) => b.pred_skill_mean - a.pred_skill_mean).map((r) => r.method).filter((m) => m !== "Mean");
  const bar = (split, color, name) => {
    const rows = order.map((m) => sum.find((r) => r.split === split && r.method === m));
    return { type: "bar", name, x: order.map((m) => METHOD_LABEL[m] || m), y: rows.map((r) => r.pred_skill_mean),
      error_y: { type: "data", array: rows.map((r) => r.pred_skill_sd), visible: true, color: COLORS.ink, thickness: 1 },
      marker: { color }, hovertemplate: "%{x}<br>predicted skill %{y:.3f}<extra>" + name + "</extra>" };
  };
  const st = state.preds.training_stats.filter((r) => r.split === "spatial");
  const cap = 1 - Math.sqrt(mean(st.map((r) => r.nugget_ratio)));
  Plotly.react("pred-plot", [bar("spatial", COLORS.soil, "held-out regions (spatial split)"), bar("random", COLORS.grey, "random split")], {
    ...BASE_LAYOUT, barmode: "group", margin: { ...BASE_LAYOUT.margin, l: 90, b: 140 },
    yaxis: { title: "predicted skill", range: [0, Math.max(0.45, cap + 0.05)] }, xaxis: { tickangle: -25 },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: cap, y1: cap, line: { dash: "dash", color: COLORS.ink, width: 1 } }],
    annotations: [{ xref: "paper", x: 1, y: cap, text: `nugget cap 1−√η ≈ ${cap.toFixed(2)}`, showarrow: false, yanchor: "bottom", xanchor: "right", font: { size: 11 } }],
  }, PLOT_CFG);
}

/* ---------- Section 6: baseline check ---------- */
function renderCheck() {
  const ds = state.site.datasets[state.ds];
  if (!ds) return;
  const methods = ["LinReg (coords)", "LinReg (coords + covariates)", "Random Forest"];
  const symbols = { spatial: "circle", random: "diamond" };
  const traces = [];
  let maxErr = 0;
  methods.forEach((m, k) => {
    ["spatial", "random"].forEach((split) => {
      const pts = state.preds.per_fold.filter((r) => r.method === m && r.split === split).map((r) => {
        const obs = ds.baseline_metrics.find((b) => b.method === m && b.split === split && b.fold === r.fold);
        maxErr = Math.max(maxErr, Math.abs(obs.skill - r.pred_skill));
        return { x: r.pred_skill, y: obs.skill, fold: r.fold + 1 };
      });
      traces.push({ x: pts.map((p) => p.x), y: pts.map((p) => p.y), mode: "markers", name: `${METHOD_LABEL[m]} · ${split}`,
        marker: { size: 10, symbol: symbols[split], color: [COLORS.water, COLORS.soil, COLORS.ok][k], line: { width: 1, color: "#fff" } },
        customdata: pts.map((p) => p.fold), hovertemplate: "fold %{customdata}<br>predicted %{x:.3f}<br>observed %{y:.3f}<extra>%{fullData.name}</extra>" });
    });
  });
  traces.push({ x: [-0.05, 0.4], y: [-0.05, 0.4], mode: "lines", name: "prediction = outcome", line: { dash: "dash", color: COLORS.grey } });
  Plotly.react("check-plot", traces, {
    ...BASE_LAYOUT, margin: { ...BASE_LAYOUT.margin, b: 110 },
    xaxis: { title: "predicted skill (training data only)", range: [-0.05, 0.4] }, yaxis: { title: "observed test skill", range: [-0.05, 0.4] },
  }, PLOT_CFG);
  const mae = (split) => {
    const errs = [];
    methods.forEach((m) => state.preds.per_fold.filter((r) => r.method === m && r.split === split).forEach((r) => {
      const obs = ds.baseline_metrics.find((b) => b.method === m && b.split === split && b.fold === r.fold);
      errs.push(Math.abs(obs.skill - r.pred_skill));
    }));
    return mean(errs);
  };
  $("check-readout").innerHTML = `Each point is one test fold. Mean absolute gap between predicted and observed skill: <b>${mae("spatial").toFixed(3)}</b> on held-out regions, <b>${mae("random").toFixed(3)}</b> on the random split (largest single gap ${maxErr.toFixed(3)}). ` +
    `The kriging predictions are checked the same way in Phase 2.`;
}

/* ---------- wiring ---------- */
document.addEventListener("DOMContentLoaded", async () => {
  ["sl-eta", "sl-range", "sl-d"].forEach((id) => $(id).addEventListener("input", updateExplainer));
  $("cb-emp").addEventListener("change", updateExplainer);
  $("btn-fit").addEventListener("click", resetExplainer);
  document.querySelectorAll("input[name=split]").forEach((el) => el.addEventListener("change", renderSplit));
  $("sel-fold").addEventListener("change", renderSplit);
  document.querySelectorAll(".ds-btn").forEach((b) => b.addEventListener("click", () => {
    if (b.disabled) return;
    document.querySelectorAll(".ds-btn").forEach((x) => x.classList.toggle("active", x === b));
    loadDataset(b.dataset.ds);
  }));
  try {
    state.site = await getJSON("data/site.json");
    if (state.site.datasets.lucas) { const lb = document.querySelector('.ds-btn[data-ds="lucas"]'); lb.disabled = false; lb.textContent = "LUCAS 2018 (dense)"; }
    await loadDataset("gemas");
  } catch (err) {
    console.error(err);
    document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="readout">Could not load data files (${err.message}). If you opened this file directly, serve the folder over HTTP instead, e.g. <code>python -m http.server -d docs</code>.</p>`);
  }
});
