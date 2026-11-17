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

const state = { ds: "gemas", site: null, stats: null, points: null, preds: null, res: null, map: null, markers: [] };

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
  const [stats, points, preds, res] = await Promise.all([
    getJSON(`data/${ds}_stats.json`), getJSON(`data/${ds}_split_points.json`), getJSON(`data/${ds}_predictions.json`), getJSON(`data/${ds}_results.json`),
  ]);
  Object.assign(state, { stats, points, preds, res });
  renderData();
  resetExplainer();
  renderSplit();
  renderPredictions();
  renderCheck();
  renderResults();
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
    `The same check for all nine methods is in Section 8.`;
}

/* ================= Sections 7-9: results from data/<ds>_results.json ================= */
const ORDER = ["Mean", "LinReg (coords)", "LinReg (coords + covariates)", "Random Forest", "OK", "GPR", "UK", "RK (linear)", "RK (RF)"];
const MCOLOR = { "Mean": "#9aa3b2", "LinReg (coords)": "#c9a227", "LinReg (coords + covariates)": "#8c6d1f", "Random Forest": "#2e7d4f",
  "OK": "#2c6fb5", "GPR": "#6a4fb5", "UK": "#1b9aaa", "RK (linear)": "#b5562c", "RK (RF)": "#d1495b" };
const FRACTIONS = [1, 0.5, 0.25, 0.1];
const checked = (name) => document.querySelector(`input[name=${name}]:checked`).value;
const pm = (m, s, d = 3) => `${fmt(m, d)} ± ${fmt(s, d)}`;
const lab = (m) => METHOD_LABEL[m] || m;

function ranks(a) { // average ranks for ties
  const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
  const r = new Array(a.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
function spearman(x, y) {
  const rx = ranks(x), ry = ranks(y), mx = mean(rx), my = mean(ry);
  let sxy = 0, sxx = 0, syy = 0;
  rx.forEach((v, i) => { sxy += (v - mx) * (ry[i] - my); sxx += (v - mx) ** 2; syy += (ry[i] - my) ** 2; });
  return sxy / Math.sqrt(sxx * syy);
}
const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const summ = (split, m) => state.res.summary.find((r) => r.split === split && r.method === m);
const verdictClass = (v) => (v === "supported" ? "supported" : v === "not supported" ? "notsupported" : v === "inconclusive" ? "inconclusive" : "pending");

function renderResults() {
  if (!state.res) return;
  renderTuning(); renderResTable(); renderSkillBar(); renderScatter(); renderVerdicts();
  renderErrDist(); renderResMap(); renderDensity(); renderTables(); renderLimits();
}

/* ---------- Section 7: tuning table ---------- */
function renderTuning() {
  const grids = { "OK": "variogram {Exponential, Spherical, Matérn} × neighbours {16, 32, 64}",
    "UK": "drift degree {1, 2} × variogram × neighbours", "RK (linear)": "variogram × neighbours", "RK (RF)": "variogram × neighbours", "GPR": "Matérn ν {0.5, 1.5, 2.5}" };
  const tally = (rows, key) => {
    const c = {};
    rows.forEach((r) => { const v = r.settings[key]; if (v !== undefined) c[v] = (c[v] || 0) + 1; });
    return Object.entries(c).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} (${n})`).join(", ");
  };
  const lbl = { variogram: "variogram", k: "neighbours", degree: "drift degree", nu: "ν" };
  document.querySelector("#tune-table tbody").innerHTML = Object.keys(grids).map((m) => {
    const rows = state.res.tuning.filter((r) => r.method === m);
    const parts = (m === "GPR" ? ["nu"] : m === "UK" ? ["degree", "variogram", "k"] : ["variogram", "k"]).map((k) => `${lbl[k]}: ${tally(rows, k)}`);
    return `<tr><td>${lab(m)}</td><td>${grids[m]}</td><td>${parts.join("<br>")}</td></tr>`;
  }).join("");
}

/* ---------- Section 8: table, bars ---------- */
function renderResTable() {
  const split = checked("rsplit");
  const rows = ORDER.map((m) => summ(split, m)).sort((a, b) => a.obs_rank - b.obs_rank);
  document.querySelector("#res-table tbody").innerHTML = rows.map((r) =>
    `<tr${r.obs_rank === 1 ? ' style="background:#fff6f1"' : ""}><td>${lab(r.method)}</td><td>${pm(r.rmse_mean, r.rmse_sd)}</td><td>${fmt(r.mae_mean)}</td>` +
    `<td><b>${pm(r.skill_mean, r.skill_sd)}</b></td><td>${pm(r.pred_skill_mean, r.pred_skill_sd)}</td><td>${r.obs_rank}</td><td>${r.pred_rank}</td></tr>`).join("");
  const top = rows[0], close = rows.slice(1).filter((r) => top.skill_mean - r.skill_mean <= top.skill_sd && r.method !== "Mean").map((r) => lab(r.method));
  const rf = summ(split, "Random Forest"), ok = summ(split, "OK");
  $("res-headline").innerHTML = `${split === "spatial" ? "On held-out regions" : "On the random split"}, <b>${lab(top.method)}</b> has the highest mean skill (<b>${fmt(top.skill_mean)}</b>, fold SD ${fmt(top.skill_sd)}). ` +
    `${close.length} other method${close.length === 1 ? " is" : "s are"} within one fold SD of it (${close.join(", ") || "none"}). ` +
    `The non-spatial Random Forest reaches ${fmt(rf.skill_mean)}; Ordinary Kriging reaches ${fmt(ok.skill_mean)}; the average itself has RMSE ${fmt(summ(split, "Mean").rmse_mean)} pH units.`;
}

function renderSkillBar() {
  const split = checked("rsplit");
  const ms = ORDER.filter((m) => m !== "Mean");
  const rows = ms.map((m) => summ(split, m));
  const perFold = (m) => state.res.metrics.filter((r) => r.split === split && r.method === m).map((r) => r.skill);
  const x = ms.map(lab);
  Plotly.react("skill-bar", [
    { type: "bar", name: "observed (test)", x, y: rows.map((r) => r.skill_mean), error_y: { type: "data", array: rows.map((r) => r.skill_sd), visible: true, color: COLORS.ink, thickness: 1 },
      marker: { color: COLORS.soil }, offsetgroup: "o", hovertemplate: "%{x}<br>observed skill %{y:.3f}<extra></extra>" },
    { type: "bar", name: "predicted (training data only)", x, y: rows.map((r) => r.pred_skill_mean), error_y: { type: "data", array: rows.map((r) => r.pred_skill_sd), visible: true, color: COLORS.ink, thickness: 1 },
      marker: { color: COLORS.water }, offsetgroup: "p", hovertemplate: "%{x}<br>predicted skill %{y:.3f}<extra></extra>" },
    { type: "scatter", mode: "markers", name: "single test folds", x: ms.flatMap((m) => perFold(m).map(() => lab(m))), y: ms.flatMap(perFold),
      offsetgroup: "o", marker: { color: "#fff", size: 6, line: { color: COLORS.ink, width: 1 } }, hovertemplate: "%{x}<br>fold skill %{y:.3f}<extra></extra>" },
  ], { ...BASE_LAYOUT, barmode: "group", margin: { ...BASE_LAYOUT.margin, l: 70, b: 140 }, xaxis: { tickangle: -25 }, yaxis: { title: "skill (1 − RMSE / RMSE of the mean)", range: [0, 0.4] } }, PLOT_CFG);
}

/* ---------- Section 8: predicted vs observed ---------- */
function renderScatter() {
  const split = checked("ssplit");
  const traces = [];
  ORDER.filter((m) => m !== "Mean").forEach((m) => {
    const pts = state.res.pred_per_fold.filter((r) => r.split === split && r.method === m).map((r) => {
      const o = state.res.metrics.find((b) => b.split === split && b.method === m && b.fold === r.fold);
      return { x: r.pred_skill, y: o.skill, fold: r.fold + 1 };
    });
    traces.push({ x: pts.map((p) => p.x), y: pts.map((p) => p.y), mode: "markers", name: lab(m), legendgroup: m, marker: { size: 8, color: MCOLOR[m], opacity: 0.6 },
      customdata: pts.map((p) => p.fold), hovertemplate: `${lab(m)}<br>fold %{customdata}<br>predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>` });
    const s = summ(split, m);
    traces.push({ x: [s.pred_skill_mean], y: [s.skill_mean], mode: "markers", name: `${lab(m)} (mean)`, legendgroup: m, showlegend: false,
      marker: { size: 15, symbol: "diamond", color: MCOLOR[m], line: { color: "#fff", width: 1.5 } }, hovertemplate: `${lab(m)} mean<br>predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>` });
  });
  traces.push({ x: [0, 0.4], y: [0, 0.4], mode: "lines", name: "prediction = outcome", line: { dash: "dash", color: COLORS.grey } });
  Plotly.react("scatter-plot", traces, { ...BASE_LAYOUT, margin: { ...BASE_LAYOUT.margin, b: 130 },
    xaxis: { title: "predicted skill (training data only)", range: [-0.02, 0.4] }, yaxis: { title: "observed test skill", range: [-0.12, 0.4] } }, PLOT_CFG);
  const ms = ORDER.map((m) => summ(split, m));
  const rho = spearman(ms.map((r) => r.pred_skill_mean), ms.map((r) => r.skill_mean));
  const perFoldRho = range(0, 4, 1).map((f) => spearman(ORDER.map((m) => state.res.pred_per_fold.find((r) => r.split === split && r.method === m && r.fold === f).pred_skill),
    ORDER.map((m) => state.res.metrics.find((r) => r.split === split && r.method === m && r.fold === f).skill)));
  $("scatter-readout").innerHTML = `Over the 9 methods, the Spearman correlation between predicted and observed mean skill is <b>${rho.toFixed(2)}</b> (pre-registered target: at least 0.7). ` +
    `Per fold it ranges from ${Math.min(...perFoldRho).toFixed(2)} to ${Math.max(...perFoldRho).toFixed(2)}.`;
  const c = state.res.apriori_absolute_check.find((r) => r.split === "spatial");
  $("apriori-check").innerHTML = `<b>Post-hoc check (spatial split, not pre-registered).</b> Ordinary Kriging was predicted to improve on the average by only ${fmt(state.res.summary.find((r) => r.split === "spatial" && r.method === "OK").pred_skill_mean)} but improved by ${fmt(summ("spatial", "OK").skill_mean)}. ` +
    `The error scale itself was predicted well: predicted RMSE ${fmt(c.predicted_ok_rmse)} against observed ${fmt(c.observed_ok_rmse)} pH units. ` +
    `The reference was off: the prediction assumed the average has RMSE √sill = ${fmt(c.sqrt_sill)}, but on held-out regions it was ${fmt(c.mean_baseline_rmse)}, because the training mean is biased for unseen regions. ` +
    `Most of the gap in skill therefore comes from the reference, not from the kriging prediction.`;
}

/* ---------- Section 8 and 6: verdicts ---------- */
function renderVerdicts() {
  const v = state.res.verdicts;
  const over = v.filter((r) => r.part === "overall");
  document.querySelector("#verdict-table tbody").innerHTML = over.map((r) =>
    `<tr><td>${r.hypothesis}</td><td>${r.statement}</td><td class="${verdictClass(r.verdict)}">${r.verdict}</td><td>${r.key_number || ""}</td></tr>`).join("");
  document.querySelector("#verdict-evidence tbody").innerHTML = v.filter((r) => r.part !== "overall").map((r) =>
    `<tr><td>${r.hypothesis}</td><td>${r.part}</td><td>${r.evidence}</td><td class="${verdictClass(r.verdict)}">${r.verdict}</td></tr>`).join("");
  document.querySelectorAll("td[data-verdict]").forEach((td) => {
    const r = over.find((o) => o.hypothesis === td.dataset.verdict);
    if (!r) return;
    td.textContent = r.verdict;
    td.className = verdictClass(r.verdict);
  });
}

/* ---------- Section 8: error against distance ---------- */
const BIN_X = { "0-50": 25, "50-100": 75, "100-150": 125, "150-250": 200, ">250": 300 };
function renderErrDist() {
  const box = $("err-methods");
  if (!box.dataset.built) {
    const def = ["Mean", "Random Forest", "OK", "UK", "RK (linear)"];
    box.innerHTML = ORDER.map((m) => `<label class="check"><input type="checkbox" value="${m}" ${def.includes(m) ? "checked" : ""}> <span class="legend-dot" style="background:${MCOLOR[m]}"></span>${lab(m)}</label>`).join("");
    box.addEventListener("change", renderErrDist);
    box.dataset.built = "1";
  }
  const split = checked("esplit"), metric = $("sel-emetric").value;
  const shown = [...box.querySelectorAll("input:checked")].map((i) => i.value);
  const d = state.res.error_by_distance.filter((r) => r.split === split);
  const traces = ORDER.filter((m) => shown.includes(m)).map((m) => {
    const rows = Object.keys(BIN_X).map((b) => d.find((r) => r.method === m && r.bin === b));
    return { x: rows.map((r) => BIN_X[r.bin]), y: rows.map((r) => r[metric]), customdata: rows.map((r) => [r.bin, r.n]), mode: "lines+markers", name: lab(m),
      line: { color: MCOLOR[m], width: m === "Mean" ? 2 : 3, dash: m === "Mean" ? "dot" : "solid" }, marker: { size: 8 },
      hovertemplate: `${lab(m)}<br>%{customdata[0]} km: ${metric.toUpperCase()} %{y:.3f}<br>%{customdata[1]} test samples<extra></extra>` };
  });
  const rg = mean(state.preds.training_stats.filter((r) => r.split === "spatial").map((r) => r.range_km));
  Plotly.react("err-dist-plot", traces, { ...BASE_LAYOUT, margin: { ...BASE_LAYOUT.margin, b: 110 },
    xaxis: { title: "distance to the nearest training sample (km); the last bin is “> 250”, drawn at 300", range: [0, rg + 40], tickvals: Object.values(BIN_X), ticktext: Object.keys(BIN_X) },
    yaxis: { title: metric === "mae" ? "mean absolute error (pH)" : "RMSE (pH)", rangemode: "tozero" },
    shapes: [{ type: "line", x0: rg, x1: rg, y0: 0, y1: 1, yref: "paper", line: { color: COLORS.grey, dash: "dash" } }],
    annotations: [{ x: rg, y: 1, yref: "paper", text: `mean fitted range ≈ ${rg.toFixed(0)} km`, showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 11 } }] }, PLOT_CFG);
  const n = Object.keys(BIN_X).map((b) => `${b}: ${d.find((r) => r.method === "Mean" && r.bin === b).n}`).join(" · ");
  const m0 = (b, m) => d.find((r) => r.method === m && r.bin === b).mae;
  $("err-dist-readout").innerHTML = `Test samples per distance bin (${split}): ${n}. ` +
    `In the 0–50 km bin Ordinary Kriging has MAE ${fmt(m0("0-50", "OK"))} against ${fmt(m0("0-50", "Mean"))} for the average (${fmt((1 - m0("0-50", "OK") / m0("0-50", "Mean")) * 100, 0)}% lower). ` +
    `Beyond 250 km it is ${fmt(m0(">250", "OK"))} against ${fmt(m0(">250", "Mean"))} (${fmt((1 - m0(">250", "OK") / m0(">250", "Mean")) * 100, 0)}% lower).`;
}

/* ---------- Section 8: residual map ---------- */
function residColor(v) {
  const t = Math.max(-1, Math.min(1, v / 1.5));
  const lerp = (a, b, k) => Math.round(a + (b - a) * k);
  return t >= 0 ? `rgb(${lerp(245, 179, t)},${lerp(245, 38, t)},${lerp(245, 30, t)})` : `rgb(${lerp(245, 44, -t)},${lerp(245, 111, -t)},${lerp(245, 181, -t)})`;
}
function renderResMap() {
  const sel = $("sel-map-method");
  if (!sel.dataset.built) {
    sel.innerHTML = ORDER.map((m) => `<option value="${m}" ${m === "RK (linear)" ? "selected" : ""}>${lab(m)}</option>`).join("");
    sel.dataset.built = "1";
    state.rmap = L.map("res-map", { preferCanvas: true, scrollWheelZoom: false }).setView([52, 13], 3);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 8, attribution: "© OpenStreetMap contributors" }).addTo(state.rmap);
    state.rlayer = L.layerGroup().addTo(state.rmap);
  }
  state.rlayer.clearLayers();
  const method = sel.value, split = $("sel-map-split").value;
  const cells = state.res.cell_residuals.filter((r) => r.split === split && r.method === method);
  cells.forEach((c) => L.circle([c.lat, c.lon], { radius: 50000, weight: 0, fillOpacity: 0.8, fillColor: residColor(c.mean_residual) })
    .bindTooltip(`${c.n} test samples · mean residual ${c.mean_residual >= 0 ? "+" : ""}${c.mean_residual.toFixed(2)} pH`).addTo(state.rlayer));
  const worst = [...cells].sort((a, b) => Math.abs(b.mean_residual) - Math.abs(a.mean_residual))[0];
  const bias = mean(cells.map((c) => c.mean_residual));
  $("res-map-readout").innerHTML = `${cells.length} cells with at least 5 test samples (${split} split, ${lab(method)}). Average cell residual: ${bias >= 0 ? "+" : ""}${bias.toFixed(2)} pH. ` +
    (worst ? `Largest cell residual: ${worst.mean_residual >= 0 ? "+" : ""}${worst.mean_residual.toFixed(2)} pH near ${worst.lat.toFixed(1)}°N, ${worst.lon.toFixed(1)}°E (${worst.n} samples).` : "");
}

/* ---------- Section 8: density slider ---------- */
function densityRow(fr, m) { return state.res.density.find((r) => r.fraction === fr && r.method === m); }
function renderDensity() {
  const i = +$("sl-frac").value, fr = FRACTIONS[i];
  const n = Math.round(densityRow(fr, "OK").n_train), nn = densityRow(fr, "OK").train_nn_median_km;
  $("o-frac").textContent = `${Math.round(fr * 100)}% (about ${n.toLocaleString()} samples)`;
  const ms = ORDER.filter((m) => m !== "Mean");
  const rows = ms.map((m) => densityRow(fr, m));
  const x = ms.map(lab);
  Plotly.react("density-plot", [
    { type: "bar", name: "observed", x, y: rows.map((r) => r.skill_mean), error_y: { type: "data", array: rows.map((r) => r.skill_sd), visible: true, color: COLORS.ink, thickness: 1 }, marker: { color: COLORS.soil } },
    { type: "bar", name: "predicted", x, y: rows.map((r) => r.pred_skill_mean), error_y: { type: "data", array: rows.map((r) => r.pred_skill_sd), visible: true, color: COLORS.ink, thickness: 1 }, marker: { color: COLORS.water } },
  ], { ...BASE_LAYOUT, barmode: "group", margin: { ...BASE_LAYOUT.margin, b: 140 }, title: { text: `Skill with ${Math.round(fr * 100)}% of the training samples`, font: { size: 14 } },
    xaxis: { tickangle: -25 }, yaxis: { title: "skill", range: [-0.05, 0.4] } }, PLOT_CFG);

  const xs = FRACTIONS.map((f) => `${Math.round(f * 100)}%`);
  const lines = ms.map((m) => ({ x: xs, y: FRACTIONS.map((f) => densityRow(f, m).skill_mean), mode: "lines+markers", name: lab(m), line: { color: MCOLOR[m], width: m === "RK (linear)" || m === "LinReg (coords + covariates)" ? 3 : 1.5 } }));
  Plotly.react("density-lines", lines, { ...BASE_LAYOUT, margin: { ...BASE_LAYOUT.margin, b: 130 }, title: { text: "Observed skill as training samples are removed", font: { size: 14 } },
    xaxis: { title: "share of training samples kept", type: "category", categoryorder: "array", categoryarray: xs }, yaxis: { title: "skill", range: [0, 0.3] },
    shapes: [{ type: "line", x0: i, x1: i, y0: 0, y1: 1, yref: "paper", line: { color: COLORS.grey, dash: "dash" } }] }, PLOT_CFG);

  const obs = ORDER.map((m) => densityRow(fr, m).skill_mean), pre = ORDER.map((m) => densityRow(fr, m).pred_skill_mean);
  const best = (a) => lab(ORDER[a.indexOf(Math.max(...a))]);
  const gain = (f) => densityRow(f, "RK (linear)").skill_mean - densityRow(f, "LinReg (coords + covariates)").skill_mean;
  $("density-readout").innerHTML = `Median distance between training samples: <b>${nn.toFixed(0)} km</b>. Best method observed: <b>${best(obs)}</b>; predicted best: <b>${best(pre)}</b>. ` +
    `Spearman correlation of predicted and observed skill over the 9 methods: <b>${spearman(pre, obs).toFixed(2)}</b>. ` +
    `Kriging gain over the linear trend (RK (linear) minus LinReg with covariates): <b>${fmt(gain(fr))}</b> now, ${fmt(gain(1))} with all samples, ${fmt(gain(0.1))} with 10%.`;
}

/* ---------- Section 8: tables ---------- */
function renderTables() {
  const r = state.res;
  const best = ORDER.map((m) => summ("spatial", m)).sort((a, b) => b.skill_mean - a.skill_mean)[0].method;
  const cols = ["Mean", "Random Forest", "OK", best === "OK" ? "UK" : best];
  const uniq = [...new Set(cols)];
  const ph = r.error_by_ph_range.filter((x) => x.split === "spatial");
  const ranges = ["pH < 4.5", "4.5 to 7.5", "pH > 7.5"];
  $("tbl-ph").innerHTML = `<tr><th>pH range (n)</th>${uniq.map((m) => `<th>${lab(m)}</th>`).join("")}</tr>` + ranges.map((g) => {
    const n = ph.find((x) => x.range === g).n;
    return `<tr><td>${g} (${n})</td>${uniq.map((m) => { const e = ph.find((x) => x.range === g && x.method === m); return `<td>${fmt(e.rmse)} (${e.bias >= 0 ? "+" : ""}${fmt(e.bias)})</td>`; }).join("")}</tr>`;
  }).join("");

  const ct = r.error_by_country;
  const countries = [...new Set(ct.map((x) => x.country))];
  const rows = countries.map((c) => ({ c, n: ct.find((x) => x.country === c && x.method === best).n, e: ct.find((x) => x.country === c && x.method === best).rmse, mean: ct.find((x) => x.country === c && x.method === "Mean").rmse }))
    .sort((a, b) => b.e - a.e).slice(0, 8);
  $("tbl-country").innerHTML = `<tr><th>Country</th><th>n</th><th>Mean</th><th>${lab(best)}</th></tr>` + rows.map((x) => `<tr><td>${x.c}</td><td>${x.n}</td><td>${fmt(x.mean)}</td><td>${fmt(x.e)}</td></tr>`).join("");

  const fe = r.error_by_fold.filter((x) => x.split === "spatial");
  $("tbl-fold").innerHTML = `<tr><th>Method</th>${range(0, 4, 1).map((f) => `<th>fold ${f + 1}</th>`).join("")}</tr>` + ORDER.map((m) => {
    const v = range(0, 4, 1).map((f) => fe.find((x) => x.fold === f && x.method === m).rmse);
    return `<tr><td>${lab(m)}</td>${v.map((e) => `<td>${fmt(e)}</td>`).join("")}</tr>`;
  }).join("") + `<tr><th>test samples</th>${range(0, 4, 1).map((f) => `<th>${fe.find((x) => x.fold === f && x.method === "Mean").n}</th>`).join("")}</tr>`;

  const cv = r.coverage;
  $("tbl-cov").innerHTML = `<tr><th>Method</th><th>Coverage, spatial</th><th>Coverage, random</th><th>Mean SD, spatial</th><th>RMSE / mean SD, spatial</th></tr>` +
    ["OK", "UK", "GPR", "RK (linear)", "RK (RF)"].map((m) => {
      const s = cv.find((x) => x.split === "spatial" && x.method === m), q = cv.find((x) => x.split === "random" && x.method === m);
      return `<tr><td>${lab(m)}</td><td>${fmt(s.coverage_95)}</td><td>${fmt(q.coverage_95)}</td><td>${fmt(s.mean_sd)}</td><td>${fmt(s.rmse_over_mean_sd)}</td></tr>`;
    }).join("");
}

/* ---------- Section 9: numbers in the limitations ---------- */
function renderLimits() {
  const top5 = ORDER.map((m) => summ("spatial", m)).sort((a, b) => b.skill_mean - a.skill_mean).slice(0, 5);
  $("lim-gap").textContent = fmt(top5[0].skill_mean - top5[4].skill_mean) + " (" + top5.map((r) => lab(r.method)).join(", ") + ")";
  const sds = top5.map((r) => r.skill_sd);
  $("lim-sd").textContent = `${fmt(Math.min(...sds), 2)} to ${fmt(Math.max(...sds), 2)}`;
}
/* ---------- wiring ---------- */
document.addEventListener("DOMContentLoaded", async () => {
  ["sl-eta", "sl-range", "sl-d"].forEach((id) => $(id).addEventListener("input", updateExplainer));
  $("cb-emp").addEventListener("change", updateExplainer);
  $("btn-fit").addEventListener("click", resetExplainer);
  document.querySelectorAll("input[name=split]").forEach((el) => el.addEventListener("change", renderSplit));
  $("sel-fold").addEventListener("change", renderSplit);
  document.querySelectorAll("input[name=rsplit]").forEach((el) => el.addEventListener("change", () => { renderResTable(); renderSkillBar(); }));
  document.querySelectorAll("input[name=ssplit]").forEach((el) => el.addEventListener("change", renderScatter));
  document.querySelectorAll("input[name=esplit]").forEach((el) => el.addEventListener("change", renderErrDist));
  $("sel-emetric").addEventListener("change", renderErrDist);
  $("sel-map-method").addEventListener("change", renderResMap);
  $("sel-map-split").addEventListener("change", renderResMap);
  $("sl-frac").addEventListener("input", renderDensity);
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
