import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import PlotImport from "react-plotly.js";
import { Radar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip as CJSTooltip,
  Legend,
} from "chart.js";
import { motion } from "framer-motion";
import {
  EntranceReveal,
  GlassCard,
  LoadingSkeleton,
} from "../components/common";
import {
  fetchModelComparison,
  fetchConfusionMatrix,
  fetchTrainingHistory,
} from "../lib/api";
import "./Models.css";

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, CJSTooltip, Legend);

/* ── Types ────────────────────────────────────────────────────── */

// Handle ESM / CJS default export mismatch
const PlotComponent =
  typeof (PlotImport as any).default === "function"
    ? (PlotImport as any).default
    : PlotImport;
const AnimatedPlot = PlotComponent as React.ComponentType<any>;

interface ModelRow {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  trainingTime: string;
  modelSize: string;
}

type SortKey = keyof Pick<ModelRow, "accuracy" | "precision" | "recall" | "f1">;
type SortDir = "asc" | "desc";

const MODEL_COLORS = ["#4361ee", "#06d6a0", "#ef476f", "#ffd166", "#8338ec"];
const MODEL_NAMES_DL = ["lstm", "bilstm", "cnn", "distilbert"];
const METRICS: SortKey[] = ["accuracy", "precision", "recall", "f1"];
const METRIC_LABELS: Record<SortKey, string> = {
  accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1: "F1 Score",
};

/* ── Helpers ──────────────────────────────────────────────────── */

const safe = (v: unknown, fb = 0): number => {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") { const n = Number(v.replace("%", "").trim()); return Number.isFinite(n) ? n : fb; }
  return fb;
};

const arr = (v: unknown): unknown[] => {
  if (Array.isArray(v)) return v;
  if (!v || typeof v !== "object") return [];
  const r = v as Record<string, unknown>;
  return (r.data ?? r.items ?? r.results ?? r.models ?? []) as unknown[];
};

const normalizeModels = (raw: unknown): ModelRow[] => {
  const items = arr(raw);
  if (!items.length && raw && typeof raw === "object") {
    // might be { models: [...] }
    const r = raw as Record<string, unknown>;
    const inner = r.models ?? r.comparison ?? r.data;
    if (Array.isArray(inner)) return normalizeModels(inner);
  }
  return items.map((item) => {
    const o = (item as Record<string, unknown>) ?? {};
    const pct = (v: unknown) => { const n = safe(v); return n <= 1 ? n * 100 : n; };
    return {
      name: String(o.name ?? o.model ?? o.model_name ?? o.modelName ?? ""),
      accuracy: pct(o.accuracy ?? o.acc),
      precision: pct(o.precision ?? o.prec),
      recall: pct(o.recall ?? o.rec),
      f1: pct(o.f1 ?? o.f1_score ?? o.f1Score),
      trainingTime: String(o.training_time ?? o.trainingTime ?? o.time ?? "—"),
      modelSize: String(o.model_size ?? o.modelSize ?? o.size ?? "—"),
    };
  });
};

const normalizeMatrix = (raw: unknown): { matrix: number[][]; labels: string[] } => {
  const d = (raw as Record<string, unknown>) ?? {};
  const m = (d.matrix ?? d.confusion_matrix ?? d.confusionMatrix ?? d.data) as number[][] | undefined;
  const labels = (d.labels ?? d.classes ?? ["Positive", "Negative", "Neutral"]) as string[];
  return { matrix: Array.isArray(m) ? m : [], labels: labels.map(l => String(l)) };
};

const normalizeHistory = (raw: unknown): {
  epochs: number[]; trainLoss: number[]; valLoss: number[];
  trainAcc: number[]; valAcc: number[];
} => {
  const d = (raw as Record<string, unknown>) ?? {};
  const toArr = (v: unknown) => (Array.isArray(v) ? v.map(safe) : []);
  const epochCount = Math.max(
    toArr(d.train_loss ?? d.trainLoss ?? d.loss).length,
    toArr(d.train_acc ?? d.trainAcc ?? d.accuracy).length
  );
  return {
    epochs: Array.from({ length: epochCount }, (_, i) => i + 1),
    trainLoss: toArr(d.train_loss ?? d.trainLoss ?? d.loss),
    valLoss: toArr(d.val_loss ?? d.valLoss ?? d.validation_loss),
    trainAcc: toArr(d.train_acc ?? d.trainAcc ?? d.accuracy),
    valAcc: toArr(d.val_acc ?? d.valAcc ?? d.validation_accuracy ?? d.val_accuracy),
  };
};

/* ── Sub-components ───────────────────────────────────────────── */

const SectionError: React.FC<{ msg: string }> = ({ msg }) => (
  <div className="models__error"><p>{msg}</p></div>
);

const PanelSkeleton: React.FC<{ h?: number }> = ({ h = 320 }) => (
  <GlassCard hoverable={false} className="models__panel">
    <div className="models__panel-hdr"><LoadingSkeleton width="180px" height="22px" borderRadius="10px" /></div>
    <LoadingSkeleton height={`${h}px`} borderRadius="14px" />
  </GlassCard>
);

const darkLayout = (extra: Record<string, unknown> = {}) => ({
  autosize: true, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 50, r: 20, t: 10, b: 50 },
  font: { color: "#e6f1ff", size: 12 },
  xaxis: { color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
    tickfont: { color: "#aab6d3", size: 11 } },
  yaxis: { color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
    tickfont: { color: "#aab6d3", size: 11 } },
  ...extra,
});

/* ═══════════════════════════════════════════════════════════════
   MAIN MODELS PAGE
   ═══════════════════════════════════════════════════════════════ */

const Models: React.FC = () => {
  /* state */
  const [compLoading, setCompLoading] = useState(true);
  const [compError, setCompError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelRow[]>([]);
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const [cmModel, setCmModel] = useState("");
  const [cmLoading, setCmLoading] = useState(false);
  const [cmError, setCmError] = useState<string | null>(null);
  const [cmData, setCmData] = useState<{ matrix: number[][]; labels: string[] }>({ matrix: [], labels: [] });

  const [histModel, setHistModel] = useState("lstm");
  const [histLoading, setHistLoading] = useState(false);
  const [histError, setHistError] = useState<string | null>(null);
  const [histData, setHistData] = useState<ReturnType<typeof normalizeHistory>>({
    epochs: [], trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] });

  /* ── load comparison ──────────────────────────────────────── */
  useEffect(() => {
    let mounted = true;
    (async () => {
      setCompLoading(true); setCompError(null);
      const res = await fetchModelComparison();
      if (!mounted) return;
      if (!res) setCompError("Unable to load model comparison.");
      else {
        const normalized = normalizeModels(res);
        setModels(normalized);
        // default CM model = best f1
        const best = normalized.reduce((a, b) => b.f1 > a.f1 ? b : a, normalized[0]);
        if (best) setCmModel(best.name.toLowerCase().replace(/\s+/g, "_"));
      }
      setCompLoading(false);
    })();
    return () => { mounted = false; };
  }, []);

  /* ── load confusion matrix ────────────────────────────────── */
  useEffect(() => {
    if (!cmModel) return;
    let mounted = true;
    (async () => {
      setCmLoading(true); setCmError(null);
      const res = await fetchConfusionMatrix(cmModel);
      if (!mounted) return;
      if (!res) setCmError("Unable to load confusion matrix.");
      else setCmData(normalizeMatrix(res));
      setCmLoading(false);
    })();
    return () => { mounted = false; };
  }, [cmModel]);

  /* ── load training history ────────────────────────────────── */
  useEffect(() => {
    if (!histModel) return;
    let mounted = true;
    (async () => {
      setHistLoading(true); setHistError(null);
      const res = await fetchTrainingHistory(histModel);
      if (!mounted) return;
      if (!res) setHistError("Unable to load training history.");
      else setHistData(normalizeHistory(res));
      setHistLoading(false);
    })();
    return () => { mounted = false; };
  }, [histModel]);

  /* ── sorting ──────────────────────────────────────────────── */
  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("desc"); }
  };

  const sorted = useMemo(() => {
    if (!sortKey) return models;
    return [...models].sort((a, b) => sortDir === "desc" ? b[sortKey] - a[sortKey] : a[sortKey] - b[sortKey]);
  }, [models, sortKey, sortDir]);

  /* ── best values per metric ───────────────────────────────── */
  const bestVals = useMemo(() => {
    const bests: Record<string, number> = {};
    METRICS.forEach(m => { bests[m] = Math.max(...models.map(r => r[m]), 0); });
    return bests;
  }, [models]);

  const bestModel = useMemo(() =>
    models.reduce((a, b) => b.f1 > a.f1 ? b : a, models[0] ?? { name: "", f1: 0 } as ModelRow), [models]);

  /* ── Chart.js radar data ──────────────────────────────────── */
  const radarData = useMemo(() => ({
    labels: METRICS.map(m => METRIC_LABELS[m]),
    datasets: models.map((m, i) => ({
      label: m.name,
      data: METRICS.map(k => m[k]),
      borderColor: MODEL_COLORS[i % MODEL_COLORS.length],
      backgroundColor: MODEL_COLORS[i % MODEL_COLORS.length] + "26",
      pointBackgroundColor: MODEL_COLORS[i % MODEL_COLORS.length],
      borderWidth: 2, pointRadius: 4,
    })),
  }), [models]);

  const radarOptions = useMemo(() => ({
    responsive: true, maintainAspectRatio: false,
    scales: {
      r: {
        angleLines: { color: "rgba(255,255,255,0.08)" },
        grid: { color: "rgba(255,255,255,0.06)" },
        pointLabels: { color: "#e6f1ff", font: { size: 12 } },
        ticks: { display: false },
        suggestedMin: 0, suggestedMax: 100,
      },
    },
    plugins: {
      legend: { position: "top" as const, labels: { color: "#e6f1ff", usePointStyle: true, pointStyle: "circle", padding: 16, font: { size: 12 } } },
      tooltip: { backgroundColor: "#141432", borderColor: "rgba(255,255,255,0.1)", borderWidth: 1, titleColor: "#e6f1ff", bodyColor: "#aab6d3" },
    },
  }), []);

  /* ── Plotly data ──────────────────────────────────────────── */
  const groupedBarData = models.map((m, i) => ({
    type: "bar", name: m.name,
    x: METRICS.map(k => METRIC_LABELS[k]),
    y: METRICS.map(k => m[k]),
    marker: { color: MODEL_COLORS[i % MODEL_COLORS.length] },
    hovertemplate: `${m.name}: %{y:.2f}%<extra></extra>`,
  }));

  const cmHeatmapData = cmData.matrix.length ? [{
    type: "heatmap", z: cmData.matrix, x: cmData.labels, y: cmData.labels,
    colorscale: [[0, "#0a0a1a"], [0.5, "#1e3a7a"], [1, "#4361ee"]],
    showscale: false,
    text: cmData.matrix.map(row => row.map(String)),
    texttemplate: "%{text}", textfont: { color: "#fff", size: 14 },
    hovertemplate: "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
  }] : [];

  const lossTraces = [
    { name: "Train Loss", x: histData.epochs, y: histData.trainLoss,
      line: { color: "#4361ee", shape: "spline" }, mode: "lines+markers",
      marker: { size: 4 }, type: "scatter" },
    { name: "Val Loss", x: histData.epochs, y: histData.valLoss,
      line: { color: "#ef476f", shape: "spline" }, mode: "lines+markers",
      marker: { size: 4 }, type: "scatter" },
  ];

  const accTraces = [
    { name: "Train Acc", x: histData.epochs, y: histData.trainAcc,
      line: { color: "#4361ee", shape: "spline" }, mode: "lines+markers",
      marker: { size: 4 }, type: "scatter" },
    { name: "Val Acc", x: histData.epochs, y: histData.valAcc,
      line: { color: "#06d6a0", shape: "spline" }, mode: "lines+markers",
      marker: { size: 4 }, type: "scatter" },
  ];

  /* ── render ───────────────────────────────────────────────── */
  const sortArrow = (key: SortKey) => {
    if (sortKey !== key) return <span className="models__sort-arrow models__sort-arrow--idle">↕</span>;
    return <span className="models__sort-arrow">{sortDir === "desc" ? "↓" : "↑"}</span>;
  };

  return (
    <div className="models-page">
      {/* ═══ SECTION 1 — Comparison Table ═══ */}
      <EntranceReveal>
        {compLoading ? <PanelSkeleton h={300} /> : (
          <GlassCard hoverable={false} className="models__panel">
            <div className="models__panel-hdr"><h3>Model Performance Comparison</h3></div>
            {compError ? <SectionError msg={compError} /> : (
              <div className="models__table-wrap">
                <table className="models__table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      {METRICS.map(m => (
                        <th key={m} className="models__th-sort" onClick={() => handleSort(m)}>
                          {METRIC_LABELS[m]} {sortArrow(m)}
                        </th>
                      ))}
                      <th>Training Time</th>
                      <th>Model Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((row, idx) => {
                      const isBest = row.name === bestModel?.name;
                      return (
                        <motion.tr key={row.name}
                          className={isBest ? "models__row--best" : ""}
                          initial={{ opacity: 0, y: 12 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.06 }}>
                          <td className="models__td-name">{row.name}</td>
                          {METRICS.map(m => (
                            <td key={m} className={`mono ${row[m] === bestVals[m] ? "models__td-best" : ""}`}>
                              {row[m].toFixed(2)}%
                            </td>
                          ))}
                          <td className="mono">{row.trainingTime}</td>
                          <td className="mono">{row.modelSize}</td>
                        </motion.tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </GlassCard>
        )}
      </EntranceReveal>

      {/* ═══ SECTION 2 — Visual Comparison ═══ */}
      <EntranceReveal>
        <h2 className="models__section-title">Visual Comparison</h2>
      </EntranceReveal>
      <div className="models__grid models__grid--2">
        {compLoading ? <PanelSkeleton h={380} /> : (
          <GlassCard hoverable={false} className="models__panel">
            <div className="models__panel-hdr"><h3>Grouped Bar Chart</h3></div>
            {compError ? <SectionError msg={compError} /> : (
              <AnimatedPlot className="models__plot" data={groupedBarData}
                layout={darkLayout({ height: 380, barmode: "group", bargap: 0.2, bargroupgap: 0.08,
                  legend: { orientation: "h", x: 0, y: 1.12, font: { color: "#e6f1ff", size: 11 } } })}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 380 }} />
            )}
          </GlassCard>
        )}
        {compLoading ? <PanelSkeleton h={380} /> : (
          <GlassCard hoverable={false} className="models__panel">
            <div className="models__panel-hdr"><h3>Radar Chart</h3></div>
            {compError ? <SectionError msg={compError} /> : (
              <div className="models__radar-wrap">
                <Radar data={radarData} options={radarOptions} />
              </div>
            )}
          </GlassCard>
        )}
      </div>

      {/* ═══ SECTION 3 — Confusion Matrix ═══ */}
      <EntranceReveal>
        <GlassCard hoverable={false} className="models__panel">
          <div className="models__panel-hdr models__panel-hdr--inline">
            <h3>Confusion Matrix</h3>
            <select className="models__select" value={cmModel}
              onChange={e => setCmModel(e.target.value)}>
              {models.map(m => {
                const val = m.name.toLowerCase().replace(/\s+/g, "_");
                return <option key={val} value={val}>{m.name}</option>;
              })}
            </select>
          </div>
          {cmLoading ? <LoadingSkeleton height="360px" borderRadius="14px" /> :
            cmError ? <SectionError msg={cmError} /> :
            cmHeatmapData.length === 0 ? <SectionError msg="No confusion matrix data." /> : (
            <AnimatedPlot className="models__plot" data={cmHeatmapData}
              layout={darkLayout({ height: 380, margin: { l: 80, r: 20, t: 10, b: 60 },
                xaxis: { title: { text: "Predicted", font: { color: "#aab6d3", size: 12 } },
                  color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
                  tickfont: { color: "#aab6d3", size: 12 } },
                yaxis: { title: { text: "Actual", font: { color: "#aab6d3", size: 12 } },
                  autorange: "reversed",
                  color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
                  tickfont: { color: "#aab6d3", size: 12 } } })}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%", height: 380 }} />
          )}
        </GlassCard>
      </EntranceReveal>

      {/* ═══ SECTION 4 — Training History ═══ */}
      <EntranceReveal>
        <div className="models__panel-hdr models__panel-hdr--section">
          <h2 className="models__section-title" style={{ marginBottom: 0 }}>Training History</h2>
          <select className="models__select" value={histModel}
            onChange={e => setHistModel(e.target.value)}>
            {MODEL_NAMES_DL.map(m => (
              <option key={m} value={m}>{m.toUpperCase()}</option>
            ))}
          </select>
        </div>
      </EntranceReveal>
      <div className="models__grid models__grid--2">
        {histLoading ? <PanelSkeleton h={320} /> : (
          <GlassCard hoverable={false} className="models__panel">
            <div className="models__panel-hdr"><h3>Loss Curves</h3></div>
            {histError ? <SectionError msg={histError} /> :
              histData.epochs.length === 0 ? <SectionError msg="No training history." /> : (
              <AnimatedPlot className="models__plot" data={lossTraces}
                layout={darkLayout({ height: 320,
                  xaxis: { title: { text: "Epoch", font: { color: "#aab6d3", size: 11 } },
                    color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
                    tickfont: { color: "#aab6d3", size: 11 } },
                  legend: { orientation: "h", x: 0, y: 1.12, font: { color: "#e6f1ff", size: 11 } } })}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 320 }} />
            )}
          </GlassCard>
        )}
        {histLoading ? <PanelSkeleton h={320} /> : (
          <GlassCard hoverable={false} className="models__panel">
            <div className="models__panel-hdr"><h3>Accuracy Curves</h3></div>
            {histError ? <SectionError msg={histError} /> :
              histData.epochs.length === 0 ? <SectionError msg="No training history." /> : (
              <AnimatedPlot className="models__plot" data={accTraces}
                layout={darkLayout({ height: 320,
                  xaxis: { title: { text: "Epoch", font: { color: "#aab6d3", size: 11 } },
                    color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
                    tickfont: { color: "#aab6d3", size: 11 } },
                  legend: { orientation: "h", x: 0, y: 1.12, font: { color: "#e6f1ff", size: 11 } } })}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 320 }} />
            )}
          </GlassCard>
        )}
      </div>

      {/* ═══ SECTION 5 — Best Model Recommendation ═══ */}
      {!compLoading && !compError && bestModel && (
        <EntranceReveal>
          <GlassCard hoverable={false} className="models__panel models__recommend">
            <span className="models__recommend-trophy">🏆</span>
            <h2 className="models__recommend-title">
              Recommended Model: {bestModel.name}
            </h2>
            <p className="models__recommend-sub">
              Highest F1 Score: {bestModel.f1.toFixed(2)}% &nbsp;|&nbsp;
              Accuracy: {bestModel.accuracy.toFixed(2)}%
            </p>
            <Link to="/predict" className="models__recommend-btn">
              Try it in Live Prediction →
            </Link>
          </GlassCard>
        </EntranceReveal>
      )}
    </div>
  );
};

export default Models;
