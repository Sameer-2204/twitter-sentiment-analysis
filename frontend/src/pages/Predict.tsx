import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PlotImport from "react-plotly.js";

// Handle ESM / CJS default export mismatch
const PlotComponent =
  typeof (PlotImport as any).default === "function"
    ? (PlotImport as any).default
    : PlotImport;
const AnimatedPlot = PlotComponent as React.ComponentType<any>;
import {
  GlassCard,
  SentimentBadge,
} from "../components/common";
import {
  getApiErrorMessage,
  predictSentiment,
  predictAllModels,
} from "../lib/api";
import "./Predict.css";


/* ── Types ────────────────────────────────────────────────────── */

type Sentiment = "positive" | "negative" | "neutral";

interface SingleResult {
  sentiment: Sentiment;
  confidence: number;
  model: string;
  positive_prob: number;
  negative_prob: number;
  neutral_prob: number;
}

interface ModelResult {
  model: string;
  sentiment: Sentiment;
  confidence: number;
}

interface HistoryEntry {
  id: number;
  text: string;
  model: string;
  sentiment: Sentiment;
  confidence: number;
  timestamp: number;
}

const SENT_COLORS: Record<Sentiment, string> = {
  positive: "#06d6a0", negative: "#ef476f", neutral: "#ffd166",
};

const MODELS = [
  { value: "logistic_regression", label: "Logistic Regression" },
  { value: "distilbert", label: "DistilBERT" },
  { value: "cnn", label: "CNN" },
  { value: "bilstm", label: "BiLSTM" },
  { value: "lstm", label: "LSTM" },
];

const HISTORY_KEY = "predictionHistory";
const MAX_HISTORY = 20;

/* ── Helpers ──────────────────────────────────────────────────── */

const safe = (v: unknown, fb = 0): number => {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") { const n = Number(v); return Number.isFinite(n) ? n : fb; }
  return fb;
};

const pct = (v: unknown) => { const n = safe(v); return n <= 1 ? n * 100 : n; };

const normSentiment = (v: unknown): Sentiment => {
  const s = String(v ?? "").toLowerCase();
  if (s.includes("pos")) return "positive";
  if (s.includes("neg")) return "negative";
  return "neutral";
};

const normSingle = (raw: unknown, model: string): SingleResult => {
  const d = (raw as Record<string, unknown>) ?? {};
  const conf = pct(d.confidence ?? d.score ?? d.probability);
  return {
    sentiment: normSentiment(d.sentiment ?? d.label ?? d.prediction),
    confidence: conf,
    model,
    positive_prob: pct(d.positive_prob ?? d.positive_probability ?? d.positive ?? d.pos_prob ?? 0),
    negative_prob: pct(d.negative_prob ?? d.negative_probability ?? d.negative ?? d.neg_prob ?? 0),
    neutral_prob: pct(d.neutral_prob ?? d.neutral_probability ?? d.neutral ?? d.neu_prob ?? 0),
  };
};

const normAllModels = (raw: unknown): ModelResult[] => {
  const items = Array.isArray(raw) ? raw
    : (raw as Record<string, unknown>)?.results ?? (raw as Record<string, unknown>)?.predictions ?? [];
  if (!Array.isArray(items)) return [];
  return items.map((item) => {
    const o = (item as Record<string, unknown>) ?? {};
    return {
      model: String(o.model ?? o.model_used ?? o.model_name ?? o.name ?? ""),
      sentiment: normSentiment(o.sentiment ?? o.label ?? o.prediction),
      confidence: pct(o.confidence ?? o.score ?? o.probability),
    };
  });
};

const loadHistory = (): HistoryEntry[] => {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
  catch { return []; }
};

const saveHistory = (entries: HistoryEntry[]) => {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
};

const truncate = (s: string, n: number) => s.length > n ? s.slice(0, n - 1) + "…" : s;

/* ── Spinner ──────────────────────────────────────────────────── */

const Spinner = () => (
  <span className="predict__spinner" />
);

/* ═══════════════════════════════════════════════════════════════
   MAIN PREDICT PAGE
   ═══════════════════════════════════════════════════════════════ */

const Predict: React.FC = () => {
  const [text, setText] = useState("");
  const [model, setModel] = useState("logistic_regression");
  const [validationError, setValidationError] = useState(false);

  const [singleLoading, setSingleLoading] = useState(false);
  const [singleResult, setSingleResult] = useState<SingleResult | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);

  const [allLoading, setAllLoading] = useState(false);
  const [allResults, setAllResults] = useState<ModelResult[] | null>(null);
  const [allError, setAllError] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);

  const addToHistory = (entry: Omit<HistoryEntry, "id" | "timestamp">) => {
    const newEntry: HistoryEntry = { ...entry, id: Date.now(), timestamp: Date.now() };
    const updated = [newEntry, ...history].slice(0, MAX_HISTORY);
    setHistory(updated);
    saveHistory(updated);
  };

  /* ── Analyze single ──────────────────────────────────────── */
  const handleAnalyze = async () => {
    if (!text.trim()) { setValidationError(true); return; }
    setValidationError(false);
    setSingleLoading(true); setSingleError(null); setSingleResult(null); setAllResults(null);
    try {
      const res = await predictSentiment(text, model);
      const result = normSingle(res, MODELS.find(m => m.value === model)?.label ?? model);
      setSingleResult(result);
      addToHistory({ text: text.trim(), model: result.model, sentiment: result.sentiment, confidence: result.confidence });
    } catch (err) {
      setSingleError(getApiErrorMessage(err, "Prediction failed."));
    }
    setSingleLoading(false);
  };

  /* ── Compare all ─────────────────────────────────────────── */
  const handleCompareAll = async () => {
    if (!text.trim()) { setValidationError(true); return; }
    setValidationError(false);
    setAllLoading(true); setAllError(null); setAllResults(null); setSingleResult(null);
    try {
      const res = await predictAllModels(text);
      const results = normAllModels(res);
      setAllResults(results);
      if (results.length) {
        const best = results.reduce((a, b) => b.confidence > a.confidence ? b : a, results[0]);
        addToHistory({ text: text.trim(), model: "All Models", sentiment: best.sentiment, confidence: best.confidence });
      }
    } catch (err) {
      setAllError(getApiErrorMessage(err, "Comparison failed."));
    }
    setAllLoading(false);
  };

  /* ── Consensus ───────────────────────────────────────────── */
  const consensus = allResults ? (() => {
    const counts: Record<string, number> = {};
    allResults.forEach(r => { counts[r.sentiment] = (counts[r.sentiment] || 0) + 1; });
    const max = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    if (max && max[1] >= Math.ceil(allResults.length / 2)) {
      return { agree: true, sentiment: max[0] as Sentiment, count: max[1], total: allResults.length };
    }
    return { agree: false, sentiment: "neutral" as Sentiment, count: 0, total: allResults.length };
  })() : null;

  return (
    <div className="predict-page">
      {/* ═══ SECTION 1 — Input ═══ */}
      <GlassCard hoverable={false} className="predict__input-card">
        <h2 className="predict__title">Analyze Sentiment</h2>
        <p className="predict__subtitle">Enter any tweet or text to analyze its sentiment</p>

        <div className={`predict__textarea-wrap ${validationError ? "predict__textarea-wrap--error" : ""}`}>
          <textarea
            className="predict__textarea"
            placeholder="Type or paste a tweet here..."
            value={text}
            onChange={e => { setText(e.target.value); setValidationError(false); }}
            maxLength={280}
          />
          <span className="predict__char-count mono">{text.length}/280</span>
        </div>
        {validationError && <span className="predict__validation">Please enter text to analyze</span>}

        <div className="predict__model-row">
          <div className="predict__model-group">
            <label className="predict__model-label">Select Model</label>
            <select className="predict__model-select" value={model} onChange={e => setModel(e.target.value)}>
              {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        </div>

        <div className="predict__btn-row">
          <button className="predict__btn-primary" onClick={handleAnalyze}
            disabled={singleLoading || allLoading}>
            {singleLoading ? <><Spinner /> Analyzing...</> : "Analyze"}
          </button>
          <button className="predict__btn-secondary" onClick={handleCompareAll}
            disabled={singleLoading || allLoading}>
            {allLoading ? <><Spinner /> Comparing...</> : "Compare All Models"}
          </button>
        </div>

        {singleError && <p className="predict__error-msg">{singleError}</p>}
        {allError && <p className="predict__error-msg">{allError}</p>}
      </GlassCard>

      {/* ═══ SECTION 2 — Single Result ═══ */}
      <AnimatePresence>
        {singleResult && (
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.5 }}>
            <GlassCard hoverable={false} className="predict__result-card"
              style={{
                borderColor: SENT_COLORS[singleResult.sentiment] + "4d",
                boxShadow: `0 0 30px ${SENT_COLORS[singleResult.sentiment]}33`,
              } as React.CSSProperties}>
              <div className="predict__result-top">
                <div className="predict__result-left">
                  <span className="predict__result-label"
                    style={{ color: SENT_COLORS[singleResult.sentiment] }}>
                    {singleResult.sentiment.toUpperCase()}
                  </span>
                  <span className="predict__result-model">Model: {singleResult.model}</span>
                  <span className="predict__result-conf mono">
                    {singleResult.confidence.toFixed(1)}%
                  </span>
                </div>
                <div className="predict__result-right">
                  <AnimatedPlot
                    data={[{
                      type: "indicator", mode: "gauge+number",
                      value: singleResult.confidence,
                      number: { suffix: "%", font: { color: "#e6f1ff", size: 28 } },
                      gauge: {
                        axis: { range: [0, 100], tickfont: { color: "#aab6d3", size: 10 } },
                        bar: { color: SENT_COLORS[singleResult.sentiment] },
                        bgcolor: "rgba(255,255,255,0.04)",
                        borderwidth: 0,
                        steps: [
                          { range: [0, 33], color: "rgba(255,255,255,0.02)" },
                          { range: [33, 66], color: "rgba(255,255,255,0.03)" },
                          { range: [66, 100], color: "rgba(255,255,255,0.04)" },
                        ],
                      },
                    }]}
                    layout={{
                      width: 250, height: 200, margin: { l: 20, r: 20, t: 30, b: 0 },
                      paper_bgcolor: "rgba(0,0,0,0)", font: { color: "#e6f1ff" },
                    }}
                    config={{ displayModeBar: false }}
                  />
                </div>
              </div>

              <div className="predict__breakdown">
                <h4 className="predict__breakdown-title">Confidence Breakdown</h4>
                {(["positive", "negative", "neutral"] as Sentiment[]).map(s => {
                  const val = singleResult[`${s}_prob` as keyof SingleResult] as number;
                  return (
                    <div className="predict__bar-row" key={s}>
                      <span className="predict__bar-label" style={{ color: SENT_COLORS[s] }}>
                        {s.charAt(0).toUpperCase() + s.slice(1)}
                      </span>
                      <div className="predict__bar-track">
                        <motion.div className="predict__bar-fill"
                          style={{ backgroundColor: SENT_COLORS[s] }}
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.max(val, 0.5)}%` }}
                          transition={{ duration: 0.8, ease: "easeOut" }} />
                      </div>
                      <span className="predict__bar-pct mono">{val.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ SECTION 3 — All Models Comparison ═══ */}
      <AnimatePresence>
        {allResults && (
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.5 }}>

            <div className="predict__all-cards">
              {allResults.map((r, i) => (
                <motion.div key={r.model} initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}>
                  <GlassCard hoverable className="predict__model-card">
                    <span className="predict__mc-name">{r.model}</span>
                    <SentimentBadge sentiment={r.sentiment} />
                    <span className="predict__mc-conf mono">{r.confidence.toFixed(1)}%</span>
                    <div className="predict__mc-bar-track">
                      <motion.div className="predict__mc-bar-fill"
                        style={{ backgroundColor: SENT_COLORS[r.sentiment] }}
                        initial={{ width: 0 }}
                        animate={{ width: `${r.confidence}%` }}
                        transition={{ duration: 0.6, delay: i * 0.08 }} />
                    </div>
                  </GlassCard>
                </motion.div>
              ))}
            </div>

            {consensus && (
              <GlassCard hoverable={false} className={`predict__consensus ${consensus.agree ? "predict__consensus--agree" : "predict__consensus--mixed"}`}>
                {consensus.agree
                  ? <span>{consensus.count} out of {consensus.total} models agree: <strong style={{ color: SENT_COLORS[consensus.sentiment] }}>
                      {consensus.sentiment.toUpperCase()}</strong> ✅</span>
                  : <span>Models disagree — mixed results ⚠️</span>}
              </GlassCard>
            )}

            <GlassCard hoverable={false} className="predict__all-chart-card">
              <AnimatedPlot
                data={[{
                  type: "bar", orientation: "h",
                  y: allResults.map(r => r.model),
                  x: allResults.map(r => r.confidence),
                  marker: { color: allResults.map(r => SENT_COLORS[r.sentiment]) },
                  text: allResults.map(r => `${r.confidence.toFixed(1)}%`),
                  textposition: "outside", textfont: { color: "#e6f1ff", size: 12 },
                  hovertemplate: "%{y}: %{x:.1f}%<extra></extra>",
                }]}
                layout={{
                  autosize: true, height: 220,
                  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 140, r: 60, t: 10, b: 30 },
                  xaxis: { range: [0, 105], color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)",
                    zeroline: false, tickfont: { color: "#aab6d3", size: 11 } },
                  yaxis: { color: "#aab6d3", tickfont: { color: "#aab6d3", size: 12 } },
                  font: { color: "#e6f1ff" },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 220 }}
              />
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ SECTION 4 — Prediction History ═══ */}
      <GlassCard hoverable={false} className="predict__history-card">
        <div className="predict__history-hdr">
          <h3>Recent Predictions</h3>
          {history.length > 0 && (
            <button className="predict__history-clear"
              onClick={() => { setHistory([]); saveHistory([]); }}>Clear</button>
          )}
        </div>
        {history.length === 0 ? (
          <div className="predict__history-empty">
            <p>No predictions yet. Try analyzing a tweet above!</p>
          </div>
        ) : (
          <div className="predict__history-table-wrap">
            <table className="predict__history-table">
              <thead><tr>
                <th>#</th><th>Text</th><th>Model</th><th>Result</th><th>Confidence</th>
              </tr></thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={h.id}>
                    <td className="mono">{i + 1}</td>
                    <td title={h.text}>{truncate(h.text, 60)}</td>
                    <td>{h.model}</td>
                    <td><SentimentBadge sentiment={h.sentiment} /></td>
                    <td className="mono">{h.confidence.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>
    </div>
  );
};

export default Predict;
