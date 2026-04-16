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
} from "../lib/api";
import "./Predict.css";


/* ── Types ────────────────────────────────────────────────────── */

type Sentiment = "positive" | "negative" | "neutral";

interface SingleResult {
  sentiment: Sentiment;
  confidence: number;
  positive_prob: number;
  negative_prob: number;
  neutral_prob: number;
}

interface HistoryEntry {
  id: number;
  text: string;
  sentiment: Sentiment;
  confidence: number;
  timestamp: number;
}

const SENT_COLORS: Record<Sentiment, string> = {
  positive: "#06d6a0", negative: "#ef476f", neutral: "#ffd166",
};

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

const normSingle = (raw: unknown): SingleResult => {
  const d = (raw as Record<string, unknown>) ?? {};
  const conf = pct(d.confidence ?? d.score ?? d.probability);

  // Backend returns { probabilities: { Positive: 0.66, Negative: 0.02, Neutral: 0.32 } }
  const probs = (d.probabilities && typeof d.probabilities === "object")
    ? d.probabilities as Record<string, unknown>
    : {};

  // Check nested probabilities object first, then flat fields
  const posProb = pct(probs.Positive ?? probs.positive ?? d.positive_prob ?? d.positive ?? 0);
  const negProb = pct(probs.Negative ?? probs.negative ?? d.negative_prob ?? d.negative ?? 0);
  const neuProb = pct(probs.Neutral ?? probs.neutral ?? d.neutral_prob ?? d.neutral ?? 0);

  return {
    sentiment: normSentiment(d.sentiment ?? d.label ?? d.prediction),
    confidence: conf,
    positive_prob: posProb,
    negative_prob: negProb,
    neutral_prob: neuProb,
  };
};

const loadHistory = (): HistoryEntry[] => {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
  catch { return []; }
};

const saveHistory = (entries: HistoryEntry[]) => {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
};

const truncate = (s: string, n: number) => s.length > n ? s.slice(0, n - 1) + "\u2026" : s;

/* ── Spinner ──────────────────────────────────────────────────── */

const Spinner = () => (
  <span className="predict__spinner" />
);

/* ═══════════════════════════════════════════════════════════════
   MAIN PREDICT PAGE
   ═══════════════════════════════════════════════════════════════ */

const Predict: React.FC = () => {
  const [text, setText] = useState("");
  const [validationError, setValidationError] = useState(false);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SingleResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);

  const addToHistory = (entry: Omit<HistoryEntry, "id" | "timestamp">) => {
    const newEntry: HistoryEntry = { ...entry, id: Date.now(), timestamp: Date.now() };
    const updated = [newEntry, ...history].slice(0, MAX_HISTORY);
    setHistory(updated);
    saveHistory(updated);
  };

  /* ── Analyze ──────────────────────────────────────────────── */
  const handleAnalyze = async () => {
    if (!text.trim()) { setValidationError(true); return; }
    setValidationError(false);
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await predictSentiment(text, "distilbert");
      const parsed = normSingle(res);
      setResult(parsed);
      addToHistory({ text: text.trim(), sentiment: parsed.sentiment, confidence: parsed.confidence });
    } catch (err) {
      setError(getApiErrorMessage(err, "Prediction failed."));
    }
    setLoading(false);
  };

  return (
    <div className="predict-page">
      {/* ═══ SECTION 1 — Input ═══ */}
      <GlassCard hoverable={false} className="predict__input-card">
        <h2 className="predict__title">Analyze Sentiment</h2>
        <p className="predict__subtitle">Enter any tweet or text to analyze its sentiment using VADER</p>

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

        <div className="predict__btn-row">
          <button className="predict__btn-primary" onClick={handleAnalyze}
            disabled={loading}>
            {loading ? <><Spinner /> Analyzing...</> : "Analyze Sentiment"}
          </button>
        </div>

        {error && <p className="predict__error-msg">{error}</p>}
      </GlassCard>

      {/* ═══ SECTION 2 — Result ═══ */}
      <AnimatePresence>
        {result && (
          <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.5 }}>
            <GlassCard hoverable={false} className="predict__result-card"
              style={{
                borderColor: SENT_COLORS[result.sentiment] + "4d",
                boxShadow: `0 0 30px ${SENT_COLORS[result.sentiment]}33`,
              } as React.CSSProperties}>
              <div className="predict__result-top">
                <div className="predict__result-left">
                  <span className="predict__result-label"
                    style={{ color: SENT_COLORS[result.sentiment] }}>
                    {result.sentiment.toUpperCase()}
                  </span>
                  <span className="predict__result-conf mono">
                    {result.confidence.toFixed(1)}%
                  </span>
                </div>
                <div className="predict__result-right">
                  <AnimatedPlot
                    data={[{
                      type: "indicator", mode: "gauge+number",
                      value: result.confidence,
                      number: { suffix: "%", font: { color: "#e6f1ff", size: 28 } },
                      gauge: {
                        axis: { range: [0, 100], tickfont: { color: "#aab6d3", size: 10 } },
                        bar: { color: SENT_COLORS[result.sentiment] },
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
                  const val = result[`${s}_prob` as keyof SingleResult] as number;
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

      {/* ═══ SECTION 3 — Prediction History ═══ */}
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
                <th>#</th><th>Text</th><th>Result</th><th>Confidence</th>
              </tr></thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={h.id}>
                    <td className="mono">{i + 1}</td>
                    <td title={h.text}>{truncate(h.text, 60)}</td>
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
