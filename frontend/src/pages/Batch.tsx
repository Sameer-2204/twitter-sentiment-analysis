import React, { useCallback, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import PlotImport from "react-plotly.js";
import {
  EntranceReveal,
  GlassCard,
  LoadingSkeleton,
  MetricCard,
  SentimentBadge,
} from "../components/common";
import { predictBatch } from "../lib/api";
import "./Batch.css";

// Handle ESM / CJS default export mismatch
const PlotComponent =
  typeof (PlotImport as any).default === "function"
    ? (PlotImport as any).default
    : PlotImport;
const AnimatedPlot = PlotComponent as React.ComponentType<any>;

/* ── Types ────────────────────────────────────────────────────── */

type Sentiment = "positive" | "negative" | "neutral";
type SortKey = "index" | "confidence";
type SortDir = "asc" | "desc";

interface BatchResult {
  index: number;
  text: string;
  sentiment: Sentiment;
  confidence: number;
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

const PAGE_SIZE = 10;

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

const normResults = (raw: unknown): BatchResult[] => {
  const d = (raw as Record<string, unknown>) ?? {};
  const items = Array.isArray(raw) ? raw
    : Array.isArray(d.results) ? d.results
    : Array.isArray(d.predictions) ? d.predictions
    : Array.isArray(d.data) ? d.data : [];
  return items.map((item, i) => {
    const o = (item as Record<string, unknown>) ?? {};
    return {
      index: i + 1,
      text: String(o.text ?? o.tweet ?? o.tweet_text ?? ""),
      sentiment: normSentiment(o.sentiment ?? o.label ?? o.prediction),
      confidence: pct(o.confidence ?? o.score ?? o.probability),
    };
  });
};

const truncate = (s: string, n: number) => s.length > n ? s.slice(0, n - 1) + "…" : s;

const formatFileSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const generateSampleCSV = () => {
  const rows = [
    "text",
    "I absolutely love this product! Best purchase ever!",
    "This is the worst experience I've had with customer service.",
    "The weather today is okay, nothing special.",
    "Just finished an amazing book, highly recommend it!",
    "Traffic was terrible this morning, ruined my day.",
  ];
  return rows.join("\n");
};

/* ═══════════════════════════════════════════════════════════════
   MAIN BATCH PAGE
   ═══════════════════════════════════════════════════════════════ */

const Batch: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState("logistic_regression");
  const [dragOver, setDragOver] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const [results, setResults] = useState<BatchResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("index");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  /* ── File handling ───────────────────────────────────────── */
  const validateFile = (f: File): boolean => {
    if (!f.name.endsWith(".csv")) { setFileError("Only CSV files are accepted."); return false; }
    if (f.size > 5 * 1024 * 1024) { setFileError("File must be under 5MB."); return false; }
    setFileError(null);
    return true;
  };

  const handleFile = (f: File) => {
    if (validateFile(f)) setFile(f);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const removeFile = () => { setFile(null); setFileError(null); };

  /* ── Sample download ─────────────────────────────────────── */
  const downloadSample = () => {
    const blob = new Blob([generateSampleCSV()], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "sample_tweets.csv";
    a.click(); URL.revokeObjectURL(url);
  };

  /* ── Upload & process ────────────────────────────────────── */
  const handleStartAnalysis = async () => {
    if (!file) return;
    setProcessing(true); setError(null); setResults(null); setProgress(0);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress(p => Math.min(p + Math.random() * 15, 90));
    }, 400);

    const formData = new FormData();
    formData.append("file", file);

    const res = await predictBatch(formData, model);
    clearInterval(interval);
    setProgress(100);

    setTimeout(() => {
      if (!res) setError("Batch processing failed. Please check your file and try again.");
      else {
        const normalized = normResults(res);
        setResults(normalized);
        setPage(1); setSearch(""); setSortKey("index"); setSortDir("asc");
      }
      setProcessing(false);
    }, 300);
  };

  /* ── Results processing ──────────────────────────────────── */
  const filtered = useMemo(() => {
    if (!results) return [];
    let data = results;
    if (search) {
      const q = search.toLowerCase();
      data = data.filter(r => r.text.toLowerCase().includes(q) || r.sentiment.includes(q));
    }
    return [...data].sort((a, b) => {
      const aVal = a[sortKey]; const bVal = b[sortKey];
      return sortDir === "asc" ? (aVal > bVal ? 1 : -1) : (aVal < bVal ? 1 : -1);
    });
  }, [results, search, sortKey, sortDir]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const counts = useMemo(() => {
    if (!results) return { positive: 0, negative: 0, neutral: 0 };
    return results.reduce((acc, r) => { acc[r.sentiment]++; return acc; },
      { positive: 0, negative: 0, neutral: 0 });
  }, [results]);

  /* ── Export ──────────────────────────────────────────────── */
  const downloadResultsCSV = () => {
    if (!results) return;
    const header = "index,text,prediction,confidence\n";
    const rows = results.map(r =>
      `${r.index},"${r.text.replace(/"/g, '""')}",${r.sentiment},${r.confidence.toFixed(2)}`
    ).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "batch_results.csv";
    a.click(); URL.revokeObjectURL(url);
  };

  const downloadSummaryJSON = () => {
    if (!results) return;
    const summary = { total: results.length, ...counts,
      avgConfidence: (results.reduce((s, r) => s + r.confidence, 0) / results.length).toFixed(2),
      model };
    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "batch_summary.json";
    a.click(); URL.revokeObjectURL(url);
  };

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  };

  return (
    <div className="batch-page">
      {/* ═══ SECTION 1 — Upload ═══ */}
      {!processing && !results && (
        <div className="batch__upload-wrap">
          <GlassCard hoverable={false} className="batch__upload-card">
            <h2 className="batch__title">Batch Analysis</h2>
            <p className="batch__subtitle">Upload a CSV file with a "text" column to analyze sentiment in bulk</p>

            {/* Drag & Drop Zone */}
            <div className={`batch__dropzone ${dragOver ? "batch__dropzone--over" : ""} ${file ? "batch__dropzone--file" : ""}`}
              onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}
              onClick={() => !file && fileInputRef.current?.click()}>
              <input ref={fileInputRef} type="file" accept=".csv" hidden
                onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }} />

              {file ? (
                <div className="batch__file-info">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#06d6a0" strokeWidth="1.5">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                  <div className="batch__file-details">
                    <span className="batch__file-name">{file.name}</span>
                    <span className="batch__file-size">{formatFileSize(file.size)}</span>
                  </div>
                  <button className="batch__file-remove" onClick={(e) => { e.stopPropagation(); removeFile(); }}>✕</button>
                </div>
              ) : (
                <>
                  <svg className="batch__upload-icon" width="48" height="48" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <span className="batch__drop-text">Drag & drop your CSV file here</span>
                  <div className="batch__drop-divider"><span>or</span></div>
                  <button className="batch__browse-btn" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                    Browse Files
                  </button>
                  <span className="batch__drop-hint">CSV file with a 'text' column, max 5MB</span>
                </>
              )}
            </div>

            {fileError && <p className="batch__file-error">{fileError}</p>}

            <button className="batch__sample-link" onClick={downloadSample}>
              ↓ Download Sample CSV
            </button>

            <div className="batch__model-row">
              <label className="batch__model-label">Select Model</label>
              <select className="batch__model-select" value={model} onChange={e => setModel(e.target.value)}>
                {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
            </div>

            <button className="batch__start-btn" disabled={!file} onClick={handleStartAnalysis}>
              Start Analysis
            </button>
          </GlassCard>
        </div>
      )}

      {/* ═══ SECTION 2 — Processing ═══ */}
      <AnimatePresence>
        {processing && (
          <motion.div className="batch__processing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <GlassCard hoverable={false} className="batch__processing-card">
              <div className="batch__progress-bar">
                <motion.div className="batch__progress-fill"
                  initial={{ width: 0 }} animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.3 }} />
              </div>
              <p className="batch__processing-status">Processing tweets...</p>
              <p className="batch__processing-hint">This may take a moment</p>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ═══ SECTION 3 — Results ═══ */}
      {error && (
        <GlassCard hoverable={false} className="batch__error-card">
          <p>{error}</p>
          <button className="batch__retry-btn" onClick={() => { setError(null); setResults(null); }}>
            Try Again
          </button>
        </GlassCard>
      )}

      {results && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          {/* Summary */}
          <EntranceReveal stagger={0.1} className="batch__summary-grid">
            <MetricCard icon={<span style={{ fontSize: 20 }}>😊</span>} value={counts.positive}
              label="Positive" color={SENT_COLORS.positive} />
            <MetricCard icon={<span style={{ fontSize: 20 }}>😠</span>} value={counts.negative}
              label="Negative" color={SENT_COLORS.negative} />
            <MetricCard icon={<span style={{ fontSize: 20 }}>😐</span>} value={counts.neutral}
              label="Neutral" color={SENT_COLORS.neutral} />
          </EntranceReveal>

          {/* Charts */}
          <div className="batch__charts-grid">
            <GlassCard hoverable={false} className="batch__panel">
              <div className="batch__panel-hdr"><h3>Distribution</h3></div>
              <AnimatedPlot data={[{
                type: "pie", hole: 0.55,
                labels: ["Positive", "Negative", "Neutral"],
                values: [counts.positive, counts.negative, counts.neutral],
                marker: { colors: [SENT_COLORS.positive, SENT_COLORS.negative, SENT_COLORS.neutral] },
                textinfo: "percent", textfont: { color: "#e6f1ff", size: 13 },
                hovertemplate: "%{label}: %{value} (%{percent})<extra></extra>",
              }]} layout={{
                autosize: true, height: 300, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 20, r: 20, t: 10, b: 40 }, font: { color: "#e6f1ff" },
                showlegend: true, legend: { orientation: "h", x: 0.1, y: -0.1, font: { color: "#e6f1ff", size: 12 } },
              }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%", height: 300 }} />
            </GlassCard>

            <GlassCard hoverable={false} className="batch__panel">
              <div className="batch__panel-hdr"><h3>Confidence Distribution</h3></div>
              <AnimatedPlot data={[{
                type: "histogram", x: results.map(r => r.confidence),
                marker: { color: "rgba(67,97,238,0.7)" }, nbinsx: 20,
                hovertemplate: "Confidence: %{x:.0f}%<br>Count: %{y}<extra></extra>",
              }]} layout={{
                autosize: true, height: 300, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 50, r: 20, t: 10, b: 40 }, font: { color: "#e6f1ff" },
                xaxis: { title: { text: "Confidence %", font: { color: "#aab6d3", size: 11 } },
                  color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false },
                yaxis: { color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false },
                bargap: 0.05,
              }} config={{ displayModeBar: false, responsive: true }} style={{ width: "100%", height: 300 }} />
            </GlassCard>
          </div>

          {/* Results Table */}
          <GlassCard hoverable={false} className="batch__panel batch__results-panel">
            <div className="batch__panel-hdr batch__panel-hdr--inline">
              <h3>Results ({filtered.length} rows)</h3>
              <input className="batch__search" type="text" placeholder="Search results..."
                value={search} onChange={e => { setSearch(e.target.value); setPage(1); }} />
            </div>
            <div className="batch__table-wrap">
              <table className="batch__table">
                <thead><tr>
                  <th className="batch__th-sort" onClick={() => handleSort("index")}># {sortKey === "index" ? (sortDir === "asc" ? "↑" : "↓") : ""}</th>
                  <th>Text</th>
                  <th>Prediction</th>
                  <th className="batch__th-sort" onClick={() => handleSort("confidence")}>Confidence {sortKey === "confidence" ? (sortDir === "asc" ? "↑" : "↓") : ""}</th>
                </tr></thead>
                <tbody>
                  {paged.map(r => (
                    <tr key={r.index}>
                      <td className="mono">{r.index}</td>
                      <td title={r.text}>{truncate(r.text, 60)}</td>
                      <td><SentimentBadge sentiment={r.sentiment} /></td>
                      <td className="mono">{r.confidence.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {totalPages > 1 && (
              <div className="batch__pagination">
                <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
                <span className="batch__page-info mono">Page {page} of {totalPages}</span>
                <button disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>Next →</button>
              </div>
            )}
          </GlassCard>

          {/* Export */}
          <div className="batch__export-row">
            <button className="batch__export-primary" onClick={downloadResultsCSV}>Download Results CSV</button>
            <button className="batch__export-secondary" onClick={downloadSummaryJSON}>Download Summary JSON</button>
            <button className="batch__export-secondary" onClick={() => { setResults(null); setFile(null); }}>New Analysis</button>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Batch;
