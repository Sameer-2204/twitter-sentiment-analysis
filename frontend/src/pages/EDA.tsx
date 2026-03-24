import React, { useCallback, useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  EntranceReveal,
  GlassCard,
  LoadingSkeleton,
} from "../components/common";
import {
  fetchBigrams,
  fetchClassDistribution,
  fetchHashtags,
  fetchMentions,
  fetchTweetLengths,
  fetchWordcloudData,
  fetchWordFrequency,
  fetchTrigrams,
} from "../lib/api";
import "./EDA.css";

/* ── Types ────────────────────────────────────────────────────── */

const AnimatedPlot = Plot as unknown as React.ComponentType<any>;

type SentimentFilter = "all" | "positive" | "negative" | "neutral";

interface BarEntry { name: string; value: number; color: string }
interface WordEntry { word: string; count: number }
interface CloudWord {
  text: string; x: number; y: number;
  rotate: number; fontSize: number; color: string; opacity: number;
}

const SENT_COLORS: Record<string, string> = {
  positive: "#06d6a0", negative: "#ef476f", neutral: "#ffd166",
};
const CLOUD_PALETTE = [
  "#06d6a0","#3a86ff","#ffd166","#ef476f","#76e5ff","#9bffb3","#f7b8ff","#ff9f68",
];
const CLOUD_H = 360;

/* ── Helpers ──────────────────────────────────────────────────── */

const safe = (v: unknown, fb = 0): number => {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") { const n = Number(v.replace("%","").trim()); return Number.isFinite(n) ? n : fb; }
  return fb;
};

const arr = (v: unknown): unknown[] => {
  if (Array.isArray(v)) return v;
  if (!v || typeof v !== "object") return [];
  const r = v as Record<string, unknown>;
  return (Array.isArray(r.data) ? r.data : Array.isArray(r.items) ? r.items :
    Array.isArray(r.words) ? r.words : Array.isArray(r.results) ? r.results : []) as unknown[];
};

const normBars = (raw: unknown): BarEntry[] => {
  const a = arr(raw);
  if (a.length) {
    return a.map((e) => {
      const o = (e as Record<string, unknown>) ?? {};
      const name = String(o.name ?? o.label ?? o.sentiment ?? o.class ?? "");
      return {
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value: safe(o.value ?? o.count ?? o.percentage ?? o.pct),
        color: SENT_COLORS[name.toLowerCase()] ?? "#4361ee",
      };
    });
  }
  if (raw && typeof raw === "object") {
    return Object.entries(raw as Record<string, unknown>).map(([k, v]) => ({
      name: k.charAt(0).toUpperCase() + k.slice(1),
      value: safe(v),
      color: SENT_COLORS[k.toLowerCase()] ?? "#4361ee",
    }));
  }
  return [];
};

const normWords = (raw: unknown): WordEntry[] => {
  const a = arr(raw);
  if (a.length) {
    return a.map((e) => {
      if (typeof e === "string") return { word: e, count: 1 };
      const o = (e as Record<string, unknown>) ?? {};
      return {
        word: String(o.word ?? o.text ?? o.ngram ?? o.keyword ?? o.term ?? ""),
        count: safe(o.count ?? o.frequency ?? o.value ?? o.weight),
      };
    }).filter((e) => e.word);
  }
  if (raw && typeof raw === "object") {
    return Object.entries(raw as Record<string, unknown>).map(([w, v]) => ({
      word: w, count: safe(v),
    }));
  }
  return [];
};

const normLengths = (raw: unknown): { charLengths: number[]; wordCounts: number[];
  avgLength: number; avgWords: number; avgPunctuation: number } => {
  const d = (raw as Record<string, unknown>) ?? {};
  return {
    charLengths: (arr(d.char_lengths ?? d.charLengths ?? d.lengths) as number[]).map(safe),
    wordCounts: (arr(d.word_counts ?? d.wordCounts ?? d.words) as number[]).map(safe),
    avgLength: safe(d.avg_length ?? d.avgLength ?? d.average_length),
    avgWords: safe(d.avg_words ?? d.avgWords ?? d.average_words),
    avgPunctuation: safe(d.avg_punctuation ?? d.avgPunctuation ?? d.average_punctuation),
  };
};

/* ── Sub-components ───────────────────────────────────────────── */

const SectionError: React.FC<{ msg: string }> = ({ msg }) => (
  <div className="eda__error"><p>{msg}</p></div>
);

const PanelSkeleton: React.FC<{ h?: number }> = ({ h = 320 }) => (
  <GlassCard hoverable={false} className="eda__panel">
    <div className="eda__panel-hdr"><LoadingSkeleton width="140px" height="20px" borderRadius="10px" /></div>
    <LoadingSkeleton height={`${h}px`} borderRadius="14px" />
  </GlassCard>
);

const darkLayout = (extra: Record<string, unknown> = {}) => ({
  autosize: true, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 50, r: 20, t: 10, b: 40 },
  font: { color: "#e6f1ff", size: 12 },
  xaxis: { color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
    tickfont: { color: "#aab6d3", size: 11 } },
  yaxis: { color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
    tickfont: { color: "#aab6d3", size: 11 } },
  ...extra,
});

/* ── Horizontal bar chart (Recharts) ──────────────────────────── */

const HBarChart: React.FC<{
  data: WordEntry[]; color?: string; maxItems?: number;
}> = ({ data, color = "#4361ee", maxItems = 20 }) => {
  const sliced = data.slice(0, maxItems);
  const mapped = sliced.map((d) => ({ name: d.word, value: d.count }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, sliced.length * 28)}>
      <BarChart data={mapped} layout="vertical" margin={{ left: 10, right: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis type="number" tick={{ fill: "#aab6d3", fontSize: 11 }} />
        <YAxis dataKey="name" type="category" width={100}
          tick={{ fill: "#aab6d3", fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: "#141432", border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 10, color: "#e6f1ff", fontSize: 13 }}
        />
        <Bar dataKey="value" radius={[0, 6, 6, 0]} animationDuration={800}>
          {mapped.map((_, i) => <Cell key={i} fill={color} opacity={0.85} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

/* ── Word Cloud (SVG) ─────────────────────────────────────────── */

const WordCloudSVG: React.FC<{ words: WordEntry[] }> = ({ words }) => {
  const hostRef = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState<CloudWord[]>([]);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!hostRef.current) return;
    const el = hostRef.current;
    const update = () => setWidth(el.clientWidth);
    update();
    const obs = new ResizeObserver(update);
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!width || !words.length) { setLayout([]); return; }
    let cancelled = false;
    (async () => {
      const mod = await import("d3-cloud");
      const create = mod.default as unknown as () => any;
      const counts = words.map((w) => w.count);
      const mn = Math.min(...counts), mx = Math.max(...counts);
      const scale = (c: number) => mn === mx ? 28 : 16 + ((c - mn) / (mx - mn)) * 38;

      create().size([width, CLOUD_H])
        .words(words.map((w, i) => ({
          text: w.word, size: scale(w.count),
          rotate: i % 7 === 0 ? 90 : 0,
          color: CLOUD_PALETTE[i % CLOUD_PALETTE.length],
          opacity: 0.7 + (i % 5) * 0.06,
        })))
        .font("Inter").padding(6).spiral("archimedean")
        .rotate((w: any) => w.rotate)
        .fontSize((w: any) => w.size)
        .on("end", (ws: any[]) => {
          if (cancelled) return;
          setLayout(ws.map((w) => ({
            text: String(w.text ?? ""), x: safe(w.x), y: safe(w.y),
            rotate: safe(w.rotate), fontSize: safe(w.size, 22),
            color: String(w.color ?? CLOUD_PALETTE[0]),
            opacity: safe(w.opacity, 0.8),
          })));
        })
        .start();
    })();
    return () => { cancelled = true; };
  }, [words, width]);

  return (
    <div ref={hostRef} className="eda__cloud-host">
      <svg width={width || "100%"} height={CLOUD_H}>
        <g transform={`translate(${width / 2},${CLOUD_H / 2})`}>
          {layout.map((w, i) => (
            <text key={i} textAnchor="middle"
              transform={`translate(${w.x},${w.y}) rotate(${w.rotate})`}
              style={{ fontSize: w.fontSize, fontFamily: "Inter", fill: w.color,
                opacity: w.opacity, transition: "all 0.4s ease" }}>
              {w.text}
            </text>
          ))}
        </g>
      </svg>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════
   MAIN EDA PAGE
   ═══════════════════════════════════════════════════════════════ */

const EDA: React.FC = () => {
  /* ── Filter state ─────────────────────────────────────────── */
  const [filterVal, setFilterVal] = useState<SentimentFilter>("all");
  const [appliedFilter, setAppliedFilter] = useState<SentimentFilter>("all");

  const sentimentParam = appliedFilter === "all" ? undefined : appliedFilter;

  const handleApply = () => setAppliedFilter(filterVal);
  const handleReset = () => { setFilterVal("all"); setAppliedFilter("all"); };

  /* ── Data states ──────────────────────────────────────────── */
  const [distLoading, setDistLoading]       = useState(true);
  const [distError, setDistError]           = useState<string | null>(null);
  const [dist, setDist]                     = useState<BarEntry[]>([]);

  const [lenLoading, setLenLoading]         = useState(true);
  const [lenError, setLenError]             = useState<string | null>(null);
  const [lengths, setLengths]               = useState<ReturnType<typeof normLengths>>({
    charLengths: [], wordCounts: [], avgLength: 0, avgWords: 0, avgPunctuation: 0 });

  const [cloudLoading, setCloudLoading]     = useState(true);
  const [cloudError, setCloudError]         = useState<string | null>(null);
  const [cloudWords, setCloudWords]         = useState<WordEntry[]>([]);

  const [freqLoading, setFreqLoading]       = useState(true);
  const [freqError, setFreqError]           = useState<string | null>(null);
  const [freqWords, setFreqWords]           = useState<WordEntry[]>([]);

  const [biLoading, setBiLoading]           = useState(true);
  const [biError, setBiError]               = useState<string | null>(null);
  const [bigrams, setBigrams]               = useState<WordEntry[]>([]);

  const [triLoading, setTriLoading]         = useState(true);
  const [triError, setTriError]             = useState<string | null>(null);
  const [trigrams, setTrigrams]             = useState<WordEntry[]>([]);

  const [hashLoading, setHashLoading]       = useState(true);
  const [hashError, setHashError]           = useState<string | null>(null);
  const [hashtags, setHashtags]             = useState<WordEntry[]>([]);

  const [mentLoading, setMentLoading]       = useState(true);
  const [mentError, setMentError]           = useState<string | null>(null);
  const [mentions, setMentions]             = useState<WordEntry[]>([]);

  /* ── Fetchers ─────────────────────────────────────────────── */

  const load = useCallback(async <T,>(
    fetcher: () => Promise<T | null>,
    normalizer: (d: T) => void,
    setLoading: (b: boolean) => void,
    setError: (e: string | null) => void,
    errMsg: string,
  ) => {
    setLoading(true); setError(null);
    const res = await fetcher();
    if (!res) setError(errMsg);
    else normalizer(res);
    setLoading(false);
  }, []);

  useEffect(() => {
    load(fetchClassDistribution, (d) => setDist(normBars(d)),
      setDistLoading, setDistError, "Unable to load class distribution.");
  }, [appliedFilter, load]);

  useEffect(() => {
    load(fetchTweetLengths, (d) => setLengths(normLengths(d)),
      setLenLoading, setLenError, "Unable to load tweet length data.");
  }, [appliedFilter, load]);

  useEffect(() => {
    load(() => fetchWordcloudData(sentimentParam), (d) => setCloudWords(normWords(d).slice(0, 60)),
      setCloudLoading, setCloudError, "Unable to load word cloud.");
  }, [appliedFilter, sentimentParam, load]);

  useEffect(() => {
    load(() => fetchWordFrequency(sentimentParam), (d) => setFreqWords(normWords(d).slice(0, 20)),
      setFreqLoading, setFreqError, "Unable to load word frequency.");
  }, [appliedFilter, sentimentParam, load]);

  useEffect(() => {
    load(() => fetchBigrams(sentimentParam), (d) => setBigrams(normWords(d).slice(0, 15)),
      setBiLoading, setBiError, "Unable to load bigrams.");
  }, [appliedFilter, sentimentParam, load]);

  useEffect(() => {
    load(() => fetchTrigrams(sentimentParam), (d) => setTrigrams(normWords(d).slice(0, 15)),
      setTriLoading, setTriError, "Unable to load trigrams.");
  }, [appliedFilter, sentimentParam, load]);

  useEffect(() => {
    load(() => fetchHashtags(sentimentParam), (d) => setHashtags(normWords(d).slice(0, 15)),
      setHashLoading, setHashError, "Unable to load hashtags.");
  }, [appliedFilter, sentimentParam, load]);

  useEffect(() => {
    load(() => fetchMentions(sentimentParam), (d) => setMentions(normWords(d).slice(0, 15)),
      setMentLoading, setMentError, "Unable to load mentions.");
  }, [appliedFilter, sentimentParam, load]);

  /* ── Plotly chart data ────────────────────────────────────── */

  const distBarData = [{
    type: "bar", x: dist.map(d => d.name), y: dist.map(d => d.value),
    marker: { color: dist.map(d => d.color), line: { width: 0 } },
    text: dist.map(d => String(d.value)), textposition: "outside",
    textfont: { color: "#e6f1ff", size: 13 },
    hovertemplate: "%{x}: %{y}<extra></extra>",
  }];

  const distPieData = [{
    type: "pie", hole: 0.55,
    labels: dist.map(d => d.name), values: dist.map(d => d.value),
    marker: { colors: dist.map(d => d.color) },
    textinfo: "percent", textfont: { color: "#e6f1ff", size: 13 },
    hovertemplate: "%{label}: %{value} (%{percent})<extra></extra>",
    sort: false, direction: "clockwise",
  }];

  const charHistData = [{
    type: "histogram", x: lengths.charLengths,
    marker: { color: "rgba(67, 97, 238, 0.7)", line: { width: 0 } },
    hovertemplate: "Length: %{x}<br>Count: %{y}<extra></extra>",
    nbinsx: 40,
  }];

  const wordHistData = [{
    type: "histogram", x: lengths.wordCounts,
    marker: { color: "rgba(6, 214, 160, 0.7)", line: { width: 0 } },
    hovertemplate: "Words: %{x}<br>Count: %{y}<extra></extra>",
    nbinsx: 30,
  }];

  /* ── Render ───────────────────────────────────────────────── */

  return (
    <div className="eda-page">
      {/* ── Filter Bar ──────────────────────────────────────── */}
      <div className="eda__filter-bar">
        <label className="eda__filter-label">Sentiment</label>
        <select className="eda__filter-select" value={filterVal}
          onChange={(e) => setFilterVal(e.target.value as SentimentFilter)}>
          <option value="all">All</option>
          <option value="positive">Positive</option>
          <option value="negative">Negative</option>
          <option value="neutral">Neutral</option>
        </select>
        <button className="eda__filter-apply" onClick={handleApply}>Apply</button>
        <button className="eda__filter-reset" onClick={handleReset}>Reset</button>
      </div>

      {/* ── Section 1: Class Distribution ───────────────────── */}
      <EntranceReveal>
        <h2 className="eda__section-title">Class Distribution</h2>
      </EntranceReveal>
      <div className="eda__grid eda__grid--2">
        {distLoading ? <PanelSkeleton /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Bar Chart</h3></div>
            {distError ? <SectionError msg={distError} /> : (
              <AnimatedPlot className="eda__plot" data={distBarData}
                layout={darkLayout({ height: 340, bargap: 0.35,
                  yaxis: { color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)",
                    zeroline: false, tickfont: { color: "#aab6d3", size: 11 } } })}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 340 }} />
            )}
          </GlassCard>
        )}
        {distLoading ? <PanelSkeleton /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Pie Chart</h3></div>
            {distError ? <SectionError msg={distError} /> : (
              <AnimatedPlot className="eda__plot" data={distPieData}
                layout={darkLayout({ height: 340, showlegend: true,
                  legend: { orientation: "h", x: 0.1, y: -0.1,
                    font: { color: "#e6f1ff", size: 12 } } })}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 340 }} />
            )}
          </GlassCard>
        )}
      </div>

      {/* ── Section 2: Text Statistics ──────────────────────── */}
      <EntranceReveal>
        <h2 className="eda__section-title">Text Statistics</h2>
      </EntranceReveal>
      <div className="eda__grid eda__grid--3">
        {lenLoading ? <PanelSkeleton /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Tweet Length</h3></div>
            {lenError ? <SectionError msg={lenError} /> : (
              <AnimatedPlot className="eda__plot" data={charHistData}
                layout={darkLayout({ height: 280,
                  xaxis: { title: { text: "Characters", font: { color: "#aab6d3", size: 11 } },
                    color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
                    tickfont: { color: "#aab6d3", size: 11 } },
                  bargap: 0.02 })}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 280 }} />
            )}
          </GlassCard>
        )}
        {lenLoading ? <PanelSkeleton /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Word Count</h3></div>
            {lenError ? <SectionError msg={lenError} /> : (
              <AnimatedPlot className="eda__plot" data={wordHistData}
                layout={darkLayout({ height: 280,
                  xaxis: { title: { text: "Words", font: { color: "#aab6d3", size: 11 } },
                    color: "#aab6d3", gridcolor: "rgba(255,255,255,0.04)", zeroline: false,
                    tickfont: { color: "#aab6d3", size: 11 } },
                  bargap: 0.02 })}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: 280 }} />
            )}
          </GlassCard>
        )}
        <GlassCard hoverable={false} className="eda__panel eda__panel--stats-card">
          <div className="eda__panel-hdr"><h3>Average Stats</h3></div>
          {lenLoading ? (
            <div className="eda__avg-skeleton">
              <LoadingSkeleton width="100%" height="20px" borderRadius="8px" />
              <LoadingSkeleton width="100%" height="20px" borderRadius="8px" />
              <LoadingSkeleton width="100%" height="20px" borderRadius="8px" />
            </div>
          ) : lenError ? <SectionError msg={lenError} /> : (
            <div className="eda__avg-rows">
              <div className="eda__avg-row">
                <span className="eda__avg-label">Avg Length</span>
                <span className="eda__avg-value mono">{lengths.avgLength.toFixed(1)} chars</span>
              </div>
              <div className="eda__avg-row">
                <span className="eda__avg-label">Avg Words</span>
                <span className="eda__avg-value mono">{lengths.avgWords.toFixed(1)}</span>
              </div>
              <div className="eda__avg-row">
                <span className="eda__avg-label">Avg Punctuation</span>
                <span className="eda__avg-value mono">{lengths.avgPunctuation.toFixed(1)}</span>
              </div>
            </div>
          )}
        </GlassCard>
      </div>

      {/* ── Section 3: Word Analysis ───────────────────────── */}
      <EntranceReveal>
        <h2 className="eda__section-title">Word Analysis</h2>
      </EntranceReveal>
      <div className="eda__grid eda__grid--2">
        {cloudLoading ? <PanelSkeleton h={CLOUD_H} /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Word Cloud</h3></div>
            {cloudError ? <SectionError msg={cloudError} /> : (
              <WordCloudSVG words={cloudWords} />
            )}
          </GlassCard>
        )}
        {freqLoading ? <PanelSkeleton h={480} /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Top 20 Words</h3></div>
            {freqError ? <SectionError msg={freqError} /> : (
              <HBarChart data={freqWords} color="#3a86ff" maxItems={20} />
            )}
          </GlassCard>
        )}
      </div>

      {/* ── Section 4: N-grams ─────────────────────────────── */}
      <EntranceReveal>
        <h2 className="eda__section-title">N-gram Analysis</h2>
      </EntranceReveal>
      <div className="eda__grid eda__grid--2">
        {biLoading ? <PanelSkeleton h={380} /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Top 15 Bigrams</h3></div>
            {biError ? <SectionError msg={biError} /> : (
              <HBarChart data={bigrams} color="#06d6a0" maxItems={15} />
            )}
          </GlassCard>
        )}
        {triLoading ? <PanelSkeleton h={380} /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Top 15 Trigrams</h3></div>
            {triError ? <SectionError msg={triError} /> : (
              <HBarChart data={trigrams} color="#ef476f" maxItems={15} />
            )}
          </GlassCard>
        )}
      </div>

      {/* ── Section 5: Hashtags & Mentions ─────────────────── */}
      <EntranceReveal>
        <h2 className="eda__section-title">Hashtags & Mentions</h2>
      </EntranceReveal>
      <div className="eda__grid eda__grid--2">
        {hashLoading ? <PanelSkeleton h={380} /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Top Hashtags</h3></div>
            {hashError ? <SectionError msg={hashError} /> : (
              <HBarChart data={hashtags} color="#ffd166" maxItems={15} />
            )}
          </GlassCard>
        )}
        {mentLoading ? <PanelSkeleton h={380} /> : (
          <GlassCard hoverable={false} className="eda__panel">
            <div className="eda__panel-hdr"><h3>Top Mentions</h3></div>
            {mentError ? <SectionError msg={mentError} /> : (
              <HBarChart data={mentions} color="#76e5ff" maxItems={15} />
            )}
          </GlassCard>
        )}
      </div>
    </div>
  );
};

export default EDA;
