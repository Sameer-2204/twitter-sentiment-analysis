import React, { lazy, Suspense, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useInView } from "framer-motion";
import { Link } from "react-router-dom";
import { useTheme } from "../lib/ThemeContext";
import { getApiErrorMessage, predictSentiment, fetchDashboardStats } from "../lib/api";
import "./Home.css";

const NeuralHeroScene = lazy(() => import("../components/three/NeuralHeroScene"));
import AetherFlowCanvas from "../components/AetherFlowCanvas";
import SplineScene from "../components/SplineScene";

import GlowCard from "../components/GlowCard";
import ScrollReveal from "../components/ScrollReveal";
import VariableProximity from "../components/VariableProximity";

/* ── Types ────────────────────────────────────────────────────── */

type Sentiment = "positive" | "negative" | "neutral";

type PredictionResult = {
  sentiment: Sentiment;
  confidence: number;
  model: string;
  inferenceTime?: number;
  probabilities: Record<Sentiment, number>;
};

/* ── Helpers ──────────────────────────────────────────────────── */

function pct(value: unknown): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
  if (!Number.isFinite(parsed)) return 0;
  return parsed <= 1 ? parsed * 100 : parsed;
}

function normalizeSentiment(value: unknown): Sentiment {
  const label = String(value ?? "").toLowerCase();
  if (label.includes("pos")) return "positive";
  if (label.includes("neg")) return "negative";
  return "neutral";
}

/* ── Count-up hook ────────────────────────────────────────────── */

function useCountUp(target: number, inView: boolean, duration = 1.8, decimals = 0): string {
  const [display, setDisplay] = useState("0");
  useEffect(() => {
    if (!inView) return;
    const startTime = performance.now();
    function animate(now: number) {
      const elapsed = Math.min((now - startTime) / (duration * 1000), 1);
      const eased = 1 - Math.pow(1 - elapsed, 3);
      setDisplay((target * eased).toFixed(decimals));
      if (elapsed < 1) requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  }, [inView, target, duration, decimals]);
  return display;
}

/* ── Animated stat ────────────────────────────────────────────── */

function AnimStat({ value, decimals = 0, suffix = "", label, color }: {
  value: number; decimals?: number; suffix?: string; label: string; color: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  const display = useCountUp(value, inView, 1.8, decimals);
  return (
    <div ref={ref} className="hp-stat">
      <span className="hp-stat__value mono" style={{ color }}>{display}{suffix}</span>
      <span className="hp-stat__label">{label}</span>
    </div>
  );
}

/* ── Mini donut chart ─────────────────────────────────────────── */

function MiniDonut({ data, size = 80 }: { data: { value: number; color: string }[]; size?: number }) {
  const total = data.reduce((s, d) => s + d.value, 0);
  const r = (size - 8) / 2;
  const circumference = 2 * Math.PI * r;
  let offset = 0;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="hp-mini-donut">
      {data.map((d, i) => {
        const pctVal = d.value / total;
        const dashArray = `${circumference * pctVal} ${circumference * (1 - pctVal)}`;
        const dashOffset = -offset;
        offset += circumference * pctVal;
        return (
          <circle
            key={i}
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={d.color}
            strokeWidth={7}
            strokeDasharray={dashArray}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            style={{ transition: "stroke-dasharray 1s ease" }}
          />
        );
      })}
    </svg>
  );
}

/* ── Theme toggle button ──────────────────────────────────────── */

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  return (
    <button
      type="button"
      className="hp-theme-toggle"
      onClick={toggleTheme}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
    >
      <motion.span
        key={theme}
        initial={{ rotate: -30, opacity: 0, scale: 0.8 }}
        animate={{ rotate: 0, opacity: 1, scale: 1 }}
        exit={{ rotate: 30, opacity: 0, scale: 0.8 }}
        transition={{ duration: 0.3 }}
        className="hp-theme-toggle__icon"
      >
        {theme === "dark" ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" />
            <line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" />
            <line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        )}
      </motion.span>
    </button>
  );
}

/* ── Data ─────────────────────────────────────────────────────── */

const liveExamples: Array<{ sentiment: Sentiment; text: string }> = [
  {
    sentiment: "positive",
    text: "The new AI feature rollout is incredible. The dashboard feels faster and way more useful today.",
  },
  {
    sentiment: "neutral",
    text: "Company reports earnings tomorrow and analysts are waiting for updated guidance from leadership.",
  },
  {
    sentiment: "negative",
    text: "Support has been painfully slow this week and the latest update introduced more bugs than fixes.",
  },
];

const models: Array<{
  key: string;
  name: string;
  type: string;
  accent: string;
  accuracy: string;
  description: string;
}> = [
    {
      key: "logistic_regression",
      name: "Logistic Regression",
      type: "Classical ML",
      accent: "#78a9ff",
      accuracy: "78.2%",
      description: "Interpretable TF-IDF baseline that keeps every deeper model honest.",
    },
    {
      key: "lstm",
      name: "LSTM",
      type: "Recurrent",
      accent: "#4cc9f0",
      accuracy: "84.1%",
      description: "Tracks how sentiment evolves through the tweet from left to right.",
    },
    {
      key: "bilstm",
      name: "BiLSTM",
      type: "Bidirectional",
      accent: "#06d6a0",
      accuracy: "86.8%",
      description: "Reads each tweet in both directions to catch nuance and reversal cues.",
    },
    {
      key: "cnn",
      name: "CNN",
      type: "Convolutional",
      accent: "#ef476f",
      accuracy: "85.3%",
      description: "Locks onto short high-signal patterns that often drive tweet polarity.",
    },
    {
      key: "distilbert",
      name: "DistilBERT",
      type: "Transformer",
      accent: "#ffd166",
      accuracy: "91%",
      description: "Fine-tuned transformer with contextual attention — best verified result.",
    },
  ];

const features = [
  {
    title: "Interactive Dashboard",
    to: "/dashboard",
    description: "Survey sentiment distributions, trend movement, and operational KPIs in one view.",
    preview: "dashboard" as const,
  },
  {
    title: "Deep EDA",
    to: "/eda",
    description: "Dive into word clouds, n-grams, hashtags, mentions, and distribution filters.",
    preview: "eda" as const,
  },
  {
    title: "Model Arena",
    to: "/models",
    description: "Compare all five architectures side by side and inspect where they diverge.",
    preview: "models" as const,
  },
  {
    title: "Live Prediction",
    to: "/predict",
    description: "Run instant sentiment analysis on any text and inspect confidence behavior.",
    preview: "predict" as const,
  },
];

const techRows = [
  ["Python", "FastAPI", "TensorFlow", "PyTorch", "scikit-learn", "NLTK", "HuggingFace", "pandas", "NumPy", "Matplotlib"],
  ["React", "TypeScript", "Vite", "Framer Motion", "Three.js", "Plotly.js", "Chart.js", "Recharts", "D3.js", "Axios"],
];

const sentimentTone: Record<Sentiment, { color: string; glow: string; emoji: string }> = {
  positive: {
    color: "var(--sentiment-positive)",
    glow: "rgba(6, 214, 160, 0.28)",
    emoji: "😊",
  },
  neutral: {
    color: "var(--sentiment-neutral)",
    glow: "rgba(255, 209, 102, 0.28)",
    emoji: "😐",
  },
  negative: {
    color: "var(--sentiment-negative)",
    glow: "rgba(239, 71, 111, 0.28)",
    emoji: "😟",
  },
};

/* ── Entrance animation wrapper ───────────────────────────────── */

function Reveal({ children, className = "", delay = 0, y = 24 }: {
  children: React.ReactNode; className?: string; delay?: number; y?: number;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.7, delay, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}

/* ── Feature preview mini-charts ──────────────────────────────── */

function FeaturePreview({ type, dashData }: { type: string; dashData: { pos: number; neu: number; neg: number } | null }) {
  if (type === "dashboard") {
    const data = dashData || { pos: 42, neu: 33, neg: 25 };
    return (
      <div className="hp-feat-preview hp-feat-preview--dashboard">
        <MiniDonut data={[
          { value: data.pos, color: "var(--sentiment-positive)" },
          { value: data.neu, color: "var(--sentiment-neutral)" },
          { value: data.neg, color: "var(--sentiment-negative)" },
        ]} size={72} />
        <div className="hp-feat-preview__legend">
          <span style={{ color: "var(--sentiment-positive)" }}>Positive {data.pos}%</span>
          <span style={{ color: "var(--sentiment-neutral)" }}>Neutral {data.neu}%</span>
          <span style={{ color: "var(--sentiment-negative)" }}>Negative {data.neg}%</span>
        </div>
      </div>
    );
  }
  if (type === "eda") {
    return (
      <div className="hp-feat-preview hp-feat-preview--eda">
        {["love", "great", "happy", "bad", "hate", "good", "sad", "nice"].map((w, i) => (
          <span key={w} className="hp-word-cloud-item" style={{
            fontSize: `${0.65 + Math.random() * 0.5}rem`,
            opacity: 0.5 + Math.random() * 0.5,
            animationDelay: `${i * 0.3}s`,
          }}>{w}</span>
        ))}
      </div>
    );
  }
  if (type === "models") {
    const bars = [78, 84, 87, 85, 90];
    const colors = ["#78a9ff", "#4cc9f0", "#06d6a0", "#ef476f", "#ffd166"];
    return (
      <div className="hp-feat-preview hp-feat-preview--models">
        {bars.map((h, i) => (
          <motion.div
            key={i}
            className="hp-feat-bar"
            style={{ backgroundColor: colors[i] }}
            initial={{ height: 0 }}
            whileInView={{ height: `${h}%` }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: i * 0.1, ease: "easeOut" }}
          />
        ))}
      </div>
    );
  }
  return (
    <div className="hp-feat-preview hp-feat-preview--predict">
      <div className="hp-gauge">
        <svg width="68" height="68" viewBox="0 0 68 68">
          <circle cx="34" cy="34" r="28" fill="none" stroke="var(--bg-card-border)" strokeWidth="6" />
          <motion.circle
            cx="34" cy="34" r="28" fill="none"
            stroke="var(--sentiment-positive)" strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${2 * Math.PI * 28 * 0.88} ${2 * Math.PI * 28 * 0.12}`}
            strokeDashoffset={2 * Math.PI * 28 * 0.25}
            initial={{ strokeDasharray: `0 ${2 * Math.PI * 28}` }}
            whileInView={{ strokeDasharray: `${2 * Math.PI * 28 * 0.88} ${2 * Math.PI * 28 * 0.12}` }}
            viewport={{ once: true }}
            transition={{ duration: 1.2, ease: "easeOut" }}
          />
        </svg>
        <span className="hp-gauge__label mono">88%</span>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   HOME COMPONENT
   ════════════════════════════════════════════════════════════════ */

const Home: React.FC = () => {
  const { theme } = useTheme();
  const [draft, setDraft] = useState("");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dashData, setDashData] = useState<{ pos: number; neu: number; neg: number } | null>(null);
  const [activeModelIdx, setActiveModelIdx] = useState(4);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const heroRef = useRef<HTMLDivElement>(null);

  // Detect scroll for navbar condensing
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    handleScroll(); // initial check
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Lock body scroll when mobile nav is open
  useEffect(() => {
    document.body.style.overflow = mobileNavOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [mobileNavOpen]);

  // Try to load dashboard stats for the mini donut
  useEffect(() => {
    fetchDashboardStats().then((data) => {
      if (data?.sentiment_distribution) {
        const dist = data.sentiment_distribution;
        const total = (dist.positive || 0) + (dist.neutral || 0) + (dist.negative || 0);
        if (total > 0) {
          setDashData({
            pos: Math.round((dist.positive / total) * 100),
            neu: Math.round((dist.neutral / total) * 100),
            neg: Math.round((dist.negative / total) * 100),
          });
        }
      }
    }).catch(() => { /* gracefully fallback */ });
  }, []);

  async function handleAnalyze() {
    if (!draft.trim()) {
      setError("Enter a tweet or tap one of the examples to run inference.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await predictSentiment(draft.trim(), "distilbert");
      const payload = response as Record<string, unknown>;
      const probabilities: Record<Sentiment, number> = { positive: 0, neutral: 0, negative: 0 };
      if (payload.probabilities && typeof payload.probabilities === "object") {
        for (const [key, value] of Object.entries(payload.probabilities as Record<string, unknown>)) {
          probabilities[normalizeSentiment(key)] = pct(value);
        }
      }
      probabilities.positive ||= pct(payload.positive_prob);
      probabilities.neutral ||= pct(payload.neutral_prob);
      probabilities.negative ||= pct(payload.negative_prob);
      const predictedSentiment = normalizeSentiment(payload.sentiment ?? payload.label ?? payload.prediction);
      setPrediction({
        sentiment: predictedSentiment,
        confidence: pct(payload.confidence ?? payload.score ?? payload.probability) || probabilities[predictedSentiment],
        model: String(payload.model_used ?? payload.model ?? "VADER"),
        inferenceTime: typeof payload.inference_time === "number" ? payload.inference_time : undefined,
        probabilities,
      });
    } catch (unknownError) {
      setPrediction(null);
      setError(getApiErrorMessage(unknownError, "Prediction failed."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="hp" data-theme={theme}>
      {/* Gradient background layers */}
      <div className="hp__bg" aria-hidden="true">
        <div className="hp__bg-gradient hp__bg-gradient--1" />
        <div className="hp__bg-gradient hp__bg-gradient--2" />
        <div className="hp__bg-gradient hp__bg-gradient--3" />
        <div className="hp__noise" />
      </div>

      {/* Aether Flow interactive particle mesh — full page */}
      <div className="hp__aether-layer" aria-hidden="true">
        <AetherFlowCanvas />
      </div>

      {/* ── NAVBAR ──────────────────────────────────────────────── */}
      <header className={`hp__nav-wrap ${scrolled && !mobileNavOpen ? "hp__nav-wrap--scrolled" : ""} ${mobileNavOpen ? "hp__nav-wrap--open" : ""}`}>
        <nav className={`hp__navbar ${scrolled ? "hp__navbar--scrolled" : ""}`}>
          <a href="#top" className="hp__logo" aria-label="Home">
            <span className="hp__logo-mark">TS</span>
            <span className="hp__logo-text">Sentiment<strong>AI</strong></span>
          </a>

          <div className="hp__nav-links">
            <Link to="/dashboard">Dashboard</Link>
            <Link to="/eda">EDA</Link>
            <Link to="/models">Models</Link>
            <Link to="/predict">Predict</Link>
          </div>

          <div className="hp__nav-actions">
            <ThemeToggle />
            <Link to="/predict" className="hp__nav-cta cursor-target">Try Prediction</Link>
          </div>

          {/* Hamburger toggle — mobile only */}
          <button
            type="button"
            className="hp__hamburger"
            onClick={() => setMobileNavOpen(!mobileNavOpen)}
            aria-label="Toggle navigation"
          >
            <span className={`hp__hamburger-bar ${mobileNavOpen ? "hp__hamburger-bar--open" : ""}`} />
            <span className={`hp__hamburger-bar ${mobileNavOpen ? "hp__hamburger-bar--open" : ""}`} />
            <span className={`hp__hamburger-bar ${mobileNavOpen ? "hp__hamburger-bar--open" : ""}`} />
          </button>
        </nav>

        {/* Mobile navigation overlay */}
        <AnimatePresence>
          {mobileNavOpen && (
            <motion.div
              className="hp__mobile-nav"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.25 }}
            >
              <div className="hp__mobile-nav-links">
                <Link to="/dashboard" onClick={() => setMobileNavOpen(false)}>Dashboard</Link>
                <Link to="/eda" onClick={() => setMobileNavOpen(false)}>EDA</Link>
                <Link to="/models" onClick={() => setMobileNavOpen(false)}>Models</Link>
                <Link to="/predict" onClick={() => setMobileNavOpen(false)}>Predict</Link>
              </div>
              <div className="hp__mobile-nav-actions">
                <Link to="/predict" className="hp__btn hp__btn--primary hp__btn--full" onClick={() => setMobileNavOpen(false)}>
                  Try Prediction
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      <main>
        {/* ── HERO ────────────────────────────────────────────── */}
        <section id="top" className="hp__hero">
          <div className="hp__hero-scene" aria-hidden="true">
            <Suspense fallback={<div className="hp__hero-scene-fallback" />}>
              <NeuralHeroScene />
            </Suspense>
          </div>

          <div className="hp__hero-split">
            {/* Left — text content */}
            <div className="hp__hero-content" ref={heroRef}>
              <Reveal>
                <span className="hp__badge">
                  <span className="hp__badge-dot" />
                  NLP + Deep Learning
                </span>
              </Reveal>

              <Reveal delay={0.1}>
                <h1 className="hp__hero-title">
                  <span className="hp__hero-line">
                    <VariableProximity
                      label="Read the emotional"
                      className="hp__hero-title-proximity"
                      fromFontVariationSettings="'wght' 420, 'opsz' 12"
                      toFontVariationSettings="'wght' 620, 'opsz' 24"
                      containerRef={heroRef}
                      radius={100}
                      falloff="linear"
                    />
                  </span>
                  <span className="hp__hero-line">
                    <VariableProximity
                      label="pulse of Twitter with a"
                      className="hp__hero-title-proximity"
                      fromFontVariationSettings="'wght' 420, 'opsz' 12"
                      toFontVariationSettings="'wght' 620, 'opsz' 24"
                      containerRef={heroRef}
                      radius={100}
                      falloff="linear"
                    />
                  </span>
                  <span className="hp__hero-line">
                    <VariableProximity
                      label="benchmarked AI stack."
                      className="hp__hero-title-proximity"
                      fromFontVariationSettings="'wght' 420, 'opsz' 12"
                      toFontVariationSettings="'wght' 620, 'opsz' 24"
                      containerRef={heroRef}
                      radius={100}
                      falloff="linear"
                    />
                  </span>
                </h1>
              </Reveal>

              <Reveal delay={0.2}>
                <p className="hp__hero-sub">
                  A full-stack sentiment analysis platform processing 1.6 million tweets
                  through five ML models — from classical baselines to fine-tuned transformers.
                </p>
              </Reveal>

              <Reveal delay={0.3}>
                <div className="hp__hero-stats">
                  <span className="hp__hero-stat"><strong>16.5K+</strong> tweets</span>
                  <span className="hp__hero-stat-dot">·</span>
                  <span className="hp__hero-stat"><strong>5</strong> models</span>
                  <span className="hp__hero-stat-dot">·</span>
                  <span className="hp__hero-stat"><strong>91%</strong> best accuracy</span>
                </div>
              </Reveal>

              <Reveal delay={0.4}>
                <div className="hp__hero-ctas">
                  <Link to="/dashboard" className="hp__btn hp__btn--primary cursor-target">
                    Explore Dashboard
                  </Link>
                  <Link to="/predict" className="hp__btn hp__btn--ghost cursor-target">
                    Run Live Prediction
                  </Link>
                </div>
              </Reveal>
            </div>

            {/* Right — Spline 3D robot */}
            <Reveal className="hp__hero-spline" delay={0.2} y={0}>
              <SplineScene
                scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                className="hp__spline-canvas"
              />
            </Reveal>
          </div>
        </section>

        {/* ── LIVE DEMO ────────────────────────────────────────── */}
        <section className="hp__demo">
          <div className="hp__container">
            <Reveal>
              <div className="hp__demo-header">
                <ScrollReveal
                  baseOpacity={0.1}
                  enableBlur
                  baseRotation={3}
                  blurStrength={4}
                  containerClassName="hp__section-title"
                  textClassName="hp__scroll-reveal-heading"
                >
                  Try it yourself
                </ScrollReveal>
                <p className="hp__section-desc">
                  Paste a tweet and watch the model analyze sentiment in real time.
                </p>
              </div>
            </Reveal>

            <div className="hp__demo-grid">
              <GlowCard glowColor="blue" className="hp__demo-input-col">
                <textarea
                  id="home-live-predictor"
                  className="hp__textarea"
                  placeholder="Type or paste a tweet here..."
                  maxLength={280}
                  value={draft}
                  onChange={(e) => { setDraft(e.target.value); setError(null); }}
                />
                <div className="hp__char-count">{draft.length}/280</div>



                <div className="hp__examples">
                  {liveExamples.map((ex) => (
                    <button
                      key={ex.text}
                      type="button"
                      className={`hp__example-btn hp__example-btn--${ex.sentiment} cursor-target`}
                      onClick={() => { setDraft(ex.text); setError(null); }}
                    >
                      <span className="hp__example-dot" />
                      {ex.text.slice(0, 60)}...
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  className="hp__btn hp__btn--primary hp__btn--full cursor-target"
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  {loading ? "Analyzing..." : "Analyze Sentiment"}
                </button>
                {error && <p className="hp__error">{error}</p>}
              </GlowCard>

              <GlowCard glowColor="purple" className="hp__demo-result-col">
                <AnimatePresence mode="wait">
                  {loading ? (
                    <motion.div
                      key="loading"
                      className="hp__result-state"
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -16 }}
                    >
                      <div className="hp__spinner" />
                      <p>Running inference...</p>
                    </motion.div>
                  ) : prediction ? (
                    <motion.div
                      key={`${prediction.model}-${prediction.sentiment}`}
                      className="hp__result-live"
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -16 }}
                    >
                      <div className="hp__result-main">
                        <span className="hp__result-emoji">{sentimentTone[prediction.sentiment].emoji}</span>
                        <div className="hp__result-info">
                          <span
                            className="hp__result-label"
                            style={{ color: sentimentTone[prediction.sentiment].color }}
                          >
                            {prediction.sentiment}
                          </span>
                          <strong className="hp__result-conf">{prediction.confidence.toFixed(1)}%</strong>
                        </div>
                      </div>

                      <div className="hp__result-bars">
                        {(["positive", "neutral", "negative"] as Sentiment[]).map((s) => (
                          <div className="hp__bar-row" key={s}>
                            <div className="hp__bar-head">
                              <span>{s}</span>
                              <span className="mono">{prediction.probabilities[s].toFixed(1)}%</span>
                            </div>
                            <div className="hp__bar-track">
                              <motion.div
                                className="hp__bar-fill"
                                style={{ backgroundColor: sentimentTone[s].color }}
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.max(prediction.probabilities[s], 2)}%` }}
                                transition={{ duration: 0.65, ease: "easeOut" }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="hp__result-meta">
                        {prediction.inferenceTime !== undefined && (
                          <span>{(prediction.inferenceTime * 1000).toFixed(0)}ms</span>
                        )}
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="idle"
                      className="hp__result-state hp__result-state--idle"
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -16 }}
                    >
                      <div className="hp__idle-bars">
                        <span /><span /><span />
                      </div>
                      <p>Results will appear here</p>
                      <span className="hp__idle-hint">Choose an example or type your own tweet</span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </GlowCard>
            </div>
          </div>
        </section>

        {/* ── NUMBERS ─────────────────────────────────────────── */}
        <section className="hp__numbers">
          <div className="hp__container">
            <Reveal>
              <div className="hp__numbers-row">
                <AnimStat value={16.5} decimals={1} suffix="K+" label="Tweets analyzed" color="var(--accent-primary)" />
                <div className="hp__numbers-divider" />
                <AnimStat value={5} label="ML models trained" color="var(--sentiment-positive)" />
                <div className="hp__numbers-divider" />
                <AnimStat value={91} decimals={1} suffix="%" label="Best accuracy" color="var(--sentiment-neutral)" />
                <div className="hp__numbers-divider" />
                <AnimStat value={3} label="Sentiment classes" color="var(--sentiment-negative)" />
              </div>
            </Reveal>
          </div>
        </section>

        {/* ── MODELS ──────────────────────────────────────────── */}
        <section className="hp__models">
          <div className="hp__container">
            <Reveal>
              <ScrollReveal
                baseOpacity={0.1}
                enableBlur
                baseRotation={3}
                blurStrength={4}
                containerClassName="hp__section-title hp__section-title--center"
                textClassName="hp__scroll-reveal-heading"
              >
                From classical ML to transformer attention
              </ScrollReveal>
              <p className="hp__section-desc hp__section-desc--center">
                Five architectures forming a progression from interpretable baselines to contextual language models.
              </p>
            </Reveal>

            <div className="hp__model-scroll">
              {models.map((m, i) => (
                <Reveal key={m.key} delay={i * 0.08}>
                  <GlowCard glowColor="purple" className="hp__model-card-glow">
                    <button
                      type="button"
                      className={`hp__model-card cursor-target ${activeModelIdx === i ? "hp__model-card--active" : ""}`}
                      onClick={() => setActiveModelIdx(i)}
                      style={{ "--model-accent": m.accent } as React.CSSProperties}
                    >
                      <span className="hp__model-num mono">0{i + 1}</span>
                      <div className="hp__model-body">
                        <div className="hp__model-top">
                          <strong>{m.name}</strong>
                          <span className="hp__model-type">{m.type}</span>
                        </div>
                        <p>{m.description}</p>
                        <span className="hp__model-acc mono">{m.accuracy}</span>
                      </div>
                      <div
                        className="hp__model-accent-bar"
                        style={{ backgroundColor: m.accent }}
                      />
                    </button>
                  </GlowCard>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ── FEATURES ────────────────────────────────────────── */}
        <section className="hp__features">
          <div className="hp__container">
            <Reveal>
              <ScrollReveal
                baseOpacity={0.1}
                enableBlur
                baseRotation={3}
                blurStrength={4}
                containerClassName="hp__section-title hp__section-title--center"
                textClassName="hp__scroll-reveal-heading"
              >
                Explore the full platform
              </ScrollReveal>
              <p className="hp__section-desc hp__section-desc--center">
                Four interactive surfaces for every stage of your sentiment analysis workflow.
              </p>
            </Reveal>

            <div className="hp__feat-list">
              {features.map((f, i) => (
                <Reveal key={f.title} delay={i * 0.08}>
                  <GlowCard glowColor="blue" className="hp__feat-item-glow">
                    <Link to={f.to} className="hp__feat-item cursor-target" style={{ flexDirection: i % 2 === 0 ? "row" : "row-reverse" }}>
                      <div className="hp__feat-text">
                        <h3>{f.title}</h3>
                        <p>{f.description}</p>
                        <span className="hp__feat-link">
                          Open →
                        </span>
                      </div>
                      <div className="hp__feat-visual">
                        <FeaturePreview type={f.preview} dashData={dashData} />
                      </div>
                    </Link>
                  </GlowCard>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* ── TECH MARQUEE (full-width) ───────────────────────── */}
        <section className="hp__marquee-section">
          {techRows.map((row, rowIdx) => (
            <div
              key={rowIdx}
              className={`hp__marquee ${rowIdx % 2 === 1 ? "hp__marquee--reverse" : ""}`}
            >
              <div className="hp__marquee-track">
                {[...row, ...row, ...row, ...row].map((item, i) => (
                  <span key={`${item}-${i}`} className="hp__tech-pill">
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </section>

        {/* ── CLOSING CTA ─────────────────────────────────────── */}
        <section className="hp__cta-section">
          <div className="hp__container">
            <Reveal>
              <div className="hp__cta-content">
                <div className="hp__cta-glow" aria-hidden="true" />
                <ScrollReveal
                  baseOpacity={0.1}
                  enableBlur
                  baseRotation={3}
                  blurStrength={4}
                  containerClassName="hp__cta-title"
                  textClassName="hp__scroll-reveal-heading hp__scroll-reveal-heading--cta"
                >
                  Ready to explore sentiment intelligence?
                </ScrollReveal>
                <p className="hp__cta-sub">
                  Dive into the dashboard, compare architectures, or test the NLP stack on your own text.
                </p>
                <div className="hp__cta-buttons">
                  <Link to="/dashboard" className="hp__btn hp__btn--primary cursor-target">Explore Dashboard</Link>
                  <Link to="/predict" className="hp__btn hp__btn--ghost cursor-target">Try Prediction</Link>
                </div>
              </div>
            </Reveal>
          </div>
        </section>

        {/* ── FOOTER ──────────────────────────────────────────── */}
        <footer className="hp__footer">
          <div className="hp__container hp__footer-inner">
            <div className="hp__footer-left">
              <strong>Twitter Sentiment Analyzer</strong>
              <span>Benchmarking ML models across 16.5K+ tweets.</span>
            </div>
            <div className="hp__footer-links">
              <Link to="/dashboard">Dashboard</Link>
              <Link to="/eda">EDA</Link>
              <Link to="/models">Models</Link>
              <Link to="/predict">Predict</Link>
              <a href="https://github.com/Sameer-2204/twitter-sentiment-analysis" target="_blank" rel="noreferrer">GitHub</a>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
};

export default Home;
