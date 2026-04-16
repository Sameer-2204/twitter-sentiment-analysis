import React from "react";
import { motion } from "framer-motion";
import { EntranceReveal, GlassCard } from "../components/common";
import "./About.css";

/* ── Data ─────────────────────────────────────────────────────── */

const BACKEND_TECH = [
  { name: "Python 3.10+", role: "Core language" },
  { name: "FastAPI", role: "REST API framework" },
  { name: "TensorFlow / Keras", role: "LSTM, BiLSTM, CNN models" },
  { name: "PyTorch", role: "DistilBERT fine-tuning" },
  { name: "HuggingFace Transformers", role: "Pre-trained transformer" },
  { name: "scikit-learn", role: "Logistic Regression, TF-IDF" },
  { name: "spaCy", role: "Lemmatization" },
  { name: "VADER Sentiment", role: "Lexicon-based scoring" },
  { name: "pandas / NumPy", role: "Data processing" },
  { name: "Docker", role: "Containerized deployment" },
];

const FRONTEND_TECH = [
  { name: "React 18", role: "UI framework" },
  { name: "TypeScript", role: "Type safety" },
  { name: "Vite", role: "Build tooling" },
  { name: "Three.js", role: "3D neural network visualization" },
  { name: "GSAP", role: "Scroll & cursor animations" },
  { name: "Framer Motion", role: "Layout animations" },
  { name: "Recharts", role: "Charts & graphs" },
  { name: "d3-cloud", role: "Word cloud rendering" },
];

const MODELS_INFO = [
  {
    icon: "📊",
    name: "Logistic Regression",
    desc: "TF-IDF vectorized baseline with RandomizedSearchCV hyperparameter tuning, sublinear TF scaling, and unigram + bigram feature space (50K max features).",
    framework: "scikit-learn",
  },
  {
    icon: "🔄",
    name: "LSTM",
    desc: "Long Short-Term Memory network with GloVe 200d pre-trained embeddings, sequence padding at 128 tokens, and early stopping for convergence.",
    framework: "TensorFlow / Keras",
  },
  {
    icon: "↔️",
    name: "Bidirectional LSTM",
    desc: "Processes sequences in both forward and backward directions for richer contextual understanding. Uses shared GloVe embedding matrix.",
    framework: "TensorFlow / Keras",
  },
  {
    icon: "🧩",
    name: "CNN (1D Convolutional)",
    desc: "Multi-channel 1D Convolutional Neural Network with parallel filter sizes for multi-scale n-gram pattern detection and GlobalMaxPooling.",
    framework: "TensorFlow / Keras",
  },
  {
    icon: "🤖",
    name: "DistilBERT",
    desc: "Fine-tuned DistilBERT transformer with attention mechanism. Dynamically quantized to int8 on CPU for reduced memory and faster inference.",
    framework: "PyTorch / HuggingFace",
  },
];

const PIPELINE_STEPS = [
  { step: "1", label: "HTML Unescape", detail: "Converts HTML entities like &amp; → &" },
  { step: "2", label: "URL Removal", detail: "Strips http/https/www links" },
  { step: "3", label: "RT / @Mention Removal", detail: "Removes retweet tags and @handles" },
  { step: "4", label: "Hashtag Normalization", detail: "Removes # symbol, preserves text" },
  { step: "5", label: "Emoji → Text", detail: "Converts emojis to text descriptions via demoji" },
  { step: "6", label: "Lowercasing", detail: "Uniform case for consistent matching" },
  { step: "7", label: "Contraction Expansion", detail: "don't → do not, can't → cannot" },
  { step: "8", label: "Elongated Word Normalization", detail: "happpyyy → happy (3+ chars → 2)" },
  { step: "9", label: "Digit & Punctuation Removal", detail: "Strips non-alpha noise" },
  { step: "10", label: "spaCy Lemmatization", detail: "running → run, better → good" },
  { step: "11", label: "Negation-Aware Stopwords", detail: "Preserves 30+ negation words (not, never, don't…)" },
  { step: "12", label: "Short Tweet Filtering", detail: "Drops tweets with < 3 tokens" },
];

const API_ENDPOINTS = [
  { method: "POST", path: "/api/predict/", desc: "Single text prediction with model selector" },
  { method: "POST", path: "/api/predict/all", desc: "Parallel 5-model inference with majority-vote consensus" },
  { method: "POST", path: "/api/predict/batch", desc: "CSV batch prediction (up to 1000 rows, 5 MB)" },
  { method: "GET", path: "/api/dashboard/stats", desc: "Aggregate tweet statistics and class distribution" },
  { method: "GET", path: "/api/eda/*", desc: "Word frequency, n-grams, word clouds, hashtags, mentions" },
  { method: "GET", path: "/api/models/*", desc: "Model comparison, confusion matrices, training history" },
  { method: "GET", path: "/api/health", desc: "Per-model load status, uptime, version" },
];

const TIMELINE = [
  {
    phase: "Phase 1",
    title: "Data Pipeline & Training",
    items: [
      "Built production-grade DataCleaner with 12-step NLP pipeline",
      "Engineered TF-IDF, GloVe, and DistilBERT feature extractors",
      "Trained 5 models with centralized TrainingConfig dataclass",
      "Created unified train_all.py orchestrator with timing & F1 comparison",
      "Built comprehensive ModelEvaluator with ROC, PR curves, McNemar's test",
    ],
  },
  {
    phase: "Phase 2",
    title: "Backend API & Services",
    items: [
      "Designed FastAPI REST API with 18+ endpoints and Pydantic schemas",
      "Implemented SentimentPredictor with eager model loading & thread pool inference",
      "Created novel VADER compound → probability calibration mapping",
      "Built pre-computed EDA engine with dictionary-based caching",
      "Added CORS, Request-ID tracing, rate limiting, and cache-control middleware",
    ],
  },
  {
    phase: "Phase 3",
    title: "Frontend Dashboard",
    items: [
      "Designed 7-page React SPA with glass-morphism design system",
      "Built interactive Three.js neural network background",
      "Implemented GSAP TargetCursor with element snapping",
      "Created scroll-aware pill-morphing navigation bar",
      "Added VariableProximity font, ScrollReveal, and GlowCard effects",
    ],
  },
  {
    phase: "Phase 4",
    title: "Deployment & Polish",
    items: [
      "Containerized backend with ARM64-optimized Docker image",
      "Deployed API to Oracle Cloud Ampere A1 VM (4 OCPU, 24 GB RAM)",
      "Deployed frontend to Vercel CDN with global distribution",
      "Configured Nginx reverse proxy with Let's Encrypt SSL",
      "Performance tuned: DistilBERT int8 quantization, model preloading",
    ],
  },
];

/* ── SVG Icons ────────────────────────────────────────────────── */

const GitHubIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
  </svg>
);

const LinkedInIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
  </svg>
);

/* ═══════════════════════════════════════════════════════════════
   MAIN ABOUT PAGE
   ═══════════════════════════════════════════════════════════════ */

const About: React.FC = () => {
  return (
    <div className="about-page">
      {/* ═══ SECTION 1 — Project Overview ═══ */}
      <EntranceReveal>
        <GlassCard hoverable={false} className="about__card about__overview">
          <h1 className="about__main-title">About This Project</h1>
          <p className="about__desc">
            A full-stack, production-grade sentiment analysis platform that classifies Twitter text
            as <strong>Positive</strong>, <strong>Negative</strong>, or <strong>Neutral</strong> using
            five distinct machine learning architectures — from classical TF-IDF logistic regression
            to a fine-tuned DistilBERT transformer. The system analyzes 1.6 million tweets from
            the Sentiment140 dataset and serves real-time predictions, batch analysis, exploratory
            data analytics, and model comparison through a modern interactive dashboard.
          </p>
          <div className="about__highlight-grid">
            <div className="about__highlight">
              <span className="about__highlight-number">5</span>
              <span className="about__highlight-label">ML Models</span>
            </div>
            <div className="about__highlight">
              <span className="about__highlight-number">1.6M</span>
              <span className="about__highlight-label">Tweets Analyzed</span>
            </div>
            <div className="about__highlight">
              <span className="about__highlight-number">18+</span>
              <span className="about__highlight-label">API Endpoints</span>
            </div>
            <div className="about__highlight">
              <span className="about__highlight-number">7</span>
              <span className="about__highlight-label">Dashboard Pages</span>
            </div>
          </div>
        </GlassCard>
      </EntranceReveal>

      {/* ═══ SECTION 2 — Architecture ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">System Architecture</h2>
      </EntranceReveal>
      <EntranceReveal>
        <GlassCard hoverable={false} className="about__card about__architecture">
          <div className="about__arch-block">
            <div className="about__arch-tier">
              <span className="about__arch-badge about__arch-badge--frontend">Frontend</span>
              <span className="about__arch-detail">React 18 + Vite + TypeScript → Vercel CDN</span>
            </div>
            <div className="about__arch-connector">↕ HTTPS / REST API</div>
            <div className="about__arch-tier">
              <span className="about__arch-badge about__arch-badge--backend">Backend</span>
              <span className="about__arch-detail">FastAPI + Python → Oracle Cloud ARM64 VM</span>
            </div>
            <div className="about__arch-connector">↕ In-Memory</div>
            <div className="about__arch-tier">
              <span className="about__arch-badge about__arch-badge--ml">ML Engine</span>
              <span className="about__arch-detail">5 Models loaded eagerly at startup (scikit-learn, TF, PyTorch)</span>
            </div>
          </div>
          <div className="about__arch-features">
            <div className="about__arch-feature">
              <span className="about__arch-feature-icon">🔒</span>
              <span>CORS + Rate Limiting</span>
            </div>
            <div className="about__arch-feature">
              <span className="about__arch-feature-icon">🏷️</span>
              <span>Request-ID Tracing</span>
            </div>
            <div className="about__arch-feature">
              <span className="about__arch-feature-icon">⚡</span>
              <span>Tiered Cache-Control</span>
            </div>
            <div className="about__arch-feature">
              <span className="about__arch-feature-icon">🐳</span>
              <span>Docker + Nginx + SSL</span>
            </div>
          </div>
        </GlassCard>
      </EntranceReveal>

      {/* ═══ SECTION 3 — Tech Stack ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">Tech Stack</h2>
      </EntranceReveal>
      <div className="about__tech-grid">
        <EntranceReveal>
          <GlassCard hoverable={false} className="about__card">
            <h3 className="about__card-title">Backend & ML</h3>
            <div className="about__tech-list">
              {BACKEND_TECH.map((t) => (
                <div className="about__tech-item" key={t.name}>
                  <span className="about__tech-name">{t.name}</span>
                  <span className="about__tech-role">{t.role}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        </EntranceReveal>
        <EntranceReveal>
          <GlassCard hoverable={false} className="about__card">
            <h3 className="about__card-title">Frontend & Visualization</h3>
            <div className="about__tech-list">
              {FRONTEND_TECH.map((t) => (
                <div className="about__tech-item" key={t.name}>
                  <span className="about__tech-name">{t.name}</span>
                  <span className="about__tech-role">{t.role}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        </EntranceReveal>
      </div>

      {/* ═══ SECTION 4 — Models ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">ML Models</h2>
      </EntranceReveal>
      <div className="about__models-list">
        {MODELS_INFO.map((m, i) => (
          <EntranceReveal key={m.name}>
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
            >
              <GlassCard hoverable className="about__card about__model-card">
                <span className="about__model-icon">{m.icon}</span>
                <div className="about__model-info">
                  <div className="about__model-header">
                    <span className="about__model-name">{m.name}</span>
                    <span className="about__model-framework">{m.framework}</span>
                  </div>
                  <span className="about__model-desc">{m.desc}</span>
                </div>
              </GlassCard>
            </motion.div>
          </EntranceReveal>
        ))}
      </div>

      {/* ═══ SECTION 5 — NLP Pipeline ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">Text Preprocessing Pipeline</h2>
      </EntranceReveal>
      <EntranceReveal>
        <GlassCard hoverable={false} className="about__card">
          <p className="about__pipeline-intro">
            A 12-step NLP pipeline cleans raw tweets for training. A lightweight variant (omitting
            lemmatization and stopwords) is used for low-latency API inference, eliminating
            training-serving skew.
          </p>
          <div className="about__pipeline-grid">
            {PIPELINE_STEPS.map((s, i) => (
              <motion.div
                className="about__pipeline-step"
                key={s.step}
                initial={{ opacity: 0, x: -12 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.04 }}
              >
                <span className="about__pipeline-num">{s.step}</span>
                <div className="about__pipeline-text">
                  <span className="about__pipeline-label">{s.label}</span>
                  <span className="about__pipeline-detail">{s.detail}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </GlassCard>
      </EntranceReveal>

      {/* ═══ SECTION 6 — Key Technical Innovations ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">Key Technical Innovations</h2>
      </EntranceReveal>
      <div className="about__innovations-grid">
        <EntranceReveal>
          <GlassCard hoverable className="about__card about__innovation">
            <span className="about__innovation-icon">⚡</span>
            <h4 className="about__innovation-title">Parallel Multi-Model Inference</h4>
            <p className="about__innovation-desc">
              All 5 models run simultaneously via <code>ThreadPoolExecutor</code> with 60s
              timeout per model. Individual failures are isolated — one model's error
              never blocks the others. A majority-vote consensus label is computed from
              all successful predictions.
            </p>
          </GlassCard>
        </EntranceReveal>
        <EntranceReveal>
          <GlassCard hoverable className="about__card about__innovation">
            <span className="about__innovation-icon">📐</span>
            <h4 className="about__innovation-title">VADER Score Calibration</h4>
            <p className="about__innovation-desc">
              A novel piecewise-linear mapping converts VADER's compound score (−1 to +1)
              into a proper 3-class probability distribution. Unlike raw token proportions,
              this produces intuitive confidence values that sum to 1.0.
            </p>
          </GlassCard>
        </EntranceReveal>
        <EntranceReveal>
          <GlassCard hoverable className="about__card about__innovation">
            <span className="about__innovation-icon">🛡️</span>
            <h4 className="about__innovation-title">Negation-Aware Stopwords</h4>
            <p className="about__innovation-desc">
              Stopword removal preserves 30+ negation words (<em>not, never, don't, isn't</em>…)
              that are critical for sentiment — preventing "not good" from being reduced
              to just "good."
            </p>
          </GlassCard>
        </EntranceReveal>
        <EntranceReveal>
          <GlassCard hoverable className="about__card about__innovation">
            <span className="about__innovation-icon">🧊</span>
            <h4 className="about__innovation-title">Pre-Computed EDA Engine</h4>
            <p className="about__innovation-desc">
              Word frequencies, n-grams, word clouds, and statistics are computed at startup
              and stored in a dictionary cache with composite keys. Subsequent requests are
              served in sub-millisecond latency.
            </p>
          </GlassCard>
        </EntranceReveal>
      </div>

      {/* ═══ SECTION 7 — API Endpoints ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">API Endpoints</h2>
      </EntranceReveal>
      <EntranceReveal>
        <GlassCard hoverable={false} className="about__card">
          <div className="about__api-list">
            {API_ENDPOINTS.map((ep) => (
              <div className="about__api-row" key={ep.path}>
                <span className={`about__api-method about__api-method--${ep.method.toLowerCase()}`}>
                  {ep.method}
                </span>
                <code className="about__api-path">{ep.path}</code>
                <span className="about__api-desc">{ep.desc}</span>
              </div>
            ))}
          </div>
        </GlassCard>
      </EntranceReveal>

      {/* ═══ SECTION 8 — Development Timeline ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">Development Journey</h2>
      </EntranceReveal>
      <div className="about__timeline">
        {TIMELINE.map((phase, i) => (
          <EntranceReveal key={phase.phase}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
            >
              <GlassCard hoverable={false} className="about__card about__timeline-card">
                <div className="about__timeline-header">
                  <span className="about__timeline-phase">{phase.phase}</span>
                  <span className="about__timeline-title">{phase.title}</span>
                </div>
                <ul className="about__timeline-items">
                  {phase.items.map((item) => (
                    <li className="about__timeline-item" key={item}>
                      {item}
                    </li>
                  ))}
                </ul>
              </GlassCard>
            </motion.div>
          </EntranceReveal>
        ))}
      </div>

      {/* ═══ SECTION 9 — Dataset ═══ */}
      <EntranceReveal>
        <GlassCard hoverable={false} className="about__card about__dataset">
          <h3 className="about__card-title">Dataset</h3>
          <div className="about__dataset-rows">
            <div className="about__dataset-row">
              <span className="about__dataset-label">Source</span>
              <span className="about__dataset-value">Sentiment140 (Kaggle)</span>
            </div>
            <div className="about__dataset-row">
              <span className="about__dataset-label">Size</span>
              <span className="about__dataset-value">1.6 million tweets</span>
            </div>
            <div className="about__dataset-row">
              <span className="about__dataset-label">Labels</span>
              <span className="about__dataset-value">Positive (4), Neutral (2), Negative (0)</span>
            </div>
            <div className="about__dataset-row">
              <span className="about__dataset-label">Split Strategy</span>
              <span className="about__dataset-value">Train / Validation / Test with stratification</span>
            </div>
            <div className="about__dataset-row">
              <span className="about__dataset-label">Preprocessing</span>
              <span className="about__dataset-value">12-step NLP pipeline (see above)</span>
            </div>
          </div>
          <a
            className="about__dataset-link"
            href="https://www.kaggle.com/datasets/kazanova/sentiment140"
            target="_blank"
            rel="noopener noreferrer"
          >
            View on Kaggle →
          </a>
        </GlassCard>
      </EntranceReveal>

      {/* ═══ SECTION 10 — Deployment ═══ */}
      <EntranceReveal>
        <h2 className="about__section-title">Deployment</h2>
      </EntranceReveal>
      <div className="about__tech-grid">
        <EntranceReveal>
          <GlassCard hoverable={false} className="about__card">
            <h3 className="about__card-title">Frontend</h3>
            <div className="about__deploy-info">
              <p>React SPA built with Vite, deployed to <strong>Vercel CDN</strong> for global edge distribution with automatic HTTPS and CI/CD from GitHub.</p>
            </div>
          </GlassCard>
        </EntranceReveal>
        <EntranceReveal>
          <GlassCard hoverable={false} className="about__card">
            <h3 className="about__card-title">Backend</h3>
            <div className="about__deploy-info">
              <p>Dockerized FastAPI on <strong>Oracle Cloud Ampere A1</strong> (4 ARM OCPU, 24 GB RAM) with Nginx reverse proxy, Let's Encrypt SSL, and DistilBERT int8 quantization.</p>
            </div>
          </GlassCard>
        </EntranceReveal>
      </div>

      {/* ═══ SECTION 11 — Author ═══ */}
      <EntranceReveal>
        <GlassCard hoverable={false} className="about__card about__author">
          <h3 className="about__author-name">Sameer Tripathi</h3>
          <p className="about__author-role">Full-Stack Developer & ML Engineer</p>
          <div className="about__author-links">
            <a
              className="about__icon-link"
              href="https://github.com/Sameer-2204"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub"
            >
              <GitHubIcon />
            </a>
            <a
              className="about__icon-link"
              href="https://linkedin.com"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="LinkedIn"
            >
              <LinkedInIcon />
            </a>
          </div>
          <a
            className="about__source-link"
            href="https://github.com/Sameer-2204/twitter-sentiment-analysis"
            target="_blank"
            rel="noopener noreferrer"
          >
            View Source Code →
          </a>
        </GlassCard>
      </EntranceReveal>
    </div>
  );
};

export default About;
