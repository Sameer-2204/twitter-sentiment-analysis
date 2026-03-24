import React, { useRef } from "react";
import { Link } from "react-router-dom";
import { motion, useScroll, useTransform, useMotionValue, useSpring } from "framer-motion";
import { EntranceReveal, GlassCard, MetricCard } from "../components/common";
import ParticleNetwork from "../components/three/ParticleNetwork";
import "./Home.css";

/* ── Stagger word animation ───────────────────────────────────── */

const WordStagger: React.FC<{ text: string; className?: string }> = ({
  text,
  className,
}) => {
  const words = text.split(" ");
  return (
    <span className={className}>
      {words.map((word, i) => (
        <motion.span
          key={i}
          className="home__word"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: 0.5,
            delay: 0.3 + i * 0.08,
            ease: "easeOut",
          }}
        >
          {word}{" "}
        </motion.span>
      ))}
    </span>
  );
};

/* ── 3D Tilt Card ─────────────────────────────────────────────── */

interface TiltCardProps {
  children: React.ReactNode;
  className?: string;
}

const TiltCard: React.FC<TiltCardProps> = ({ children, className }) => {
  const rotateX = useMotionValue(0);
  const rotateY = useMotionValue(0);
  const smoothX = useSpring(rotateX, { stiffness: 150, damping: 20 });
  const smoothY = useSpring(rotateY, { stiffness: 150, damping: 20 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    rotateX.set(y * -10);
    rotateY.set(x * 10);
  };

  const handleMouseLeave = () => {
    rotateX.set(0);
    rotateY.set(0);
  };

  return (
    <motion.div
      className={`home__tilt-wrapper ${className || ""}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateX: smoothX,
        rotateY: smoothY,
        transformStyle: "preserve-3d",
        perspective: 1000,
      }}
    >
      {children}
    </motion.div>
  );
};

/* ── Scroll Chevron ───────────────────────────────────────────── */

const ScrollChevron = () => (
  <motion.div
    className="home__scroll-indicator"
    animate={{ y: [0, 10, 0], opacity: [0.3, 1, 0.3] }}
    transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
  >
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  </motion.div>
);

/* ── Feature data ─────────────────────────────────────────────── */

const FEATURES = [
  {
    icon: "📊",
    title: "Interactive Dashboard",
    desc: "Explore sentiment trends, distributions, and word patterns across 1.6M tweets",
  },
  {
    icon: "🤖",
    title: "5 ML Models",
    desc: "Compare Logistic Regression, LSTM, BiLSTM, CNN, and DistilBERT performance",
  },
  {
    icon: "🔮",
    title: "Live Prediction",
    desc: "Type any text and get instant sentiment analysis with confidence scores",
  },
  {
    icon: "📁",
    title: "Batch Analysis",
    desc: "Upload CSV files for bulk sentiment prediction and downloadable results",
  },
];

const TECH_STACK = [
  { emoji: "🐍", name: "Python" },
  { emoji: "🔥", name: "TensorFlow" },
  { emoji: "⚡", name: "PyTorch" },
  { emoji: "🧠", name: "scikit-learn" },
  { emoji: "🚀", name: "FastAPI" },
  { emoji: "⚛️", name: "React" },
  { emoji: "📈", name: "Plotly.js" },
  { emoji: "📊", name: "Chart.js" },
  { emoji: "🎨", name: "Three.js" },
  { emoji: "▲", name: "Vercel" },
  { emoji: "🚂", name: "Railway" },
];

/* ── Main Component ───────────────────────────────────────────── */

const Home: React.FC = () => {
  const heroRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLElement>(null);

  const { scrollYProgress } = useScroll();
  const orbY1 = useTransform(scrollYProgress, [0, 1], [0, -150]);
  const orbY2 = useTransform(scrollYProgress, [0, 1], [0, -100]);
  const orbY3 = useTransform(scrollYProgress, [0, 1], [0, -80]);

  return (
    <div className="home">
      {/* ═══════ SECTION 1 — Hero ═══════ */}
      <section className="home__hero" ref={heroRef}>
        {/* Three.js background */}
        <div className="home__hero-canvas">
          <ParticleNetwork />
        </div>

        {/* Content */}
        <div className="home__hero-content">
          <motion.span
            className="home__eyebrow"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            NLP + DEEP LEARNING
          </motion.span>

          <h1 className="home__title">
            <WordStagger text="Twitter Sentiment Analyzer" />
          </h1>

          <motion.p
            className="home__subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            Analyzing 1.6 million tweets using 5 ML models to decode public
            opinion
          </motion.p>

          <motion.div
            className="home__cta-group"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.0 }}
          >
            <Link to="/dashboard" className="home__cta-primary">
              Explore Dashboard
            </Link>
            <Link to="/predict" className="home__cta-secondary">
              Try Live Prediction
            </Link>
          </motion.div>
        </div>

        <ScrollChevron />
      </section>

      {/* ═══════ SECTION 2 — Stats ═══════ */}
      <section className="home__stats" ref={statsRef}>
        {/* Parallax orbs */}
        <motion.div
          className="home__orb home__orb--1"
          style={{ y: orbY1 }}
        />
        <motion.div
          className="home__orb home__orb--2"
          style={{ y: orbY2 }}
        />
        <motion.div
          className="home__orb home__orb--3"
          style={{ y: orbY3 }}
        />

        <EntranceReveal stagger={0.1} className="home__stats-grid">
          <MetricCard
            icon={<span style={{ fontSize: 22 }}>📊</span>}
            value={1.6}
            suffix="M+"
            label="Tweets Analyzed"
            color="#4361ee"
            decimals={1}
          />
          <MetricCard
            icon={<span style={{ fontSize: 22 }}>🤖</span>}
            value={5}
            label="Models Trained"
            color="#06d6a0"
          />
          <MetricCard
            icon={<span style={{ fontSize: 22 }}>🎯</span>}
            value={93}
            suffix="%+"
            label="Best Accuracy"
            color="#ef476f"
          />
          <MetricCard
            icon={<span style={{ fontSize: 22 }}>🏷️</span>}
            value={3}
            label="Sentiment Classes"
            color="#ffd166"
          />
        </EntranceReveal>
      </section>

      {/* ═══════ SECTION 3 — Features ═══════ */}
      <section className="home__features">
        <EntranceReveal>
          <h2 className="home__section-title">
            Powerful <span className="text-gradient">Features</span>
          </h2>
        </EntranceReveal>

        <EntranceReveal stagger={0.12} className="home__features-grid">
          {FEATURES.map((f) => (
            <TiltCard key={f.title}>
              <GlassCard className="home__feature-card" hoverable={false}>
                <span className="home__feature-icon">{f.icon}</span>
                <h3 className="home__feature-title">{f.title}</h3>
                <p className="home__feature-desc">{f.desc}</p>
              </GlassCard>
            </TiltCard>
          ))}
        </EntranceReveal>
      </section>

      {/* ═══════ SECTION 4 — Tech Stack Marquee ═══════ */}
      <section className="home__marquee-section">
        <EntranceReveal>
          <h2 className="home__section-title">
            Built With <span className="text-gradient">Modern Tech</span>
          </h2>
        </EntranceReveal>

        <div className="home__marquee">
          <div className="home__marquee-track">
            {/* Duplicate for seamless loop */}
            {[...TECH_STACK, ...TECH_STACK].map((tech, i) => (
              <div className="home__marquee-item" key={`${tech.name}-${i}`}>
                <span className="home__marquee-emoji">{tech.emoji}</span>
                <span className="home__marquee-name">{tech.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════ SECTION 5 — CTA Banner ═══════ */}
      <section className="home__cta-section">
        <EntranceReveal>
          <GlassCard className="home__cta-banner" hoverable={false}>
            <h2 className="home__cta-heading">
              Ready to explore the data?
            </h2>
            <p className="home__cta-text">
              Dive into our comprehensive sentiment analysis dashboard
            </p>
            <Link to="/dashboard" className="home__cta-primary">
              Go to Dashboard →
            </Link>
          </GlassCard>
        </EntranceReveal>
      </section>

      {/* ═══════ FOOTER ═══════ */}
      <footer className="home__footer">
        <span className="home__footer-copy">
          Twitter Sentiment Analyzer © 2025
        </span>
        <span className="home__footer-tech">
          Built with Python, React, FastAPI
        </span>
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="home__footer-github"
          aria-label="GitHub"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
          </svg>
        </a>
      </footer>
    </div>
  );
};

export default Home;
