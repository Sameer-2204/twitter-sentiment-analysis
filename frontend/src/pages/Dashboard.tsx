import React, { useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { motion, useInView } from "framer-motion";
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
  MetricCard,
  SentimentBadge,
} from "../components/common";
import {
  fetchDashboardStats,
  fetchRecentTweets,
  fetchWordFrequency,
  fetchWordcloudData,
} from "../lib/api";
import "./Dashboard.css";

const AnimatedPlot = Plot as unknown as React.ComponentType<any>;

type SentimentType = "positive" | "negative" | "neutral";
type FilterType = "all" | SentimentType;

interface TrendPoint {
  batch: number;
  positive: number;
  negative: number;
  neutral: number;
}

interface KeywordEntry {
  word: string;
  count: number;
  sentiment: FilterType;
}

interface RecentTweet {
  id: string;
  text: string;
  sentiment: SentimentType;
  confidence: number;
}

interface DashboardStatsNormalized {
  totalTweets: number;
  positivePct: number;
  negativePct: number;
  neutralPct: number;
  trend: TrendPoint[];
}

interface WordCloudLayoutWord {
  text: string;
  x: number;
  y: number;
  rotate: number;
  fontSize: number;
  color: string;
  opacity: number;
}

const PAGE_SIZE = 10;
const WORD_CLOUD_HEIGHT = 320;
const sentimentColors: Record<SentimentType, string> = {
  positive: "#06d6a0",
  negative: "#ef476f",
  neutral: "#ffd166",
};
const wordCloudPalette = [
  "#06d6a0",
  "#3a86ff",
  "#ffd166",
  "#ef476f",
  "#76e5ff",
  "#9bffb3",
  "#f7b8ff",
  "#ff9f68",
];

const safeNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value.replace("%", "").trim());
    return Number.isFinite(parsed) ? parsed : fallback;
  }
  return fallback;
};

const toPercent = (value: unknown): number => {
  const numeric = safeNumber(value);
  return numeric > 1 && numeric <= 100 ? numeric : numeric * 100;
};

const resolveArray = (value: unknown): unknown[] => {
  if (Array.isArray(value)) return value;
  if (!value || typeof value !== "object") return [];

  const record = value as Record<string, unknown>;
  if (Array.isArray(record.data)) return record.data as unknown[];
  if (Array.isArray(record.items)) return record.items as unknown[];
  if (Array.isArray(record.words)) return record.words as unknown[];
  if (Array.isArray(record.results)) return record.results as unknown[];
  return [];
};

const normalizeSentiment = (value: unknown): SentimentType => {
  const sentiment = String(value ?? "")
    .toLowerCase()
    .trim();
  if (sentiment.includes("pos")) return "positive";
  if (sentiment.includes("neg")) return "negative";
  return "neutral";
};

const normalizeTrend = (raw: unknown): TrendPoint[] => {
  if (!raw || typeof raw !== "object") return [];
  const source = raw as Record<string, unknown>;
  const trendSource =
    source.trend ??
    source.sentiment_trend ??
    source.sentimentTrend ??
    source.trends ??
    source.timeline;

  if (Array.isArray(trendSource)) {
    return trendSource.map((point, index) => {
      const item = point as Record<string, unknown>;
      return {
        batch: safeNumber(
          item.batch ?? item.batch_index ?? item.index ?? item.x,
          index + 1
        ),
        positive: safeNumber(item.positive ?? item.pos ?? item.positive_pct),
        negative: safeNumber(item.negative ?? item.neg ?? item.negative_pct),
        neutral: safeNumber(item.neutral ?? item.neu ?? item.neutral_pct),
      };
    });
  }

  if (trendSource && typeof trendSource === "object") {
    const trend = trendSource as Record<string, unknown>;
    const positive = resolveArray(trend.positive ?? trend.pos).map((item) =>
      safeNumber(
        typeof item === "object"
          ? (item as Record<string, unknown>).value ?? item
          : item
      )
    );
    const negative = resolveArray(trend.negative ?? trend.neg).map((item) =>
      safeNumber(
        typeof item === "object"
          ? (item as Record<string, unknown>).value ?? item
          : item
      )
    );
    const neutral = resolveArray(trend.neutral ?? trend.neu).map((item) =>
      safeNumber(
        typeof item === "object"
          ? (item as Record<string, unknown>).value ?? item
          : item
      )
    );
    const xAxis = resolveArray(trend.x ?? trend.batch ?? trend.labels);
    const length = Math.max(positive.length, negative.length, neutral.length);

    return Array.from({ length }, (_, index) => ({
      batch: safeNumber(xAxis[index], index + 1),
      positive: positive[index] ?? 0,
      negative: negative[index] ?? 0,
      neutral: neutral[index] ?? 0,
    }));
  }

  return [];
};

const normalizeStats = (raw: unknown): DashboardStatsNormalized => {
  const stats = (raw as Record<string, unknown>) ?? {};
  return {
    totalTweets: safeNumber(stats.total_tweets ?? stats.totalTweets ?? stats.total),
    positivePct: toPercent(stats.positive_pct ?? stats.positivePct ?? stats.positive),
    negativePct: toPercent(stats.negative_pct ?? stats.negativePct ?? stats.negative),
    neutralPct: toPercent(stats.neutral_pct ?? stats.neutralPct ?? stats.neutral),
    trend: normalizeTrend(raw),
  };
};

const normalizeWordEntries = (
  raw: unknown,
  fallbackSentiment: FilterType = "all"
): KeywordEntry[] => {
  const source = resolveArray(raw);

  if (source.length > 0) {
    return source
      .map((entry) => {
        if (typeof entry === "string") {
          return {
            word: entry,
            count: 1,
            sentiment: fallbackSentiment,
          };
        }

        const item = (entry as Record<string, unknown>) ?? {};
        return {
          word: String(item.word ?? item.text ?? item.keyword ?? item.term ?? ""),
          count: safeNumber(item.count ?? item.frequency ?? item.value ?? item.weight),
          sentiment:
            fallbackSentiment === "all"
              ? (item.sentiment ? normalizeSentiment(item.sentiment) : "neutral")
              : fallbackSentiment,
        };
      })
      .filter((entry) => entry.word);
  }

  if (raw && typeof raw === "object") {
    return Object.entries(raw as Record<string, unknown>)
      .map(([word, value]) => ({
        word,
        count: safeNumber(value),
        sentiment: fallbackSentiment,
      }))
      .filter((entry) => entry.word);
  }

  return [];
};

const normalizeTweetsResponse = (
  raw: unknown
): { tweets: RecentTweet[]; totalPages: number } => {
  const data = (raw as Record<string, unknown>) ?? {};
  const tweetsSource = Array.isArray(raw)
    ? raw
    : resolveArray(data.tweets ?? data.items ?? data.results ?? data.data);
  const tweets = tweetsSource.map((tweet, index) => {
    const item = (tweet as Record<string, unknown>) ?? {};
    const confidenceRaw =
      item.confidence ?? item.score ?? item.probability ?? item.confidence_score;
    const confidence = safeNumber(confidenceRaw);
    const normalizedConfidence = confidence <= 1 ? confidence * 100 : confidence;

    return {
      id: String(item.id ?? item.tweet_id ?? item.uuid ?? index + 1),
      text: String(item.text ?? item.tweet_text ?? item.tweet ?? ""),
      sentiment: normalizeSentiment(item.sentiment ?? item.label ?? item.prediction),
      confidence: Math.max(0, Math.min(100, normalizedConfidence)),
    };
  });

  const pagination = data.pagination as Record<string, unknown> | undefined;
  const totalPages = safeNumber(
    data.total_pages ?? data.totalPages ?? pagination?.total_pages ?? pagination?.totalPages,
    1
  );

  return { tweets, totalPages: Math.max(1, totalPages || 1) };
};

const truncateText = (text: string, maxLength: number) =>
  text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;

const formatMetricDecimals = (value: number) => (Number.isInteger(value) ? 0 : 1);

const SectionError: React.FC<{ message: string; minHeight?: number }> = ({
  message,
  minHeight = 220,
}) => (
  <div className="dashboard__section-error" style={{ minHeight }}>
    <p>{message}</p>
  </div>
);

const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 350 }) => (
  <GlassCard hoverable={false} className="dashboard__panel dashboard__panel--chart">
    <div className="dashboard__panel-header">
      <LoadingSkeleton width="160px" height="22px" borderRadius="10px" />
    </div>
    <LoadingSkeleton height={`${height}px`} borderRadius="18px" />
  </GlassCard>
);

const TableSkeleton: React.FC = () => (
  <GlassCard hoverable={false} className="dashboard__panel">
    <div className="dashboard__panel-header">
      <LoadingSkeleton width="150px" height="22px" borderRadius="10px" />
    </div>
    <div className="dashboard__table-shell">
      {Array.from({ length: 5 }, (_, index) => (
        <div className="dashboard__table-skeleton-row" key={index}>
          <LoadingSkeleton width="36px" height="18px" />
          <LoadingSkeleton width="100%" height="18px" />
          <LoadingSkeleton width="92px" height="28px" borderRadius="20px" />
          <LoadingSkeleton width="120px" height="10px" borderRadius="999px" />
        </div>
      ))}
    </div>
  </GlassCard>
);

const DashboardIcon: React.FC<{
  type: "message" | "smile" | "frown" | "meh";
}> = ({ type }) => {
  const common = {
    width: 22,
    height: 22,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.9,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  if (type === "message") {
    return (
      <svg {...common}>
        <path d="M7 10h10" />
        <path d="M7 14h6" />
        <path d="M21 12c0 4.4-4 8-9 8a9.8 9.8 0 0 1-4-.8L3 21l1.8-4.1A7.7 7.7 0 0 1 3 12c0-4.4 4-8 9-8s9 3.6 9 8Z" />
      </svg>
    );
  }

  if (type === "smile") {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="9" />
        <path d="M8.5 14.5c.9 1.4 2.3 2 3.5 2s2.6-.6 3.5-2" />
        <path d="M9 10h.01" />
        <path d="M15 10h.01" />
      </svg>
    );
  }

  if (type === "frown") {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="9" />
        <path d="M8.5 16c.9-1.4 2.3-2 3.5-2s2.6.6 3.5 2" />
        <path d="M9 10h.01" />
        <path d="M15 10h.01" />
      </svg>
    );
  }

  return (
    <svg {...common}>
      <circle cx="12" cy="12" r="9" />
      <path d="M8.5 15h7" />
      <path d="M9 10h.01" />
      <path d="M15 10h.01" />
    </svg>
  );
};

const Dashboard: React.FC = () => {
  const trendRef = useRef<HTMLDivElement>(null);
  const donutRef = useRef<HTMLDivElement>(null);
  const wordCloudHostRef = useRef<HTMLDivElement>(null);
  const trendInView = useInView(trendRef, { once: true, amount: 0.25 });
  const donutInView = useInView(donutRef, { once: true, amount: 0.25 });

  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [stats, setStats] = useState<DashboardStatsNormalized>({
    totalTweets: 0,
    positivePct: 0,
    negativePct: 0,
    neutralPct: 0,
    trend: [],
  });

  const [wordCloudFilter, setWordCloudFilter] = useState<FilterType>("all");
  const [wordCloudLoading, setWordCloudLoading] = useState(true);
  const [wordCloudError, setWordCloudError] = useState<string | null>(null);
  const [wordCloudData, setWordCloudData] = useState<KeywordEntry[]>([]);
  const [wordCloudLayout, setWordCloudLayout] = useState<WordCloudLayoutWord[]>([]);
  const [wordCloudWidth, setWordCloudWidth] = useState(0);

  const [keywordsLoading, setKeywordsLoading] = useState(true);
  const [keywordsError, setKeywordsError] = useState<string | null>(null);
  const [keywords, setKeywords] = useState<KeywordEntry[]>([]);

  const [tweetsLoading, setTweetsLoading] = useState(true);
  const [tweetsError, setTweetsError] = useState<string | null>(null);
  const [tweets, setTweets] = useState<RecentTweet[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [distributionHoverIndex, setDistributionHoverIndex] = useState<number | null>(
    null
  );

  useEffect(() => {
    let isMounted = true;

    const loadStats = async () => {
      setStatsLoading(true);
      setStatsError(null);
      const response = await fetchDashboardStats();
      if (!isMounted) return;

      if (!response) {
        setStatsError("Unable to load dashboard stats right now.");
      } else {
        setStats(normalizeStats(response));
      }
      setStatsLoading(false);
    };

    loadStats();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;

    const loadKeywords = async () => {
      setKeywordsLoading(true);
      setKeywordsError(null);
      const response = await fetchWordFrequency("all");
      if (!isMounted) return;

      if (!response) {
        setKeywordsError("Unable to load top keywords right now.");
      } else {
        setKeywords(normalizeWordEntries(response, "all").slice(0, 15));
      }
      setKeywordsLoading(false);
    };

    loadKeywords();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;

    const loadWordCloud = async () => {
      setWordCloudLoading(true);
      setWordCloudError(null);
      const sentiment = wordCloudFilter === "all" ? undefined : wordCloudFilter;
      const response = await fetchWordcloudData(sentiment);
      if (!isMounted) return;

      if (!response) {
        setWordCloudError("Unable to load word cloud data right now.");
        setWordCloudData([]);
      } else {
        setWordCloudData(normalizeWordEntries(response, wordCloudFilter).slice(0, 60));
      }
      setWordCloudLoading(false);
    };

    loadWordCloud();

    return () => {
      isMounted = false;
    };
  }, [wordCloudFilter]);

  useEffect(() => {
    let isMounted = true;

    const loadTweets = async () => {
      setTweetsLoading(true);
      setTweetsError(null);
      const response = await fetchRecentTweets(PAGE_SIZE, currentPage);
      if (!isMounted) return;

      if (!response) {
        setTweetsError("Unable to load recent tweets right now.");
        setTweets([]);
      } else {
        const normalized = normalizeTweetsResponse(response);
        setTweets(normalized.tweets);
        setTotalPages(normalized.totalPages);
      }
      setTweetsLoading(false);
    };

    loadTweets();

    return () => {
      isMounted = false;
    };
  }, [currentPage]);

  useEffect(() => {
    if (!wordCloudHostRef.current) return;
    const element = wordCloudHostRef.current;
    const updateSize = () => {
      setWordCloudWidth(element.clientWidth);
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    let stopTimeout: number | null = null;

    if (!wordCloudWidth || wordCloudLoading || wordCloudData.length === 0) {
      setWordCloudLayout([]);
      return () => undefined;
    }

    const buildLayout = async () => {
      const module = await import("d3-cloud");
      const createCloud = module.default as unknown as () => any;
      const counts = wordCloudData.map((entry) => entry.count);
      const min = Math.min(...counts);
      const max = Math.max(...counts);
      const scaleSize = (count: number) => {
        if (min === max) return 28;
        return 18 + ((count - min) / (max - min)) * 36;
      };

      const layout = createCloud()
        .size([wordCloudWidth, WORD_CLOUD_HEIGHT])
        .words(
          wordCloudData.map((entry, index) => ({
            text: entry.word,
            size: scaleSize(entry.count),
            rotate: index % 7 === 0 ? 90 : 0,
            color: wordCloudPalette[index % wordCloudPalette.length],
            opacity: 0.68 + ((index % 5) + 1) * 0.05,
          }))
        )
        .font("Inter")
        .padding(8)
        .spiral("archimedean")
        .rotate((word: { rotate: number }) => word.rotate)
        .fontSize((word: { size: number }) => word.size)
        .on("end", (words: Array<Record<string, unknown>>) => {
          if (cancelled) return;
          setWordCloudLayout(
            words.map((word) => ({
              text: String(word.text ?? ""),
              x: safeNumber(word.x),
              y: safeNumber(word.y),
              rotate: safeNumber(word.rotate),
              fontSize: safeNumber(word.size, 24),
              color: String(word.color ?? wordCloudPalette[0]),
              opacity: safeNumber(word.opacity, 0.85),
            }))
          );
        });

      layout.start();
      stopTimeout = window.setTimeout(() => {
        try {
          layout.stop();
        } catch {
          return;
        }
      }, 3000);
    };

    buildLayout();

    return () => {
      cancelled = true;
      if (stopTimeout) window.clearTimeout(stopTimeout);
    };
  }, [wordCloudData, wordCloudLoading, wordCloudWidth]);

  const metricCards = [
    {
      icon: <DashboardIcon type="message" />,
      value: stats.totalTweets,
      label: "Total Tweets",
      color: "#4361ee",
      suffix: "",
    },
    {
      icon: <DashboardIcon type="smile" />,
      value: stats.positivePct,
      label: "Positive",
      color: sentimentColors.positive,
      suffix: "%",
    },
    {
      icon: <DashboardIcon type="frown" />,
      value: stats.negativePct,
      label: "Negative",
      color: sentimentColors.negative,
      suffix: "%",
    },
    {
      icon: <DashboardIcon type="meh" />,
      value: stats.neutralPct,
      label: "Neutral",
      color: sentimentColors.neutral,
      suffix: "%",
    },
  ];

  const trendXAxis = stats.trend.map((point) => point.batch);
  const lineChartData = stats.trend.length
    ? [
        {
          name: "Positive",
          x: trendInView ? trendXAxis : [],
          y: trendInView ? stats.trend.map((point) => point.positive) : [],
          line: { color: sentimentColors.positive, shape: "spline", smoothing: 1.12 },
          fill: "tozeroy",
          fillcolor: "rgba(6, 214, 160, 0.1)",
          fillgradient: {
            type: "vertical",
            colorscale: [
              [0, "rgba(6, 214, 160, 0.14)"],
              [1, "rgba(6, 214, 160, 0.02)"],
            ],
          },
          mode: "lines+markers",
          marker: { size: 5, color: sentimentColors.positive },
          hovertemplate: "Positive: %{y}<extra></extra>",
          type: "scatter",
        },
        {
          name: "Negative",
          x: trendInView ? trendXAxis : [],
          y: trendInView ? stats.trend.map((point) => point.negative) : [],
          line: { color: sentimentColors.negative, shape: "spline", smoothing: 1.12 },
          fill: "tozeroy",
          fillcolor: "rgba(239, 71, 111, 0.1)",
          fillgradient: {
            type: "vertical",
            colorscale: [
              [0, "rgba(239, 71, 111, 0.14)"],
              [1, "rgba(239, 71, 111, 0.02)"],
            ],
          },
          mode: "lines+markers",
          marker: { size: 5, color: sentimentColors.negative },
          hovertemplate: "Negative: %{y}<extra></extra>",
          type: "scatter",
        },
        {
          name: "Neutral",
          x: trendInView ? trendXAxis : [],
          y: trendInView ? stats.trend.map((point) => point.neutral) : [],
          line: { color: sentimentColors.neutral, shape: "spline", smoothing: 1.12 },
          fill: "tozeroy",
          fillcolor: "rgba(255, 209, 102, 0.1)",
          fillgradient: {
            type: "vertical",
            colorscale: [
              [0, "rgba(255, 209, 102, 0.16)"],
              [1, "rgba(255, 209, 102, 0.03)"],
            ],
          },
          mode: "lines+markers",
          marker: { size: 5, color: sentimentColors.neutral },
          hovertemplate: "Neutral: %{y}<extra></extra>",
          type: "scatter",
        },
      ]
    : [];

  const distributionValues = [
    stats.positivePct,
    stats.negativePct,
    stats.neutralPct,
  ].map((value) => Math.max(value, 0.0001));
  const donutValues = donutInView ? distributionValues : [0.0001, 0.0001, 0.0001];
  const pieChartData = [
    {
      type: "pie",
      hole: 0.6,
      labels: ["Positive", "Negative", "Neutral"],
      values: donutValues,
      sort: false,
      direction: "clockwise",
      marker: {
        colors: [
          sentimentColors.positive,
          sentimentColors.negative,
          sentimentColors.neutral,
        ],
      },
      textinfo: "none",
      hovertemplate: "%{label}: %{value:.1f}%<extra></extra>",
      pull: [0, 1, 2].map((index) => (distributionHoverIndex === index ? 0.06 : 0)),
      automargin: true,
    },
  ];

  const derivedTotalPages =
    totalPages || Math.ceil(Math.max(stats.totalTweets, tweets.length) / PAGE_SIZE) || 1;
  const paginationCount = Math.max(1, Math.min(3, derivedTotalPages));
  const visiblePages = Array.from({ length: paginationCount }, (_, index) => index + 1);

  return (
    <div className="dashboard-page">
      <EntranceReveal stagger={0.1} className="dashboard__metrics-grid">
        {statsLoading
          ? Array.from({ length: 4 }, (_, index) => (
              <GlassCard
                hoverable={false}
                className="dashboard__metric-skeleton"
                key={index}
              >
                <LoadingSkeleton width="52px" height="52px" borderRadius="50%" />
                <LoadingSkeleton width="120px" height="38px" borderRadius="12px" />
                <LoadingSkeleton width="88px" height="14px" borderRadius="8px" />
              </GlassCard>
            ))
          : statsError
            ? Array.from({ length: 4 }, (_, index) => (
                <GlassCard
                  hoverable={false}
                  className="dashboard__metric-error"
                  key={index}
                >
                  <p>{statsError}</p>
                </GlassCard>
              ))
            : metricCards.map((card) => (
                <MetricCard
                  key={card.label}
                  icon={card.icon}
                  value={card.value}
                  label={card.label}
                  color={card.color}
                  suffix={card.suffix}
                  decimals={formatMetricDecimals(card.value)}
                />
              ))}
      </EntranceReveal>

      <EntranceReveal delay={0.1}>
        <div className="dashboard__row dashboard__row--charts">
          <div ref={trendRef}>
            {statsLoading ? (
              <ChartSkeleton />
            ) : (
              <GlassCard
                hoverable={false}
                className="dashboard__panel dashboard__panel--chart"
              >
                <div className="dashboard__panel-header">
                  <h3>Sentiment Trend</h3>
                </div>
                {statsError ? (
                  <SectionError message={statsError} minHeight={350} />
                ) : lineChartData.length === 0 ? (
                  <SectionError
                    message="No sentiment trend data is available yet."
                    minHeight={350}
                  />
                ) : (
                  <AnimatedPlot
                    className="dashboard__plot"
                    data={lineChartData as any}
                    layout={{
                      autosize: true,
                      height: 350,
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                      hovermode: "x unified",
                      margin: { l: 50, r: 24, t: 10, b: 42 },
                      legend: {
                        orientation: "h",
                        x: 0,
                        y: 1.12,
                        font: { color: "#e6f1ff", size: 12 },
                      },
                      xaxis: {
                        title: { text: "Batch Index", font: { color: "#e6f1ff", size: 12 } },
                        color: "#e6f1ff",
                        showgrid: false,
                        zeroline: false,
                        tickfont: { color: "#aab6d3", size: 11 },
                        showspikes: true,
                        spikemode: "across",
                        spikecolor: "rgba(255,255,255,0.25)",
                        spikethickness: 1,
                      },
                      yaxis: {
                        title: { text: "Count / %", font: { color: "#e6f1ff", size: 12 } },
                        color: "#e6f1ff",
                        showgrid: false,
                        zeroline: false,
                        tickfont: { color: "#aab6d3", size: 11 },
                      },
                      hoverlabel: {
                        bgcolor: "rgba(14,18,40,0.94)",
                        bordercolor: "rgba(255,255,255,0.08)",
                        font: { color: "#f7fbff" },
                      },
                      transition: { duration: 800, easing: "cubic-in-out" },
                    }}
                    config={{
                      displayModeBar: false,
                      responsive: true,
                    }}
                    useResizeHandler
                    animate
                    animation={{ duration: 900, easing: "cubic-in-out" }}
                  />
                )}
              </GlassCard>
            )}
          </div>

          <div ref={donutRef}>
            {statsLoading ? (
              <ChartSkeleton />
            ) : (
              <GlassCard
                hoverable={false}
                className="dashboard__panel dashboard__panel--chart"
              >
                <div className="dashboard__panel-header">
                  <h3>Distribution</h3>
                </div>
                {statsError ? (
                  <SectionError message={statsError} minHeight={350} />
                ) : (
                  <AnimatedPlot
                    className="dashboard__plot"
                    data={pieChartData as any}
                    layout={{
                      autosize: true,
                      height: 350,
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                      margin: { l: 10, r: 10, t: 10, b: 10 },
                      showlegend: true,
                      legend: {
                        orientation: "h",
                        x: 0.08,
                        y: -0.06,
                        font: { color: "#e6f1ff", size: 12 },
                      },
                      annotations: [
                        {
                          x: 0.5,
                          y: 0.5,
                          xref: "paper",
                          yref: "paper",
                          showarrow: false,
                          text: `<span style="font-size:12px;color:#94a3c3;">Tweets</span><br><span style="font-size:28px;color:#ffffff;font-weight:700;">${stats.totalTweets.toLocaleString()}</span>`,
                        },
                      ],
                      transition: { duration: 800, easing: "cubic-in-out" },
                    }}
                    config={{
                      displayModeBar: false,
                      responsive: true,
                    }}
                    useResizeHandler
                    animate
                    animation={{ duration: 850, easing: "cubic-in-out" }}
                    onHover={(event: any) => {
                      const point = event.points?.[0];
                      if (typeof point?.pointNumber === "number") {
                        setDistributionHoverIndex(point.pointNumber);
                      }
                    }}
                    onUnhover={() => setDistributionHoverIndex(null)}
                  />
                )}
              </GlassCard>
            )}
          </div>
        </div>
      </EntranceReveal>

      <EntranceReveal delay={0.15}>
        <div className="dashboard__row dashboard__row--insights">
          <GlassCard hoverable={false} className="dashboard__panel">
            <div className="dashboard__panel-header dashboard__panel-header--between">
              <h3>Word Cloud</h3>
              <label className="dashboard__filter-shell">
                <span className="dashboard__sr-only">Filter word cloud</span>
                <select
                  className="dashboard__filter"
                  value={wordCloudFilter}
                  onChange={(event) =>
                    setWordCloudFilter(event.target.value as FilterType)
                  }
                >
                  <option value="all">All</option>
                  <option value="positive">Positive</option>
                  <option value="negative">Negative</option>
                  <option value="neutral">Neutral</option>
                </select>
              </label>
            </div>

            <div ref={wordCloudHostRef} className="dashboard__word-cloud-shell">
              {wordCloudLoading ? (
                <LoadingSkeleton height={`${WORD_CLOUD_HEIGHT}px`} borderRadius="18px" />
              ) : wordCloudError ? (
                <SectionError message={wordCloudError} minHeight={WORD_CLOUD_HEIGHT} />
              ) : wordCloudLayout.length === 0 ? (
                <SectionError
                  message="No word cloud terms are available for this filter."
                  minHeight={WORD_CLOUD_HEIGHT}
                />
              ) : (
                <motion.svg
                  key={wordCloudFilter}
                  className="dashboard__word-cloud"
                  viewBox={`0 0 ${Math.max(wordCloudWidth, 1)} ${WORD_CLOUD_HEIGHT}`}
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.35, ease: "easeOut" }}
                >
                  <defs>
                    <radialGradient id="word-cloud-glow" cx="50%" cy="50%" r="70%">
                      <stop offset="0%" stopColor="rgba(67, 97, 238, 0.22)" />
                      <stop offset="100%" stopColor="rgba(67, 97, 238, 0)" />
                    </radialGradient>
                  </defs>
                  <rect
                    x="0"
                    y="0"
                    width={Math.max(wordCloudWidth, 1)}
                    height={WORD_CLOUD_HEIGHT}
                    fill="url(#word-cloud-glow)"
                    opacity="0.55"
                  />
                  <g
                    transform={`translate(${Math.max(wordCloudWidth, 1) / 2}, ${WORD_CLOUD_HEIGHT / 2})`}
                  >
                    {wordCloudLayout.map((word) => (
                      <text
                        key={`${word.text}-${word.x}-${word.y}`}
                        x={word.x}
                        y={word.y}
                        transform={`rotate(${word.rotate}, ${word.x}, ${word.y})`}
                        fontSize={word.fontSize}
                        fill={word.color}
                        fillOpacity={word.opacity}
                        textAnchor="middle"
                        className="dashboard__word-cloud-text"
                      >
                        {word.text}
                      </text>
                    ))}
                  </g>
                </motion.svg>
              )}
            </div>
          </GlassCard>

          <GlassCard hoverable={false} className="dashboard__panel">
            <div className="dashboard__panel-header">
              <h3>Top Keywords</h3>
            </div>

            <div className="dashboard__keywords-shell">
              {keywordsLoading ? (
                <LoadingSkeleton height="320px" borderRadius="18px" />
              ) : keywordsError ? (
                <SectionError message={keywordsError} minHeight={320} />
              ) : keywords.length === 0 ? (
                <SectionError
                  message="No keyword frequency data is available yet."
                  minHeight={320}
                />
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    data={[...keywords].reverse()}
                    layout="vertical"
                    margin={{ top: 4, right: 16, left: 8, bottom: 0 }}
                  >
                    <CartesianGrid
                      stroke="rgba(255,255,255,0.04)"
                      horizontal={false}
                      vertical={false}
                    />
                    <XAxis
                      type="number"
                      stroke="#93a2c5"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#93a2c5", fontSize: 11 }}
                    />
                    <YAxis
                      dataKey="word"
                      type="category"
                      width={94}
                      stroke="#93a2c5"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#dce8ff", fontSize: 11 }}
                    />
                    <Tooltip
                      cursor={{ fill: "rgba(255,255,255,0.04)" }}
                      contentStyle={{
                        background: "rgba(10, 14, 32, 0.96)",
                        border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "14px",
                        boxShadow: "0 18px 40px rgba(0,0,0,0.28)",
                      }}
                      labelStyle={{ color: "#ffffff", fontWeight: 600 }}
                      itemStyle={{ color: "#dbe8ff" }}
                    />
                    <Bar
                      dataKey="count"
                      radius={[0, 4, 4, 0]}
                      isAnimationActive
                      animationDuration={900}
                      animationBegin={120}
                      activeBar={{
                        fillOpacity: 1,
                        stroke: "rgba(255,255,255,0.25)",
                        strokeWidth: 1,
                      }}
                    >
                      {[...keywords].reverse().map((entry) => (
                        <Cell
                          key={entry.word}
                          fill={
                            entry.sentiment === "all"
                              ? "#7aa4ff"
                              : sentimentColors[entry.sentiment]
                          }
                          fillOpacity={0.9}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </GlassCard>
        </div>
      </EntranceReveal>

      <EntranceReveal delay={0.2}>
        {tweetsLoading ? (
          <TableSkeleton />
        ) : (
          <GlassCard hoverable={false} className="dashboard__panel">
            <div className="dashboard__panel-header">
              <h3>Recent Tweets</h3>
            </div>

            {tweetsError ? (
              <SectionError message={tweetsError} minHeight={260} />
            ) : tweets.length === 0 ? (
              <SectionError message="No recent tweets are available yet." minHeight={260} />
            ) : (
              <>
                <div className="dashboard__table-wrap">
                  <table className="dashboard__table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Tweet Text</th>
                        <th>Sentiment</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tweets.map((tweet, index) => (
                        <motion.tr
                          key={`${tweet.id}-${index}`}
                          initial={{ opacity: 0, y: 18 }}
                          whileInView={{ opacity: 1, y: 0 }}
                          viewport={{ once: true, amount: 0.2 }}
                          transition={{ duration: 0.35, delay: index * 0.05 }}
                        >
                          <td>{(currentPage - 1) * PAGE_SIZE + index + 1}</td>
                          <td className="dashboard__tweet-text">
                            {truncateText(tweet.text, 100)}
                          </td>
                          <td>
                            <SentimentBadge sentiment={tweet.sentiment} />
                          </td>
                          <td>
                            <div className="dashboard__confidence">
                              <div className="dashboard__confidence-bar">
                                <span
                                  className="dashboard__confidence-fill"
                                  style={{
                                    width: `${tweet.confidence}%`,
                                    background:
                                      tweet.sentiment === "neutral"
                                        ? "linear-gradient(90deg, rgba(255, 209, 102, 0.9), rgba(255, 209, 102, 0.55))"
                                        : tweet.sentiment === "positive"
                                          ? "linear-gradient(90deg, rgba(6, 214, 160, 0.95), rgba(6, 214, 160, 0.55))"
                                          : "linear-gradient(90deg, rgba(239, 71, 111, 0.95), rgba(239, 71, 111, 0.55))",
                                  }}
                                />
                              </div>
                              <span className="dashboard__confidence-value mono">
                                {tweet.confidence.toFixed(1)}%
                              </span>
                            </div>
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="dashboard__pagination">
                  <button
                    type="button"
                    className="dashboard__pagination-button"
                    onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </button>
                  <div className="dashboard__pagination-pages" aria-label="Pages">
                    {visiblePages.map((page) => (
                      <button
                        key={page}
                        type="button"
                        className={`dashboard__pagination-page ${page === currentPage ? "is-active" : ""}`}
                        onClick={() => setCurrentPage(page)}
                      >
                        {page}
                      </button>
                    ))}
                  </div>
                  <button
                    type="button"
                    className="dashboard__pagination-button"
                    onClick={() =>
                      setCurrentPage((page) => Math.min(derivedTotalPages, page + 1))
                    }
                    disabled={currentPage >= derivedTotalPages}
                  >
                    Next
                  </button>
                </div>
              </>
            )}
          </GlassCard>
        )}
      </EntranceReveal>
    </div>
  );
};

export default Dashboard;
