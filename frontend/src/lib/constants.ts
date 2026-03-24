/* ═══════════════════════════════════════════════════════════════
   Application Constants
   ═══════════════════════════════════════════════════════════════ */

export const SENTIMENT_COLORS = {
  positive: "#06d6a0",
  negative: "#ef476f",
  neutral: "#ffd166",
} as const;

export const SENTIMENT_BG_COLORS = {
  positive: "rgba(6, 214, 160, 0.12)",
  negative: "rgba(239, 71, 111, 0.12)",
  neutral: "rgba(255, 209, 102, 0.12)",
} as const;

export const SENTIMENT_MAP: Record<number, string> = {
  0: "negative",
  1: "neutral",
  2: "positive",
};

export const SENTIMENT_LABELS: Record<string, string> = {
  positive: "Positive",
  negative: "Negative",
  neutral: "Neutral",
};

export const MODEL_NAMES = [
  "logistic",
  "lstm",
  "bilstm",
  "cnn",
  "distilbert",
] as const;

export type ModelName = (typeof MODEL_NAMES)[number];

export const MODEL_DISPLAY_NAMES: Record<ModelName, string> = {
  logistic: "Logistic Regression",
  lstm: "LSTM",
  bilstm: "BiLSTM",
  cnn: "CNN",
  distilbert: "DistilBERT",
};

export const CHART_COLORS = [
  "#4361ee",
  "#3a86ff",
  "#06d6a0",
  "#ef476f",
  "#ffd166",
  "#7209b7",
  "#f72585",
  "#4cc9f0",
  "#fb5607",
  "#8338ec",
] as const;

export const NAV_ITEMS = [
  { path: "/", label: "Home", icon: "home" },
  { path: "/dashboard", label: "Dashboard", icon: "dashboard" },
  { path: "/eda", label: "EDA", icon: "eda" },
  { path: "/models", label: "Models", icon: "models" },
  { path: "/predict", label: "Predict", icon: "predict" },
  { path: "/batch", label: "Batch", icon: "batch" },
  { path: "/about", label: "About", icon: "about" },
] as const;
