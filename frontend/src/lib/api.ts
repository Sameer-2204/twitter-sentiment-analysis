import axios, { AxiosError } from "axios";

/* ── API Configuration ──────────────────────────────────────── */

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

console.log("🔗 API Configuration:");
console.log(`   Base URL: ${API_BASE}`);
console.log(`   Environment: ${import.meta.env.MODE}`);

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

/* ── Error helpers ──────────────────────────────────────────── */

function formatError(err: unknown, context: string): null {
  if (err instanceof AxiosError) {
    if (err.code === "ERR_NETWORK" || err.message === "Network Error") {
      console.error(
        `${context}: Backend unreachable at ${API_BASE}.\n` +
        `  → Is the backend running? (python run.py)\n` +
        `  → Is the Cloudflare tunnel active? (cloudflared tunnel --url http://localhost:8000)\n` +
        `  → Is VITE_API_BASE correct in .env?`
      );
    } else if (err.response) {
      console.error(
        `${context}: HTTP ${err.response.status} — ${err.response.statusText}`
      );
    } else {
      console.error(`${context}: ${err.message}`);
    }
  } else {
    console.error(`${context}:`, err);
  }
  return null;
}

/* ── Connection test ────────────────────────────────────────── */

export async function testConnection(): Promise<{
  connected: boolean;
  latency?: number;
  error?: string;
}> {
  const start = performance.now();
  try {
    const response = await api.get("/api/health");
    const latency = Math.round(performance.now() - start);
    if (response.status === 200) {
      console.log(`✅ Backend connected (${latency}ms)`);
      return { connected: true, latency };
    }
    return { connected: false, error: `HTTP ${response.status}` };
  } catch (err) {
    const msg =
      err instanceof AxiosError
        ? err.code === "ERR_NETWORK"
          ? "Backend unreachable"
          : err.message
        : "Unknown error";
    console.error("❌ Backend connection failed:", msg);
    return { connected: false, error: msg };
  }
}

/* ── Dashboard ────────────────────────────────────────────────── */

export async function fetchDashboardStats() {
  try {
    const { data } = await api.get("/api/dashboard/stats");
    return data;
  } catch (err) {
    return formatError(err, "fetchDashboardStats");
  }
}

export async function fetchRecentTweets(limit = 10, page = 1) {
  try {
    const { data } = await api.get("/api/dashboard/recent-tweets", {
      params: { limit, page },
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchRecentTweets");
  }
}

export async function fetchSentimentTrend(batchSize = 1000) {
  try {
    const { data } = await api.get("/api/dashboard/sentiment-trend", {
      params: { batch_size: batchSize },
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchSentimentTrend");
  }
}

/* ── EDA ──────────────────────────────────────────────────────── */

export async function fetchClassDistribution() {
  try {
    const { data } = await api.get("/api/eda/class-distribution");
    return data;
  } catch (err) {
    return formatError(err, "fetchClassDistribution");
  }
}

export async function fetchWordFrequency(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/word-frequency", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchWordFrequency");
  }
}

export async function fetchBigrams(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/bigrams", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchBigrams");
  }
}

export async function fetchTrigrams(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/trigrams", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchTrigrams");
  }
}

export async function fetchTweetLengths() {
  try {
    const { data } = await api.get("/api/eda/tweet-lengths");
    return data;
  } catch (err) {
    return formatError(err, "fetchTweetLengths");
  }
}

export async function fetchWordcloudData(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/wordcloud-data", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchWordcloudData");
  }
}

export async function fetchHashtags(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/hashtags", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchHashtags");
  }
}

export async function fetchMentions(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/mentions", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    return formatError(err, "fetchMentions");
  }
}

/* ── Models ───────────────────────────────────────────────────── */

export async function fetchModelComparison() {
  try {
    const { data } = await api.get("/api/models/comparison");
    return data;
  } catch (err) {
    return formatError(err, "fetchModelComparison");
  }
}

export async function fetchConfusionMatrix(modelName: string) {
  try {
    const { data } = await api.get(
      `/api/models/confusion-matrix/${modelName}`
    );
    return data;
  } catch (err) {
    return formatError(err, "fetchConfusionMatrix");
  }
}

export async function fetchTrainingHistory(modelName: string) {
  try {
    const { data } = await api.get(
      `/api/models/training-history/${modelName}`
    );
    return data;
  } catch (err) {
    return formatError(err, "fetchTrainingHistory");
  }
}

/* ── Predict ──────────────────────────────────────────────────── */

export async function predictSentiment(text: string, model?: string) {
  try {
    const body = model ? { text, model_name: model } : { text };
    const { data } = await api.post("/api/predict/", body);
    return data;
  } catch (err) {
    return formatError(err, "predictSentiment");
  }
}

export async function predictAllModels(text: string) {
  try {
    const { data } = await api.post("/api/predict/all", { text });
    return data;
  } catch (err) {
    return formatError(err, "predictAllModels");
  }
}

export async function predictBatch(formData: FormData, modelName?: string) {
  try {
    const { data } = await api.post("/api/predict/batch", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      params: modelName ? { model_name: modelName } : {},
    });
    return data;
  } catch (err) {
    return formatError(err, "predictBatch");
  }
}
