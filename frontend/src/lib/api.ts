import axios from "axios";

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

/* ── Dashboard ────────────────────────────────────────────────── */

export async function fetchDashboardStats() {
  try {
    const { data } = await api.get("/api/dashboard/stats");
    return data;
  } catch (err) {
    console.error("fetchDashboardStats failed:", err);
    return null;
  }
}

export async function fetchRecentTweets(limit = 10, page = 1) {
  try {
    const { data } = await api.get("/api/dashboard/recent-tweets", {
      params: { limit, page },
    });
    return data;
  } catch (err) {
    console.error("fetchRecentTweets failed:", err);
    return null;
  }
}

/* ── EDA ──────────────────────────────────────────────────────── */

export async function fetchClassDistribution() {
  try {
    const { data } = await api.get("/api/eda/class-distribution");
    return data;
  } catch (err) {
    console.error("fetchClassDistribution failed:", err);
    return null;
  }
}

export async function fetchWordFrequency(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/word-frequency", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    console.error("fetchWordFrequency failed:", err);
    return null;
  }
}

export async function fetchBigrams(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/bigrams", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    console.error("fetchBigrams failed:", err);
    return null;
  }
}

export async function fetchTrigrams(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/trigrams", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    console.error("fetchTrigrams failed:", err);
    return null;
  }
}

export async function fetchTweetLengths() {
  try {
    const { data } = await api.get("/api/eda/tweet-lengths");
    return data;
  } catch (err) {
    console.error("fetchTweetLengths failed:", err);
    return null;
  }
}

export async function fetchWordcloudData(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/wordcloud-data", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    console.error("fetchWordcloudData failed:", err);
    return null;
  }
}

export async function fetchHashtags(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/hashtags", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    console.error("fetchHashtags failed:", err);
    return null;
  }
}

export async function fetchMentions(sentiment?: string) {
  try {
    const { data } = await api.get("/api/eda/mentions", {
      params: sentiment ? { sentiment } : {},
    });
    return data;
  } catch (err) {
    console.error("fetchMentions failed:", err);
    return null;
  }
}

/* ── Models ───────────────────────────────────────────────────── */

export async function fetchModelComparison() {
  try {
    const { data } = await api.get("/api/models/comparison");
    return data;
  } catch (err) {
    console.error("fetchModelComparison failed:", err);
    return null;
  }
}

export async function fetchConfusionMatrix(modelName: string) {
  try {
    const { data } = await api.get(
      `/api/models/confusion-matrix/${modelName}`
    );
    return data;
  } catch (err) {
    console.error("fetchConfusionMatrix failed:", err);
    return null;
  }
}

export async function fetchTrainingHistory(modelName: string) {
  try {
    const { data } = await api.get(
      `/api/models/training-history/${modelName}`
    );
    return data;
  } catch (err) {
    console.error("fetchTrainingHistory failed:", err);
    return null;
  }
}

/* ── Predict ──────────────────────────────────────────────────── */

export async function predictSentiment(text: string, model?: string) {
  try {
    const { data } = await api.post("/api/predict", { text, model });
    return data;
  } catch (err) {
    console.error("predictSentiment failed:", err);
    return null;
  }
}

export async function predictAllModels(text: string) {
  try {
    const { data } = await api.post("/api/predict/all", { text });
    return data;
  } catch (err) {
    console.error("predictAllModels failed:", err);
    return null;
  }
}

export async function predictBatch(formData: FormData) {
  try {
    const { data } = await api.post("/api/predict/batch", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  } catch (err) {
    console.error("predictBatch failed:", err);
    return null;
  }
}
