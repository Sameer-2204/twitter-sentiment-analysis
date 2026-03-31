import React, { useState, useEffect, useCallback } from "react";
import { testConnection } from "../../lib/api";
import "./ConnectionStatus.css";

type Status = "checking" | "connected" | "disconnected";

interface ConnectionStatusProps {
  /** Check interval in ms (default: 30 000). Set 0 to disable polling. */
  pollInterval?: number;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  pollInterval = 30_000,
}) => {
  const [status, setStatus] = useState<Status>("checking");
  const [latency, setLatency] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const check = useCallback(async () => {
    const result = await testConnection();
    if (result.connected) {
      setStatus("connected");
      setLatency(result.latency ?? null);
      setError(null);
    } else {
      setStatus("disconnected");
      setLatency(null);
      setError(result.error ?? "Backend unreachable");
    }
  }, []);

  useEffect(() => {
    check();
    if (pollInterval > 0) {
      const id = setInterval(check, pollInterval);
      return () => clearInterval(id);
    }
  }, [check, pollInterval]);

  const label =
    status === "checking"
      ? "Checking…"
      : status === "connected"
        ? `API Connected${latency ? ` (${latency}ms)` : ""}`
        : "API Offline";

  return (
    <div
      className={`connection-status ${status}`}
      onClick={check}
      title="Click to re-check"
      role="status"
    >
      <span className="connection-dot" />
      <span>{label}</span>

      {status === "disconnected" && (
        <div className="connection-tooltip">
          <strong>Backend is unreachable</strong>
          <br />
          {error && <>{error}<br /></>}
          <br />
          Make sure:
          <br />
          • Backend is running (<code>python run.py</code>)
          <br />
          • Cloudflare tunnel is active
          <br />
          • <code>VITE_API_BASE</code> is set correctly
        </div>
      )}
    </div>
  );
};

export default ConnectionStatus;
