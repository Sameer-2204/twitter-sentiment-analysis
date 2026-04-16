import React from "react";
import { GlassCard } from "./index";
import "./ErrorState.css";

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

const ErrorState: React.FC<ErrorStateProps> = ({
  message = "Failed to load data",
  onRetry,
}) => {
  return (
    <GlassCard hoverable={false} className="error-state">
      <svg className="error-state__icon" width="40" height="40" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="15" y1="9" x2="9" y2="15" />
        <line x1="9" y1="9" x2="15" y2="15" />
      </svg>
      <p className="error-state__message">{message}</p>
      {onRetry && (
        <button className="error-state__retry" onClick={onRetry}>
          Retry
        </button>
      )}
    </GlassCard>
  );
};

export default ErrorState;
