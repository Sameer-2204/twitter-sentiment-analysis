import React from "react";
import { GlassCard } from "./index";
import "./EmptyState.css";

interface EmptyStateProps {
  message?: string;
  icon?: React.ReactNode;
}

const EmptyState: React.FC<EmptyStateProps> = ({
  message = "No data available",
  icon,
}) => {
  return (
    <GlassCard hoverable={false} className="empty-state">
      {icon || (
        <svg className="empty-state__icon" width="40" height="40" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 002 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0022 16z" />
          <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
          <line x1="12" y1="22.08" x2="12" y2="12" />
        </svg>
      )}
      <p className="empty-state__message">{message}</p>
    </GlassCard>
  );
};

export default EmptyState;
