import React from "react";
import "./SentimentBadge.css";

interface SentimentBadgeProps {
  sentiment: "positive" | "negative" | "neutral";
  className?: string;
}

const SentimentBadge: React.FC<SentimentBadgeProps> = ({
  sentiment,
  className = "",
}) => {
  const labels: Record<string, string> = {
    positive: "Positive",
    negative: "Negative",
    neutral: "Neutral",
  };

  return (
    <span className={`sentiment-badge sentiment-badge--${sentiment} ${className}`}>
      {labels[sentiment]}
    </span>
  );
};

export default SentimentBadge;
