import React from "react";
import "./LoadingSkeleton.css";

interface LoadingSkeletonProps {
  width?: string;
  height?: string;
  borderRadius?: string;
  className?: string;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  width = "100%",
  height = "20px",
  borderRadius = "8px",
  className = "",
}) => {
  return (
    <div
      className={`loading-skeleton ${className}`}
      style={{ width, height, borderRadius }}
    />
  );
};

export default LoadingSkeleton;
