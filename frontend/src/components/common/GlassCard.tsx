import React from "react";
import { motion } from "framer-motion";
import "./GlassCard.css";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hoverable?: boolean;
  glowColor?: string;
  onClick?: () => void;
  style?: React.CSSProperties;
}

const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = "",
  hoverable = true,
  glowColor,
  onClick,
  style,
}) => {
  const mergedStyle: React.CSSProperties = {
    ...(glowColor ? { "--glow-color": glowColor } as React.CSSProperties : {}),
    ...style,
  };

  return (
    <motion.div
      className={`glass-card ${hoverable ? "glass-card--hoverable" : ""} ${className}`}
      onClick={onClick}
      style={Object.keys(mergedStyle).length ? mergedStyle : undefined}
      whileHover={hoverable ? { y: -4 } : undefined}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
};

export default GlassCard;

