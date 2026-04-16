import React, { useEffect, useRef, useState } from "react";
import { motion, useInView } from "framer-motion";
import GlassCard from "./GlassCard";
import "./MetricCard.css";

interface MetricCardProps {
  icon: React.ReactNode;
  value: number;
  label: string;
  color: string;
  trend?: { value: number; positive: boolean };
  prefix?: string;
  suffix?: string;
  decimals?: number;
  className?: string;
}

function useCountUp(
  target: number,
  inView: boolean,
  duration = 1.5,
  decimals = 0
): string {
  const [display, setDisplay] = useState("0");

  useEffect(() => {
    if (!inView) return;

    let start = 0;
    const startTime = performance.now();

    function animate(now: number) {
      const elapsed = Math.min((now - startTime) / (duration * 1000), 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - elapsed, 3);
      const current = start + (target - start) * eased;
      setDisplay(current.toFixed(decimals));
      if (elapsed < 1) requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);
  }, [inView, target, duration, decimals]);

  return display;
}

const MetricCard: React.FC<MetricCardProps> = ({
  icon,
  value,
  label,
  color,
  trend,
  prefix = "",
  suffix = "",
  decimals = 0,
  className = "",
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, amount: 0.2 });
  const displayValue = useCountUp(value, isInView, 1.5, decimals);

  return (
    <GlassCard className={`metric-card ${className}`} glowColor={color}>
      <div
        className="metric-card__accent"
        style={{ backgroundColor: color }}
      />
      <div ref={ref} className="metric-card__body">
        <div
          className="metric-card__icon"
          style={{ backgroundColor: `${color}26` }}
        >
          {icon}
        </div>
        <div className="metric-card__content">
          <motion.span
            className="metric-card__value mono"
            initial={{ opacity: 0 }}
            animate={isInView ? { opacity: 1 } : {}}
          >
            {prefix}
            {displayValue}
            {suffix}
          </motion.span>
          <span className="metric-card__label caption">{label}</span>
          {trend && (
            <span
              className={`metric-card__trend ${trend.positive ? "metric-card__trend--up" : "metric-card__trend--down"}`}
            >
              {trend.positive ? "▲" : "▼"} {Math.abs(trend.value)}%
            </span>
          )}
        </div>
      </div>
    </GlassCard>
  );
};

export default MetricCard;
