import React, { useEffect, useRef, ReactNode } from "react";
import "./GlowCard.css";

interface GlowCardProps {
  children: ReactNode;
  className?: string;
  glowColor?: "blue" | "purple" | "green" | "red" | "orange";
}

const glowColorMap: Record<string, { base: number; spread: number }> = {
  blue: { base: 220, spread: 26 },
  purple: { base: 280, spread: 22 },
  green: { base: 140, spread: 18 },
  red: { base: 8, spread: 16 },
  orange: { base: 28, spread: 14 },
};

const GlowCard: React.FC<GlowCardProps> = ({
  children,
  className = "",
  glowColor = "blue",
}) => {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const clamp = (value: number) => Math.min(Math.max(value, 0), 1);

    const resetPointer = () => {
      if (!cardRef.current) return;

      const { width, height } = cardRef.current.getBoundingClientRect();
      cardRef.current.style.setProperty("--x", (width / 2).toFixed(2));
      cardRef.current.style.setProperty("--y", (height / 2).toFixed(2));
      cardRef.current.style.setProperty("--xp", "0.50");
      cardRef.current.style.setProperty("--yp", "0.50");
    };

    const syncPointer = (e: PointerEvent) => {
      if (!cardRef.current) return;

      const rect = cardRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const xp = rect.width === 0 ? 0.5 : clamp(x / rect.width);
      const yp = rect.height === 0 ? 0.5 : clamp(y / rect.height);

      cardRef.current.style.setProperty("--x", x.toFixed(2));
      cardRef.current.style.setProperty("--xp", xp.toFixed(2));
      cardRef.current.style.setProperty("--y", y.toFixed(2));
      cardRef.current.style.setProperty("--yp", yp.toFixed(2));
    };

    const card = cardRef.current;
    if (!card) return;

    resetPointer();
    card.addEventListener("pointermove", syncPointer);
    card.addEventListener("pointerleave", resetPointer);
    window.addEventListener("resize", resetPointer);

    return () => {
      card.removeEventListener("pointermove", syncPointer);
      card.removeEventListener("pointerleave", resetPointer);
      window.removeEventListener("resize", resetPointer);
    };
  }, []);

  const { base, spread } = glowColorMap[glowColor];

  const inlineStyles: React.CSSProperties & Record<string, string> = {
    "--base": String(base),
    "--spread": String(spread),
    "--radius": "14",
    "--border": "3",
    "--backdrop": "hsl(0 0% 60% / 0.12)",
    "--backup-border": "var(--backdrop)",
    "--size": "172",
    "--outer": "0.56",
    "--bg-spot-opacity": "0.08",
    "--border-spot-opacity": "0.72",
    "--border-light-opacity": "0.45",
    "--border-size": "calc(var(--border, 2) * 1px)",
    "--spotlight-size": "calc(var(--size, 150) * 1px)",
    "--hue": "calc(var(--base) + ((var(--xp, 0.5) - 0.5) * var(--spread, 0)))",
    backgroundImage: `radial-gradient(
      var(--spotlight-size) var(--spotlight-size) at
      calc(var(--x, 0) * 1px)
      calc(var(--y, 0) * 1px),
      hsl(var(--hue, 210) calc(var(--saturation, 100) * 1%) calc(var(--lightness, 70) * 1%) / var(--bg-spot-opacity, 0.1)), transparent
    )`,
    backgroundColor: "var(--backdrop, transparent)",
    backgroundSize:
      "calc(100% + (2 * var(--border-size))) calc(100% + (2 * var(--border-size)))",
    backgroundPosition: "50% 50%",
    border: "var(--border-size) solid var(--backup-border)",
    position: "relative",
    touchAction: "none",
  };

  return (
    <div
      ref={cardRef}
      data-glow
      style={inlineStyles}
      className={`glow-card ${className}`}
    >
      <div data-glow />
      {children}
    </div>
  );
};

export { GlowCard };
export default GlowCard;
