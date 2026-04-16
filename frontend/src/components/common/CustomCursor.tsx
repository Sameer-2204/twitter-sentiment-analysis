import React, { useEffect, useCallback } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import "./CustomCursor.css";

const CustomCursor: React.FC = () => {
  const cursorX = useMotionValue(-100);
  const cursorY = useMotionValue(-100);
  const springX = useSpring(cursorX, { damping: 20, stiffness: 150 });
  const springY = useSpring(cursorY, { damping: 20, stiffness: 150 });
  const scale = useMotionValue(1);
  const springScale = useSpring(scale, { damping: 15, stiffness: 200 });

  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      cursorX.set(e.clientX);
      cursorY.set(e.clientY);
    },
    [cursorX, cursorY]
  );

  useEffect(() => {
    // Hide on touch devices
    const isTouchDevice =
      "ontouchstart" in window || navigator.maxTouchPoints > 0;
    if (isTouchDevice) return;

    window.addEventListener("mousemove", onMouseMove);

    const handleHoverIn = () => scale.set(2);
    const handleHoverOut = () => scale.set(1);

    const interactiveSelector =
      "a, button, [role='button'], input, textarea, select, [data-cursor-hover]";

    const observer = new MutationObserver(() => {
      document.querySelectorAll(interactiveSelector).forEach((el) => {
        el.addEventListener("mouseenter", handleHoverIn);
        el.addEventListener("mouseleave", handleHoverOut);
      });
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Initial pass
    document.querySelectorAll(interactiveSelector).forEach((el) => {
      el.addEventListener("mouseenter", handleHoverIn);
      el.addEventListener("mouseleave", handleHoverOut);
    });

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      observer.disconnect();
    };
  }, [onMouseMove, scale]);

  // Don't render on mobile
  if (typeof window !== "undefined") {
    const isTouchDevice =
      "ontouchstart" in window || navigator.maxTouchPoints > 0;
    if (isTouchDevice) return null;
  }

  return (
    <motion.div
      className="custom-cursor"
      style={{
        x: springX,
        y: springY,
        scale: springScale,
      }}
    />
  );
};

export default CustomCursor;
