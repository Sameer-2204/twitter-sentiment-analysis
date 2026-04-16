import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import "./SmoothLoader.css";

interface SmoothLoaderProps {
  duration?: number;
}

const SmoothLoader: React.FC<SmoothLoaderProps> = ({ duration = 1800 }) => {
  const [visible, setVisible] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const start = performance.now();

    function tick(now: number) {
      const elapsed = now - start;
      const pct = Math.min(elapsed / duration, 1);
      setProgress(pct * 100);
      if (pct < 1) {
        requestAnimationFrame(tick);
      } else {
        setTimeout(() => setVisible(false), 200);
      }
    }

    requestAnimationFrame(tick);
  }, [duration]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          className="smooth-loader"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.05 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
        >
          <div className="smooth-loader__content">
            {/* Pulsing circle */}
            <motion.div
              className="smooth-loader__circle"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.6, 1, 0.6],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
            <motion.div
              className="smooth-loader__ring"
              animate={{ rotate: 360 }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "linear",
              }}
            />
            <span className="smooth-loader__text">Loading</span>
          </div>

          {/* Gradient progress bar */}
          <div className="smooth-loader__bar-track">
            <motion.div
              className="smooth-loader__bar-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default SmoothLoader;
