import React from "react";
import { motion } from "framer-motion";
import type { Variants } from "framer-motion";

interface EntranceRevealProps {
  children: React.ReactNode;
  delay?: number;
  direction?: "up" | "down" | "left" | "right";
  duration?: number;
  className?: string;
  stagger?: number;
}

const directionOffset: Record<string, { x: number; y: number }> = {
  up: { x: 0, y: 30 },
  down: { x: 0, y: -30 },
  left: { x: 30, y: 0 },
  right: { x: -30, y: 0 },
};

const EntranceReveal: React.FC<EntranceRevealProps> = ({
  children,
  delay = 0,
  direction = "up",
  duration = 0.6,
  className = "",
  stagger,
}) => {
  const offset = directionOffset[direction];

  if (stagger !== undefined) {
    const containerVariants: Variants = {
      hidden: {},
      visible: {
        transition: {
          staggerChildren: stagger,
          delayChildren: delay,
        },
      },
    };

    const childVariants: Variants = {
      hidden: { opacity: 0, x: offset.x, y: offset.y },
      visible: {
        opacity: 1,
        x: 0,
        y: 0,
        transition: { duration, ease: "easeOut" },
      },
    };

    return (
      <motion.div
        className={className}
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
      >
        {React.Children.map(children, (child) => (
          <motion.div variants={childVariants}>{child}</motion.div>
        ))}
      </motion.div>
    );
  }

  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, x: offset.x, y: offset.y }}
      whileInView={{ opacity: 1, x: 0, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration, delay, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
};

export default EntranceReveal;
