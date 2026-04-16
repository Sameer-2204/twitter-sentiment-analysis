import React from "react";
import { Outlet } from "react-router-dom";
import { motion } from "framer-motion";
import "./LandingLayout.css";

const LandingLayout: React.FC = () => {
  return (
    <motion.div
      className="landing-layout"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Outlet />
    </motion.div>
  );
};

export default LandingLayout;
