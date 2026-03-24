import React, { useState, useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";
import "./DashboardLayout.css";

const pageTitles: Record<string, { title: string; breadcrumb: string }> = {
  "/dashboard": { title: "Dashboard", breadcrumb: "Home / Dashboard" },
  "/eda": { title: "Exploratory Analysis", breadcrumb: "Home / EDA" },
  "/models": { title: "Model Performance", breadcrumb: "Home / Models" },
  "/predict": { title: "Live Prediction", breadcrumb: "Home / Predict" },
  "/batch": { title: "Batch Prediction", breadcrumb: "Home / Batch" },
  "/about": { title: "About", breadcrumb: "Home / About" },
};

const DashboardLayout: React.FC = () => {
  const location = useLocation();
  const page = pageTitles[location.pathname] || {
    title: "Dashboard",
    breadcrumb: "Home",
  };

  const [mobileOpen, setMobileOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Close mobile sidebar on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  return (
    <div className="dashboard-layout">
      <Sidebar mobileOpen={mobileOpen} onMobileClose={() => setMobileOpen(false)} />
      <div className="dashboard-layout__main">
        <TopBar
          title={page.title}
          breadcrumb={page.breadcrumb}
          showHamburger={isMobile}
          onHamburgerClick={() => setMobileOpen(true)}
        />
        <motion.main
          className="dashboard-layout__content"
          key={location.pathname}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        >
          <Outlet />
        </motion.main>
      </div>
    </div>
  );
};

export default DashboardLayout;
