import React, { useState, useEffect, lazy, Suspense } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import { CustomCursor, SmoothLoader } from "./components/common";
import { DashboardLayout, LandingLayout } from "./components/Layout";
import "./styles/globals.css";

/* Lazy-loaded pages */
const Home = lazy(() => import("./pages/Home"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const EDA = lazy(() => import("./pages/EDA"));
const Models = lazy(() => import("./pages/Models"));
const Predict = lazy(() => import("./pages/Predict"));
const Batch = lazy(() => import("./pages/Batch"));
const About = lazy(() => import("./pages/About"));

const PageFallback = () => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "40vh",
      color: "var(--text-muted)",
      fontSize: 14,
    }}
  >
    Loading…
  </div>
);

const App: React.FC = () => {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 1900);
    return () => clearTimeout(timer);
  }, []);

  return (
    <BrowserRouter>
      {/* Initial loader */}
      {loading && <SmoothLoader duration={1800} />}

      {/* Custom cursor (desktop only) */}
      <CustomCursor />

      <AnimatePresence mode="wait">
        <Suspense fallback={<PageFallback />}>
          <Routes>
            {/* Landing — no sidebar */}
            <Route element={<LandingLayout />}>
              <Route index element={<Home />} />
            </Route>

            {/* Dashboard layout — sidebar + topbar */}
            <Route element={<DashboardLayout />}>
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="eda" element={<EDA />} />
              <Route path="models" element={<Models />} />
              <Route path="predict" element={<Predict />} />
              <Route path="batch" element={<Batch />} />
              <Route path="about" element={<About />} />
            </Route>
          </Routes>
        </Suspense>
      </AnimatePresence>
    </BrowserRouter>
  );
};

export default App;
