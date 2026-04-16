import React, { Suspense, lazy } from "react";
const Spline = lazy(() => import("@splinetool/react-spline"));

interface SplineSceneProps {
  scene: string;
  className?: string;
}

const SplineScene: React.FC<SplineSceneProps> = ({ scene, className }) => {
  return (
    <Suspense
      fallback={
        <div className="hp__spline-loader">
          <span className="hp__spline-loader-spinner" />
        </div>
      }
    >
      <Spline scene={scene} className={className} />
    </Suspense>
  );
};

export default SplineScene;
