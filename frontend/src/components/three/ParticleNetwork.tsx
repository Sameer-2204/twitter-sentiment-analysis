import React, { useRef, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/* ── Particle System (inside Canvas) ──────────────────────────── */

interface ParticlesProps {
  count: number;
  mouseRef: React.MutableRefObject<{ x: number; y: number }>;
}

function Particles({ count, mouseRef }: ParticlesProps) {
  const meshRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  const groupRef = useRef<THREE.Group>(null);

  // Generate random positions + velocities
  const { positions, velocities, colors } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    const nodeColor = new THREE.Color("#4361ee");

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      pos[i3] = (Math.random() - 0.5) * 10;
      pos[i3 + 1] = (Math.random() - 0.5) * 10;
      pos[i3 + 2] = (Math.random() - 0.5) * 10;

      vel[i3] = (Math.random() - 0.5) * 0.002;
      vel[i3 + 1] = (Math.random() - 0.5) * 0.002;
      vel[i3 + 2] = (Math.random() - 0.5) * 0.002;

      col[i3] = nodeColor.r;
      col[i3 + 1] = nodeColor.g;
      col[i3 + 2] = nodeColor.b;
    }

    return { positions: pos, velocities: vel, colors: col };
  }, [count]);

  // Line geometry for connections
  const lineGeometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    // Max possible connections (allocate generously)
    const maxLines = count * 6;
    const linePositions = new Float32Array(maxLines * 6);
    const lineColors = new Float32Array(maxLines * 6);
    geo.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
    geo.setDrawRange(0, 0);
    return geo;
  }, [count]);

  const connectionColor = useMemo(() => new THREE.Color("#06d6a0"), []);

  useFrame((state) => {
    if (!meshRef.current || !groupRef.current) return;

    const time = state.clock.elapsedTime;
    const posAttr = meshRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const posArray = posAttr.array as Float32Array;

    // Drift particles with sin/cos oscillation
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      posArray[i3] += Math.sin(time * 0.3 + i * 0.1) * 0.001 + velocities[i3];
      posArray[i3 + 1] +=
        Math.cos(time * 0.2 + i * 0.15) * 0.001 + velocities[i3 + 1];
      posArray[i3 + 2] +=
        Math.sin(time * 0.4 + i * 0.05) * 0.001 + velocities[i3 + 2];

      // Bounds check — wrap around
      for (let j = 0; j < 3; j++) {
        if (posArray[i3 + j] > 5) posArray[i3 + j] = -5;
        if (posArray[i3 + j] < -5) posArray[i3 + j] = 5;
      }
    }
    posAttr.needsUpdate = true;

    // Update connections
    const linePosAttr = lineGeometry.attributes
      .position as THREE.BufferAttribute;
    const lineColAttr = lineGeometry.attributes
      .color as THREE.BufferAttribute;
    const linePos = linePosAttr.array as Float32Array;
    const lineCol = lineColAttr.array as Float32Array;
    let lineIdx = 0;
    const maxDist = 2.0;

    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const i3 = i * 3;
        const j3 = j * 3;
        const dx = posArray[i3] - posArray[j3];
        const dy = posArray[i3 + 1] - posArray[j3 + 1];
        const dz = posArray[i3 + 2] - posArray[j3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < maxDist && lineIdx < linePos.length - 6) {
          const alpha = 1 - dist / maxDist;

          linePos[lineIdx] = posArray[i3];
          linePos[lineIdx + 1] = posArray[i3 + 1];
          linePos[lineIdx + 2] = posArray[i3 + 2];
          linePos[lineIdx + 3] = posArray[j3];
          linePos[lineIdx + 4] = posArray[j3 + 1];
          linePos[lineIdx + 5] = posArray[j3 + 2];

          lineCol[lineIdx] = connectionColor.r * alpha;
          lineCol[lineIdx + 1] = connectionColor.g * alpha;
          lineCol[lineIdx + 2] = connectionColor.b * alpha;
          lineCol[lineIdx + 3] = connectionColor.r * alpha;
          lineCol[lineIdx + 4] = connectionColor.g * alpha;
          lineCol[lineIdx + 5] = connectionColor.b * alpha;

          lineIdx += 6;
        }
      }
    }
    lineGeometry.setDrawRange(0, lineIdx / 3);
    linePosAttr.needsUpdate = true;
    lineColAttr.needsUpdate = true;

    // Auto-rotation
    groupRef.current.rotation.y += 0.001;

    // Mouse parallax
    const targetRotX = -mouseRef.current.y * 0.1;
    const targetRotZ = mouseRef.current.x * 0.1;
    groupRef.current.rotation.x +=
      (targetRotX - groupRef.current.rotation.x) * 0.02;
    groupRef.current.rotation.z +=
      (targetRotZ - groupRef.current.rotation.z) * 0.02;
  });

  return (
    <group ref={groupRef}>
      <points ref={meshRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[colors, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.04}
          vertexColors
          transparent
          opacity={0.9}
          sizeAttenuation
          depthWrite={false}
        />
      </points>
      <lineSegments ref={linesRef} geometry={lineGeometry}>
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={0.3}
          depthWrite={false}
        />
      </lineSegments>
    </group>
  );
}

/* ── Exported Canvas Wrapper ──────────────────────────────────── */

interface ParticleNetworkProps {
  className?: string;
}

const ParticleNetwork: React.FC<ParticleNetworkProps> = ({ className }) => {
  const mouseRef = useRef({ x: 0, y: 0 });

  // Reduce particles on mobile
  const isMobile =
    typeof window !== "undefined" && window.innerWidth < 768;
  const particleCount = isMobile ? 80 : 200;

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      mouseRef.current.x =
        ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      mouseRef.current.y =
        ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    },
    []
  );

  return (
    <div
      className={className}
      onPointerMove={handlePointerMove}
      style={{ width: "100%", height: "100%" }}
    >
      <Canvas
        camera={{ position: [0, 0, 6], fov: 60 }}
        dpr={[1, 1.5]}
        gl={{ antialias: false, alpha: true }}
        style={{ background: "transparent" }}
      >
        <Particles count={particleCount} mouseRef={mouseRef} />
      </Canvas>
    </div>
  );
};

export default ParticleNetwork;
