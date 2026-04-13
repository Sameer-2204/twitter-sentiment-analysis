import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

type MouseState = React.MutableRefObject<{ x: number; y: number }>;

type NodeSeed = {
  cluster: number;
  phase: number;
  radius: number;
  lift: number;
  speed: number;
  drift: number;
};

const CLUSTERS = [
  {
    center: new THREE.Vector3(-2.6, 0.8, -0.4),
    color: new THREE.Color("#06d6a0"),
    size: 0.16,
  },
  {
    center: new THREE.Vector3(2.7, -1, 0.6),
    color: new THREE.Color("#ef476f"),
    size: 0.15,
  },
  {
    center: new THREE.Vector3(0.3, 2.1, -0.8),
    color: new THREE.Color("#ffd166"),
    size: 0.14,
  },
  {
    center: new THREE.Vector3(0, -0.1, 0.2),
    color: new THREE.Color("#4361ee"),
    size: 0.19,
  },
];

function NeuralField({
  count,
  mouseRef,
}: {
  count: number;
  mouseRef: MouseState;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const pointsRef = useRef<THREE.Points>(null);

  const { colors, lineGeometry, pointGeometry, seeds } = useMemo(() => {
    const pointPositions = new Float32Array(count * 3);
    const pointColors = new Float32Array(count * 3);
    const localSeeds: NodeSeed[] = [];

    for (let index = 0; index < count; index += 1) {
      const cluster = index % CLUSTERS.length;
      const color = CLUSTERS[cluster].color;
      const offset = index * 3;

      pointColors[offset] = color.r;
      pointColors[offset + 1] = color.g;
      pointColors[offset + 2] = color.b;

      localSeeds.push({
        cluster,
        phase: Math.random() * Math.PI * 2,
        radius: 0.65 + Math.random() * 0.9,
        lift: (Math.random() - 0.5) * 1.1,
        speed: 0.35 + Math.random() * 0.4,
        drift: 0.12 + Math.random() * 0.18,
      });
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(pointColors, 3));

    const maxConnections = count * 10;
    const lines = new THREE.BufferGeometry();
    lines.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(maxConnections * 6), 3)
    );
    lines.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(maxConnections * 6), 3)
    );
    lines.setDrawRange(0, 0);

    return {
      colors: pointColors,
      lineGeometry: lines,
      pointGeometry: geometry,
      seeds: localSeeds,
    };
  }, [count]);

  useFrame((state) => {
    if (!groupRef.current || !pointsRef.current) return;

    const time = state.clock.elapsedTime;
    const pointAttribute = pointsRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const pointPositions = pointAttribute.array as Float32Array;

    for (let index = 0; index < count; index += 1) {
      const seed = seeds[index];
      const cluster = CLUSTERS[seed.cluster];
      const offset = index * 3;
      const orbit = time * seed.speed + seed.phase;

      pointPositions[offset] =
        cluster.center.x +
        Math.cos(orbit) * seed.radius +
        Math.sin(time * seed.drift + seed.phase) * 0.28 +
        mouseRef.current.x * 0.24;
      pointPositions[offset + 1] =
        cluster.center.y +
        Math.sin(orbit * 1.2) * seed.radius * 0.45 +
        seed.lift +
        mouseRef.current.y * 0.18;
      pointPositions[offset + 2] =
        cluster.center.z +
        Math.cos(orbit * 0.8) * 0.85 +
        Math.sin(orbit * 1.4) * 0.2;
    }

    pointAttribute.needsUpdate = true;

    const linePositions = lineGeometry.attributes.position as THREE.BufferAttribute;
    const lineColors = lineGeometry.attributes.color as THREE.BufferAttribute;
    const lineArray = linePositions.array as Float32Array;
    const colorArray = lineColors.array as Float32Array;
    const mixColor = new THREE.Color();
    let cursor = 0;

    for (let first = 0; first < count; first += 1) {
      for (let second = first + 1; second < count; second += 1) {
        const firstSeed = seeds[first];
        const secondSeed = seeds[second];
        const linked =
          firstSeed.cluster === secondSeed.cluster ||
          firstSeed.cluster === 3 ||
          secondSeed.cluster === 3;

        if (!linked) continue;

        const a = first * 3;
        const b = second * 3;
        const dx = pointPositions[a] - pointPositions[b];
        const dy = pointPositions[a + 1] - pointPositions[b + 1];
        const dz = pointPositions[a + 2] - pointPositions[b + 2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (distance > 1.55 || cursor > lineArray.length - 6) continue;

        const alpha = 1 - distance / 1.55;
        mixColor
          .copy(CLUSTERS[firstSeed.cluster].color)
          .lerp(CLUSTERS[secondSeed.cluster].color, 0.5);

        lineArray[cursor] = pointPositions[a];
        lineArray[cursor + 1] = pointPositions[a + 1];
        lineArray[cursor + 2] = pointPositions[a + 2];
        lineArray[cursor + 3] = pointPositions[b];
        lineArray[cursor + 4] = pointPositions[b + 1];
        lineArray[cursor + 5] = pointPositions[b + 2];

        colorArray[cursor] = mixColor.r * alpha;
        colorArray[cursor + 1] = mixColor.g * alpha;
        colorArray[cursor + 2] = mixColor.b * alpha;
        colorArray[cursor + 3] = mixColor.r * alpha;
        colorArray[cursor + 4] = mixColor.g * alpha;
        colorArray[cursor + 5] = mixColor.b * alpha;

        cursor += 6;
      }
    }

    lineGeometry.setDrawRange(0, cursor / 3);
    linePositions.needsUpdate = true;
    lineColors.needsUpdate = true;

    groupRef.current.rotation.y = THREE.MathUtils.lerp(
      groupRef.current.rotation.y,
      mouseRef.current.x * 0.28 + time * 0.04,
      0.04
    );
    groupRef.current.rotation.x = THREE.MathUtils.lerp(
      groupRef.current.rotation.x,
      -mouseRef.current.y * 0.18,
      0.04
    );
  });

  return (
    <group ref={groupRef}>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.1, 0]}>
        <torusGeometry args={[1.7, 0.025, 12, 120]} />
        <meshBasicMaterial color="#4361ee" transparent opacity={0.24} />
      </mesh>

      <mesh position={[0, -0.1, 0]}>
        <sphereGeometry args={[0.18, 24, 24]} />
        <meshStandardMaterial
          color="#4361ee"
          emissive="#4361ee"
          emissiveIntensity={1.6}
          transparent
          opacity={0.95}
        />
      </mesh>

      {CLUSTERS.slice(0, 3).map((cluster) => (
        <group key={cluster.color.getHexString()} position={cluster.center.toArray()}>
          <mesh>
            <sphereGeometry args={[cluster.size, 24, 24]} />
            <meshStandardMaterial
              color={cluster.color}
              emissive={cluster.color}
              emissiveIntensity={1.25}
              transparent
              opacity={0.92}
            />
          </mesh>
          <mesh>
            <sphereGeometry args={[cluster.size * 2.8, 20, 20]} />
            <meshBasicMaterial color={cluster.color} transparent opacity={0.08} />
          </mesh>
        </group>
      ))}

      <points ref={pointsRef} geometry={pointGeometry}>
        <pointsMaterial
          size={0.05}
          vertexColors
          transparent
          opacity={0.95}
          sizeAttenuation
          depthWrite={false}
        />
      </points>

      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial vertexColors transparent opacity={0.24} depthWrite={false} />
      </lineSegments>
    </group>
  );
}

const NeuralHeroScene: React.FC = () => {
  const [count, setCount] = useState(160);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const media = window.matchMedia("(max-width: 860px)");
    const updateCount = () => setCount(media.matches ? 84 : 160);
    updateCount();

    media.addEventListener("change", updateCount);
    return () => media.removeEventListener("change", updateCount);
  }, []);

  return (
    <div
      style={{ width: "100%", height: "100%" }}
      onPointerMove={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        mouseRef.current.x = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
        mouseRef.current.y = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
      }}
      onPointerLeave={() => {
        mouseRef.current.x = 0;
        mouseRef.current.y = 0;
      }}
    >
      <Canvas
        camera={{ position: [0, 0.2, 7.3], fov: 52 }}
        dpr={[1, 1.5]}
        gl={{ antialias: false, alpha: true }}
        style={{ background: "transparent" }}
      >
        <fog attach="fog" args={["#050816", 7, 14]} />
        <ambientLight intensity={0.75} />
        <pointLight position={[3.4, 2.4, 5.8]} color="#4361ee" intensity={6.5} />
        <pointLight position={[-3.8, -1.4, 4.4]} color="#06d6a0" intensity={5} />
        <pointLight position={[2.4, -3.1, 2.2]} color="#ef476f" intensity={4.4} />
        <NeuralField count={count} mouseRef={mouseRef} />
      </Canvas>
    </div>
  );
};

export default NeuralHeroScene;
