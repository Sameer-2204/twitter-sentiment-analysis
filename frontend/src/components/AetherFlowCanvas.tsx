import React, { useEffect, useRef } from "react";

/**
 * AetherFlowCanvas â€” Interactive particle-mesh network that responds to cursor.
 * Renders on a transparent canvas so it layers with the Three.js NeuralHeroScene.
 * Purple-tinted particles connect with proximity lines that brighten near the mouse.
 */
const AetherFlowCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let particles: Particle[] = [];
    const mouse = { x: null as number | null, y: null as number | null, radius: 200 };

    class Particle {
      x: number;
      y: number;
      dx: number;
      dy: number;
      size: number;
      color: string;

      constructor(x: number, y: number, dx: number, dy: number, size: number, color: string) {
        this.x = x;
        this.y = y;
        this.dx = dx;
        this.dy = dy;
        this.size = size;
        this.color = color;
      }

      draw() {
        ctx!.beginPath();
        ctx!.arc(this.x, this.y, this.size, 0, Math.PI * 2, false);
        ctx!.fillStyle = this.color;
        ctx!.fill();
      }

      update() {
        if (this.x > canvas!.width || this.x < 0) this.dx = -this.dx;
        if (this.y > canvas!.height || this.y < 0) this.dy = -this.dy;

        // Mouse repulsion
        if (mouse.x !== null && mouse.y !== null) {
          const dxM = mouse.x - this.x;
          const dyM = mouse.y - this.y;
          const dist = Math.sqrt(dxM * dxM + dyM * dyM);
          if (dist < mouse.radius + this.size) {
            const force = (mouse.radius - dist) / mouse.radius;
            this.x -= (dxM / dist) * force * 5;
            this.y -= (dyM / dist) * force * 5;
          }
        }

        this.x += this.dx;
        this.y += this.dy;
        this.draw();
      }
    }

    function init() {
      particles = [];
      // fewer particles for performance (overlay, not primary bg)
      const count = Math.min((canvas!.width * canvas!.height) / 12000, 400);
      for (let i = 0; i < count; i++) {
        const size = Math.random() * 2 + 1;
        const x = Math.random() * (canvas!.width - size * 4) + size * 2;
        const y = Math.random() * (canvas!.height - size * 4) + size * 2;
        const dx = (Math.random() - 0.5) * 0.4;
        const dy = (Math.random() - 0.5) * 0.4;
        const color = `rgba(191, 128, 255, ${0.5 + Math.random() * 0.4})`;
        particles.push(new Particle(x, y, dx, dy, size, color));
      }
    }

    function connect() {
      const threshold = (canvas!.width / 7) * (canvas!.height / 7);
      for (let a = 0; a < particles.length; a++) {
        for (let b = a + 1; b < particles.length; b++) {
          const dxP = particles[a].x - particles[b].x;
          const dyP = particles[a].y - particles[b].y;
          const distSq = dxP * dxP + dyP * dyP;

          if (distSq < threshold) {
            const opacity = 1 - distSq / 20000;
            // brighten lines near cursor
            if (mouse.x !== null && mouse.y !== null) {
              const dxA = particles[a].x - mouse.x;
              const dyA = particles[a].y - mouse.y;
              const distA = Math.sqrt(dxA * dxA + dyA * dyA);
              ctx!.strokeStyle =
                distA < mouse.radius
                  ? `rgba(255, 255, 255, ${opacity})`
                  : `rgba(200, 150, 255, ${opacity})`;
            } else {
              ctx!.strokeStyle = `rgba(200, 150, 255, ${opacity})`;
            }
            ctx!.lineWidth = 1;
            ctx!.beginPath();
            ctx!.moveTo(particles[a].x, particles[a].y);
            ctx!.lineTo(particles[b].x, particles[b].y);
            ctx!.stroke();
          }
        }
      }
    }

    function animate() {
      animId = requestAnimationFrame(animate);
      // Transparent clear â€” lets layers beneath show through
      ctx!.clearRect(0, 0, canvas!.width, canvas!.height);
      for (const p of particles) p.update();
      connect();
    }

    function resize() {
      const parent = canvas!.parentElement;
      const w = parent ? parent.clientWidth : window.innerWidth;
      const h = parent ? parent.clientHeight : window.innerHeight;
      canvas!.width = w;
      canvas!.height = h;
      init();
    }

    const onMove = (e: MouseEvent) => {
      // Adjust mouse position relative to canvas (accounting for scroll)
      const rect = canvas!.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    };
    const onOut = () => {
      mouse.x = null;
      mouse.y = null;
    };

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseout", onOut);

    resize();
    animate();

    // Re-measure when page content changes height
    const resizeObserver = new ResizeObserver(() => resize());
    if (canvas!.parentElement) resizeObserver.observe(canvas!.parentElement);

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseout", onOut);
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="hp__aether-canvas" />;
};

export default AetherFlowCanvas;
