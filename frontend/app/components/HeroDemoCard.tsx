"use client";

import { useState, useEffect } from "react";

const STAGES = [
  { name: "Wake", label: "Wakefulness", color: "#EF4444", angle: -65, conf: 96.2 },
  { name: "N1", label: "Light Sleep", color: "#F97316", angle: -32, conf: 82.4 },
  { name: "N2", label: "Intermediate", color: "#EAB308", angle: 0, conf: 91.8 },
  { name: "N3", label: "Deep Slow-Wave", color: "#3B82F6", angle: 32, conf: 95.7 },
  { name: "REM", label: "Dreaming (REM)", color: "#8B5CF6", angle: 65, conf: 89.3 },
];

export default function HeroDemoCard() {
  const [selectedIdx, setSelectedIdx] = useState(3); // Start with N3 Deep Sleep
  const [isHovered, setIsHovered] = useState(false);

  // Auto-cycle gently when not hovered for dynamic realism
  useEffect(() => {
    if (isHovered) return;
    const interval = setInterval(() => {
      setSelectedIdx((prev) => (prev + 1) % STAGES.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [isHovered]);

  const current = STAGES[selectedIdx];

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        background: "#ffffff",
        borderRadius: 24,
        padding: "32px 28px 28px",
        boxShadow: "0 25px 60px -15px rgba(0, 0, 0, 0.4), 0 0 30px rgba(225, 77, 69, 0.12)",
        position: "relative",
        border: "1px solid rgba(255, 255, 255, 0.8)",
        color: "#0f172a",
        width: "100%",
        maxWidth: 480,
        margin: "0 auto",
      }}
    >
      {/* Live Demo floating badge (top right as in AyuPulse screenshot) */}
      <div
        style={{
          position: "absolute",
          top: -12,
          right: 28,
          background: "#E14D45",
          color: "#ffffff",
          padding: "4px 14px",
          borderRadius: 9999,
          fontSize: "0.75rem",
          fontWeight: 700,
          boxShadow: "0 4px 14px rgba(225, 77, 69, 0.45)",
          display: "flex",
          alignItems: "center",
          gap: 6,
          letterSpacing: "0.02em",
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: "#ffffff",
            animation: "pulse 1.5s infinite",
          }}
        />
        Live Demo
      </div>

      {/* Model Title */}
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <h3
          style={{
            fontSize: "1.1rem",
            fontWeight: 700,
            color: "#0f172a",
            margin: 0,
            letterSpacing: "-0.01em",
          }}
        >
          Sleep Architecture Model
        </h3>
        <p style={{ fontSize: "0.78rem", color: "#64748b", margin: "4px 0 0" }}>
          Interactive Real-Time Staging Gauge
        </p>
      </div>

      {/* Semi-Circular Radial Gauge */}
      <div style={{ position: "relative", width: 260, height: 145, margin: "0 auto 16px" }}>
        <svg viewBox="0 0 200 110" style={{ width: "100%", height: "100%", overflow: "visible" }}>
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#EF4444" />
              <stop offset="25%" stopColor="#F97316" />
              <stop offset="50%" stopColor="#EAB308" />
              <stop offset="75%" stopColor="#3B82F6" />
              <stop offset="100%" stopColor="#8B5CF6" />
            </linearGradient>
          </defs>

          {/* Background Track */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#f1f5f9"
            strokeWidth="14"
            strokeLinecap="round"
          />

          {/* Colored Arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="14"
            strokeLinecap="round"
          />

          {/* Center Hub */}
          <circle cx="100" cy="100" r="9" fill="#0f172a" />
          <circle cx="100" cy="100" r="4" fill="#ffffff" />

          {/* Needle with smooth transition */}
          <g
            style={{
              transformOrigin: "100px 100px",
              transform: `rotate(${current.angle}deg)`,
              transition: "transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)",
            }}
          >
            <line
              x1="100"
              y1="100"
              x2="100"
              y2="32"
              stroke="#0f172a"
              strokeWidth="3.5"
              strokeLinecap="round"
            />
          </g>
        </svg>

        {/* Current Stage Indicator */}
        <div
          style={{
            position: "absolute",
            bottom: -6,
            left: 0,
            right: 0,
            textAlign: "center",
          }}
        >
          <div
            style={{
              fontSize: "1.1rem",
              fontWeight: 800,
              color: current.color,
              transition: "color 0.3s ease",
            }}
          >
            {current.name} · {current.label}
          </div>
          <div style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: 500 }}>
            Model Confidence: <strong style={{ color: "#0f172a" }}>{current.conf}%</strong>
          </div>
        </div>
      </div>

      {/* Stage Selector Pills (Interactive) */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: 6,
          marginTop: 22,
          marginBottom: 20,
        }}
      >
        {STAGES.map((s, idx) => {
          const isSelected = selectedIdx === idx;
          return (
            <button
              key={s.name}
              type="button"
              onClick={() => setSelectedIdx(idx)}
              style={{
                background: isSelected ? s.color : "#f8fafc",
                color: isSelected ? "#ffffff" : "#64748b",
                border: `1px solid ${isSelected ? s.color : "#e2e8f0"}`,
                borderRadius: 9999,
                padding: "3px 10px",
                fontSize: "0.75rem",
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              {s.name}
            </button>
          );
        })}
      </div>

      {/* Bottom Progress Bar (As in AyuPulse screenshot: "Clinical Data Analyzed — Complete") */}
      <div style={{ borderTop: "1px solid #f1f5f9", paddingTop: 16 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "0.78rem",
            marginBottom: 8,
          }}
        >
          <span style={{ color: "#64748b", fontWeight: 500 }}>Clinical EEG Data Analyzed</span>
          <span style={{ color: "#10b981", fontWeight: 700 }}>Complete (100 Hz FIR)</span>
        </div>
        <div
          style={{
            width: "100%",
            height: 7,
            background: "#f1f5f9",
            borderRadius: 9999,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: "100%",
              height: "100%",
              background: "linear-gradient(90deg, #E14D45 0%, #3B82F6 100%)",
              borderRadius: 9999,
            }}
          />
        </div>

        {/* Micro statistics */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: 10,
            fontSize: "0.72rem",
            color: "#94a3b8",
          }}
        >
          <span>Context: 20 Epochs (10 min)</span>
          <span>Latency: 24 ms</span>
          <span>AASM v2.6 Compliant</span>
        </div>
      </div>
    </div>
  );
}
