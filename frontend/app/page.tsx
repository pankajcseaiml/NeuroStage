"use client";

import Link from "next/link";

const STAGE_COLORS = {
  Wake: "var(--stage-wake)",
  N1: "var(--stage-n1)",
  N2: "var(--stage-n2)",
  N3: "var(--stage-n3)",
  REM: "var(--stage-rem)",
};

const ARCHITECTURE_STEPS = [
  { icon: "📡", label: "Raw EEG", sub: "Single channel" },
  { icon: "🔧", label: "Filter & Norm", sub: "0.3–35 Hz, z-score" },
  { icon: "⏱️", label: "30s Epochs", sub: "AASM standard" },
  { icon: "🔬", label: "1D-CNN", sub: "Depthwise-separable" },
  { icon: "🧩", label: "Embeddings", sub: "128-dim per epoch" },
  { icon: "🔍", label: "Transformer", sub: "2-layer, 4-head" },
  { icon: "🎯", label: "5-Class Output", sub: "Wake/N1/N2/N3/REM" },
];

const FEATURES = [
  {
    icon: "⚡",
    title: "Lightweight CNN-Transformer",
    description:
      "Depthwise-separable 1D-CNN with a compact 2-layer self-attention encoder (~94K params). Engineered for minimal memory footprint and fast CPU execution without requiring high-power GPU clusters.",
    badge: "Efficiency",
    color: "#2563EB",
  },
  {
    icon: "📊",
    title: "Subject-Independent CV",
    description:
      "k-fold cross-validation partitioned strictly by subject ID to prevent data leakage. Reports overall accuracy, Cohen's Kappa, and per-class F1 (with critical focus on the minority N1 transition stage).",
    badge: "Rigor",
    color: "#7C3AED",
  },
  {
    icon: "🗜️",
    title: "Model Compression Pipeline",
    description:
      "Post-training INT8 dynamic quantization and teacher-student knowledge distillation to compress model footprint down to ~0.17 MB while preserving classification fidelity.",
    badge: "Compression",
    color: "#0891B2",
  },
  {
    icon: "👁️",
    title: "Inter-Epoch Attention Mapping",
    description:
      "Exposes multi-head self-attention matrices across the 20-epoch sequence, interpreting how preceding and succeeding sleep cycles drive stage transition decisions.",
    badge: "Explainability",
    color: "#059669",
  },
  {
    icon: "🔄",
    title: "Deterministic Tracking",
    description:
      "Centralized YAML configurations, fixed random seeds across all data splits, and automated MLflow artifact logging to ensure fully reproducible training runs and metric audits.",
    badge: "Reproducibility",
    color: "#D97706",
  },
  {
    icon: "📈",
    title: "Empirical Baseline Benchmarking",
    description:
      "Multi-dimensional comparative analysis against published baselines (DeepSleepNet, TinySleepNet, AttnSleep) across parameter scale, memory footprint, and classification accuracy on Sleep-EDF.",
    badge: "Evaluation",
    color: "#E11D48",
  },
];

const METRICS = [
  { label: "Sleep Stage Classes", value: "5", sub: "Wake · N1 · N2 · N3 · REM", target: false },
  { label: "Target Accuracy", value: "~83%", sub: "On Sleep-EDF (to be measured)", target: true },
  { label: "Epoch Window", value: "30s", sub: "AASM standard segmentation", target: false },
  { label: "Model Variants", value: "6+", sub: "Ablation + compression configs", target: false },
];

export default function Home() {
  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 24px" }}>
      {/* ── HERO ── */}
      <section
        className="grid-pattern"
        style={{
          padding: "96px 0 76px",
          textAlign: "center",
          position: "relative",
          borderRadius: 24,
          margin: "16px 0 32px",
          border: "1px solid var(--border)",
          background: "linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%)",
          boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04), 0 10px 24px -6px rgba(15, 23, 42, 0.03)",
        }}
      >
        {/* Subtle ambient light blooms */}
        <div style={{
          position: "absolute", top: "15%", left: "12%", width: 340, height: 340,
          background: "radial-gradient(circle, rgba(225, 77, 69, 0.06) 0%, transparent 70%)",
          borderRadius: "50%", pointerEvents: "none",
        }} />
        <div style={{
          position: "absolute", top: "25%", right: "12%", width: 340, height: 340,
          background: "radial-gradient(circle, rgba(37, 99, 235, 0.05) 0%, transparent 70%)",
          borderRadius: "50%", pointerEvents: "none",
        }} />

        <div className="fade-in-up" style={{ position: "relative", zIndex: 1, padding: "0 20px" }}>
          <div className="badge badge-coral" style={{ marginBottom: 20, display: "inline-flex" }}>
            🎓 B.Tech Final Year Project · KIT Kolhapur · 2026–27
          </div>
          <h1
            style={{
              fontSize: "clamp(2.5rem, 6vw, 4.4rem)",
              fontWeight: 900,
              letterSpacing: "-0.04em",
              lineHeight: 1.08,
              margin: "0 0 20px",
              color: "var(--text-primary)",
            }}
          >
            <span className="gradient-text">NeuroStage</span>
          </h1>
          <p
            style={{
              fontSize: "clamp(1.05rem, 2vw, 1.35rem)",
              color: "var(--text-secondary)",
              maxWidth: 720,
              margin: "0 auto 16px",
              lineHeight: 1.65,
              fontWeight: 500,
            }}
          >
            Lightweight CNN-Transformer Framework for{" "}
            <strong style={{ color: "var(--text-primary)", fontWeight: 700 }}>
              Real-Time EEG-Based Sleep Stage Classification
            </strong>
          </p>
          <p style={{ fontSize: "0.92rem", color: "var(--text-muted)", marginBottom: 38, maxWidth: 640, margin: "0 auto 38px", lineHeight: 1.6 }}>
            Core question: Can we hold competitive accuracy while being small and fast enough for edge & wearable deployment?
          </p>

          <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap" }}>
            <Link href="/predict">
              <button className="btn-primary">🚀 Try Live Demo</button>
            </Link>
            <Link href="/compare">
              <button className="btn-secondary">📊 Compare Models</button>
            </Link>
            <Link href="/experiments">
              <button className="btn-secondary">🧪 View Experiments</button>
            </Link>
          </div>
        </div>
      </section>

      {/* ── METRICS ── */}
      <section style={{ padding: "0 0 64px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
          {METRICS.map((m) => (
            <div key={m.label} className="metric-card">
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
                <div className="metric-value">{m.value}</div>
                {m.target && <span className="target-badge">⚠ Target</span>}
              </div>
              <div className="metric-label">{m.label}</div>
              <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", fontWeight: 500 }}>{m.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── SLEEP STAGES ── */}
      <section style={{ padding: "0 0 64px" }}>
        <h2 style={{ fontSize: "1.45rem", fontWeight: 800, marginBottom: 20, letterSpacing: "-0.02em", color: "var(--text-primary)" }}>
          5 Sleep Stage Classes
        </h2>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {Object.entries(STAGE_COLORS).map(([stage, color]) => (
            <div
              key={stage}
              className="glass-card"
              style={{ padding: "16px 22px", display: "flex", alignItems: "center", gap: 12 }}
            >
              <div style={{ width: 12, height: 12, borderRadius: "50%", background: color, flexShrink: 0 }} />
              <span style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.95rem" }}>{stage}</span>
              <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                {stage === "Wake" && "Wakefulness"}
                {stage === "N1" && "Light onset"}
                {stage === "N2" && "Spindles, K-complexes"}
                {stage === "N3" && "Deep / Slow-wave"}
                {stage === "REM" && "Rapid Eye Movement"}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ── ARCHITECTURE PIPELINE ── */}
      <section style={{ padding: "0 0 64px" }}>
        <h2 style={{ fontSize: "1.45rem", fontWeight: 800, marginBottom: 24, letterSpacing: "-0.02em", color: "var(--text-primary)" }}>
          Model Pipeline
        </h2>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            overflowX: "auto",
            padding: "22px",
            background: "#FFFFFF",
            borderRadius: 18,
            border: "1px solid var(--border)",
            boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04), 0 6px 18px -4px rgba(15, 23, 42, 0.03)",
          }}
        >
          {ARCHITECTURE_STEPS.map((step, i) => (
            <div key={step.label} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  textAlign: "center",
                  minWidth: 118,
                  padding: "16px 14px",
                  background: "#F8FAFC",
                  borderRadius: 12,
                  border: "1px solid var(--border)",
                  flexShrink: 0,
                  transition: "all 0.2s ease",
                }}
              >
                <div style={{ fontSize: 28, marginBottom: 8 }}>{step.icon}</div>
                <div style={{ fontWeight: 700, fontSize: "0.84rem", color: "var(--text-primary)" }}>
                  {step.label}
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4, fontWeight: 500 }}>
                  {step.sub}
                </div>
              </div>
              {i < ARCHITECTURE_STEPS.length - 1 && (
                <div style={{ color: "var(--brand-coral)", fontSize: "1.1rem", fontWeight: 700, flexShrink: 0 }}>
                  →
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section style={{ padding: "0 0 80px" }}>
        <h2 style={{ fontSize: "1.45rem", fontWeight: 800, marginBottom: 28, letterSpacing: "-0.02em", color: "var(--text-primary)" }}>
          What Makes This a Research-Grade Project
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
          {FEATURES.map((f) => (
            <div key={f.title} className="glass-card" style={{ padding: 28 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 16 }}>
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: 12,
                    background: `${f.color}12`,
                    border: `1px solid ${f.color}30`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 22,
                    flexShrink: 0,
                  }}
                >
                  {f.icon}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: "1.02rem", color: "var(--text-primary)" }}>
                    {f.title}
                  </div>
                  <span style={{ fontSize: "0.72rem", color: f.color, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                    {f.badge}
                  </span>
                </div>
              </div>
              <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.68, margin: 0, fontWeight: 450 }}>
                {f.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section
        style={{
          padding: "64px 32px",
          textAlign: "center",
          background: "#FFFFFF",
          borderRadius: 22,
          border: "1px solid var(--border)",
          boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04), 0 10px 28px -4px rgba(15, 23, 42, 0.04)",
          marginBottom: 80,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{
          position: "absolute", inset: 0,
          background: "radial-gradient(ellipse at center, rgba(225, 77, 69, 0.04) 0%, transparent 70%)",
          pointerEvents: "none",
        }} />
        <h2 style={{ fontSize: "2rem", fontWeight: 800, marginBottom: 12, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Explore the Research
        </h2>
        <p style={{ color: "var(--text-secondary)", marginBottom: 32, fontSize: "1rem", maxWidth: 580, margin: "0 auto 32px", lineHeight: 1.65, fontWeight: 450 }}>
          Upload an EEG recording, view the predicted hypnogram, inspect model attention, and compare
          NeuroStage against published baselines.
        </p>
        <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap", position: "relative", zIndex: 1 }}>
          <Link href="/predict">
            <button className="btn-primary pulse-glow">Start with Upload & Predict →</button>
          </Link>
        </div>
      </section>
    </div>
  );
}
