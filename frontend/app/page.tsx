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
    title: "Lightweight Architecture",
    description:
      "Depthwise-separable 1D-CNN with compact Transformer encoder. Designed for edge & wearable deployment — not just GPU servers.",
    badge: "Innovation",
    color: "var(--accent-blue)",
  },
  {
    icon: "📊",
    title: "Research-Grade Evaluation",
    description:
      "Subject-independent k-fold CV, per-class F1 (including N1), Cohen's Kappa, and mean±std across folds. No information leakage.",
    badge: "Rigor",
    color: "var(--accent-violet)",
  },
  {
    icon: "🗜️",
    title: "Model Compression",
    description:
      "Pruning (L1 magnitude), INT8 quantization, and knowledge distillation — each independently benchmarked for accuracy vs. efficiency.",
    badge: "Compression",
    color: "var(--accent-cyan)",
  },
  {
    icon: "👁️",
    title: "Interpretability",
    description:
      "Attention weights from the Transformer's last layer are exposed and visualized — showing which epochs influenced each prediction.",
    badge: "Explainability",
    color: "var(--accent-emerald)",
  },
  {
    icon: "🔄",
    title: "Reproducible Experiments",
    description:
      "YAML config system, fixed seeds, MLflow tracking. Any reviewer can rerun the entire pipeline and get the same numbers back.",
    badge: "Reproducibility",
    color: "var(--accent-amber)",
  },
  {
    icon: "📈",
    title: "Accuracy-vs-Efficiency Analysis",
    description:
      "The core research output: where does NeuroStage sit on the trade-off curve vs. DeepSleepNet, TinySleepNet, and AttnSleep?",
    badge: "Research",
    color: "var(--accent-rose)",
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
          padding: "100px 0 80px",
          textAlign: "center",
          position: "relative",
        }}
      >
        {/* Glow blobs */}
        <div style={{
          position: "absolute", top: "20%", left: "10%", width: 400, height: 400,
          background: "radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%)",
          borderRadius: "50%", pointerEvents: "none",
        }} />
        <div style={{
          position: "absolute", top: "30%", right: "10%", width: 300, height: 300,
          background: "radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%)",
          borderRadius: "50%", pointerEvents: "none",
        }} />

        <div className="fade-in-up" style={{ position: "relative" }}>
          <div className="badge badge-blue" style={{ marginBottom: 20, display: "inline-flex" }}>
            🎓 B.Tech Final Year Project · KIT Kolhapur · 2026–27
          </div>
          <h1
            style={{
              fontSize: "clamp(2.5rem, 6vw, 4.5rem)",
              fontWeight: 900,
              letterSpacing: "-0.04em",
              lineHeight: 1.05,
              margin: "0 0 20px",
            }}
          >
            <span className="gradient-text">NeuroStage</span>
          </h1>
          <p
            style={{
              fontSize: "clamp(1rem, 2vw, 1.35rem)",
              color: "var(--text-secondary)",
              maxWidth: 720,
              margin: "0 auto 16px",
              lineHeight: 1.7,
              fontWeight: 400,
            }}
          >
            Lightweight CNN-Transformer Framework for{" "}
            <strong style={{ color: "var(--text-primary)" }}>Real-Time EEG-Based Sleep Stage Classification</strong>
          </p>
          <p style={{ fontSize: "0.9rem", color: "var(--text-muted)", marginBottom: 40 }}>
            Core question: Can we hold competitive accuracy while being small and fast enough for edge & wearable deployment?
          </p>

          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
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
                <div className="metric-value gradient-text">{m.value}</div>
                {m.target && <span className="target-badge">⚠ Target</span>}
              </div>
              <div className="metric-label">{m.label}</div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>{m.sub}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── SLEEP STAGES ── */}
      <section style={{ padding: "0 0 64px" }}>
        <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: 24, letterSpacing: "-0.02em" }}>
          5 Sleep Stage Classes
        </h2>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {Object.entries(STAGE_COLORS).map(([stage, color]) => (
            <div
              key={stage}
              className="glass-card"
              style={{ padding: "16px 24px", display: "flex", alignItems: "center", gap: 12 }}
            >
              <div style={{ width: 12, height: 12, borderRadius: "50%", background: color }} />
              <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{stage}</span>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
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
        <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: 32, letterSpacing: "-0.02em" }}>
          Model Pipeline
        </h2>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            overflowX: "auto",
            padding: "20px",
            background: "var(--bg-secondary)",
            borderRadius: 16,
            border: "1px solid var(--border)",
          }}
        >
          {ARCHITECTURE_STEPS.map((step, i) => (
            <div key={step.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  textAlign: "center",
                  minWidth: 110,
                  padding: "16px 12px",
                  background: "var(--bg-card)",
                  borderRadius: 12,
                  border: "1px solid var(--border)",
                  flexShrink: 0,
                }}
              >
                <div style={{ fontSize: 28, marginBottom: 6 }}>{step.icon}</div>
                <div style={{ fontWeight: 600, fontSize: "0.8rem", color: "var(--text-primary)" }}>
                  {step.label}
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: 4 }}>
                  {step.sub}
                </div>
              </div>
              {i < ARCHITECTURE_STEPS.length - 1 && (
                <div style={{ color: "var(--accent-blue)", fontSize: "1.2rem", flexShrink: 0 }}>→</div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section style={{ padding: "0 0 80px" }}>
        <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: 32, letterSpacing: "-0.02em" }}>
          What Makes This a Research-Grade Project
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 20 }}>
          {FEATURES.map((f) => (
            <div key={f.title} className="glass-card" style={{ padding: 28 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: 12,
                    background: `${f.color}20`,
                    border: `1px solid ${f.color}40`,
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
                  <div style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)" }}>
                    {f.title}
                  </div>
                  <span style={{ fontSize: "0.7rem", color: f.color, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                    {f.badge}
                  </span>
                </div>
              </div>
              <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.7, margin: 0 }}>
                {f.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section
        style={{
          padding: "60px",
          textAlign: "center",
          background: "var(--gradient-card)",
          borderRadius: 20,
          border: "1px solid var(--border)",
          marginBottom: 80,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div style={{
          position: "absolute", inset: 0,
          background: "radial-gradient(ellipse at center, rgba(59,130,246,0.06) 0%, transparent 70%)",
          pointerEvents: "none",
        }} />
        <h2 style={{ fontSize: "2rem", fontWeight: 800, marginBottom: 12, letterSpacing: "-0.03em" }}>
          Explore the Research
        </h2>
        <p style={{ color: "var(--text-secondary)", marginBottom: 32, fontSize: "1rem", maxWidth: 560, margin: "0 auto 32px" }}>
          Upload an EEG recording, view the predicted hypnogram, inspect model attention, and compare
          NeuroStage against published baselines.
        </p>
        <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <Link href="/predict">
            <button className="btn-primary pulse-glow">Start with Upload & Predict →</button>
          </Link>
        </div>
      </section>
    </div>
  );
}
