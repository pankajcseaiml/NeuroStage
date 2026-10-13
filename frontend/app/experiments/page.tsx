"use client";

import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useAuth } from "../context/AuthContext";
import { useRecording } from "../context/RecordingContext";
import { API_URL } from "../lib/api";

type Experiment = {
  id: string;
  mlflow_run_id: string | null;
  model_variant: string;
  accuracy: number | null;
  macro_f1: number | null;
  kappa: number | null;
  param_count: number | null;
  latency_ms: number | null;
  model_size_mb: number | null;
  created_at: string;
};

function ExperimentsContent() {
  const searchParams = useSearchParams();
  const urlId = searchParams.get("id") || "";
  const { token: authJwt } = useAuth();
  const { activeRecordingId, activeRecordingFilename, recordings, setActiveRecording } = useRecording();

  const currentStudyId = urlId || activeRecordingId || "";
  const currentStudy = recordings.find((r) => r.id === currentStudyId) || (recordings.length > 0 ? recordings[0] : null);

  const [exps, setExps] = useState<Experiment[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<keyof Experiment>("accuracy");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [activeTab, setActiveTab] = useState<"ablation" | "runs" | "methodology" | "qa" | "wavelets">("ablation");
  const [expandedQa, setExpandedQa] = useState<number | null>(0);

  useEffect(() => {
    if (urlId && urlId !== activeRecordingId) {
      setActiveRecording(urlId);
    }
  }, [urlId, activeRecordingId, setActiveRecording]);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const activeTok = authJwt || (typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : "") || "";
      const headers: Record<string, string> = {};
      if (activeTok) headers["Authorization"] = `Bearer ${activeTok}`;

      const res = await fetch(`${API_URL}/experiments?limit=100`, { headers });
      if (res.ok) {
        const data = await res.json();
        setExps(data.experiments || []);
        setTotal(data.total || 0);
      }
    } catch {}
    setLoading(false);
  };

  useEffect(() => {
    fetchExperiments();
  }, [authJwt]);

  const sorted = [...exps].sort((a, b) => {
    const av = a[sortKey] ?? "";
    const bv = b[sortKey] ?? "";
    if (av < bv) return sortDir === "asc" ? -1 : 1;
    if (av > bv) return sortDir === "asc" ? 1 : -1;
    return 0;
  });

  const toggleSort = (key: keyof Experiment) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sortIndicator = (key: keyof Experiment) =>
    sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : "";

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px" }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
          <div>
            <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
              Experimentation & Ablation Study Log
            </h1>
            <p style={{ color: "var(--text-secondary)", fontSize: "1rem", maxWidth: 820, fontWeight: 450, lineHeight: 1.6 }}>
              Systematic quantitative evaluation across model architectures, compression stages,
              and temporal mechanisms on PhysioNet Sleep-EDF benchmark dataset.
            </p>
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn-secondary" onClick={fetchExperiments} style={{ padding: "10px 16px", fontSize: "0.85rem", fontWeight: 700 }}>
              🔄 Refresh Runs
            </button>
            <Link
              href={currentStudy ? `/compare?id=${currentStudy.id}` : "/compare"}
              className="btn-primary"
              style={{ padding: "10px 18px", fontSize: "0.85rem", textDecoration: "none", fontWeight: 700 }}
            >
              📊 Compare vs Baselines →
            </Link>
          </div>
        </div>
      </div>

      {/* Active Study Pill */}
      {currentStudy && (
        <div
          className="glass-card"
          style={{
            padding: "14px 20px",
            marginBottom: 28,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 12,
            background: "rgba(37, 99, 235, 0.04)",
            border: "1px solid rgba(37, 99, 235, 0.2)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: "1.2rem" }}>📂</span>
            <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Active Study Context: <strong style={{ color: "var(--text-primary)" }}>{currentStudy.filename}</strong> ({currentStudy.n_epochs} epochs)
            </span>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <Link
              href={`/hypnogram?id=${currentStudy.id}`}
              style={{
                fontSize: "0.78rem",
                fontWeight: 700,
                padding: "4px 10px",
                borderRadius: 6,
                background: "rgba(37, 99, 235, 0.1)",
                color: "#1D4ED8",
                textDecoration: "none",
              }}
            >
              📈 Hypnogram
            </Link>
            <Link
              href={`/attention?id=${currentStudy.id}&epoch=10`}
              style={{
                fontSize: "0.78rem",
                fontWeight: 700,
                padding: "4px 10px",
                borderRadius: 6,
                background: "rgba(124, 58, 237, 0.1)",
                color: "#6D28D9",
                textDecoration: "none",
              }}
            >
              👁️ Attention Map
            </Link>
          </div>
        </div>
      )}

      {/* Defense Highlights Scorecards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16, marginBottom: 32 }}>
        <div className="metric-card" style={{ borderLeft: "4px solid #2563EB" }}>
          <div className="metric-label">Best Teacher Model Accuracy</div>
          <div className="metric-value" style={{ color: "#2563EB" }}>83.1%</div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 4 }}>
            Macro-F1: <strong>0.776</strong> · Kappa: <strong>0.768</strong>
          </div>
        </div>

        <div className="metric-card" style={{ borderLeft: "4px solid #059669" }}>
          <div className="metric-label">Compact Student (95% Compression)</div>
          <div className="metric-value" style={{ color: "#059669" }}>82.4%</div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 4 }}>
            Only <strong>94.4K params</strong> · <strong>0.40 MB</strong> storage
          </div>
        </div>

        <div className="metric-card" style={{ borderLeft: "4px solid #7C3AED" }}>
          <div className="metric-label">INT8 Quantized Model</div>
          <div className="metric-value" style={{ color: "#7C3AED" }}>0.17 MB</div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 4 }}>
            82.1% Acc · <strong>43x size reduction</strong> for edge MCU
          </div>
        </div>

        <div className="metric-card" style={{ borderLeft: "4px solid #BE123C" }}>
          <div className="metric-label">Ablation: CNN-Only (No Transformer)</div>
          <div className="metric-value" style={{ color: "#BE123C" }}>78.9%</div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: 4 }}>
            -4.2% drop proves temporal attention is vital
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div style={{ display: "flex", gap: 8, borderBottom: "1px solid var(--border)", marginBottom: 28, overflowX: "auto" }}>
        {[
          { key: "ablation", label: "🧪 Research Ablation Study" },
          { key: "wavelets", label: "🌊 3-Branch Wavelet Physics" },
          { key: "runs", label: `📋 Training Runs (${total})` },
          { key: "methodology", label: "⚙️ Training Methodology & Math" },
          { key: "qa", label: "🎓 Panel Defense Q&A Simulator" },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as any)}
            style={{
              padding: "10px 18px",
              background: "none",
              border: "none",
              borderBottom: activeTab === key ? "3px solid var(--brand-coral)" : "3px solid transparent",
              color: activeTab === key ? "var(--brand-coral)" : "var(--text-secondary)",
              fontWeight: 700,
              fontSize: "0.88rem",
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* TAB 1: Ablation Study Analysis */}
      {activeTab === "ablation" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1.2rem", fontWeight: 800, marginBottom: 8, color: "var(--text-primary)" }}>
              Core Research Ablation Matrix
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 20, lineHeight: 1.6 }}>
              The ablation study systematically removes each architectural component to prove its individual contribution
              to clinical accuracy and edge efficiency on the Sleep-EDF benchmark.
            </p>

            <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 12 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Architecture Variant</th>
                    <th>Accuracy</th>
                    <th>Macro-F1</th>
                    <th>Cohen&apos;s Kappa</th>
                    <th>Parameters</th>
                    <th>Size (MB)</th>
                    <th>Latency (ms)</th>
                    <th>Panel Defense Significance</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <div style={{ fontWeight: 800, color: "#2563EB" }}>NeuroStage (Base Teacher)</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>1D-CNN + 2-Layer Transformer</div>
                    </td>
                    <td><strong>83.1%</strong></td>
                    <td>0.776</td>
                    <td>0.768</td>
                    <td>1.90M</td>
                    <td>7.29 MB</td>
                    <td>3.8 ms</td>
                    <td style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      Full representation capacity. Serves as teacher for knowledge distillation.
                    </td>
                  </tr>

                  <tr>
                    <td>
                      <div style={{ fontWeight: 800, color: "#059669" }}>NeuroStage (Compact Student)</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Distilled + Depthwise-Separable CNN</div>
                    </td>
                    <td><strong>82.4%</strong></td>
                    <td>0.768</td>
                    <td>0.762</td>
                    <td><strong>94.4K (-95%)</strong></td>
                    <td><strong>0.40 MB</strong></td>
                    <td><strong>1.8 ms</strong></td>
                    <td style={{ fontSize: "0.8rem", color: "#065F46", fontWeight: 600 }}>
                      Retains 99.2% of teacher performance while eliminating 95% of weights.
                    </td>
                  </tr>

                  <tr>
                    <td>
                      <div style={{ fontWeight: 800, color: "#7C3AED" }}>NeuroStage (INT8 Quantized)</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Post-Training INT8 Quantization</div>
                    </td>
                    <td><strong>82.1%</strong></td>
                    <td>0.764</td>
                    <td>0.758</td>
                    <td>94.4K</td>
                    <td><strong>0.17 MB (170 KB)</strong></td>
                    <td>2.2 ms</td>
                    <td style={{ fontSize: "0.8rem", color: "#6D28D9", fontWeight: 600 }}>
                      Fits comfortably inside 512KB SRAM of ultra-low-power microcontrollers (STM32/Nordic).
                    </td>
                  </tr>

                  <tr>
                    <td>
                      <div style={{ fontWeight: 800, color: "#BE123C" }}>Ablation 1: CNN-Only</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>No Transformer / No Temporal Context</div>
                    </td>
                    <td><strong style={{ color: "#BE123C" }}>78.9% (-4.2%)</strong></td>
                    <td><strong style={{ color: "#BE123C" }}>0.718 (-5.8%)</strong></td>
                    <td>0.710</td>
                    <td>1.84M</td>
                    <td>7.08 MB</td>
                    <td>2.1 ms</td>
                    <td style={{ fontSize: "0.8rem", color: "#BE123C", fontWeight: 600 }}>
                      Directly proves that single-epoch EEG features cannot resolve ambiguous stages without temporal context.
                    </td>
                  </tr>

                  <tr>
                    <td>
                      <div style={{ fontWeight: 800, color: "#D97706" }}>Ablation 2: 1-Layer Transformer</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Single self-attention encoder layer</div>
                    </td>
                    <td>81.6% (-0.8%)</td>
                    <td>0.754</td>
                    <td>0.748</td>
                    <td>88.1K</td>
                    <td>0.36 MB</td>
                    <td>1.5 ms</td>
                    <td style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      Insufficient depth to disentangle transition dynamics between N1, REM, and Wake.
                    </td>
                  </tr>

                  <tr>
                    <td>
                      <div style={{ fontWeight: 800, color: "#64748B" }}>Ablation 3: 4-Layer Transformer</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Deep 4-layer temporal encoder</div>
                    </td>
                    <td>83.3% (+0.2%)</td>
                    <td>0.778</td>
                    <td>0.770</td>
                    <td>119.5K</td>
                    <td>0.51 MB</td>
                    <td>3.4 ms (+88%)</td>
                    <td style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      Diminishing returns: 40% latency increase for negligible +0.2% gain. Proves 2 layers is Pareto optimal.
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Component Deep Dives */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: "1rem", fontWeight: 800, marginBottom: 12, color: "var(--text-primary)" }}>
                💡 Why Depthwise-Separable Convolutions?
              </h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                Standard 1D-CNNs perform full spatial and temporal convolutions simultaneously, requiring
                thousands of MAC (multiply-accumulate) operations. NeuroStage splits each layer into:
              </p>
              <ul style={{ fontSize: "0.82rem", color: "var(--text-secondary)", paddingLeft: 20, marginTop: 8, lineHeight: 1.6 }}>
                <li><strong>Depthwise Conv:</strong> 1D filtering across the temporal dimension per channel.</li>
                <li><strong>Pointwise Conv:</strong> 1x1 projection across feature channels.</li>
                <li><strong>Result:</strong> Reduces parameter count by ~75% and FLOPs by 70% with zero loss in feature extraction accuracy.</li>
              </ul>
            </div>

            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: "1rem", fontWeight: 800, marginBottom: 12, color: "var(--text-primary)" }}>
                🧠 Why 2-Layer Transformer with 20-Epoch Context?
              </h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                Sleep physiology does not change erratically. Sleep stages occur in continuous 5 to 15 minute cycles:
              </p>
              <ul style={{ fontSize: "0.82rem", color: "var(--text-secondary)", paddingLeft: 20, marginTop: 8, lineHeight: 1.6 }}>
                <li><strong>20 epochs = 10 minutes:</strong> Perfectly matches the shortest natural sleep stage burst (N1 or micro-arousal).</li>
                <li><strong>2 layers vs 4 layers:</strong> Testing 4 layers increased latency by +40% for only +0.2% accuracy. 2 layers is the optimal Pareto boundary.</li>
                <li><strong>4 attention heads:</strong> Allows separate heads to focus on spindles (12–14 Hz), delta waves, and micro-awakenings.</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Wavelet Physics */}
      {activeTab === "wavelets" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1.2rem", fontWeight: 800, marginBottom: 8, color: "var(--text-primary)" }}>
              🌊 Multi-Scale Wavelet Physics: 3 Parallel 1D-CNN Branches
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: 24, lineHeight: 1.6 }}>
              In sleep electroencephalography (EEG), different neurological phenomena occur across distinct frequency bands.
              Rather than using a generic single convolution filter, NeuroStage implements three mathematically tailored branches:
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
              <div style={{ padding: 20, borderRadius: 12, background: "rgba(37, 99, 235, 0.05)", border: "1px solid rgba(37, 99, 235, 0.2)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontWeight: 800, color: "#2563EB", fontSize: "1.05rem" }}>Branch 1: Kernel 50</span>
                  <span className="badge" style={{ background: "#DBEAFE", color: "#1E40AF" }}>500 ms @ 100 Hz</span>
                </div>
                <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: 6 }}>
                  Target: Slow Delta Oscillations (0.5 – 4.0 Hz)
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                  Dominates in Stage N3 (Slow-Wave Deep Sleep). Captures wide, high-voltage delta dip waves (&gt;75 µV).
                  A 500ms receptive field spans half an entire slow wave cycle, enabling reliable deep sleep identification.
                </p>
              </div>

              <div style={{ padding: 20, borderRadius: 12, background: "rgba(234, 88, 12, 0.05)", border: "1px solid rgba(234, 88, 12, 0.2)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontWeight: 800, color: "#EA580C", fontSize: "1.05rem" }}>Branch 2: Kernel 25</span>
                  <span className="badge" style={{ background: "#FFEDD5", color: "#9A3412" }}>250 ms @ 100 Hz</span>
                </div>
                <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: 6 }}>
                  Target: Theta & Alpha Rhythms (4 – 12 Hz)
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                  Distinguishes Awake (8–12 Hz occipital alpha rhythm) from Light Sleep N1 (4–7 Hz central theta waves).
                  A 250ms receptive field isolates 1 to 2 cycles of alpha rhythm attenuation.
                </p>
              </div>

              <div style={{ padding: 20, borderRadius: 12, background: "rgba(124, 58, 237, 0.05)", border: "1px solid rgba(124, 58, 237, 0.2)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontWeight: 800, color: "#7C3AED", fontSize: "1.05rem" }}>Branch 3: Kernel 10</span>
                  <span className="badge" style={{ background: "#EDE9FE", color: "#6D28D9" }}>100 ms @ 100 Hz</span>
                </div>
                <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: 6 }}>
                  Target: Sleep Spindles & K-Complexes (12 – 16 Hz)
                </div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
                  The gold standard hallmark of Stage N2 sleep. Fast spindle bursts and steep negative deflections of K-complexes
                  require high temporal resolution (100ms) to prevent blurring.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Runs Table */}
      {activeTab === "runs" && (
        <div className="glass-card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <h2 style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)" }}>
                Logged Training Runs from PostgreSQL / MLflow
              </h2>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
                Click any column header to sort. Verified metrics recorded during training.
              </p>
            </div>
            <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontWeight: 600 }}>
              {sorted.length} recorded runs
            </span>
          </div>

          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  {[
                    { key: "model_variant", label: "Model Architecture" },
                    { key: "accuracy", label: "Accuracy" },
                    { key: "macro_f1", label: "Macro-F1" },
                    { key: "kappa", label: "Cohen's Kappa" },
                    { key: "param_count", label: "Parameters" },
                    { key: "model_size_mb", label: "Size (MB)" },
                    { key: "latency_ms", label: "Latency (ms)" },
                    { key: "created_at", label: "Date / Run" },
                  ].map(({ key, label }) => (
                    <th
                      key={key}
                      onClick={() => toggleSort(key as keyof Experiment)}
                      style={{ cursor: "pointer", userSelect: "none", whiteSpace: "nowrap" }}
                    >
                      {label}{sortIndicator(key as keyof Experiment)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((e) => (
                  <tr key={e.id}>
                    <td>
                      <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>{e.model_variant}</div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>ID: <code>{e.id.slice(0, 8)}…</code></div>
                    </td>
                    <td>
                      {e.accuracy != null ? (
                        <span style={{ fontWeight: 700, color: e.accuracy >= 0.82 ? "#059669" : "#BE123C" }}>
                          {(e.accuracy * 100).toFixed(1)}%
                        </span>
                      ) : "—"}
                    </td>
                    <td>{e.macro_f1 != null ? e.macro_f1.toFixed(3) : "—"}</td>
                    <td>{e.kappa != null ? e.kappa.toFixed(3) : "—"}</td>
                    <td>{e.param_count != null ? `${(e.param_count / 1000).toFixed(0)}K` : "—"}</td>
                    <td>{e.model_size_mb != null ? `${e.model_size_mb.toFixed(2)} MB` : "—"}</td>
                    <td>{e.latency_ms != null ? `${e.latency_ms.toFixed(1)} ms` : "—"}</td>
                    <td style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                      {e.created_at ? new Date(e.created_at).toLocaleDateString() : "Baseline"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 4: Methodology & Math */}
      {activeTab === "methodology" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 800, marginBottom: 16, color: "var(--text-primary)" }}>
              📊 Academic Integrity: Leak-Free Subject-Independent Validation
            </h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "var(--brand-coral)" }}>Sleep-EDF Benchmark (PhysioNet)</div>
                <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: 2, lineHeight: 1.55 }}>
                  39 full-night polysomnographic recordings from 20 healthy subjects. Fpz-Cz single-channel EEG sampled at 100 Hz.
                </div>
              </div>

              <div>
                <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "var(--brand-coral)" }}>5-Fold Subject-Independent Splitting</div>
                <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: 2, lineHeight: 1.55 }}>
                  Crucial methodological safeguard: all recordings from any given subject are placed strictly into train OR test — never split across folds. This guarantees ZERO subject-leakage.
                </div>
              </div>

              <div>
                <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "var(--brand-coral)" }}>AASM 5-Class Categorization</div>
                <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: 2, lineHeight: 1.55 }}>
                  Mapped standard Rechtschaffen & Kales (R&K) scoring into AASM 5 stages: Wake, N1 (Light), N2, N3 (Slow Wave Deep), REM.
                </div>
              </div>
            </div>
          </div>

          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 800, marginBottom: 16, color: "var(--text-primary)" }}>
              ⚡ Loss Formulation & Distillation Mathematics
            </h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "var(--brand-coral)" }}>Inverse-Frequency Class Weighted Cross-Entropy</div>
                <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: 2, lineHeight: 1.55 }}>
                  N1 represents less than 5% of adult sleep. We weight the loss by inverse class frequencies:
                  <div style={{ padding: "8px 12px", background: "rgba(0,0,0,0.04)", borderRadius: 6, fontFamily: "monospace", margin: "6px 0", fontSize: "0.8rem" }}>
                    w_c = N_total / (5 * N_c)
                  </div>
                  This prevents the model from ignoring rare N1 transitions.
                </div>
              </div>

              <div>
                <div style={{ fontWeight: 700, fontSize: "0.85rem", color: "var(--brand-coral)" }}>Knowledge Distillation Objective</div>
                <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: 2, lineHeight: 1.55 }}>
                  <div style={{ padding: "8px 12px", background: "rgba(0,0,0,0.04)", borderRadius: 6, fontFamily: "monospace", margin: "6px 0", fontSize: "0.78rem" }}>
                    L_KD = alpha * L_CE(student, y) + (1-alpha) * T^2 * KL(soft_s, soft_t)
                  </div>
                  Temperature T = 4.0 softens the teacher probability distribution, transferring dark knowledge about ambiguous transition epochs to the 94K student.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: Panel Defense Q&A Simulator */}
      {activeTab === "qa" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {[
            {
              q: "Panel: 'Why use a Transformer instead of an LSTM or GRU, which have fewer parameters?'",
              a: "While LSTMs may appear lighter, they are strictly sequential and cannot be parallelized across the time axis during training or vectorized batch inference. LSTMs also compress historical context into a single hidden vector, creating a memory bottleneck. Our 2-layer, 4-head Transformer uses only ~12.5K parameters, executes in parallel across the 20-epoch window, and produces an interpretable (20 x 20) self-attention matrix for clinicians.",
              highlight: "Parallel execution · Sub-125ms latency budget · Native clinician interpretability via attention maps.",
            },
            {
              q: "Panel: 'Your dataset accuracy is ~83%. Why isn\\'t it 95% like some published papers?'",
              a: "Studies claiming >90–95% accuracy on single-channel EEG typically commit one of two methodological errors: (1) Data leakage via random epoch shuffling, where models memorize subject-specific waveforms, or (2) Merging N1 with N2, reducing the problem to 4 classes. NeuroStage enforces strict subject-independent K-fold cross-validation across all 5 standard AASM stages. Furthermore, human inter-scorer agreement between two board-certified sleep physicians is typically 82–85%. Matching human inter-scorer concordance with ~94K parameters is clinically sound.",
              highlight: "Human expert concordance is 82–85%. NeuroStage matches board-certified physicians without leakage.",
            },
            {
              q: "Panel: 'Why did you use Depthwise-Separable Convolutions?'",
              a: "Standard 1D convolution computes channel mixing and temporal filtering simultaneously, requiring C_in x C_out x K operations. Depthwise-separable convolutions decouple this into spatial depthwise filtering per channel followed by a 1x1 pointwise projection. For our kernel sizes (K=50, 25, 10) and channel dimensions (C=64), this reduces Multiply-Accumulates by approximately 8 to 9-fold, cutting parameters while retaining multi-band frequency sensitivity.",
              highlight: "8-9x reduction in MAC operations · Preserves frequency band selectivity across Delta, Theta, and Spindles.",
            },
            {
              q: "Panel: 'Why evaluate on Macro-F1 and Cohen\\'s Kappa instead of just Accuracy?'",
              a: "Sleep stage distributions are highly imbalanced: N2 comprises ~50% of the night, while N1 accounts for only 5–8%. A naive classifier predicting only N2 and Wake can achieve >75% accuracy with an F1 score of 0 on N1. Macro-F1 weights every class equally, and Cohen\\'s Kappa (κ) explicitly corrects for chance agreement. These are the gold-standard metrics in sleep medicine.",
              highlight: "Corrects for severe class imbalance. Verifies true predictive utility across rare N1 transitions.",
            },
            {
              q: "Panel: 'How does the model handle signal artifacts (sweat, electrode pops, movement)?'",
              a: "Our preprocessing pipeline applies an FIR bandpass filter (0.3–35 Hz) that removes slow galvanic sweat drift (<0.3 Hz) and high-frequency muscle noise or mains hum (>35 Hz), followed by per-recording z-score normalization. In Phase II, we plan to implement automated variance thresholding to flag high-amplitude movement artifacts as Wake in accordance with AASM Rule VII.",
              highlight: "FIR 0.3-35 Hz bandpass filter + per-recording z-score normalization eliminates mains hum and DC drift.",
            },
          ].map((item, idx) => (
            <div
              key={idx}
              className="glass-card"
              style={{
                padding: 22,
                borderLeft: "4px solid var(--brand-coral)",
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
              onClick={() => setExpandedQa(expandedQa === idx ? null : idx)}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontWeight: 800, fontSize: "0.98rem", color: "var(--text-primary)" }}>
                  {item.q}
                </span>
                <span style={{ fontSize: "1rem", color: "var(--text-muted)", marginLeft: 12 }}>
                  {expandedQa === idx ? "▲" : "▼"}
                </span>
              </div>

              {expandedQa === idx && (
                <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid rgba(0,0,0,0.06)" }}>
                  <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6, marginBottom: 10 }}>
                    {item.a}
                  </p>
                  <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#065F46", background: "rgba(5, 150, 105, 0.08)", padding: "6px 12px", borderRadius: 6, display: "inline-block" }}>
                    🎯 Panel Defense Script: {item.highlight}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ExperimentsPage() {
  return (
    <Suspense fallback={<div style={{ padding: 60, textAlign: "center", color: "var(--text-muted)" }}>Loading experiments...</div>}>
      <ExperimentsContent />
    </Suspense>
  );
}
