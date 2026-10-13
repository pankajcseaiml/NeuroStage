"use client";

import { useEffect, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { API_URL } from "../lib/api";

type BenchmarkModel = {
  model: string;
  accuracy: number | null;
  macro_f1: number | null;
  kappa: number | null;
  param_count: number | null;
  latency_ms: number | null;
  model_size_mb: number | null;
  source: string;
  note?: string;
  is_target?: boolean;
};

const MODEL_COLORS: Record<string, string> = {
  DeepSleepNet: "#DC2626",
  TinySleepNet: "#EA580C",
  AttnSleep: "#D97706",
  neurostage: "#2563EB",
  neurostage_base: "#2563EB",
  ablation_cnn_only: "#0891B2",
  ablation_transformer_1l: "#7C3AED",
  ablation_transformer_4l: "#9333EA",
};

function getColor(name: string) {
  for (const [key, color] of Object.entries(MODEL_COLORS)) {
    if (name.toLowerCase().includes(key.toLowerCase())) return color;
  }
  return "#64748B";
}

type CompareTooltipProps = {
  active?: boolean;
  payload?: Array<{
    payload: BenchmarkModel;
  }>;
};

const CustomTooltip = ({ active, payload }: CompareTooltipProps) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: "#FFFFFF",
      border: "1px solid #CBD5E1",
      borderRadius: 12,
      padding: "14px 18px",
      fontSize: "0.82rem",
      maxWidth: 270,
      boxShadow: "0 4px 16px rgba(15, 23, 42, 0.1)",
    }}>
      <div style={{ fontWeight: 800, fontSize: "0.95rem", color: getColor(d.model), marginBottom: 8 }}>{d.model}</div>
      <div style={{ color: "var(--text-secondary)", marginBottom: 3 }}>Accuracy: <strong style={{ color: "var(--text-primary)" }}>{d.accuracy != null ? `${(d.accuracy * 100).toFixed(1)}%` : "—"}</strong></div>
      <div style={{ color: "var(--text-secondary)", marginBottom: 3 }}>Macro-F1: <strong style={{ color: "var(--text-primary)" }}>{d.macro_f1 != null ? (d.macro_f1).toFixed(3) : "—"}</strong></div>
      <div style={{ color: "var(--text-secondary)", marginBottom: 3 }}>Params: <strong style={{ color: "var(--text-primary)" }}>{d.param_count != null ? `${(d.param_count / 1000).toFixed(0)}K` : "—"}</strong></div>
      <div style={{ color: "var(--text-secondary)", marginBottom: 3 }}>Size: <strong style={{ color: "var(--text-primary)" }}>{d.model_size_mb != null ? `${d.model_size_mb.toFixed(1)} MB` : "—"}</strong></div>
      <div style={{ color: "var(--text-muted)", marginTop: 8, fontSize: "0.74rem" }}>{d.source}</div>
      {d.note && <div style={{ color: "#D97706", fontSize: "0.72rem", marginTop: 4, fontWeight: 600 }}>ℹ️ {d.note}</div>}
    </div>
  );
};

export default function ComparePage() {
  const [benchmarks, setBenchmarks] = useState<{ neurostage_results: BenchmarkModel[]; baselines: BenchmarkModel[]; note: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [xAxis, setXAxis] = useState<"param_count" | "model_size_mb" | "latency_ms">("param_count");

  useEffect(() => {
    fetch(`${API_URL}/benchmark`)
      .then((r) => r.json())
      .then((d) => {
        setBenchmarks(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const allModels: BenchmarkModel[] = [
    ...(benchmarks?.baselines || []),
    ...(benchmarks?.neurostage_results || []),
  ];

  const scatterData = allModels.filter((m) => m.accuracy != null && m[xAxis] != null);

  const X_AXIS_LABELS: Record<string, string> = {
    param_count: "Parameter Count (K)",
    model_size_mb: "Model Size (MB)",
    latency_ms: "Inference Latency (ms)",
  };

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 36 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Model Comparison Dashboard
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", maxWidth: 720, fontWeight: 450 }}>
          NeuroStage variants vs. published baselines — accuracy, macro-F1, parameter count, and latency.
          This scatter plot is the core research output of the project.
        </p>
      </div>

      {/* Important labeling notice */}
      <div style={{
        padding: "16px 22px",
        borderRadius: 14,
        background: "rgba(217, 119, 6, 0.08)",
        border: "1px solid rgba(217, 119, 6, 0.25)",
        marginBottom: 32,
        fontSize: "0.875rem",
        color: "#92400E",
        lineHeight: 1.65,
        fontWeight: 500,
      }}>
        ⚠️ <strong>Data Integrity Notice:</strong> Baseline values (DeepSleepNet, TinySleepNet, AttnSleep)
        are published results from papers — not measured in this repository. NeuroStage values are
        experimentally measured results from training runs in this repository. Mixing these without
        clear labeling would be a research ethics violation. {benchmarks?.note}
      </div>

      {loading && <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)", fontWeight: 500 }}>⏳ Loading benchmarks...</div>}

      {benchmarks && (
        <>
          {/* X-axis selector */}
          <div style={{ display: "flex", gap: 10, marginBottom: 24, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 600 }}>X-axis metric:</span>
            {(["param_count", "model_size_mb", "latency_ms"] as const).map((key) => (
              <button
                key={key}
                className={xAxis === key ? "btn-primary" : "btn-secondary"}
                onClick={() => setXAxis(key)}
                style={{ padding: "8px 18px", fontSize: "0.82rem" }}
              >
                {X_AXIS_LABELS[key]}
              </button>
            ))}
          </div>

          {/* Scatter chart */}
          <div className="glass-card" style={{ padding: 32, marginBottom: 32 }}>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 800, marginBottom: 6, color: "var(--text-primary)" }}>
              Accuracy vs. {X_AXIS_LABELS[xAxis]}
            </h2>
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginBottom: 24, fontWeight: 500 }}>
              Upper-left = optimal frontier (higher accuracy, lower computational cost).
            </p>
            {scatterData.length > 0 ? (
              <ResponsiveContainer width="100%" height={380}>
                <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis
                    type="number"
                    dataKey={xAxis}
                    name={X_AXIS_LABELS[xAxis]}
                    tick={{ fill: "#64748B", fontSize: 11 }}
                    tickFormatter={(v) => xAxis === "param_count" ? `${(v / 1000).toFixed(0)}K` : v.toFixed(1)}
                    axisLine={{ stroke: "#E2E8F0" }}
                    label={{ value: X_AXIS_LABELS[xAxis], position: "insideBottom", offset: -10, fill: "#475569", fontSize: 12, fontWeight: 600 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="accuracy"
                    name="Accuracy"
                    domain={[0.7, 1.0]}
                    tick={{ fill: "#64748B", fontSize: 11 }}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    axisLine={{ stroke: "#E2E8F0" }}
                    label={{ value: "Accuracy", angle: -90, position: "insideLeft", fill: "#475569", fontSize: 12, fontWeight: 600 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  {scatterData.map((m) => (
                    <Scatter
                      key={m.model}
                      name={m.model}
                      data={[m]}
                      fill={getColor(m.model)}
                      shape={(props: { cx?: number; cy?: number }) => {
                        const { cx = 0, cy = 0 } = props;
                        const isNeuroStage = m.model.toLowerCase().includes("neurostage");
                        return (
                          <g>
                            <circle
                              cx={cx} cy={cy}
                              r={isNeuroStage ? 12 : 9}
                              fill={getColor(m.model)}
                              fillOpacity={isNeuroStage ? 1 : 0.8}
                              stroke="#FFFFFF"
                              strokeWidth={2}
                            />
                            <text x={cx + 14} y={cy + 4} fill="#1E293B" fontSize={11} fontWeight={600}>
                              {m.model.length > 15 ? m.model.slice(0, 15) + "…" : m.model}
                            </text>
                          </g>
                        );
                      }}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)" }}>
                No NeuroStage experiment data yet. Run training first.
              </div>
            )}
          </div>

          {/* Table */}
          <div className="glass-card" style={{ padding: 28, overflow: "hidden" }}>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 800, marginBottom: 20, color: "var(--text-primary)" }}>
              Full Comparison Table
            </h2>
            <div style={{ overflowX: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Macro-F1</th>
                    <th>Kappa</th>
                    <th>Params</th>
                    <th>Size (MB)</th>
                    <th>Latency (ms)</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {allModels.map((m) => (
                    <tr key={m.model}>
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <div style={{ width: 9, height: 9, borderRadius: "50%", background: getColor(m.model), flexShrink: 0 }} />
                          <span style={{ fontWeight: 700, color: "var(--text-primary)" }}>{m.model}</span>
                          {m.note && <span title={m.note} style={{ cursor: "help", fontSize: "0.75rem" }}>ℹ️</span>}
                        </div>
                      </td>
                      <td>{m.accuracy != null ? <strong>{(m.accuracy * 100).toFixed(1)}%</strong> : "—"}</td>
                      <td>{m.macro_f1 != null ? m.macro_f1.toFixed(3) : "—"}</td>
                      <td>{m.kappa != null ? m.kappa.toFixed(3) : "—"}</td>
                      <td>{m.param_count != null ? `${(m.param_count / 1000).toFixed(0)}K` : "—"}</td>
                      <td>{m.model_size_mb != null ? `${m.model_size_mb.toFixed(1)}` : "—"}</td>
                      <td>
                        {m.latency_ms != null ? (
                          `${m.latency_ms.toFixed(1)}`
                        ) : (
                          <span className="target-badge">Not measured</span>
                        )}
                      </td>
                      <td style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{m.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
