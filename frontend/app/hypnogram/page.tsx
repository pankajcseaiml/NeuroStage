"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { API_URL } from "../lib/api";

const STAGE_COLORS: Record<string, string> = {
  Wake: "#DC2626",
  N1: "#EA580C",
  N2: "#D97706",
  N3: "#2563EB",
  REM: "#7C3AED",
};

const STAGE_INT: Record<string, number> = { Wake: 4, N1: 3, N2: 2, N3: 1, REM: 0 };
const STAGE_FROM_INT: Record<number, string> = { 4: "Wake", 3: "N1", 2: "N2", 1: "N3", 0: "REM" };

type EpochPrediction = {
  epoch_index: number;
  time_start_sec: number;
  predicted_class: number;
  stage_name: string;
  confidence: number;
};

type HypnogramData = {
  recording_id: string;
  n_epochs: number;
  status: string;
  hypnogram: EpochPrediction[];
};

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h${m.toString().padStart(2, "0")}m`;
}

type TooltipProps = {
  active?: boolean;
  payload?: Array<{
    payload: {
      stage_level: number;
      epoch_index: number;
      time_start_sec: number;
      confidence: number;
    };
  }>;
};

const CustomTooltip = ({ active, payload }: TooltipProps) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const stage = STAGE_FROM_INT[d.stage_level];
  return (
    <div style={{
      background: "#FFFFFF",
      border: "1px solid #CBD5E1",
      borderRadius: 12,
      padding: "12px 16px",
      fontSize: "0.82rem",
      boxShadow: "0 4px 16px rgba(15, 23, 42, 0.1)",
    }}>
      <div style={{ color: STAGE_COLORS[stage], fontWeight: 800, fontSize: "0.95rem", marginBottom: 6 }}>
        {stage}
      </div>
      <div style={{ color: "var(--text-secondary)", marginBottom: 2 }}>Epoch {d.epoch_index}</div>
      <div style={{ color: "var(--text-secondary)", marginBottom: 2 }}>Time: {formatTime(d.time_start_sec)}</div>
      <div style={{ color: "var(--text-secondary)" }}>
        Confidence: <strong style={{ color: "var(--text-primary)" }}>{(d.confidence * 100).toFixed(1)}%</strong>
      </div>
    </div>
  );
};

function HypnogramContent() {
  const params = useSearchParams();
  const urlId = params.get("id") || "";
  const [recordingId, setRecordingId] = useState(urlId);
  const [data, setData] = useState<HypnogramData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [generatingToken, setGeneratingToken] = useState(false);

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : null;
    if (saved) {
      setTimeout(() => setToken(saved), 0);
    }
    if (urlId) {
      setTimeout(() => setRecordingId(urlId), 0);
    }
  }, [urlId]);

  const generateDemoToken = async () => {
    setGeneratingToken(true);
    setError(null);
    try {
      await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: "demo@neurostage.com", password: "demopassword123" }),
      });
      const form = new URLSearchParams();
      form.append("username", "demo@neurostage.com");
      form.append("password", "demopassword123");
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString(),
      });
      if (!res.ok) throw new Error("Failed to get token.");
      const d = await res.json();
      setToken(d.access_token);
      localStorage.setItem("neurostage_jwt_token", d.access_token);
      if (recordingId) fetchHypnogram(recordingId, d.access_token);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to generate token.";
      setError("Auto-token error: " + msg);
    } finally {
      setGeneratingToken(false);
    }
  };

  const fetchHypnogram = async (rid: string, tok: string) => {
    if (!rid) {
      setError("Please enter a Recording ID.");
      return;
    }
    if (!tok) {
      setError("Please enter or generate an Access Token.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/hypnogram/${rid}`, {
        headers: { Authorization: `Bearer ${tok}` },
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || "Failed to fetch hypnogram.");
      }
      const d = await res.json();
      if (d.status === "processing") {
        setError("Recording is still processing on backend. Please wait a few moments and click Load again.");
      } else if (!d.hypnogram || d.hypnogram.length === 0) {
        setError("No predictions found for this recording.");
      } else {
        setData(d);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to fetch hypnogram.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (urlId && token) {
      fetchHypnogram(urlId, token);
    }
  }, [urlId, token]);

  // Stage count summary
  const stageCounts: Record<string, number> = {};
  data?.hypnogram?.forEach((e) => {
    stageCounts[e.stage_name] = (stageCounts[e.stage_name] || 0) + 1;
  });

  // Chart data
  const chartData = data?.hypnogram?.map((e) => ({
    ...e,
    stage_level: STAGE_INT[e.stage_name] ?? 2,
  }));

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 36 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Hypnogram Viewer
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", fontWeight: 450 }}>
          Predicted sleep stage sequence across the full recording night.
        </p>
      </div>

      {/* Controls Bar */}
      <div className="glass-card" style={{ padding: 24, marginBottom: 32 }}>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-end" }}>
          <div style={{ flex: 1, minWidth: 240 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, display: "block", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              RECORDING ID
            </label>
            <input
              className="input-field"
              placeholder="e.g. 4166f4f8-7eed-43d8-..."
              value={recordingId}
              onChange={(e) => setRecordingId(e.target.value)}
              style={{ width: "100%" }}
            />
          </div>
          <div style={{ flex: 1, minWidth: 240 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                ACCESS TOKEN
              </label>
              <button
                type="button"
                onClick={generateDemoToken}
                disabled={generatingToken}
                style={{
                  background: "rgba(37,99,235,0.08)",
                  border: "1px solid rgba(37,99,235,0.25)",
                  color: "#1D4ED8",
                  borderRadius: 6,
                  padding: "2px 8px",
                  fontSize: "0.72rem",
                  fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                {generatingToken ? "Generating..." : "🔑 1-Click Demo Token"}
              </button>
            </div>
            <input
              className="input-field"
              placeholder="JWT Token"
              type="password"
              value={token}
              onChange={(e) => {
                setToken(e.target.value);
                localStorage.setItem("neurostage_jwt_token", e.target.value);
              }}
              style={{ width: "100%" }}
            />
          </div>
          <button
            className="btn-primary"
            onClick={() => fetchHypnogram(recordingId, token)}
            style={{ padding: "12px 24px" }}
          >
            📊 Load Hypnogram
          </button>
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)", fontWeight: 500 }}>
          ⏳ Loading hypnogram...
        </div>
      )}

      {error && (
        <div style={{
          padding: "16px 20px",
          borderRadius: 12,
          background: "rgba(225,29,72,0.08)",
          border: "1px solid rgba(225,29,72,0.25)",
          color: "#BE123C",
          fontWeight: 600,
          marginBottom: 24,
        }}>
          ❌ {error}
        </div>
      )}

      {data?.hypnogram && (
        <>
          {/* Stage summary */}
          <div style={{ display: "flex", gap: 12, marginBottom: 32, flexWrap: "wrap" }}>
            {Object.entries(STAGE_COLORS).map(([stage, color]) => {
              const count = stageCounts[stage] || 0;
              const pct = data.n_epochs > 0 ? (count / data.n_epochs * 100).toFixed(1) : "0";
              return (
                <div key={stage} className="metric-card" style={{ minWidth: 140, flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, flexShrink: 0 }} />
                    <span style={{ fontWeight: 700, fontSize: "0.9rem", color: "var(--text-primary)" }}>{stage}</span>
                  </div>
                  <div className="metric-value" style={{ fontSize: "1.8rem", color }}>{pct}%</div>
                  <div className="metric-label">{count} epochs · {(count * 0.5).toFixed(0)} min</div>
                </div>
              );
            })}
          </div>

          {/* Hypnogram chart */}
          <div className="glass-card" style={{ padding: 32, marginBottom: 28 }}>
            <h2 style={{ fontSize: "1.15rem", fontWeight: 800, marginBottom: 24, color: "var(--text-primary)" }}>
              Predicted Hypnogram — {data.n_epochs} epochs ({(data.n_epochs * 0.5 / 60).toFixed(1)} hours)
            </h2>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                <defs>
                  <linearGradient id="stageGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563EB" stopOpacity={0.16} />
                    <stop offset="95%" stopColor="#2563EB" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis
                  dataKey="time_start_sec"
                  tickFormatter={formatTime}
                  tick={{ fill: "#64748B", fontSize: 11 }}
                  axisLine={{ stroke: "#E2E8F0" }}
                />
                <YAxis
                  domain={[0, 4]}
                  ticks={[0, 1, 2, 3, 4]}
                  tickFormatter={(v) => STAGE_FROM_INT[v]}
                  tick={{ fill: "#64748B", fontSize: 11 }}
                  axisLine={{ stroke: "#E2E8F0" }}
                  width={48}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="stepAfter"
                  dataKey="stage_level"
                  stroke="#2563EB"
                  strokeWidth={2.5}
                  fill="url(#stageGrad)"
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 16, textAlign: "center", fontWeight: 500 }}>
              Y-axis: deeper sleep at the bottom (REM/N3), wakefulness at the top.
              Each step = one 30-second epoch.
            </p>
          </div>

          {/* Epoch table */}
          <div className="glass-card" style={{ padding: 24, overflow: "hidden" }}>
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 16, color: "var(--text-primary)" }}>
              Per-Epoch Predictions
            </h2>
            <div style={{ maxHeight: 400, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Time</th>
                    <th>Stage</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {data.hypnogram.slice(0, 200).map((e) => (
                    <tr key={e.epoch_index}>
                      <td className="code">{e.epoch_index}</td>
                      <td>{formatTime(e.time_start_sec)}</td>
                      <td>
                        <span style={{ color: STAGE_COLORS[e.stage_name], fontWeight: 700 }}>
                          {e.stage_name}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <div style={{ flex: 1, height: 6, background: "#E2E8F0", borderRadius: 3, maxWidth: 80 }}>
                            <div style={{
                              height: "100%",
                              width: `${e.confidence * 100}%`,
                              background: STAGE_COLORS[e.stage_name],
                              borderRadius: 3,
                            }} />
                          </div>
                          <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                            {(e.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
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

export default function HypnogramPage() {
  return (
    <Suspense fallback={<div style={{ padding: 60, textAlign: "center", color: "var(--text-muted)" }}>Loading...</div>}>
      <HypnogramContent />
    </Suspense>
  );
}
