"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useAuth } from "../context/AuthContext";
import { useRecording } from "../context/RecordingContext";
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

type RecordingOption = {
  id: string;
  filename: string;
  status: string;
  uploaded_at: string;
  n_epochs: number;
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
  const { token: authJwt } = useAuth();
  const { activeRecordingId, setActiveRecording } = useRecording();

  const [recordingId, setRecordingId] = useState(urlId || activeRecordingId || "");
  const [recordingsList, setRecordingsList] = useState<RecordingOption[]>([]);
  const [data, setData] = useState<HypnogramData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [token, setToken] = useState("");

  // Load recordings list on mount
  useEffect(() => {
    const fetchRecordings = async () => {
      try {
        const activeTok = authJwt || (typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : "") || "";
        const headers: Record<string, string> = {};
        if (activeTok) headers["Authorization"] = `Bearer ${activeTok}`;

        const res = await fetch(`${API_URL}/recordings`, { headers });
        if (res.ok) {
          const d = await res.json();
          const recs: RecordingOption[] = d.recordings || [];
          setRecordingsList(recs);

          const targetId = urlId || activeRecordingId;
          if (targetId) {
            setRecordingId(targetId);
            const found = recs.find((r) => r.id === targetId);
            if (found) setActiveRecording(found.id, found.filename);
          } else if (recs.length > 0) {
            const firstDone = recs.find((r) => r.status === "done") || recs[0];
            setRecordingId(firstDone.id);
            setActiveRecording(firstDone.id, firstDone.filename);
          }
        }
      } catch {}
    };
    fetchRecordings();
  }, [urlId, activeRecordingId, authJwt, setActiveRecording]);

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : null;
    if (saved) setToken(saved);
    if (authJwt) setToken(authJwt);
    if (urlId) {
      setRecordingId(urlId);
      setActiveRecording(urlId);
    }
  }, [urlId, authJwt, setActiveRecording]);

  const fetchHypnogram = async (rid: string) => {
    if (!rid) return;
    setLoading(true);
    setError(null);
    try {
      const activeTok = token || authJwt || (typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : "") || "";
      const headers: Record<string, string> = {};
      if (activeTok) headers["Authorization"] = `Bearer ${activeTok}`;

      const res = await fetch(`${API_URL}/hypnogram/${rid}`, { headers });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || "Failed to fetch hypnogram.");
      }
      const d = await res.json();
      if (d.status === "processing") {
        setError("Recording is still processing on backend. Please wait a few moments and click Refresh.");
      } else if (!d.hypnogram || d.hypnogram.length === 0) {
        setError("No predictions found for this recording yet.");
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

  // Auto-fetch when recordingId is set
  useEffect(() => {
    if (recordingId) {
      fetchHypnogram(recordingId);
    }
  }, [recordingId]);

  // Stage count summary
  const stageCounts: Record<string, number> = {};
  data?.hypnogram?.forEach((e) => {
    stageCounts[e.stage_name] = (stageCounts[e.stage_name] || 0) + 1;
  });

  // Clinical sleep metrics
  const totalEpochs = data?.n_epochs || 0;
  const wakeEpochs = stageCounts["Wake"] || 0;
  const n1Epochs = stageCounts["N1"] || 0;
  const n2Epochs = stageCounts["N2"] || 0;
  const n3Epochs = stageCounts["N3"] || 0;
  const remEpochs = stageCounts["REM"] || 0;
  const sleepEpochs = totalEpochs - wakeEpochs;

  const tstHours = (sleepEpochs * 0.5) / 60;
  const sleepEfficiency = totalEpochs > 0 ? (sleepEpochs / totalEpochs) * 100 : 0;
  const deepSleepPct = sleepEpochs > 0 ? (n3Epochs / sleepEpochs) * 100 : 0;
  const remSleepPct = sleepEpochs > 0 ? (remEpochs / sleepEpochs) * 100 : 0;
  const lightSleepPct = sleepEpochs > 0 ? ((n1Epochs + n2Epochs) / sleepEpochs) * 100 : 0;

  // Chart data
  const chartData = data?.hypnogram?.map((e) => ({
    ...e,
    stage_level: STAGE_INT[e.stage_name] ?? 2,
  }));

  const selectedRec = recordingsList.find((r) => r.id === recordingId);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 36 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Hypnogram Viewer
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", fontWeight: 450 }}>
          Predicted sleep stage sequence across the entire overnight EEG recording (AASM 30-second standard epochs).
        </p>
      </div>

      {/* Recording Selector Bar */}
      <div className="glass-card" style={{ padding: 24, marginBottom: 32 }}>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-end" }}>
          <div style={{ flex: 2, minWidth: 280 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, display: "block", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              SELECT RECORDING
            </label>
            <select
              className="input-field"
              value={recordingId}
              onChange={(e) => {
                const nextId = e.target.value;
                setRecordingId(nextId);
                const rec = recordingsList.find((r) => r.id === nextId);
                setActiveRecording(nextId, rec?.filename);
              }}
              style={{ width: "100%", padding: "10px 14px", fontWeight: 600 }}
            >
              {recordingsList.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.filename} — {r.n_epochs > 0 ? `${r.n_epochs} epochs` : r.status} ({r.id.slice(0, 8)}…)
                </option>
              ))}
              {recordingId && !recordingsList.some((r) => r.id === recordingId) && (
                <option value={recordingId}>Current ID: {recordingId.slice(0, 12)}…</option>
              )}
            </select>
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button
              className="btn-primary"
              onClick={() => fetchHypnogram(recordingId)}
              disabled={loading}
              style={{ padding: "11px 20px" }}
            >
              {loading ? "⏳ Loading..." : "🔄 Refresh Data"}
            </button>
            {recordingId && (
              <Link
                href={`/attention?id=${recordingId}&epoch=10`}
                className="btn-secondary"
                style={{
                  padding: "11px 20px",
                  textDecoration: "none",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  fontWeight: 700,
                  color: "#6D28D9",
                }}
              >
                👁️ View Attention Map →
              </Link>
            )}
          </div>
        </div>

        {selectedRec && (
          <div style={{ marginTop: 14, fontSize: "0.8rem", color: "var(--text-secondary)", display: "flex", gap: 16, flexWrap: "wrap" }}>
            <span>File: <strong>{selectedRec.filename}</strong></span>
            <span>Status: <strong style={{ color: selectedRec.status === "done" ? "#059669" : "#D97706" }}>{selectedRec.status.toUpperCase()}</strong></span>
            <span>Total Epochs: <strong>{selectedRec.n_epochs}</strong></span>
            <span>UUID: <code>{selectedRec.id}</code></span>
          </div>
        )}
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)", fontWeight: 500 }}>
          ⏳ Loading neural hypnogram predictions...
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
          {/* Clinical Polysomnography Scorecard */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14, marginBottom: 28 }}>
            <div className="metric-card">
              <div className="metric-label">Total Sleep Time (TST)</div>
              <div className="metric-value" style={{ color: "#2563EB", fontSize: "1.8rem" }}>
                {tstHours.toFixed(1)} hrs
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4 }}>
                {sleepEpochs} valid sleep epochs
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Sleep Efficiency</div>
              <div className="metric-value" style={{ color: sleepEfficiency >= 85 ? "#059669" : "#D97706", fontSize: "1.8rem" }}>
                {sleepEfficiency.toFixed(1)}%
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4 }}>
                {sleepEfficiency >= 85 ? "Normal (>85%)" : "Reduced efficiency"}
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Deep Sleep (N3 Slow Wave)</div>
              <div className="metric-value" style={{ color: "#2563EB", fontSize: "1.8rem" }}>
                {deepSleepPct.toFixed(1)}%
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4 }}>
                {n3Epochs} epochs · restorative sleep
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">REM Sleep</div>
              <div className="metric-value" style={{ color: "#7C3AED", fontSize: "1.8rem" }}>
                {remSleepPct.toFixed(1)}%
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4 }}>
                {remEpochs} epochs · dream stage
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-label">Light Sleep (N1 + N2)</div>
              <div className="metric-value" style={{ color: "#EA580C", fontSize: "1.8rem" }}>
                {lightSleepPct.toFixed(1)}%
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4 }}>
                {n1Epochs + n2Epochs} epochs (spindles/K-complex)
              </div>
            </div>
          </div>

          {/* Stage summary pills */}
          <div style={{ display: "flex", gap: 12, marginBottom: 28, flexWrap: "wrap" }}>
            {Object.entries(STAGE_COLORS).map(([stage, color]) => {
              const count = stageCounts[stage] || 0;
              const pct = data.n_epochs > 0 ? ((count / data.n_epochs) * 100).toFixed(1) : "0";
              return (
                <div key={stage} className="metric-card" style={{ minWidth: 140, flex: 1, padding: "16px 20px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, flexShrink: 0 }} />
                    <span style={{ fontWeight: 700, fontSize: "0.9rem", color: "var(--text-primary)" }}>{stage}</span>
                  </div>
                  <div className="metric-value" style={{ fontSize: "1.6rem", color }}>{pct}%</div>
                  <div className="metric-label">{count} epochs · {(count * 0.5).toFixed(0)} min</div>
                </div>
              );
            })}
          </div>

          {/* Hypnogram chart */}
          <div className="glass-card" style={{ padding: 32, marginBottom: 28 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <div>
                <h2 style={{ fontSize: "1.15rem", fontWeight: 800, color: "var(--text-primary)" }}>
                  Full-Night Architecture — {data.n_epochs} epochs ({(data.n_epochs * 0.5 / 60).toFixed(1)} hours)
                </h2>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Standard clinical staircase progression: Wake at top, N3 slow-wave and REM at lower tiers.
                </p>
              </div>
              <Link
                href={`/attention?id=${recordingId}&epoch=10`}
                className="btn-secondary"
                style={{ fontSize: "0.82rem", textDecoration: "none", padding: "8px 14px", fontWeight: 700, color: "#6D28D9" }}
              >
                👁️ View Attention Map
              </Link>
            </div>

            <ResponsiveContainer width="100%" height={290}>
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
              Y-axis: Wake (top) → N1 → N2 → N3 Deep Sleep → REM (bottom). Each step represents one 30-second epoch.
            </p>
          </div>

          {/* Epoch table with direct Attention links */}
          <div className="glass-card" style={{ padding: 24, overflow: "hidden" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div>
                <h2 style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)" }}>
                  Per-Epoch Predictions & Attention Inspection
                </h2>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  Click &ldquo;Inspect Attention&rdquo; on any epoch to see which temporal context epochs the Transformer relied on.
                </p>
              </div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>
                Showing first 150 epochs of {data.n_epochs}
              </span>
            </div>

            <div style={{ maxHeight: 420, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 10 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Time</th>
                    <th>Predicted Stage</th>
                    <th>Confidence</th>
                    <th style={{ textAlign: "right" }}>Transformer Attention</th>
                  </tr>
                </thead>
                <tbody>
                  {data.hypnogram.slice(0, 150).map((e) => (
                    <tr key={e.epoch_index}>
                      <td className="code">{e.epoch_index}</td>
                      <td>{formatTime(e.time_start_sec)}</td>
                      <td>
                        <span
                          style={{
                            color: STAGE_COLORS[e.stage_name],
                            fontWeight: 700,
                            background: `${STAGE_COLORS[e.stage_name]}15`,
                            padding: "3px 8px",
                            borderRadius: 6,
                          }}
                        >
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
                      <td style={{ textAlign: "right" }}>
                        <Link
                          href={`/attention?id=${recordingId}&epoch=${e.epoch_index}`}
                          style={{
                            padding: "4px 10px",
                            borderRadius: 6,
                            background: "rgba(124, 58, 237, 0.08)",
                            border: "1px solid rgba(124, 58, 237, 0.25)",
                            color: "#6D28D9",
                            fontSize: "0.75rem",
                            fontWeight: 700,
                            textDecoration: "none",
                          }}
                        >
                          👁️ Inspect Epoch {e.epoch_index} Attention
                        </Link>
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
