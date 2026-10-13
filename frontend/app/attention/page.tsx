"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "../context/AuthContext";
import { useRecording } from "../context/RecordingContext";
import { API_URL } from "../lib/api";

const STAGE_COLORS: Record<number, string> = {
  0: "#DC2626", 1: "#EA580C", 2: "#D97706", 3: "#2563EB", 4: "#7C3AED",
};
const STAGE_NAMES: Record<number, string> = {
  0: "Wake", 1: "N1 (Drowsy)", 2: "N2 (Light)", 3: "N3 (Deep/SWS)", 4: "REM (Dream)",
};

type AttentionResponse = {
  epoch_index: number;
  predicted_class: number;
  confidence: number;
  attention_weights?: {
    weights: number[][];
    epoch_weights?: number[];
    context_epochs: number;
  };
};

type RecordingOption = {
  id: string;
  filename: string;
  status: string;
  uploaded_at: string;
  n_epochs: number;
};

function AttentionContent() {
  const params = useSearchParams();
  const urlId = params.get("id") || "";
  const urlEpoch = params.get("epoch") ? parseInt(params.get("epoch")!) : 10;
  const { token: authJwt } = useAuth();
  const { activeRecordingId, setActiveRecording } = useRecording();

  const [recordingId, setRecordingId] = useState(urlId || activeRecordingId || "");
  const [recordingsList, setRecordingsList] = useState<RecordingOption[]>([]);
  const [epochIndex, setEpochIndex] = useState(urlEpoch);
  const [attnData, setAttnData] = useState<AttentionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    if (urlId) {
      setRecordingId(urlId);
      setActiveRecording(urlId);
    }
    if (params.get("epoch")) setEpochIndex(parseInt(params.get("epoch")!));
  }, [urlId, params, setActiveRecording]);

  const fetchAttention = useCallback(async (rid: string, ep: number) => {
    if (!rid) return;
    setLoading(true);
    setError(null);
    try {
      const activeTok = authJwt || (typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : "") || "";
      const headers: Record<string, string> = {};
      if (activeTok) headers["Authorization"] = `Bearer ${activeTok}`;

      const res = await fetch(`${API_URL}/attention/${rid}/${ep}`, { headers });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Failed to fetch attention data for epoch ${ep}.`);
      }
      const data = await res.json();
      setAttnData(data);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to fetch attention data.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [authJwt]);

  // Auto-fetch when recordingId or epochIndex changes
  useEffect(() => {
    if (recordingId) {
      fetchAttention(recordingId, epochIndex);
    }
  }, [recordingId, epochIndex, fetchAttention]);

  const rawAttn = attnData?.attention_weights as { weights?: number[][] | number[]; epoch_weights?: number[] } | undefined;
  let weights: number[] | null = null;
  if (rawAttn) {
    if (Array.isArray(rawAttn.epoch_weights)) {
      weights = rawAttn.epoch_weights;
    } else if (Array.isArray(rawAttn.weights)) {
      if (Array.isArray(rawAttn.weights[0])) {
        weights = rawAttn.weights[0] as number[];
      } else if (typeof rawAttn.weights[0] === "number") {
        weights = rawAttn.weights as number[];
      }
    }
  }

  const selectedRec = recordingsList.find((r) => r.id === recordingId);
  const maxEpochs = selectedRec && selectedRec.n_epochs > 0 ? selectedRec.n_epochs - 1 : 2800;

  return (
    <div style={{ maxWidth: 1150, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Self-Attention Interpretability Map
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", maxWidth: 780, fontWeight: 450 }}>
          The Transformer&apos;s multi-head self-attention weights reveal how NeuroStage looks across
          surrounding 30-second context epochs to disambiguate sleep stages.
        </p>
      </div>

      {/* Recording & Epoch Navigation Bar */}
      <div className="glass-card" style={{ padding: 24, marginBottom: 32 }}>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 20 }}>
          <div style={{ flex: 2, minWidth: 260 }}>
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

          <div style={{ width: 140 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, display: "block", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              EPOCH #
            </label>
            <input
              className="input-field"
              type="number"
              min={0}
              max={maxEpochs}
              value={epochIndex}
              onChange={(e) => setEpochIndex(Math.max(0, parseInt(e.target.value) || 0))}
              style={{ width: "100%", padding: "10px 14px", fontWeight: 700 }}
            />
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              className="btn-secondary"
              onClick={() => setEpochIndex((e) => Math.max(0, e - 1))}
              disabled={epochIndex <= 0 || loading}
              style={{ padding: "10px 14px", fontSize: "0.85rem", fontWeight: 700 }}
            >
              ◀ Prev
            </button>
            <button
              className="btn-secondary"
              onClick={() => setEpochIndex((e) => Math.min(maxEpochs, e + 1))}
              disabled={epochIndex >= maxEpochs || loading}
              style={{ padding: "10px 14px", fontSize: "0.85rem", fontWeight: 700 }}
            >
              Next ▶
            </button>
            <Link
              href={`/hypnogram?id=${recordingId}`}
              className="btn-primary"
              style={{
                padding: "10px 16px",
                textDecoration: "none",
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontWeight: 700,
                fontSize: "0.85rem",
              }}
            >
              📈 Full Hypnogram →
            </Link>
          </div>
        </div>

        {/* Epoch Scrubber Slider */}
        <div style={{ marginTop: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "var(--text-secondary)", marginBottom: 6, fontWeight: 600 }}>
            <span>Timeline Scrubber: <strong>Epoch {epochIndex}</strong> (approx {Math.floor((epochIndex * 30) / 3600)}h {Math.floor(((epochIndex * 30) % 3600) / 60)}m into recording)</span>
            <span>Total: {maxEpochs + 1} epochs</span>
          </div>
          <input
            type="range"
            min={0}
            max={maxEpochs}
            value={epochIndex}
            onChange={(e) => setEpochIndex(parseInt(e.target.value))}
            style={{ width: "100%", accentColor: "var(--brand-coral)", cursor: "pointer" }}
          />

          {/* Quick Jump Bookmarks */}
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700 }}>QUICK PRESETS:</span>
            {[
              { label: "Epoch 10 (Wake/N1)", ep: 10 },
              { label: "Epoch 120 (N2 Spindles)", ep: 120 },
              { label: "Epoch 240 (N3 Deep SWS)", ep: 240 },
              { label: "Epoch 360 (REM Dream)", ep: 360 },
              { label: "Epoch 500 (Mid-Night)", ep: 500 },
            ].map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setEpochIndex(p.ep)}
                style={{
                  background: epochIndex === p.ep ? "var(--brand-coral)" : "rgba(0,0,0,0.04)",
                  color: epochIndex === p.ep ? "#FFFFFF" : "var(--text-secondary)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  padding: "4px 10px",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)", fontWeight: 500 }}>
          ⏳ Computing attention weights for Epoch {epochIndex}...
        </div>
      )}

      {error && (
        <div style={{
          padding: "16px 20px", borderRadius: 12,
          background: "rgba(225,29,72,0.08)", border: "1px solid rgba(225,29,72,0.25)",
          color: "#BE123C", marginBottom: 24, fontSize: "0.875rem", fontWeight: 600,
        }}>
          ❌ {error}
        </div>
      )}

      {attnData && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.3fr", gap: 24 }}>
          {/* Classification outcome */}
          <div className="glass-card" style={{ padding: 28 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)" }}>
                Epoch {epochIndex} Classification
              </h2>
              <span className="code" style={{ fontSize: "0.75rem" }}>
                Time: {Math.floor((epochIndex * 30) / 3600)}h {Math.floor(((epochIndex * 30) % 3600) / 60)}m
              </span>
            </div>

            <div style={{
              padding: "24px 20px",
              borderRadius: 14,
              background: `${STAGE_COLORS[attnData.predicted_class]}0C`,
              border: `1px solid ${STAGE_COLORS[attnData.predicted_class]}30`,
              marginBottom: 20,
              textAlign: "center",
            }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>
                Predicted Sleep Stage
              </div>
              <div style={{ fontSize: "2.4rem", fontWeight: 900, color: STAGE_COLORS[attnData.predicted_class], letterSpacing: "-0.03em" }}>
                {STAGE_NAMES[attnData.predicted_class]}
              </div>
              <div style={{ fontSize: "1rem", color: "var(--text-secondary)", marginTop: 6, fontWeight: 600 }}>
                Confidence: <strong style={{ color: "var(--text-primary)", fontWeight: 800 }}>
                  {(attnData.confidence * 100).toFixed(1)}%
                </strong>
              </div>
            </div>

            <div style={{ padding: "16px 18px", borderRadius: 10, background: "rgba(37,99,235,0.06)", border: "1px solid rgba(37,99,235,0.2)", marginBottom: 16 }}>
              <div style={{ fontSize: "0.82rem", fontWeight: 700, color: "#1D4ED8", marginBottom: 4 }}>
                🧠 Neurophysiological Rationale
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.55, margin: 0 }}>
                {attnData.predicted_class === 0 && "High alpha/beta rhythm with rapid ocular blinks. Model places emphasis on immediate awake state."}
                {attnData.predicted_class === 1 && "Vertex sharp waves and theta rhythm (4–7 Hz). Crucial transition state between wakefulness and deep sleep."}
                {attnData.predicted_class === 2 && "Sleep spindles (11–16 Hz) and K-complexes. Self-attention looks back into preceding epochs to confirm spindle burst regularity."}
                {attnData.predicted_class === 3 && "High-amplitude slow waves (>75 µV, 0.5–2 Hz). Transformer confirms delta dominance across continuous 30-second windows."}
                {attnData.predicted_class === 4 && "Desynchronized low-voltage mixed EEG with rapid eye movements and muscle atonia. Attention verifies cyclic REM recurrence."}
              </p>
            </div>

            <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.5, margin: 0 }}>
              Under AASM scoring manual standards, sleep stage scoring requires evaluating temporal sequence context,
              preventing isolated artifacts from misclassifying whole nights.
            </p>
          </div>

          {/* Attention Heatmap & Context Weights */}
          <div className="glass-card" style={{ padding: 28 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h2 style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)" }}>
                Context Window Attention (Layer 2)
              </h2>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>
                20-Epoch Context Window
              </span>
            </div>

            {weights && weights.length > 0 ? (
              <div>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: 16 }}>
                  Heatmap shows attention energy across the 20-epoch sequence. Darker color indicates
                  greater influence on the current epoch prediction.
                </p>

                {/* Grid Heatmap */}
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 24 }}>
                  {weights.map((w, i) => {
                    const maxW = Math.max(...weights!);
                    const intensity = maxW > 0 ? w / maxW : 0;
                    const posInWindow = epochIndex % 20;
                    const isTarget = i === posInWindow;
                    return (
                      <div
                        key={i}
                        title={`Window Position ${i} (Offset: ${i - posInWindow}): ${(w * 100).toFixed(2)}%`}
                        style={{
                          width: 38,
                          height: 38,
                          borderRadius: 8,
                          background: isTarget
                            ? "var(--brand-coral)"
                            : `rgba(37, 99, 235, ${0.1 + intensity * 0.85})`,
                          border: isTarget
                            ? "2px solid #BE123C"
                            : `1px solid rgba(37, 99, 235, ${0.2 + intensity * 0.5})`,
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.68rem",
                          fontWeight: 800,
                          color: isTarget || intensity > 0.5 ? "#FFFFFF" : "#1E293B",
                          cursor: "help",
                          transition: "all 0.15s ease",
                          position: "relative",
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.15)")}
                        onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
                      >
                        <div>{i}</div>
                        {isTarget && (
                          <div style={{ fontSize: "0.5rem", textTransform: "uppercase" }}>HERE</div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Ranked Attention Distribution */}
                <div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 12 }}>
                    Most Influential Context Epochs
                  </div>
                  {[...weights.map((w, i) => ({ w, i }))].sort((a, b) => b.w - a.w).slice(0, 5).map(({ w, i }) => {
                    const posInWindow = epochIndex % 20;
                    const offset = i - posInWindow;
                    return (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                        <span className="code" style={{ fontSize: "0.75rem", width: 44, textAlign: "center" }}>
                          Pos {i}
                        </span>
                        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", width: 70 }}>
                          {offset === 0 ? "Target epoch" : offset > 0 ? `+${offset} ahead` : `${offset} behind`}
                        </span>
                        <div style={{ flex: 1, height: 8, background: "#E2E8F0", borderRadius: 4 }}>
                          <div style={{
                            height: "100%",
                            width: `${(w / Math.max(...weights!)) * 100}%`,
                            background: i === posInWindow
                              ? "var(--brand-coral)"
                              : "linear-gradient(90deg, #2563EB, #7C3AED)",
                            borderRadius: 4,
                          }} />
                        </div>
                        <span style={{ fontSize: "0.82rem", color: "var(--text-primary)", fontWeight: 700, width: 48, textAlign: "right" }}>
                          {(w * 100).toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: 40, color: "var(--text-muted)", fontSize: "0.875rem" }}>
                No attention weights returned for this epoch.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AttentionPage() {
  return (
    <Suspense fallback={<div style={{ padding: 60, textAlign: "center", color: "var(--text-muted)" }}>Loading...</div>}>
      <AttentionContent />
    </Suspense>
  );
}
