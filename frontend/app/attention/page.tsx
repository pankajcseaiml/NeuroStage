"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { API_URL } from "../lib/api";

const STAGE_COLORS: Record<number, string> = {
  0: "#DC2626", 1: "#EA580C", 2: "#D97706", 3: "#2563EB", 4: "#7C3AED",
};
const STAGE_NAMES: Record<number, string> = {
  0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM",
};

type AttentionResponse = {
  epoch_index: number;
  predicted_class: number;
  stage_name: string;
  confidence: number;
  attention_weights?: {
    weights: number[][];
    context_epochs: number;
  };
};

function AttentionContent() {
  const params = useSearchParams();
  const recordingId = params.get("id");
  const [epochIndex, setEpochIndex] = useState(10);
  const [attnData, setAttnData] = useState<AttentionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localToken, setLocalToken] = useState("");
  const [localId, setLocalId] = useState(recordingId || "");

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : null;
    if (saved) {
      setTimeout(() => setLocalToken(saved), 0);
    }
    if (recordingId) {
      setTimeout(() => setLocalId(recordingId), 0);
    }
  }, [recordingId]);

  const fetchAttention = useCallback(async () => {
    if (!localId || !localToken) {
      setError("Please enter both Recording ID and JWT Token.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/attention/${localId}/${epochIndex}`, {
        headers: { Authorization: `Bearer ${localToken}` },
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || "Failed to fetch attention data.");
      }
      setAttnData(await res.json());
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to fetch attention data.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [localId, localToken, epochIndex]);

  useEffect(() => {
    if (recordingId && localToken) {
      const timer = setTimeout(() => {
        fetchAttention();
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [recordingId, localToken, fetchAttention]);

  const [generatingToken, setGeneratingToken] = useState(false);

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
      if (!res.ok) throw new Error("Failed to get demo token.");
      const d = await res.json();
      setLocalToken(d.access_token);
      localStorage.setItem("neurostage_jwt_token", d.access_token);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to generate token.";
      setError("Token error: " + msg);
    } finally {
      setGeneratingToken(false);
    }
  };

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

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 36 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Attention Visualization
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", maxWidth: 680, fontWeight: 450 }}>
          The Transformer&apos;s last-layer attention weights show which other epochs in the context window
          influenced the prediction for the selected epoch.
        </p>
      </div>

      {/* Controls */}
      <div className="glass-card" style={{ padding: 28, marginBottom: 32 }}>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "flex-end" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, display: "block", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              RECORDING ID
            </label>
            <input
              className="input-field"
              placeholder="e.g. abc123..."
              value={localId}
              onChange={(e) => setLocalId(e.target.value)}
              style={{ width: "100%" }}
            />
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                JWT TOKEN
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
              type="password"
              placeholder="Bearer token"
              value={localToken}
              onChange={(e) => {
                setLocalToken(e.target.value);
                localStorage.setItem("neurostage_jwt_token", e.target.value);
              }}
              style={{ width: "100%" }}
            />
          </div>
          <div style={{ minWidth: 140 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, display: "block", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              EPOCH INDEX
            </label>
            <input
              className="input-field"
              type="number"
              min={0}
              value={epochIndex}
              onChange={(e) => setEpochIndex(parseInt(e.target.value) || 0)}
              style={{ width: "100%" }}
            />
          </div>
          <button className="btn-primary" onClick={fetchAttention} style={{ padding: "12px 24px" }}>
            Load
          </button>
        </div>
      </div>

      {loading && <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)", fontWeight: 500 }}>⏳ Loading attention weights...</div>}
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
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          {/* Prediction info */}
          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 20, color: "var(--text-primary)" }}>Prediction at Epoch {epochIndex}</h2>
            <div style={{
              padding: "24px 20px",
              borderRadius: 14,
              background: `${STAGE_COLORS[attnData.predicted_class]}0C`,
              border: `1px solid ${STAGE_COLORS[attnData.predicted_class]}30`,
              marginBottom: 20,
              textAlign: "center",
            }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>
                Predicted Stage
              </div>
              <div style={{ fontSize: "2.6rem", fontWeight: 900, color: STAGE_COLORS[attnData.predicted_class], letterSpacing: "-0.03em" }}>
                {STAGE_NAMES[attnData.predicted_class]}
              </div>
              <div style={{ fontSize: "1rem", color: "var(--text-secondary)", marginTop: 8, fontWeight: 500 }}>
                Confidence: <strong style={{ color: "var(--text-primary)", fontWeight: 800 }}>
                  {(attnData.confidence * 100).toFixed(1)}%
                </strong>
              </div>
            </div>
            <div>
              <p style={{ fontSize: "0.84rem", color: "var(--text-secondary)", lineHeight: 1.65, fontWeight: 450 }}>
                The heatmap (right) shows how much the Transformer attended to each of the{" "}
                <strong style={{ color: "var(--text-primary)" }}>preceding epochs</strong> when
                classifying this epoch. Darker/richer blue = higher attention weight.
              </p>
            </div>
          </div>

          {/* Attention heatmap */}
          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 20, color: "var(--text-primary)" }}>
              Attention Weights (Last Transformer Layer)
            </h2>
            {weights ? (
              <div>
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 20 }}>
                  {weights.map((w, i) => {
                    const maxW = Math.max(...weights);
                    const intensity = maxW > 0 ? w / maxW : 0;
                    return (
                      <div
                        key={i}
                        title={`Epoch ${i}: ${(w * 100).toFixed(2)}%`}
                        style={{
                          width: 30,
                          height: 30,
                          borderRadius: 6,
                          background: `rgba(37, 99, 235, ${0.08 + intensity * 0.85})`,
                          border: `1px solid rgba(37, 99, 235, ${0.15 + intensity * 0.5})`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.62rem",
                          fontWeight: 700,
                          color: intensity > 0.5 ? "#FFFFFF" : "#1E293B",
                          cursor: "help",
                          transition: "all 0.15s ease",
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.2)")}
                        onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
                      >
                        {i}
                      </div>
                    );
                  })}
                </div>
                {/* Top attended epochs */}
                <div style={{ marginTop: 20 }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 12 }}>
                    Most Attended Epochs
                  </div>
                  {[...weights.map((w, i) => ({ w, i }))].sort((a, b) => b.w - a.w).slice(0, 5).map(({ w, i }) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                      <span className="code" style={{ fontSize: "0.8rem", width: 34, textAlign: "center" }}>E{i}</span>
                      <div style={{ flex: 1, height: 7, background: "#E2E8F0", borderRadius: 4 }}>
                        <div style={{
                          height: "100%",
                          width: `${(w / Math.max(...weights)) * 100}%`,
                          background: "linear-gradient(90deg, #E14D45, #EA580C)",
                          borderRadius: 4,
                        }} />
                      </div>
                      <span style={{ fontSize: "0.84rem", color: "var(--text-primary)", fontWeight: 700, width: 48, textAlign: "right" }}>
                        {(w * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: 40, color: "var(--text-muted)", fontSize: "0.875rem" }}>
                No attention weights available for this epoch.
                <br />
                Ensure the model was trained with the Transformer variant.
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
