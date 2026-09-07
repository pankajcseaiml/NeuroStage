"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const STAGE_COLORS: Record<number, string> = {
  0: "#ef4444", 1: "#f97316", 2: "#eab308", 3: "#3b82f6", 4: "#8b5cf6",
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

  const weights: number[] | null = attnData?.attention_weights?.weights?.[epochIndex] ?? null;

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 40 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em" }}>
          Attention Visualization
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", maxWidth: 680 }}>
          The Transformer&apos;s last-layer attention weights show which other epochs in the context window
          influenced the prediction for the selected epoch.
        </p>
      </div>

      {/* Controls */}
      <div className="glass-card" style={{ padding: 28, marginBottom: 32 }}>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 500, display: "block", marginBottom: 6 }}>
              RECORDING ID
            </label>
            <input
              className="input-field"
              placeholder="e.g. abc123..."
              value={localId}
              onChange={(e) => setLocalId(e.target.value)}
            />
          </div>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 500 }}>
                JWT TOKEN
              </label>
              <button
                type="button"
                onClick={generateDemoToken}
                disabled={generatingToken}
                style={{
                  background: "rgba(59,130,246,0.15)",
                  border: "1px solid rgba(59,130,246,0.3)",
                  color: "var(--accent-blue)",
                  borderRadius: 6,
                  padding: "2px 6px",
                  fontSize: "0.7rem",
                  fontWeight: 600,
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
            />
          </div>
          <div style={{ minWidth: 140 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 500, display: "block", marginBottom: 6 }}>
              EPOCH INDEX
            </label>
            <input
              className="input-field"
              type="number"
              min={0}
              value={epochIndex}
              onChange={(e) => setEpochIndex(parseInt(e.target.value) || 0)}
            />
          </div>
          <button className="btn-primary" onClick={fetchAttention}>
            Load
          </button>
        </div>
      </div>

      {loading && <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)" }}>⏳ Loading...</div>}
      {error && (
        <div style={{
          padding: "14px 18px", borderRadius: 10,
          background: "rgba(244,63,94,0.1)", border: "1px solid rgba(244,63,94,0.3)",
          color: "#fda4af", marginBottom: 24, fontSize: "0.875rem",
        }}>
          ❌ {error}
        </div>
      )}

      {attnData && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          {/* Prediction info */}
          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: 20 }}>Prediction at Epoch {epochIndex}</h2>
            <div style={{
              padding: "20px",
              borderRadius: 12,
              background: `${STAGE_COLORS[attnData.predicted_class]}15`,
              border: `1px solid ${STAGE_COLORS[attnData.predicted_class]}40`,
              marginBottom: 20,
              textAlign: "center",
            }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>
                Predicted Stage
              </div>
              <div style={{ fontSize: "2.5rem", fontWeight: 800, color: STAGE_COLORS[attnData.predicted_class] }}>
                {STAGE_NAMES[attnData.predicted_class]}
              </div>
              <div style={{ fontSize: "1rem", color: "var(--text-secondary)", marginTop: 8 }}>
                Confidence: <strong style={{ color: "var(--text-primary)" }}>
                  {(attnData.confidence * 100).toFixed(1)}%
                </strong>
              </div>
            </div>
            <div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", lineHeight: 1.7 }}>
                The heatmap (right) shows how much the Transformer attended to each of the{" "}
                <strong style={{ color: "var(--text-primary)" }}>preceding epochs</strong> when
                classifying this epoch. Brighter = more attention.
              </p>
            </div>
          </div>

          {/* Attention heatmap */}
          <div className="glass-card" style={{ padding: 28 }}>
            <h2 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: 20 }}>
              Attention Weights (Last Transformer Layer)
            </h2>
            {weights ? (
              <div>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 16 }}>
                  {weights.map((w, i) => {
                    const maxW = Math.max(...weights);
                    const intensity = maxW > 0 ? w / maxW : 0;
                    return (
                      <div
                        key={i}
                        title={`Epoch ${i}: ${(w * 100).toFixed(2)}%`}
                        style={{
                          width: 28,
                          height: 28,
                          borderRadius: 4,
                          background: `rgba(59, 130, 246, ${0.1 + intensity * 0.9})`,
                          border: `1px solid rgba(59, 130, 246, ${0.2 + intensity * 0.4})`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.55rem",
                          color: intensity > 0.6 ? "white" : "var(--text-muted)",
                          cursor: "help",
                          transition: "transform 0.15s",
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.3)")}
                        onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
                      >
                        {i}
                      </div>
                    );
                  })}
                </div>
                {/* Top attended epochs */}
                <div style={{ marginTop: 16 }}>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 10 }}>
                    Most Attended Epochs
                  </div>
                  {[...weights.map((w, i) => ({ w, i }))].sort((a, b) => b.w - a.w).slice(0, 5).map(({ w, i }) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                      <span className="code" style={{ fontSize: "0.8rem", color: "var(--text-muted)", width: 28 }}>E{i}</span>
                      <div style={{ flex: 1, height: 6, background: "var(--border)", borderRadius: 3 }}>
                        <div style={{
                          height: "100%",
                          width: `${(w / Math.max(...weights)) * 100}%`,
                          background: "var(--gradient-accent)",
                          borderRadius: 3,
                        }} />
                      </div>
                      <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", width: 40, textAlign: "right" }}>
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
