"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { useAuth } from "../context/AuthContext";
import { API_URL } from "../lib/api";

type Status = "idle" | "uploading" | "processing" | "done" | "error";

export default function PredictPage() {
  const { token: authJwt, user, login } = useAuth();
  const [status, setStatus] = useState<Status>("idle");
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [token, setToken] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [generatingToken, setGeneratingToken] = useState(false);
  const [showManualToken, setShowManualToken] = useState(false);

  const activeToken = token || authJwt || "";

  const generateDemoToken = async () => {
    setGeneratingToken(true);
    setError(null);
    try {
      // 1. Try to register demo user (ignore 409 if already registered)
      await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: "demo@neurostage.com", password: "demopassword123" }),
      });
      // 2. Login through global context
      const res = await login("demo@neurostage.com", "demopassword123");
      if (!res.success) {
        throw new Error(res.error || "Failed to log in demo user.");
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to generate token.";
      setError("Auto-token generation error: " + msg);
    } finally {
      setGeneratingToken(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFileName(f.name);
  };

  const handleUpload = async () => {
    if (!fileRef.current?.files?.[0]) {
      setError("Please select an EDF file.");
      return;
    }
    if (!activeToken) {
      setError("Please sign in or generate an access token before uploading.");
      return;
    }
    setStatus("uploading");
    setError(null);
    setUploadProgress(30);

    try {
      const formData = new FormData();
      formData.append("file", fileRef.current.files[0]);

      const res = await fetch(`${API_URL}/eeg/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${activeToken}` },
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Upload failed.");
      }

      const data = await res.json();
      setRecordingId(data.recording_id);
      setUploadProgress(60);

      // Trigger prediction
      const predRes = await fetch(`${API_URL}/predict/${data.recording_id}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${activeToken}` },
      });

      if (!predRes.ok) {
        const predData = await predRes.json();
        throw new Error(predData.detail || "Inference failed.");
      }

      setUploadProgress(100);
      setStatus("processing");

      // Poll for completion
      await pollForCompletion(data.recording_id);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "An error occurred during upload.";
      setError(msg);
      setStatus("error");
      setUploadProgress(0);
    }
  };

  const pollForCompletion = async (rid: string) => {
    let attempts = 0;
    const maxAttempts = 60;
    while (attempts < maxAttempts) {
      await new Promise((r) => setTimeout(r, 3000));
      try {
        const res = await fetch(`${API_URL}/hypnogram/${rid}`, {
          headers: { Authorization: `Bearer ${activeToken}` },
        });
        const data = await res.json();
        if (data.status === "done") {
          setStatus("done");
          return;
        }
        if (data.status === "error") {
          setStatus("error");
          setError("Model inference failed on the server.");
          return;
        }
      } catch {}
      attempts++;
    }
    setStatus("error");
    setError("Timed out waiting for inference to complete.");
  };

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 36 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Upload & Predict
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", lineHeight: 1.65, fontWeight: 450 }}>
          Upload an EEG recording in EDF format. The system will preprocess it, run inference with the
          trained NeuroStage model, and generate a predicted hypnogram.
        </p>
        <div className="badge badge-amber" style={{ marginTop: 12 }}>
          ⚠️ Research prototype — not for clinical use
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {/* Upload panel */}
        <div className="glass-card" style={{ padding: 32 }}>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 20, color: "var(--text-primary)" }}>
            📁 Upload EDF File
          </h2>

          {/* User Auth Banner */}
          <div style={{ marginBottom: 20 }}>
            {user ? (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "12px 14px",
                  background: "rgba(5, 150, 105, 0.08)",
                  border: "1px solid rgba(5, 150, 105, 0.25)",
                  borderRadius: 10,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ color: "#059669", fontSize: "1.1rem", fontWeight: 800 }}>✓</span>
                  <div>
                    <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#065F46" }}>
                      Authenticated as {user.email}
                    </div>
                    <div style={{ fontSize: "0.75rem", color: "#047857" }}>
                      Session active · Ready for EEG inference
                    </div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setShowManualToken(!showManualToken)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--text-muted)",
                    fontSize: "0.75rem",
                    cursor: "pointer",
                    textDecoration: "underline",
                    fontWeight: 500,
                  }}
                >
                  {showManualToken ? "Hide Token" : "View Token"}
                </button>
              </div>
            ) : (
              <div
                style={{
                  padding: "14px 16px",
                  background: "rgba(37, 99, 235, 0.06)",
                  border: "1px solid rgba(37, 99, 235, 0.2)",
                  borderRadius: 12,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--text-primary)" }}>
                    Authentication Required
                  </span>
                  <button
                    type="button"
                    onClick={generateDemoToken}
                    disabled={generatingToken}
                    style={{
                      background: "rgba(37, 99, 235, 0.12)",
                      border: "1px solid rgba(37, 99, 235, 0.3)",
                      color: "#1D4ED8",
                      borderRadius: 6,
                      padding: "4px 10px",
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      cursor: "pointer",
                    }}
                  >
                    {generatingToken ? "Generating..." : "⚡ 1-Click Demo Login"}
                  </button>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: "0.82rem" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Already have an account?</span>
                  <Link href="/login" style={{ color: "#1D4ED8", fontWeight: 700, textDecoration: "none" }}>
                    Sign In
                  </Link>
                  <span style={{ color: "var(--text-muted)" }}>•</span>
                  <Link href="/register" style={{ color: "var(--brand-coral)", fontWeight: 700, textDecoration: "none" }}>
                    Sign Up
                  </Link>
                </div>
              </div>
            )}

            {/* Optional / Manual Token Input (only if requested or unauthenticated) */}
            {(showManualToken || !user) && (
              <div style={{ marginTop: 12 }}>
                <input
                  className="input-field"
                  type="password"
                  placeholder="Paste custom JWT token here"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  style={{ width: "100%", fontSize: "0.8rem", padding: "8px 12px" }}
                />
              </div>
            )}
          </div>

          {/* File drop zone */}
          <div
            onClick={() => fileRef.current?.click()}
            style={{
              border: `2px dashed ${fileName ? "var(--brand-coral)" : "var(--border-hover)"}`,
              borderRadius: 14,
              padding: "40px 20px",
              textAlign: "center",
              cursor: "pointer",
              marginBottom: 20,
              transition: "all 0.2s",
              background: fileName ? "rgba(225, 77, 69, 0.04)" : "#F8FAFC",
            }}
          >
            <div style={{ fontSize: 40, marginBottom: 12 }}>📂</div>
            <p style={{ color: fileName ? "var(--brand-coral)" : "var(--text-primary)", fontWeight: 600, fontSize: "0.95rem" }}>
              {fileName || "Click to select EDF file"}
            </p>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6, fontWeight: 500 }}>
              .edf or .bdf · max 500 MB
            </p>
          </div>
          <input ref={fileRef} type="file" accept=".edf,.bdf" style={{ display: "none" }} onChange={handleFileChange} />

          {/* Progress bar */}
          {status !== "idle" && status !== "error" && (
            <div style={{ marginBottom: 16 }}>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 6, fontWeight: 500 }}>
                {status === "uploading" && "Uploading EDF recording..."}
                {status === "processing" && "Running inference (polling)..."}
                {status === "done" && "Complete!"}
              </p>
            </div>
          )}

          {error && (
            <div style={{
              padding: "12px 16px",
              borderRadius: 10,
              background: "rgba(225, 29, 72, 0.08)",
              border: "1px solid rgba(225, 29, 72, 0.25)",
              color: "#BE123C",
              fontSize: "0.875rem",
              fontWeight: 600,
              marginBottom: 16,
            }}>
              ❌ {error}
            </div>
          )}

          {status === "done" && (
            <div style={{
              padding: "12px 16px",
              borderRadius: 10,
              background: "rgba(5, 150, 105, 0.08)",
              border: "1px solid rgba(5, 150, 105, 0.25)",
              color: "#065F46",
              fontSize: "0.875rem",
              fontWeight: 600,
              marginBottom: 16,
            }}>
              ✅ Inference complete! Recording ID: <code className="code">{recordingId}</code>
            </div>
          )}

          <button
            className="btn-coral"
            onClick={handleUpload}
            disabled={status === "uploading" || status === "processing"}
            style={{
              width: "100%",
              padding: "14px",
              fontSize: "0.98rem",
              borderRadius: 12,
              opacity: status === "uploading" || status === "processing" ? 0.6 : 1,
            }}
          >
            {status === "uploading" ? "⏳ Uploading EDF..." : status === "processing" ? "🧠 Processing Sleep Stages..." : "🚀 Upload & Run Inference"}
          </button>
        </div>

        {/* Info panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* What happens */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: 16, color: "var(--text-primary)" }}>
              🔬 What happens after upload?
            </h3>
            {[
              ["1. Preprocessing", "Band-pass filter (0.3–35 Hz), z-score normalize, segment into 30s epochs"],
              ["2. CNN Feature Extraction", "Depthwise-separable 1D-CNN extracts embeddings per epoch"],
              ["3. Transformer Encoding", "2-layer, 4-head self-attention over 20-epoch context windows"],
              ["4. Classification", "Dense + softmax → 5-class prediction per epoch"],
              ["5. Hypnogram", "Full-night prediction sequence stored and returned"],
            ].map(([title, desc]) => (
              <div key={title} style={{ marginBottom: 14 }}>
                <div style={{ fontWeight: 700, fontSize: "0.82rem", color: "var(--brand-coral)" }}>{title}</div>
                <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: 2, lineHeight: 1.5 }}>{desc}</div>
              </div>
            ))}
          </div>

          {/* Next steps */}
          {status === "done" && recordingId && (
            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: 16, color: "var(--text-primary)" }}>
                📊 View Results
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <a
                  href={`/hypnogram?id=${recordingId}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "14px 16px",
                    borderRadius: 10,
                    background: "rgba(37, 99, 235, 0.08)",
                    border: "1px solid rgba(37, 99, 235, 0.25)",
                    color: "#1D4ED8",
                    textDecoration: "none",
                    fontWeight: 700,
                    fontSize: "0.875rem",
                    transition: "all 0.2s",
                  }}
                >
                  📈 View Hypnogram
                </a>
                <a
                  href={`/attention?id=${recordingId}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "14px 16px",
                    borderRadius: 10,
                    background: "rgba(124, 58, 237, 0.08)",
                    border: "1px solid rgba(124, 58, 237, 0.25)",
                    color: "#6D28D9",
                    textDecoration: "none",
                    fontWeight: 700,
                    fontSize: "0.875rem",
                    transition: "all 0.2s",
                  }}
                >
                  👁️ View Attention Map
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
