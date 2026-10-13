"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { useAuth } from "../context/AuthContext";
import { useRecording } from "../context/RecordingContext";
import { API_URL, getFriendlyErrorMessage } from "../lib/api";

type Status = "idle" | "uploading" | "processing" | "done" | "error";

export default function PredictPage() {
  const { token: authJwt, user, login } = useAuth();
  const { setActiveRecording } = useRecording();
  const [status, setStatus] = useState<Status>("idle");
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
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
      const msg = getFriendlyErrorMessage(e, "Failed to generate demo token.");
      setError(msg);
    } finally {
      setGeneratingToken(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFileName(f.name);
  };

  const handleUpload = async () => {
    const selectedFile = fileRef.current?.files?.[0];
    if (!selectedFile) {
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
      formData.append("file", selectedFile);

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
      setActiveRecording(data.recording_id, selectedFile.name);
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

      const predJson = await predRes.json();
      setUploadProgress(100);

      if (predJson.status === "done") {
        setStatus("done");
        fetchRecentRecordings();
        return;
      }

      setStatus("processing");

      // Poll for completion
      await pollForCompletion(data.recording_id, selectedFile.name);
    } catch (e: unknown) {
      const msg = getFriendlyErrorMessage(e, "An error occurred during upload.");
      setError(msg);
      setStatus("error");
      setUploadProgress(0);
    }
  };

  const [recentRecordings, setRecentRecordings] = useState<
    Array<{ id: string; filename: string; status: string; uploaded_at: string; n_epochs: number }>
  >([]);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const fetchRecentRecordings = async () => {
    try {
      const res = await fetch(`${API_URL}/recordings`, {
        headers: activeToken ? { Authorization: `Bearer ${activeToken}` } : {},
      });
      if (res.ok) {
        const d = await res.json();
        setRecentRecordings(d.recordings || []);
      }
    } catch {}
  };

  useEffect(() => {
    fetchRecentRecordings();
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        setBackendOnline(res.ok);
      } catch {
        setBackendOnline(false);
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 15000);
    return () => clearInterval(interval);
  }, [activeToken]);

  const pollForCompletion = async (rid: string, fname?: string) => {
    let attempts = 0;
    const maxAttempts = 120;
    setElapsedSeconds(0);
    const intervalTimer = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);

    try {
      while (attempts < maxAttempts) {
        await new Promise((r) => setTimeout(r, 1500));
        try {
          const res = await fetch(`${API_URL}/hypnogram/${rid}`, {
            headers: activeToken ? { Authorization: `Bearer ${activeToken}` } : {},
          });
          const data = await res.json();
          if (data.status === "done") {
            clearInterval(intervalTimer);
            setStatus("done");
            setActiveRecording(rid, fname || fileName || undefined);
            fetchRecentRecordings();
            return;
          }
          if (data.status === "error") {
            clearInterval(intervalTimer);
            setStatus("error");
            setError("Model inference failed on the server.");
            return;
          }
        } catch {}
        attempts++;
      }
      clearInterval(intervalTimer);
      setStatus("error");
      setError("Timed out waiting for inference to complete.");
    } finally {
      clearInterval(intervalTimer);
    }
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
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
            <h2 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0, color: "var(--text-primary)" }}>
              📁 Upload EDF File
            </h2>
            {backendOnline !== null && (
              <span
                style={{
                  fontSize: "0.72rem",
                  fontWeight: 700,
                  padding: "3px 8px",
                  borderRadius: 6,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 5,
                  background: backendOnline ? "rgba(5, 150, 105, 0.1)" : "rgba(225, 29, 72, 0.1)",
                  color: backendOnline ? "#065F46" : "#BE123C",
                  border: `1px solid ${backendOnline ? "rgba(5, 150, 105, 0.25)" : "rgba(225, 29, 72, 0.25)"}`,
                }}
              >
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: backendOnline ? "#059669" : "#E11D48" }} />
                {backendOnline ? "API Online" : "API Offline"}
              </span>
            )}
          </div>

          {backendOnline === false && (
            <div
              style={{
                marginBottom: 18,
                padding: "12px 14px",
                background: "rgba(225, 29, 72, 0.08)",
                border: "1px solid rgba(225, 29, 72, 0.3)",
                borderRadius: 10,
                fontSize: "0.82rem",
                color: "#BE123C",
                lineHeight: 1.5,
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: 4 }}>⚠️ Backend API server not detected</div>
              <div>
                The frontend cannot reach <code>{API_URL}</code>. Start the backend in your terminal with:
              </div>
              <pre
                style={{
                  marginTop: 6,
                  padding: "6px 10px",
                  background: "rgba(0,0,0,0.06)",
                  borderRadius: 6,
                  fontSize: "0.78rem",
                  fontFamily: "monospace",
                }}
              >
                uvicorn backend.main:app --port 8000
              </pre>
            </div>
          )}

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

          {status === "processing" && (
            <div style={{
              padding: "12px 16px",
              borderRadius: 10,
              background: "rgba(37, 99, 235, 0.08)",
              border: "1px solid rgba(37, 99, 235, 0.25)",
              color: "#1D4ED8",
              fontSize: "0.85rem",
              fontWeight: 600,
              marginBottom: 16,
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}>
              <span style={{ fontSize: "1.2rem" }}>⏳</span>
              <div>
                <div>Running PyTorch CNN-Transformer ({elapsedSeconds}s elapsed)</div>
                <div style={{ fontSize: "0.75rem", fontWeight: 400, color: "var(--text-secondary)" }}>
                  Filtering 0.3–35 Hz FIR, segmenting 30s epochs, batching self-attention...
                </div>
              </div>
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
            {status === "uploading"
              ? "⏳ Uploading EDF..."
              : status === "processing"
              ? `🧠 Inferring Stages... (${elapsedSeconds}s)`
              : "🚀 Upload & Run Inference"}
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
              ["1. Vectorized Preprocessing", "Band-pass filter (0.3–35 Hz), z-score normalize, vector slice into 30s epochs"],
              ["2. CNN Feature Extraction", "Depthwise-separable 1D-CNN extracts embeddings per epoch (75% FLOP reduction)"],
              ["3. Transformer Encoding", "2-layer, 4-head self-attention over 20-epoch context windows (~10 min sleep dynamics)"],
              ["4. Classification", "Dense + softmax → 5-class sleep stage prediction (W, N1, N2, N3, REM)"],
              ["5. Hypnogram & Attention", "Full-night sequence + self-attention weights stored in PostgreSQL"],
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
                📊 View Results for This Upload
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <Link
                  href={`/hypnogram?id=${recordingId}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
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
                  <span>📈 View Hypnogram</span>
                  <span style={{ fontSize: "0.75rem", background: "#2563EB", color: "#FFF", padding: "2px 8px", borderRadius: 4 }}>Full Night Sequence →</span>
                </Link>
                <Link
                  href={`/attention?id=${recordingId}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
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
                  <span>👁️ View Attention Map</span>
                  <span style={{ fontSize: "0.75rem", background: "#7C3AED", color: "#FFF", padding: "2px 8px", borderRadius: 4 }}>Context Weights →</span>
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Available Recordings History Launcher */}
      {recentRecordings.length > 0 && (
        <div className="glass-card" style={{ marginTop: 36, padding: 28 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <h2 style={{ fontSize: "1.15rem", fontWeight: 800, color: "var(--text-primary)" }}>
                📂 Available Processed Recordings
              </h2>
              <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: 2 }}>
                Preloaded overnight sleep studies ready for instant panel presentation. No re-upload required.
              </p>
            </div>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>
              {recentRecordings.length} recordings stored
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14 }}>
            {recentRecordings.map((rec) => (
              <div
                key={rec.id}
                style={{
                  padding: 16,
                  borderRadius: 12,
                  background: "var(--card-bg, #FFFFFF)",
                  border: "1px solid var(--border)",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.03)",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, fontSize: "0.88rem", color: "var(--text-primary)" }}>
                      {rec.filename}
                    </span>
                    <span
                      style={{
                        fontSize: "0.68rem",
                        fontWeight: 700,
                        padding: "2px 6px",
                        borderRadius: 4,
                        background: rec.status === "done" ? "rgba(5, 150, 105, 0.1)" : "rgba(217, 119, 6, 0.1)",
                        color: rec.status === "done" ? "#065F46" : "#92400E",
                      }}
                    >
                      {rec.status.toUpperCase()}
                    </span>
                  </div>
                  <div style={{ fontSize: "0.74rem", color: "var(--text-muted)", marginBottom: 12 }}>
                    ID: <code>{rec.id.slice(0, 12)}…</code> · {rec.n_epochs > 0 ? `${rec.n_epochs} epochs` : "0 epochs"}
                  </div>
                </div>

                <div style={{ display: "flex", gap: 8 }}>
                  <Link
                    href={`/hypnogram?id=${rec.id}`}
                    style={{
                      flex: 1,
                      textAlign: "center",
                      padding: "6px 10px",
                      borderRadius: 6,
                      background: "rgba(37, 99, 235, 0.08)",
                      border: "1px solid rgba(37, 99, 235, 0.2)",
                      color: "#1D4ED8",
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      textDecoration: "none",
                    }}
                  >
                    📈 Hypnogram
                  </Link>
                  <Link
                    href={`/attention?id=${rec.id}`}
                    style={{
                      flex: 1,
                      textAlign: "center",
                      padding: "6px 10px",
                      borderRadius: 6,
                      background: "rgba(124, 58, 237, 0.08)",
                      border: "1px solid rgba(124, 58, 237, 0.2)",
                      color: "#6D28D9",
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      textDecoration: "none",
                    }}
                  >
                    👁️ Attention
                  </Link>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
