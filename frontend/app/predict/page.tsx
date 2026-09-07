"use client";

import { useState, useRef } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Status = "idle" | "uploading" | "processing" | "done" | "error";

export default function PredictPage() {
  const [status, setStatus] = useState<Status>("idle");
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [token, setToken] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const [generatingToken, setGeneratingToken] = useState(false);

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
      // 2. Login
      const form = new URLSearchParams();
      form.append("username", "demo@neurostage.com");
      form.append("password", "demopassword123");
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString(),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Failed to get token.");
      }
      const data = await res.json();
      setToken(data.access_token);
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
    if (!token) {
      setError("Please enter your access token.");
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
        headers: { Authorization: `Bearer ${token}` },
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
        headers: { Authorization: `Bearer ${token}` },
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
          headers: { Authorization: `Bearer ${token}` },
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
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 40 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em" }}>
          Upload & Predict
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", lineHeight: 1.7 }}>
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
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: 24 }}>
            📁 Upload EDF File
          </h2>

          {/* Token input */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 500 }}>
                ACCESS TOKEN
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
                  padding: "3px 8px",
                  fontSize: "0.75rem",
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
              placeholder="Paste JWT token or click '1-Click Demo Token'"
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 6 }}>
              Click <strong>1-Click Demo Token</strong> above to auto-authenticate with the backend.
            </p>
          </div>

          {/* File drop zone */}
          <div
            onClick={() => fileRef.current?.click()}
            style={{
              border: "2px dashed var(--border)",
              borderRadius: 12,
              padding: "40px 20px",
              textAlign: "center",
              cursor: "pointer",
              marginBottom: 20,
              transition: "all 0.2s",
              background: fileName ? "rgba(59,130,246,0.05)" : "transparent",
              borderColor: fileName ? "var(--accent-blue)" : "var(--border)",
            }}
          >
            <div style={{ fontSize: 40, marginBottom: 12 }}>📂</div>
            <p style={{ color: fileName ? "var(--accent-blue)" : "var(--text-secondary)", fontWeight: 500 }}>
              {fileName || "Click to select EDF file"}
            </p>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6 }}>
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
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6 }}>
                {status === "uploading" && "Uploading..."}
                {status === "processing" && "Running inference (polling)..."}
                {status === "done" && "Complete!"}
              </p>
            </div>
          )}

          {error && (
            <div style={{
              padding: "12px 16px",
              borderRadius: 10,
              background: "rgba(244,63,94,0.1)",
              border: "1px solid rgba(244,63,94,0.3)",
              color: "#fda4af",
              fontSize: "0.875rem",
              marginBottom: 16,
            }}>
              ❌ {error}
            </div>
          )}

          {status === "done" && (
            <div style={{
              padding: "12px 16px",
              borderRadius: 10,
              background: "rgba(16,185,129,0.1)",
              border: "1px solid rgba(16,185,129,0.3)",
              color: "#6ee7b7",
              fontSize: "0.875rem",
              marginBottom: 16,
            }}>
              ✅ Inference complete! Recording ID: <code className="code">{recordingId}</code>
            </div>
          )}

          <button
            className="btn-primary"
            onClick={handleUpload}
            disabled={status === "uploading" || status === "processing"}
            style={{ width: "100%" }}
          >
            {status === "uploading" ? "Uploading..." : status === "processing" ? "Processing..." : "🚀 Upload & Run Inference"}
          </button>
        </div>

        {/* Info panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* What happens */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: 16 }}>
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
                <div style={{ fontWeight: 600, fontSize: "0.8rem", color: "var(--accent-blue)" }}>{title}</div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 2 }}>{desc}</div>
              </div>
            ))}
          </div>

          {/* Next steps */}
          {status === "done" && recordingId && (
            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: 16 }}>
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
                    background: "rgba(59,130,246,0.1)",
                    border: "1px solid rgba(59,130,246,0.2)",
                    color: "var(--accent-blue)",
                    textDecoration: "none",
                    fontWeight: 600,
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
                    background: "rgba(139,92,246,0.1)",
                    border: "1px solid rgba(139,92,246,0.2)",
                    color: "var(--accent-violet)",
                    textDecoration: "none",
                    fontWeight: 600,
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
