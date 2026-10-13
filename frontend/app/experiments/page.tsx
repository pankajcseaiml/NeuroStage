"use client";

import { useState } from "react";
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

export default function ExperimentsPage() {
  const [exps, setExps] = useState<Experiment[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [token, setToken] = useState("");
  const [sortKey, setSortKey] = useState<keyof Experiment>("created_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const fetchExperiments = async () => {
    if (!token) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/experiments?limit=100`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setExps(data.experiments || []);
      setTotal(data.total || 0);
    } catch {}
    setLoading(false);
  };

  const sorted = [...exps].sort((a, b) => {
    const av = a[sortKey] ?? "";
    const bv = b[sortKey] ?? "";
    if (av < bv) return sortDir === "asc" ? -1 : 1;
    if (av > bv) return sortDir === "asc" ? 1 : -1;
    return 0;
  });

  const toggleSort = (key: keyof Experiment) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
  };

  const sortIndicator = (key: keyof Experiment) =>
    sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : "";

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 24px" }}>
      <div style={{ marginBottom: 36 }}>
        <h1 style={{ fontSize: "2.2rem", fontWeight: 800, marginBottom: 8, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>
          Experiment Log
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "1rem", fontWeight: 450 }}>
          All training runs with metrics, model variants, and efficiency data. Sortable by any column.
        </p>
      </div>

      {/* Auth */}
      <div className="glass-card" style={{ padding: 24, marginBottom: 28 }}>
        <div style={{ display: "flex", gap: 14, alignItems: "flex-end" }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 700, display: "block", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              ACCESS TOKEN
            </label>
            <input
              className="input-field"
              type="password"
              placeholder="JWT token"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              style={{ width: "100%" }}
            />
          </div>
          <button className="btn-primary" onClick={fetchExperiments} style={{ padding: "12px 24px" }}>
            Load Experiments
          </button>
        </div>
      </div>

      {loading && <div style={{ textAlign: "center", padding: 60, color: "var(--text-muted)", fontWeight: 500 }}>⏳ Loading experiments...</div>}

      {exps.length === 0 && !loading && (
        <div className="glass-card" style={{ padding: 60, textAlign: "center" }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>🧪</div>
          <p style={{ color: "var(--text-secondary)", fontSize: "1rem", fontWeight: 500 }}>
            No experiments found in database. Run training first:
          </p>
          <code className="code" style={{
            display: "inline-block",
            marginTop: 16,
            padding: "10px 24px",
            background: "#F8FAFC",
            borderRadius: 8,
            border: "1px solid var(--border)",
            color: "#2563EB",
            fontSize: "0.9rem",
            fontWeight: 600,
          }}>
            python train.py
          </code>
        </div>
      )}

      {exps.length > 0 && (
        <>
          <div style={{ marginBottom: 16, fontSize: "0.875rem", color: "var(--text-muted)", fontWeight: 500 }}>
            Showing {exps.length} of {total} experiments
          </div>
          <div className="glass-card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ overflowX: "auto" }}>
              <table className="data-table">
                <thead>
                  <tr>
                    {[
                      { key: "model_variant", label: "Model Variant" },
                      { key: "accuracy", label: "Accuracy" },
                      { key: "macro_f1", label: "Macro-F1" },
                      { key: "kappa", label: "Kappa" },
                      { key: "param_count", label: "Params" },
                      { key: "model_size_mb", label: "Size (MB)" },
                      { key: "latency_ms", label: "Latency (ms)" },
                      { key: "created_at", label: "Date" },
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
                        {e.mlflow_run_id && (
                          <div className="code" style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "inline-block", marginTop: 2 }}>
                            {e.mlflow_run_id.slice(0, 8)}…
                          </div>
                        )}
                      </td>
                      <td>{e.accuracy != null ? <strong>{(e.accuracy * 100).toFixed(1)}%</strong> : "—"}</td>
                      <td>{e.macro_f1 != null ? e.macro_f1.toFixed(3) : "—"}</td>
                      <td>{e.kappa != null ? e.kappa.toFixed(3) : "—"}</td>
                      <td>{e.param_count != null ? `${(e.param_count / 1000).toFixed(0)}K` : "—"}</td>
                      <td>{e.model_size_mb != null ? `${e.model_size_mb.toFixed(1)}` : "—"}</td>
                      <td>
                        {e.latency_ms != null ? (
                          `${e.latency_ms.toFixed(1)}`
                        ) : (
                          <span className="target-badge">N/A</span>
                        )}
                      </td>
                      <td style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                        {new Date(e.created_at).toLocaleDateString()}
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
