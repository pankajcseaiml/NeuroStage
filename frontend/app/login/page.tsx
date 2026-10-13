"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!email || !password) {
      setError("Please fill in both email and password.");
      return;
    }

    setLoading(true);
    const res = await login(email, password);
    setLoading(false);

    if (res.success) {
      router.push("/predict");
    } else {
      setError(res.error || "Login failed. Please check your credentials.");
    }
  };

  const handleDemoFill = () => {
    setEmail("test@gmail.com");
    setPassword("test@123");
    setError(null);
  };

  return (
    <div
      style={{
        minHeight: "calc(100vh - 160px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "48px 20px",
        background: "radial-gradient(ellipse at 50% 20%, rgba(225, 77, 69, 0.04) 0%, transparent 60%)",
      }}
    >
      <div
        className="glass-card"
        style={{
          width: "100%",
          maxWidth: 440,
          padding: "44px 36px",
          color: "var(--text-primary)",
        }}
      >
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div
            style={{
              width: 52,
              height: 52,
              borderRadius: 14,
              background: "linear-gradient(135deg, #E14D45 0%, #D9383A 100%)",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 26,
              marginBottom: 16,
              boxShadow: "0 2px 8px rgba(225, 77, 69, 0.3)",
            }}
          >
            🧠
          </div>
          <h1
            style={{
              fontSize: "1.75rem",
              fontWeight: 800,
              letterSpacing: "-0.03em",
              marginBottom: 8,
              color: "var(--text-primary)",
            }}
          >
            Welcome Back
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.92rem", lineHeight: 1.55, margin: 0, fontWeight: 450 }}>
            Sign in to access your EEG sleep staging recordings & attention maps.
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div
            style={{
              background: "rgba(225, 29, 72, 0.08)",
              border: "1px solid rgba(225, 29, 72, 0.25)",
              color: "#BE123C",
              padding: "12px 16px",
              borderRadius: 10,
              fontSize: "0.85rem",
              fontWeight: 600,
              marginBottom: 24,
              display: "flex",
              alignItems: "center",
              gap: 10,
            }}
          >
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div>
            <label
              style={{
                display: "block",
                fontSize: "0.78rem",
                fontWeight: 700,
                color: "var(--text-secondary)",
                marginBottom: 8,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              Email Address
            </label>
            <input
              type="email"
              className="input-field"
              placeholder="e.g. test@gmail.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={{ width: "100%", padding: "12px 14px", fontSize: "0.95rem" }}
            />
          </div>

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <label
                style={{
                  fontSize: "0.78rem",
                  fontWeight: 700,
                  color: "var(--text-secondary)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Password
              </label>
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--brand-coral)",
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  cursor: "pointer",
                  padding: 0,
                }}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
            <input
              type={showPassword ? "text" : "password"}
              className="input-field"
              placeholder="••••••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{ width: "100%", padding: "12px 14px", fontSize: "0.95rem" }}
            />
          </div>

          <button
            type="submit"
            className="btn-primary"
            disabled={loading}
            style={{ width: "100%", padding: "13px", marginTop: 6 }}
          >
            {loading ? "Signing in..." : "Sign In →"}
          </button>
        </form>

        {/* Quick Demo Fill Helper */}
        <div
          style={{
            marginTop: 24,
            padding: "16px",
            background: "#F8FAFC",
            border: "1px dashed #CBD5E1",
            borderRadius: 14,
            textAlign: "center",
          }}
        >
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: 8, margin: 0, fontWeight: 500 }}>
            Quick fill your registered test account:
          </p>
          <button
            type="button"
            onClick={handleDemoFill}
            style={{
              marginTop: 10,
              background: "#FFFFFF",
              border: "1px solid var(--brand-coral)",
              color: "var(--brand-coral)",
              padding: "7px 18px",
              borderRadius: 9999,
              fontSize: "0.82rem",
              fontWeight: 700,
              cursor: "pointer",
              transition: "all 0.2s ease",
              boxShadow: "0 1px 2px rgba(15, 23, 42, 0.04)",
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "var(--brand-coral)";
              e.currentTarget.style.color = "#ffffff";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "#FFFFFF";
              e.currentTarget.style.color = "var(--brand-coral)";
            }}
          >
            ⚡ Auto-Fill test@gmail.com
          </button>
        </div>

        {/* Footer switch */}
        <div style={{ textAlign: "center", marginTop: 28, fontSize: "0.88rem", color: "var(--text-secondary)" }}>
          Don&apos;t have an account yet?{" "}
          <Link
            href="/register"
            style={{ color: "var(--brand-coral)", fontWeight: 700, textDecoration: "none" }}
          >
            Create an account
          </Link>
        </div>
      </div>
    </div>
  );
}
