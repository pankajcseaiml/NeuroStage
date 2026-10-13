"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "../context/AuthContext";
import { useRecording } from "../context/RecordingContext";

const navLinks = [
  { href: "/", label: "Overview" },
  { href: "/predict", label: "Predict" },
  { href: "/hypnogram", label: "Hypnogram" },
  { href: "/attention", label: "Attention" },
  { href: "/compare", label: "Compare" },
  { href: "/experiments", label: "Experiments" },
];

export default function Navbar() {
  const pathname = usePathname();
  const { user, isLoading, logout } = useAuth();
  const { activeRecordingId, activeRecordingFilename } = useRecording();

  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        background: "rgba(255, 255, 255, 0.92)",
        backdropFilter: "blur(16px)",
        borderBottom: "1px solid var(--border)",
        boxShadow: "0 1px 3px rgba(15, 23, 42, 0.04)",
      }}
    >
      <div
        style={{
          maxWidth: 1280,
          margin: "0 auto",
          padding: "0 24px",
          height: 66,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
        }}
      >
        {/* Logo */}
        <Link href="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 10,
              background: "linear-gradient(135deg, #E14D45 0%, #D9383A 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
              boxShadow: "0 2px 8px rgba(225, 77, 69, 0.3)",
            }}
          >
            🧠
          </div>
          <span style={{ fontWeight: 800, fontSize: "1.15rem", color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
            Neuro<span style={{ color: "var(--brand-coral)" }}>Stage</span>
          </span>
        </Link>

        {/* Center Nav Links with active recording persistence */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {navLinks.map((link) => {
            const isActive = pathname === link.href;
            const targetHref =
              activeRecordingId && link.href !== "/" && link.href !== "/predict"
                ? `${link.href}?id=${activeRecordingId}`
                : link.href;

            return (
              <Link
                key={link.href}
                href={targetHref}
                className={`nav-link ${isActive ? "active" : ""}`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>

        {/* Right Section: Auth & Active PSG Badge */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {activeRecordingId && (
            <Link
              href={`/hypnogram?id=${activeRecordingId}`}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 10px",
                borderRadius: 8,
                background: "rgba(37, 99, 235, 0.08)",
                border: "1px solid rgba(37, 99, 235, 0.2)",
                fontSize: "0.75rem",
                color: "#1D4ED8",
                fontWeight: 700,
                textDecoration: "none",
                maxWidth: 180,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={`Active Study: ${activeRecordingFilename || activeRecordingId}`}
            >
              <span>📂</span>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                {activeRecordingFilename || activeRecordingId.slice(0, 8) + "…"}
              </span>
            </Link>
          )}
          {isLoading ? (
            <div
              style={{
                width: 80,
                height: 34,
                borderRadius: 9999,
                background: "#E2E8F0",
                animation: "pulse 1.5s infinite",
              }}
            />
          ) : user ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "5px 14px",
                  borderRadius: 9999,
                  background: "#F1F5F9",
                  border: "1px solid var(--border)",
                  fontSize: "0.82rem",
                  color: "var(--text-primary)",
                  fontWeight: 600,
                }}
              >
                <span
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: "50%",
                    background: "#059669",
                    boxShadow: "0 0 6px #059669",
                  }}
                />
                <span style={{ maxWidth: 150, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {user.email}
                </span>
              </div>
              <button
                onClick={logout}
                style={{
                  background: "#FFFFFF",
                  border: "1px solid var(--border)",
                  color: "var(--text-secondary)",
                  borderRadius: 8,
                  padding: "6px 12px",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.color = "var(--brand-coral)";
                  e.currentTarget.style.borderColor = "var(--brand-coral)";
                  e.currentTarget.style.background = "var(--brand-coral-subtle)";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.color = "var(--text-secondary)";
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.background = "#FFFFFF";
                }}
              >
                Sign Out
              </button>
            </div>
          ) : (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Link
                href="/login"
                style={{
                  color: "var(--text-secondary)",
                  textDecoration: "none",
                  padding: "6px 12px",
                  fontSize: "0.88rem",
                  fontWeight: 600,
                  transition: "color 0.2s ease",
                }}
                onMouseOver={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
                onMouseOut={(e) => (e.currentTarget.style.color = "var(--text-secondary)")}
              >
                Log in
              </Link>
              <Link
                href="/register"
                style={{
                  background: "linear-gradient(135deg, #E14D45 0%, #D9383A 100%)",
                  color: "#FFFFFF",
                  textDecoration: "none",
                  padding: "8px 18px",
                  borderRadius: 9999,
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  boxShadow: "0 2px 6px rgba(225, 77, 69, 0.25)",
                  transition: "all 0.2s ease",
                  display: "inline-flex",
                  alignItems: "center",
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = "translateY(-1px)";
                  e.currentTarget.style.boxShadow = "0 4px 12px rgba(225, 77, 69, 0.35)";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.boxShadow = "0 2px 6px rgba(225, 77, 69, 0.25)";
                }}
              >
                Get Started
              </Link>
            </div>
          )}

          {/* Research prototype badge */}
          <div className="badge badge-amber" style={{ fontSize: "0.75rem", padding: "4px 10px" }}>
            Research Prototype
          </div>
        </div>
      </div>
    </nav>
  );
}
