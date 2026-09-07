import type { Metadata } from "next";
import Link from "next/link";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "NeuroStage — EEG Sleep Stage Classification",
  description:
    "Lightweight CNN-Transformer framework for real-time EEG-based sleep stage classification. Research prototype — not a clinical diagnostic tool.",
  keywords: "EEG, sleep staging, deep learning, CNN, transformer, AASM, polysomnography",
};

const navLinks = [
  { href: "/", label: "Overview" },
  { href: "/predict", label: "Predict" },
  { href: "/hypnogram", label: "Hypnogram" },
  { href: "/attention", label: "Attention" },
  { href: "/compare", label: "Compare" },
  { href: "/experiments", label: "Experiments" },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body>
        {/* Navigation */}
        <nav
          style={{
            position: "sticky",
            top: 0,
            zIndex: 50,
            background: "rgba(7, 10, 18, 0.85)",
            backdropFilter: "blur(20px)",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <div
            style={{
              maxWidth: 1280,
              margin: "0 auto",
              padding: "0 24px",
              height: 64,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            {/* Logo */}
            <Link href="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 8,
                  background: "var(--gradient-accent)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 16,
                }}
              >
                🧠
              </div>
              <span style={{ fontWeight: 700, fontSize: "1.1rem", color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
                Neuro<span style={{ color: "var(--accent-blue)" }}>Stage</span>
              </span>
            </Link>

            {/* Nav Links */}
            <div style={{ display: "flex", gap: 4 }}>
              {navLinks.map((link) => (
                <Link key={link.href} href={link.href} className="nav-link">
                  {link.label}
                </Link>
              ))}
            </div>

            {/* Research badge */}
            <div className="badge badge-amber">
              Research Prototype
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main style={{ minHeight: "calc(100vh - 64px)" }}>
          {children}
        </main>

        {/* Footer */}
        <footer
          style={{
            borderTop: "1px solid var(--border)",
            padding: "24px",
            textAlign: "center",
          }}
        >
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            NeuroStage — Research Prototype &nbsp;|&nbsp; KIT College of Engineering, Kolhapur &nbsp;|&nbsp; B.Tech AIML 2026–27
          </p>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: 4 }}>
            ⚠️ Not a clinically certified diagnostic system. All predictions are for research purposes only.
          </p>
        </footer>
      </body>
    </html>
  );
}
