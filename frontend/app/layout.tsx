import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "./context/AuthContext";
import Navbar from "./components/Navbar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "NeuroStage — EEG Sleep Stage Classification",
  description:
    "Lightweight CNN-Transformer framework for real-time EEG-based sleep stage classification. Research prototype — not a clinical diagnostic tool.",
  keywords: "EEG, sleep staging, deep learning, CNN, transformer, AASM, polysomnography",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body>
        <AuthProvider>
          <Navbar />
          {/* Main content */}
          <main style={{ minHeight: "calc(100vh - 64px)" }}>
            {children}
          </main>
        </AuthProvider>

        {/* Footer */}
        <footer
          style={{
            borderTop: "1px solid var(--border)",
            padding: "36px 24px",
            textAlign: "center",
            background: "#FFFFFF",
          }}
        >
          <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", margin: "0 0 6px", fontWeight: 500 }}>
            NeuroStage — Research Prototype &nbsp;|&nbsp; KIT College of Engineering, Kolhapur &nbsp;|&nbsp; B.Tech AIML 2026–27
          </p>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: 0 }}>
            ⚠️ Not a clinically certified diagnostic system. All predictions are for research purposes only.
          </p>
        </footer>
      </body>
    </html>
  );
}
