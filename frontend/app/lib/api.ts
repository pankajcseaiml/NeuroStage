/**
 * Centralized API configuration for NeuroStage frontend.
 * Always targets the running FastAPI backend (default port 8000).
 */
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL && !process.env.NEXT_PUBLIC_API_URL.includes("8085")
    ? process.env.NEXT_PUBLIC_API_URL
    : "http://localhost:8000";
