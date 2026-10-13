/**
 * Centralized API configuration for NeuroStage frontend.
 * Prioritizes NEXT_PUBLIC_API_URL, then dynamically defaults to 127.0.0.1:8000
 * to prevent Windows IPv6 [::1] vs IPv4 127.0.0.1 localhost connection refused mismatches.
 */
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname === "localhost" ? "127.0.0.1" : window.location.hostname}:8000`
    : "http://127.0.0.1:8000");

export function getFriendlyErrorMessage(err: unknown, fallback: string = "An error occurred"): string {
  const msg = err instanceof Error ? err.message : String(err || fallback);
  if (msg.toLowerCase().includes("failed to fetch") || msg.toLowerCase().includes("networkerror")) {
    return `Could not connect to the backend server at ${API_URL}. Please ensure the FastAPI backend is running (python -m uvicorn backend.main:app --port 8000).`;
  }
  return msg;
}

