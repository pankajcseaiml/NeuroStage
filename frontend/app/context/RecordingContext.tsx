"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { API_URL } from "../lib/api";

export type RecordingMeta = {
  id: string;
  filename: string;
  status: string;
  uploaded_at: string;
  n_epochs: number;
};

type RecordingContextType = {
  activeRecordingId: string | null;
  activeRecordingFilename: string | null;
  recordings: RecordingMeta[];
  isLoading: boolean;
  setActiveRecording: (id: string, filename?: string) => void;
  refreshRecordings: () => Promise<RecordingMeta[]>;
  getRecordingUrl: (path: string) => string;
};

const RecordingContext = createContext<RecordingContextType | undefined>(undefined);

export function RecordingProvider({ children }: { children: React.ReactNode }) {
  const [activeRecordingId, setActiveRecordingIdState] = useState<string | null>(null);
  const [activeRecordingFilename, setActiveRecordingFilenameState] = useState<string | null>(null);
  const [recordings, setRecordings] = useState<RecordingMeta[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Restore from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedId = localStorage.getItem("neurostage_active_recording_id");
      const savedName = localStorage.getItem("neurostage_active_recording_filename");
      if (savedId) {
        setActiveRecordingIdState(savedId);
        if (savedName) setActiveRecordingFilenameState(savedName);
      }
    }
  }, []);

  const setActiveRecording = useCallback((id: string, filename?: string) => {
    setActiveRecordingIdState(id);
    if (typeof window !== "undefined") {
      localStorage.setItem("neurostage_active_recording_id", id);
      if (filename) {
        setActiveRecordingFilenameState(filename);
        localStorage.setItem("neurostage_active_recording_filename", filename);
      }
    }
  }, []);

  const refreshRecordings = useCallback(async (): Promise<RecordingMeta[]> => {
    setIsLoading(true);
    try {
      const token = typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : null;
      const headers: Record<string, string> = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const res = await fetch(`${API_URL}/recordings`, { headers });
      if (res.ok) {
        const data = await res.json();
        const list: RecordingMeta[] = data.recordings || [];
        setRecordings(list);

        // If no active ID yet, or current active ID isn't in list, select the newest done recording
        if (list.length > 0) {
          const savedId = typeof window !== "undefined" ? localStorage.getItem("neurostage_active_recording_id") : null;
          const matching = list.find((r) => r.id === savedId);
          if (matching) {
            setActiveRecordingIdState(matching.id);
            setActiveRecordingFilenameState(matching.filename);
          } else if (!savedId) {
            const firstDone = list.find((r) => r.status === "done") || list[0];
            setActiveRecording(firstDone.id, firstDone.filename);
          }
        }
        setIsLoading(false);
        return list;
      }
    } catch {
      // Fallback
    }
    setIsLoading(false);
    return [];
  }, [setActiveRecording]);

  useEffect(() => {
    refreshRecordings();
  }, [refreshRecordings]);

  const getRecordingUrl = useCallback((path: string): string => {
    if (!activeRecordingId) return path;
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}id=${activeRecordingId}`;
  }, [activeRecordingId]);

  return (
    <RecordingContext.Provider
      value={{
        activeRecordingId,
        activeRecordingFilename,
        recordings,
        isLoading,
        setActiveRecording,
        refreshRecordings,
        getRecordingUrl,
      }}
    >
      {children}
    </RecordingContext.Provider>
  );
}

export function useRecording() {
  const ctx = useContext(RecordingContext);
  if (!ctx) {
    throw new Error("useRecording must be used within a RecordingProvider");
  }
  return ctx;
}
