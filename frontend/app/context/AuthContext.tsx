"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { API_URL } from "../lib/api";

export type UserProfile = {
  id: string;
  email: string;
  role: string;
};

type AuthContextType = {
  user: UserProfile | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, pass: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  refreshToken: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchProfile = useCallback(async (jwt: string): Promise<UserProfile | null> => {
    try {
      const res = await fetch(`${API_URL}/auth/me`, {
        headers: { Authorization: `Bearer ${jwt}` },
      });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }, []);

  // Restore token on initial mount
  useEffect(() => {
    const initAuth = async () => {
      const savedToken = typeof window !== "undefined" ? localStorage.getItem("neurostage_jwt_token") : null;
      if (savedToken) {
        setToken(savedToken);
        const profile = await fetchProfile(savedToken);
        if (profile) {
          setUser(profile);
        } else {
          // Token is invalid/expired
          localStorage.removeItem("neurostage_jwt_token");
          setToken(null);
          setUser(null);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, [fetchProfile]);

  const login = async (email: string, pass: string) => {
    setIsLoading(true);
    try {
      const form = new URLSearchParams();
      form.append("username", email);
      form.append("password", pass);

      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: form.toString(),
      });

      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setIsLoading(false);
        return { success: false, error: d.detail || "Invalid email or password." };
      }

      const data = await res.json();
      const jwt = data.access_token;
      setToken(jwt);
      localStorage.setItem("neurostage_jwt_token", jwt);

      const profile = await fetchProfile(jwt);
      if (profile) {
        setUser(profile);
      } else {
        setUser({ id: "user", email, role: "user" });
      }

      setIsLoading(false);
      return { success: true };
    } catch (e: unknown) {
      setIsLoading(false);
      const msg = e instanceof Error ? e.message : "Failed to connect to authentication server.";
      return {
        success: false,
        error: `Could not connect to server (${msg}). Ensure backend is running at ${API_URL}`,
      };
    }
  };

  const register = async (email: string, pass: string) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password: pass }),
      });

      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setIsLoading(false);
        if (res.status === 409) {
          return { success: false, error: "This email is already registered. Please sign in instead." };
        }
        return { success: false, error: d.detail || "Registration failed." };
      }

      // Automatically login upon successful registration
      return await login(email, pass);
    } catch (e: unknown) {
      setIsLoading(false);
      const msg = e instanceof Error ? e.message : "Connection failed.";
      return {
        success: false,
        error: `Could not reach backend (${msg}). Ensure backend is active at ${API_URL}`,
      };
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    if (typeof window !== "undefined") {
      localStorage.removeItem("neurostage_jwt_token");
    }
  };

  const refreshToken = async () => {
    if (token) {
      const profile = await fetchProfile(token);
      if (profile) setUser(profile);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        login,
        register,
        logout,
        refreshToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
