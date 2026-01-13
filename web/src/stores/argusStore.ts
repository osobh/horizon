import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export interface MetricSample {
  metric: Record<string, string>;
  value?: [number, string];
  values: [number, string][];
}

export interface QueryResult {
  result_type: string;
  data: MetricSample[];
}

export interface Alert {
  name: string;
  state: 'firing' | 'pending' | 'resolved';
  severity: 'critical' | 'warning' | 'info';
  summary: string;
  description: string;
  labels: Record<string, string>;
  annotations: Record<string, string>;
  active_at: number;
  resolved_at: number | null;
  fingerprint: string;
}

export interface Target {
  labels: Record<string, string>;
  scrape_url: string;
  health: 'up' | 'down' | 'unknown';
  last_scrape: number;
  last_scrape_duration: number;
  last_error: string | null;
  job: string;
  instance: string;
}

export interface ArgusStatus {
  connected: boolean;
  server_url: string;
  version: string;
  active_alerts: number;
  total_targets: number;
  healthy_targets: number;
  last_query: number | null;
}

interface ArgusState {
  status: ArgusStatus | null;
  alerts: Alert[];
  targets: Target[];
  queryResult: QueryResult | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  setServerUrl: (url: string) => Promise<void>;
  fetchAlerts: () => Promise<void>;
  fetchTargets: () => Promise<void>;
  queryInstant: (query: string) => Promise<void>;
  queryRange: (query: string, start: number, end: number, step: number) => Promise<void>;
}

export const useArgusStore = create<ArgusState>((set) => ({
  status: null,
  alerts: [],
  targets: [],
  queryResult: null,
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<ArgusStatus>('get_argus_status');
      set({ status, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  setServerUrl: async (url: string) => {
    try {
      await invoke('set_argus_server_url', { url });
      // Refresh status after URL change
      const status = await invoke<ArgusStatus>('get_argus_status');
      set({ status });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchAlerts: async () => {
    try {
      set({ loading: true, error: null });
      const alerts = await invoke<Alert[]>('get_argus_alerts');
      set({ alerts, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchTargets: async () => {
    try {
      set({ loading: true, error: null });
      const targets = await invoke<Target[]>('get_argus_targets');
      set({ targets, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  queryInstant: async (query: string) => {
    try {
      set({ loading: true, error: null });
      const queryResult = await invoke<QueryResult>('query_argus_metrics', { query });
      set({ queryResult, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  queryRange: async (query: string, start: number, end: number, step: number) => {
    try {
      set({ loading: true, error: null });
      const queryResult = await invoke<QueryResult>('query_argus_metrics_range', {
        query,
        start,
        end,
        step,
      });
      set({ queryResult, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },
}));
