import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types for cost intelligence
export interface CostAttribution {
  id: string;
  name: string;
  type: 'team' | 'project' | 'resource' | 'user';
  cost_usd: number;
  cost_trend_pct: number;
  gpu_hours: number;
  storage_gb: number;
  network_gb: number;
  period: string;
}

export interface CostForecast {
  week: number;
  date: string;
  predicted_cost_usd: number;
  confidence_low: number;
  confidence_high: number;
  trend: 'up' | 'down' | 'stable';
}

export interface BudgetAlert {
  id: string;
  name: string;
  threshold_usd: number;
  current_usd: number;
  percentage_used: number;
  status: 'ok' | 'warning' | 'critical';
  alert_at_pct: number;
}

export interface CostSummary {
  total_cost_usd: number;
  total_cost_trend_pct: number;
  gpu_cost_usd: number;
  storage_cost_usd: number;
  network_cost_usd: number;
  compute_cost_usd: number;
  period: string;
  attributions: CostAttribution[];
  forecasts: CostForecast[];
  alerts: BudgetAlert[];
}

export interface ChargebackReport {
  id: string;
  name: string;
  generated_at: string;
  period_start: string;
  period_end: string;
  total_cost_usd: number;
  format: 'csv' | 'json' | 'pdf';
  download_url: string;
}

interface CostsState {
  summary: CostSummary | null;
  attributions: CostAttribution[];
  forecasts: CostForecast[];
  alerts: BudgetAlert[];
  reports: ChargebackReport[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchSummary: () => Promise<void>;
  fetchAttributions: (type?: string) => Promise<void>;
  fetchForecasts: (weeks?: number) => Promise<void>;
  fetchAlerts: () => Promise<void>;
  fetchReports: () => Promise<void>;
  generateChargebackReport: (periodStart: string, periodEnd: string, format: string) => Promise<ChargebackReport>;
  generateShowbackReport: (teamId: string, periodStart: string, periodEnd: string) => Promise<ChargebackReport>;
  setBudgetThreshold: (name: string, threshold: number, alertAtPct: number) => Promise<void>;
}

export const useCostsStore = create<CostsState>((set, get) => ({
  summary: null,
  attributions: [],
  forecasts: [],
  alerts: [],
  reports: [],
  loading: false,
  error: null,

  fetchSummary: async () => {
    try {
      set({ loading: true, error: null });
      const summary = await invoke<CostSummary>('get_cost_summary');
      set({
        summary,
        attributions: summary.attributions,
        forecasts: summary.forecasts,
        alerts: summary.alerts,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchAttributions: async (type?: string) => {
    try {
      const attributions = await invoke<CostAttribution[]>('get_cost_attributions', { attributionType: type });
      set({ attributions });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchForecasts: async (weeks = 13) => {
    try {
      const forecasts = await invoke<CostForecast[]>('get_cost_forecasts', { weeks });
      set({ forecasts });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchAlerts: async () => {
    try {
      const alerts = await invoke<BudgetAlert[]>('get_budget_alerts');
      set({ alerts });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchReports: async () => {
    try {
      const reports = await invoke<ChargebackReport[]>('list_cost_reports');
      set({ reports });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  generateChargebackReport: async (periodStart: string, periodEnd: string, format: string) => {
    try {
      set({ loading: true, error: null });
      const report = await invoke<ChargebackReport>('generate_chargeback_report', {
        periodStart,
        periodEnd,
        format,
      });
      await get().fetchReports();
      set({ loading: false });
      return report;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  generateShowbackReport: async (teamId: string, periodStart: string, periodEnd: string) => {
    try {
      set({ loading: true, error: null });
      const report = await invoke<ChargebackReport>('generate_showback_report', {
        teamId,
        periodStart,
        periodEnd,
      });
      await get().fetchReports();
      set({ loading: false });
      return report;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  setBudgetThreshold: async (name: string, threshold: number, alertAtPct: number) => {
    try {
      await invoke('set_budget_threshold', { name, thresholdUsd: threshold, alertAtPct });
      await get().fetchAlerts();
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
