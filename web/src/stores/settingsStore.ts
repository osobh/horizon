import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types for settings (governor + quota-manager)
export type PolicyType = 'rbac' | 'abac' | 'quota' | 'rate_limit';
export type PolicyAction = 'allow' | 'deny' | 'require_approval';

export interface Policy {
  id: string;
  name: string;
  description: string;
  type: PolicyType;
  action: PolicyAction;
  conditions: PolicyCondition[];
  enabled: boolean;
  priority: number;
  created_at: string;
  updated_at: string;
  // Extended for view
  policy_type: PolicyType;
}

export interface PolicyCondition {
  field: string;
  operator: 'equals' | 'not_equals' | 'contains' | 'greater_than' | 'less_than' | 'in' | 'not_in';
  value: string | number | string[];
}

export interface Quota {
  id: string;
  name: string;
  resource_type: 'gpu' | 'cpu' | 'memory' | 'storage' | 'network';
  limit: number;
  used: number;
  unit: string;
  scope: 'user' | 'team' | 'project' | 'global';
  scope_id: string | null;
  reset_period: 'hourly' | 'daily' | 'weekly' | 'monthly' | 'never';
  enabled: boolean;
  // Extended for view
  current_usage: number;
  soft_limit?: number;
}

export interface AppSettings {
  theme: 'light' | 'dark' | 'system';
  notifications_enabled: boolean;
  auto_refresh_interval_secs: number;
  default_cluster: string | null;
  telemetry_enabled: boolean;
  log_level: 'debug' | 'info' | 'warn' | 'error';
  // Extended for view
  auto_refresh: boolean;
  refresh_interval_seconds: number;
  sound_enabled: boolean;
}

export interface SettingsSummary {
  policies: Policy[];
  quotas: Quota[];
  app_settings: AppSettings;
  policy_evaluation_count: number;
  quota_violations_count: number;
  // Extended for view
  active_policies: number;
  total_policies: number;
  total_quotas: number;
  theme: 'light' | 'dark' | 'system';
  auto_refresh: boolean;
}

interface SettingsState {
  summary: SettingsSummary | null;
  policies: Policy[];
  quotas: Quota[];
  appSettings: AppSettings | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchSummary: () => Promise<void>;
  fetchPolicies: () => Promise<void>;
  fetchQuotas: () => Promise<void>;
  fetchAppSettings: () => Promise<void>;
  createPolicy: (policy: Omit<Policy, 'id' | 'created_at' | 'updated_at'>) => Promise<Policy>;
  updatePolicy: (id: string, policy: Partial<Policy>) => Promise<void>;
  deletePolicy: (id: string) => Promise<void>;
  togglePolicy: (id: string, enabled: boolean) => Promise<void>;
  setQuota: (quota: Omit<Quota, 'id' | 'used'>) => Promise<Quota>;
  updateQuota: (id: string, quota: Partial<Quota>) => Promise<void>;
  deleteQuota: (id: string) => Promise<void>;
  updateAppSettings: (settings: Partial<AppSettings>) => Promise<void>;
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  summary: null,
  policies: [],
  quotas: [],
  appSettings: null,
  loading: false,
  error: null,

  fetchSummary: async () => {
    try {
      set({ loading: true, error: null });
      const summary = await invoke<SettingsSummary>('get_settings_summary');
      set({
        summary,
        policies: summary.policies,
        quotas: summary.quotas,
        appSettings: summary.app_settings,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchPolicies: async () => {
    try {
      const policies = await invoke<Policy[]>('get_policies');
      set({ policies });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchQuotas: async () => {
    try {
      const quotas = await invoke<Quota[]>('get_quotas');
      set({ quotas });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchAppSettings: async () => {
    try {
      const appSettings = await invoke<AppSettings>('get_app_settings');
      set({ appSettings });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  createPolicy: async (policy) => {
    try {
      set({ loading: true, error: null });
      const newPolicy = await invoke<Policy>('create_policy', { policy });
      await get().fetchPolicies();
      set({ loading: false });
      return newPolicy;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  updatePolicy: async (id: string, policy: Partial<Policy>) => {
    try {
      await invoke('update_policy', { id, policy });
      await get().fetchPolicies();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  deletePolicy: async (id: string) => {
    try {
      await invoke('delete_policy', { id });
      await get().fetchPolicies();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  togglePolicy: async (id: string, enabled: boolean) => {
    try {
      await invoke('toggle_policy', { id, enabled });
      await get().fetchPolicies();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  setQuota: async (quota) => {
    try {
      set({ loading: true, error: null });
      const newQuota = await invoke<Quota>('set_quota', { quota });
      await get().fetchQuotas();
      set({ loading: false });
      return newQuota;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  updateQuota: async (id: string, quota: Partial<Quota>) => {
    try {
      await invoke('update_quota', { id, quota });
      await get().fetchQuotas();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  deleteQuota: async (id: string) => {
    try {
      await invoke('delete_quota', { id });
      await get().fetchQuotas();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  updateAppSettings: async (settings: Partial<AppSettings>) => {
    try {
      await invoke('update_app_settings', { settings });
      await get().fetchAppSettings();
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
