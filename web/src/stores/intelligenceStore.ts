import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types for executive intelligence
export type ResourceStatus = 'idle' | 'underutilized' | 'optimal' | 'overutilized';
export type Severity = 'info' | 'warning' | 'critical';

export interface IdleResource {
  id: string;
  name: string;
  type: 'gpu' | 'cpu' | 'storage' | 'network';
  node_id: string;
  hostname: string;
  idle_since: string;
  idle_hours: number;
  potential_savings_usd: number;
  status: ResourceStatus;
  recommended_action: string;
  // Extended for view
  resource_type: 'gpu' | 'cpu' | 'storage' | 'network';
  wasted_cost_usd: number;
  recommendation: string;
}

export interface ProfitMargin {
  id: string;
  name: string;
  type: 'service' | 'tenant' | 'project';
  revenue_usd: number;
  cost_usd: number;
  margin_usd: number;
  margin_pct: number;
  trend_pct: number;
  period: string;
  // Extended for view
  service_name: string;
}

export interface VendorUtilization {
  vendor_id: string;
  vendor_name: string;
  contract_value_usd: number;
  used_value_usd: number;
  utilization_pct: number;
  contract_end: string;
  days_remaining: number;
  status: 'underutilized' | 'optimal' | 'overutilized' | 'expiring';
  recommendations: string[];
  // Extended for view
  id: string;
  contract_type: string;
  cost_usd: number;
}

export interface ExecutiveKpi {
  name: string;
  value: number;
  unit: string;
  trend_pct: number;
  trend_direction: 'up' | 'down' | 'stable';
  target: number | null;
  status: 'on_track' | 'at_risk' | 'off_track';
  // Extended for view
  id: string;
  trend: 'up' | 'down' | 'stable';
  change_pct: number;
}

// Alias for view compatibility
export type ExecutiveKPI = ExecutiveKpi;

export interface IntelligenceSummary {
  idle_resources: IdleResource[];
  total_potential_savings_usd: number;
  profit_margins: ProfitMargin[];
  overall_margin_pct: number;
  vendor_utilizations: VendorUtilization[];
  kpis: ExecutiveKpi[];
  alerts: IntelligenceAlert[];
  // Extended for view
  overall_efficiency: number;
  idle_resource_count: number;
  potential_savings_usd: number;
  average_margin: number;
}

export interface IntelligenceAlert {
  id: string;
  title: string;
  description: string;
  severity: Severity;
  source: 'efficiency' | 'margin' | 'vendor' | 'executive';
  created_at: string;
  acknowledged: boolean;
  // Extended for view
  message: string;
}

interface IntelligenceState {
  summary: IntelligenceSummary | null;
  idleResources: IdleResource[];
  profitMargins: ProfitMargin[];
  vendorUtilizations: VendorUtilization[];
  kpis: ExecutiveKpi[];
  executiveKpis: ExecutiveKpi[]; // Alias for kpis
  alerts: IntelligenceAlert[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchSummary: () => Promise<void>;
  fetchIdleResources: () => Promise<void>;
  fetchProfitMargins: (type?: string) => Promise<void>;
  fetchVendorUtilizations: () => Promise<void>;
  fetchKpis: () => Promise<void>;
  fetchExecutiveKpis: () => Promise<void>; // Alias for fetchKpis
  fetchAlerts: () => Promise<void>;
  acknowledgeAlert: (alertId: string) => Promise<void>;
  terminateIdleResource: (resourceId: string) => Promise<void>;
}

export const useIntelligenceStore = create<IntelligenceState>((set, get) => ({
  summary: null,
  idleResources: [],
  profitMargins: [],
  vendorUtilizations: [],
  kpis: [],
  executiveKpis: [],
  alerts: [],
  loading: false,
  error: null,

  fetchSummary: async () => {
    try {
      set({ loading: true, error: null });
      const summary = await invoke<IntelligenceSummary>('get_intelligence_summary');
      set({
        summary,
        idleResources: summary.idle_resources,
        profitMargins: summary.profit_margins,
        vendorUtilizations: summary.vendor_utilizations,
        kpis: summary.kpis,
        executiveKpis: summary.kpis,
        alerts: summary.alerts,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchIdleResources: async () => {
    try {
      const idleResources = await invoke<IdleResource[]>('get_idle_resources');
      set({ idleResources });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchProfitMargins: async (type?: string) => {
    try {
      const profitMargins = await invoke<ProfitMargin[]>('get_profit_margins', { marginType: type });
      set({ profitMargins });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchVendorUtilizations: async () => {
    try {
      const vendorUtilizations = await invoke<VendorUtilization[]>('get_vendor_utilizations');
      set({ vendorUtilizations });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchKpis: async () => {
    try {
      const kpis = await invoke<ExecutiveKpi[]>('get_executive_kpis');
      set({ kpis, executiveKpis: kpis });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchExecutiveKpis: async () => {
    await get().fetchKpis();
  },

  fetchAlerts: async () => {
    try {
      const alerts = await invoke<IntelligenceAlert[]>('get_intelligence_alerts');
      set({ alerts });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  acknowledgeAlert: async (alertId: string) => {
    try {
      await invoke('acknowledge_intelligence_alert', { alertId });
      await get().fetchAlerts();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  terminateIdleResource: async (resourceId: string) => {
    try {
      await invoke('terminate_idle_resource', { resourceId });
      await get().fetchIdleResources();
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
