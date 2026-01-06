import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export type RoutingReason = 'best_availability' | 'data_affinity' | 'lowest_utilization' | 'closest_proximity' | 'label_match' | 'low_carbon' | 'load_balance';
export type HealthStatus = 'healthy' | 'degraded' | 'at_risk' | 'failed' | 'unknown';

export interface ProtocolStats {
  source_protocol: string;
  target_protocol: string;
  request_count: number;
  avg_latency_ms: number;
  success_rate_pct: number;
}

export interface RoutingDecision {
  request_id: string;
  path: string;
  source_protocol: string;
  target_protocol: string;
  target_node: string;
  reason: RoutingReason;
  decision_latency_ms: number;
  timestamp: number;
}

export interface BackendHealth {
  node_id: string;
  hostname: string;
  status: HealthStatus;
  gpu_utilization_pct: number;
  memory_utilization_pct: number;
  cpu_utilization_pct: number;
  active_jobs: number;
  failure_probability: number;
  last_heartbeat_secs: number;
}

export interface EdgeProxyStatus {
  active_connections: number;
  requests_per_second: number;
  protocols: ProtocolStats[];
  routing_decisions: RoutingDecision[];
  backend_health: BackendHealth[];
  uptime_seconds: number;
  total_requests: number;
}

export interface BrainStatus {
  registered_nodes: number;
  healthy_gpus: number;
  at_risk_gpus: number;
  failed_gpus: number;
  predictions_made: number;
  migrations_triggered: number;
  jobs_saved: number;
  model_accuracy_pct: number;
  active_monitors: number;
}

export interface FailurePrediction {
  gpu_id: string;
  node_id: string;
  probability: number;
  estimated_ttf_secs: number | null;
  primary_factor: string;
  recommended_action: string;
  jobs_at_risk: number;
}

export interface EdgeProxyBrainStatus {
  proxy: EdgeProxyStatus;
  brain: BrainStatus;
  predictions: FailurePrediction[];
}

interface EdgeProxyState {
  status: EdgeProxyBrainStatus | null;
  proxyStatus: EdgeProxyStatus | null;
  brainStatus: BrainStatus | null;
  predictions: FailurePrediction[];
  backendHealth: BackendHealth[];
  routingDecisions: RoutingDecision[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  fetchProxyStatus: () => Promise<void>;
  fetchBrainStatus: () => Promise<void>;
  fetchPredictions: () => Promise<void>;
  fetchBackendHealth: () => Promise<void>;
  fetchRoutingDecisions: () => Promise<void>;
  simulateActivity: () => Promise<void>;
}

export const useEdgeProxyStore = create<EdgeProxyState>((set) => ({
  status: null,
  proxyStatus: null,
  brainStatus: null,
  predictions: [],
  backendHealth: [],
  routingDecisions: [],
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<EdgeProxyBrainStatus>('get_edge_proxy_status');
      set({
        status,
        proxyStatus: status.proxy,
        brainStatus: status.brain,
        predictions: status.predictions,
        backendHealth: status.proxy.backend_health,
        routingDecisions: status.proxy.routing_decisions,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchProxyStatus: async () => {
    try {
      const proxyStatus = await invoke<EdgeProxyStatus>('get_proxy_status');
      set({
        proxyStatus,
        backendHealth: proxyStatus.backend_health,
        routingDecisions: proxyStatus.routing_decisions,
      });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchBrainStatus: async () => {
    try {
      const brainStatus = await invoke<BrainStatus>('get_brain_status');
      set({ brainStatus });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchPredictions: async () => {
    try {
      const predictions = await invoke<FailurePrediction[]>('get_failure_predictions');
      set({ predictions });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchBackendHealth: async () => {
    try {
      const backendHealth = await invoke<BackendHealth[]>('get_backend_health');
      set({ backendHealth });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchRoutingDecisions: async () => {
    try {
      const routingDecisions = await invoke<RoutingDecision[]>('get_routing_decisions');
      set({ routingDecisions });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  simulateActivity: async () => {
    try {
      await invoke('simulate_edge_proxy_activity');
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
