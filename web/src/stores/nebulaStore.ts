import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export type MeshNodeType = 'gpu_server' | 'workstation' | 'edge' | 'relay';
export type ConnectionType = 'rdma' | 'roce' | 'tcp' | 'local';
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

export interface RdmaStats {
  peak_bandwidth_gbps: number;
  current_bandwidth_gbps: number;
  avg_latency_ns: number;
  min_latency_ns: number;
  bytes_transferred: number;
  operations_completed: number;
  active_connections: number;
  gpu_direct_enabled: boolean;
  queue_pairs: number;
}

export interface ZkStats {
  proofs_generated: number;
  proofs_per_second: number;
  gpu_utilization: number;
  cpu_utilization: number;
  gpu_proof_ratio: number;
  avg_proof_time_ms: number;
  total_verification_time_ms: number;
  distributed_enabled: boolean;
  coordinators: number;
  proof_types: string[];
}

export interface MeshNode {
  id: string;
  hostname: string;
  ip_address: string;
  node_type: MeshNodeType;
  gpu_count: number;
  rdma_capable: boolean;
  zk_enabled: boolean;
  load: number;
  peer_count: number;
}

export interface MeshConnection {
  source: string;
  target: string;
  connection_type: ConnectionType;
  bandwidth_gbps: number;
  latency_us: number;
  utilization: number;
}

export interface MeshTopology {
  nodes: MeshNode[];
  connections: MeshConnection[];
  total_bandwidth_gbps: number;
  active_transfers: number;
}

export interface NebulaStatus {
  rdma: RdmaStats;
  zk: ZkStats;
  topology: MeshTopology;
  health: HealthStatus;
}

interface NebulaState {
  status: NebulaStatus | null;
  rdmaStats: RdmaStats | null;
  zkStats: ZkStats | null;
  topology: MeshTopology | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  fetchRdmaStats: () => Promise<void>;
  fetchZkStats: () => Promise<void>;
  fetchTopology: () => Promise<void>;
  simulateActivity: () => Promise<void>;
}

export const useNebulaStore = create<NebulaState>((set) => ({
  status: null,
  rdmaStats: null,
  zkStats: null,
  topology: null,
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<NebulaStatus>('get_nebula_status');
      set({
        status,
        rdmaStats: status.rdma,
        zkStats: status.zk,
        topology: status.topology,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchRdmaStats: async () => {
    try {
      const rdmaStats = await invoke<RdmaStats>('get_rdma_stats');
      set({ rdmaStats });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchZkStats: async () => {
    try {
      const zkStats = await invoke<ZkStats>('get_zk_stats');
      set({ zkStats });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchTopology: async () => {
    try {
      const topology = await invoke<MeshTopology>('get_mesh_topology');
      set({ topology });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  simulateActivity: async () => {
    try {
      await invoke('simulate_nebula_activity');
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
