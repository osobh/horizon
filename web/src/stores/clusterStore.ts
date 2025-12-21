import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

interface NodeInfo {
  id: string;
  hostname: string;
  node_type: 'datacenter' | 'workstation' | 'laptop' | 'edge' | 'storage';
  status: 'online' | 'offline' | 'degraded' | 'starting';
  gpu_count: number;
  gpu_memory_gb: number;
  cpu_cores: number;
  memory_gb: number;
}

interface ClusterStatus {
  connected: boolean;
  endpoint: string | null;
  node_count: number;
  total_gpus: number;
  total_memory_gb: number;
  healthy_nodes: number;
}

interface ClusterState {
  connected: boolean;
  endpoint: string | null;
  nodeCount: number;
  totalGpus: number;
  totalMemoryGb: number;
  healthyNodes: number;
  nodes: NodeInfo[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  fetchNodes: () => Promise<void>;
  connect: (endpoint: string) => Promise<void>;
  disconnect: () => Promise<void>;
}

export const useClusterStore = create<ClusterState>((set, get) => ({
  connected: false,
  endpoint: null,
  nodeCount: 0,
  totalGpus: 0,
  totalMemoryGb: 0,
  healthyNodes: 0,
  nodes: [],
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<ClusterStatus>('get_cluster_status');
      set({
        connected: status.connected,
        endpoint: status.endpoint,
        nodeCount: status.node_count,
        totalGpus: status.total_gpus,
        totalMemoryGb: status.total_memory_gb,
        healthyNodes: status.healthy_nodes,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchNodes: async () => {
    try {
      set({ loading: true, error: null });
      const nodes = await invoke<NodeInfo[]>('list_nodes');
      set({ nodes, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  connect: async (endpoint: string) => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<ClusterStatus>('connect_cluster', { endpoint });
      set({
        connected: status.connected,
        endpoint: status.endpoint,
        nodeCount: status.node_count,
        totalGpus: status.total_gpus,
        totalMemoryGb: status.total_memory_gb,
        healthyNodes: status.healthy_nodes,
        loading: false,
      });
      // Fetch nodes after connecting
      await get().fetchNodes();
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  disconnect: async () => {
    try {
      set({ loading: true, error: null });
      await invoke('disconnect_cluster');
      set({
        connected: false,
        endpoint: null,
        nodeCount: 0,
        totalGpus: 0,
        totalMemoryGb: 0,
        healthyNodes: 0,
        nodes: [],
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },
}));
