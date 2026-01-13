/**
 * StratoSwarm Store
 *
 * Zustand store for StratoSwarm agent visualization and evolution tracking.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export interface StratoSwarmStatus {
  connected: boolean;
  cluster_url: string;
  cluster_name: string | null;
  version: string | null;
  last_error: string | null;
}

export type AgentTier = 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond';
export type AgentStatus = 'idle' | 'working' | 'learning' | 'evolving' | 'offline' | 'error';
export type AgentSpecialization = 'general' | 'compute' | 'storage' | 'network' | 'security' | 'analytics' | 'machine_learning';

export interface AgentSkill {
  name: string;
  proficiency: number; // 0.0 - 1.0
  xp: number;
}

export interface SwarmAgent {
  id: string;
  name: string;
  status: AgentStatus;
  tier: AgentTier;
  specialization: AgentSpecialization;
  xp: number;
  xp_to_next_tier: number;
  tasks_completed: number;
  tasks_failed: number;
  success_rate: number;
  uptime_hours: number;
  skills: AgentSkill[];
  current_task: string | null;
  node_id: string | null;
  created_at: string;
  last_active: string;
}

export interface SwarmStats {
  total_agents: number;
  active_agents: number;
  idle_agents: number;
  evolving_agents: number;
  offline_agents: number;
  total_xp: number;
  tasks_completed_today: number;
  tasks_in_progress: number;
  average_success_rate: number;
  tier_distribution: Record<string, number>;
}

export interface EvolutionEvent {
  id: string;
  agent_id: string;
  agent_name: string;
  from_tier: AgentTier;
  to_tier: AgentTier;
  xp_at_evolution: number;
  timestamp: string;
}

export interface AgentTask {
  id: string;
  agent_id: string;
  task_type: string;
  description: string;
  status: string;
  progress: number;
  xp_reward: number;
  started_at: string;
  estimated_completion: string | null;
}

interface StratoSwarmState {
  // Connection state
  status: StratoSwarmStatus;
  isLoading: boolean;
  error: string | null;

  // Data
  stats: SwarmStats | null;
  agents: SwarmAgent[];
  selectedAgent: SwarmAgent | null;
  evolutionEvents: EvolutionEvent[];
  activeTasks: AgentTask[];

  // Filters
  statusFilter: AgentStatus | null;
  tierFilter: AgentTier | null;

  // Actions
  fetchStatus: () => Promise<void>;
  setClusterUrl: (url: string) => Promise<void>;
  fetchStats: () => Promise<void>;
  fetchAgents: (status?: AgentStatus | null, tier?: AgentTier | null) => Promise<void>;
  fetchAgent: (id: string) => Promise<void>;
  fetchEvolutionEvents: (limit?: number) => Promise<void>;
  fetchActiveTasks: () => Promise<void>;
  triggerEvolution: (agentId: string) => Promise<EvolutionEvent>;
  simulateActivity: () => Promise<void>;
  selectAgent: (agent: SwarmAgent | null) => void;
  setStatusFilter: (status: AgentStatus | null) => void;
  setTierFilter: (tier: AgentTier | null) => void;
  clearError: () => void;
}

export const useStratoSwarmStore = create<StratoSwarmState>((set, get) => ({
  // Initial state
  status: {
    connected: false,
    cluster_url: 'http://localhost:9090',
    cluster_name: null,
    version: null,
    last_error: null,
  },
  isLoading: false,
  error: null,
  stats: null,
  agents: [],
  selectedAgent: null,
  evolutionEvents: [],
  activeTasks: [],
  statusFilter: null,
  tierFilter: null,

  // Actions
  fetchStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      const status = await invoke<StratoSwarmStatus>('get_stratoswarm_status');
      set({ status, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  setClusterUrl: async (url: string) => {
    set({ isLoading: true, error: null });
    try {
      await invoke('set_stratoswarm_cluster_url', { url });
      await get().fetchStatus();
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchStats: async () => {
    set({ isLoading: true, error: null });
    try {
      const stats = await invoke<SwarmStats>('get_swarm_stats');
      set({ stats, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchAgents: async (status?: AgentStatus | null, tier?: AgentTier | null) => {
    set({ isLoading: true, error: null });
    try {
      const { statusFilter, tierFilter } = get();
      const agents = await invoke<SwarmAgent[]>('list_swarm_agents', {
        status: status !== undefined ? status : statusFilter,
        tier: tier !== undefined ? tier : tierFilter,
      });
      set({ agents, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchAgent: async (id: string) => {
    set({ isLoading: true, error: null });
    try {
      const agent = await invoke<SwarmAgent>('get_swarm_agent', { id });
      set({ selectedAgent: agent, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchEvolutionEvents: async (limit?: number) => {
    set({ isLoading: true, error: null });
    try {
      const events = await invoke<EvolutionEvent[]>('get_swarm_evolution_events', { limit });
      set({ evolutionEvents: events, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchActiveTasks: async () => {
    set({ isLoading: true, error: null });
    try {
      const tasks = await invoke<AgentTask[]>('get_active_agent_tasks');
      set({ activeTasks: tasks, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  triggerEvolution: async (agentId: string) => {
    set({ isLoading: true, error: null });
    try {
      const event = await invoke<EvolutionEvent>('trigger_agent_evolution', { agentId });
      // Refresh data
      await get().fetchAgents();
      await get().fetchEvolutionEvents();
      await get().fetchStats();
      set({ isLoading: false });
      return event;
    } catch (err) {
      set({ error: String(err), isLoading: false });
      throw err;
    }
  },

  simulateActivity: async () => {
    set({ isLoading: true, error: null });
    try {
      await invoke('simulate_swarm_activity');
      // Refresh data
      await get().fetchAgents();
      await get().fetchStats();
      await get().fetchActiveTasks();
      set({ isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  selectAgent: (agent: SwarmAgent | null) => {
    set({ selectedAgent: agent });
  },

  setStatusFilter: (status: AgentStatus | null) => {
    set({ statusFilter: status });
    get().fetchAgents(status, get().tierFilter);
  },

  setTierFilter: (tier: AgentTier | null) => {
    set({ tierFilter: tier });
    get().fetchAgents(get().statusFilter, tier);
  },

  clearError: () => set({ error: null }),
}));

// Helper functions
export function getTierXpThreshold(tier: AgentTier): number {
  const thresholds: Record<AgentTier, number> = {
    bronze: 0,
    silver: 1000,
    gold: 5000,
    platinum: 15000,
    diamond: 50000,
  };
  return thresholds[tier];
}

export function getNextTier(tier: AgentTier): AgentTier | null {
  const order: AgentTier[] = ['bronze', 'silver', 'gold', 'platinum', 'diamond'];
  const idx = order.indexOf(tier);
  return idx < order.length - 1 ? order[idx + 1] : null;
}

export function getTierColor(tier: AgentTier): string {
  const colors: Record<AgentTier, string> = {
    bronze: '#cd7f32',
    silver: '#c0c0c0',
    gold: '#ffd700',
    platinum: '#e5e4e2',
    diamond: '#b9f2ff',
  };
  return colors[tier];
}
