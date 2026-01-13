import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export type EngineType = 'adas' | 'dgm' | 'swarm';
export type EventType = 'generation_complete' | 'new_best_found' | 'design_discovered' | 'self_assessment' | 'code_modification' | 'growth_pattern' | 'population_update' | 'cluster_rebalance';

export interface EvolutionMetrics {
  generation: number;
  total_evaluations: number;
  best_fitness: number;
  average_fitness: number;
  diversity_score: number;
  convergence_rate: number;
  elapsed_secs: number;
  custom_metrics: Record<string, number>;
}

export interface AdasMetrics extends EvolutionMetrics {
  designs_explored: number;
  best_design_score: number;
  current_design: string;
  architecture_complexity: number;
  // Extended for view
  architectures_explored: number;
  best_architecture_score: number;
  search_space_coverage: number;
  mutations_applied: number;
}

export interface DgmMetrics extends EvolutionMetrics {
  self_assessment_score: number;
  code_modifications: number;
  improvement_capability: number;
  growth_patterns: string[];
  recommendations: string[];
  // Extended for view
  total_generations: number;
  mutation_rate: number;
  crossover_rate: number;
  selection_pressure: number;
  diversity_index: number;
  population_size: number;
  elite_count: number;
}

export interface SwarmMetrics extends EvolutionMetrics {
  population_size: number;
  active_particles: number;
  global_best_fitness: number;
  velocity_diversity: number;
  cluster_nodes: number;
  // Extended for view
  active_agents: number;
  tasks_completed: number;
  coordination_score: number;
  average_response_ms: number;
  communication_overhead: number;
}

export interface EvolutionEvent {
  timestamp: string;
  engine: EngineType;
  event_type: EventType;
  description: string;
  metrics_delta: Record<string, number> | null;
  // Extended for view
  id: string;
  generation: number;
}

export interface EngineStatus {
  running: boolean;
  generation: number;
  best_fitness: number;
  improvement_pct: number;
}

export interface EvolutionStatus {
  adas: EngineStatus;
  dgm: EngineStatus;
  swarm: EngineStatus;
  recent_events: EvolutionEvent[];
  // Extended for view
  active: boolean;
  current_generation: number;
  best_fitness: number;
  population_size: number;
}

interface EvolutionState {
  status: EvolutionStatus | null;
  adasMetrics: AdasMetrics | null;
  dgmMetrics: DgmMetrics | null;
  swarmMetrics: SwarmMetrics | null;
  events: EvolutionEvent[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  fetchAdasMetrics: () => Promise<void>;
  fetchDgmMetrics: () => Promise<void>;
  fetchSwarmMetrics: () => Promise<void>;
  fetchEvents: (limit?: number) => Promise<void>;
  simulateStep: () => Promise<void>;
}

export const useEvolutionStore = create<EvolutionState>((set) => ({
  status: null,
  adasMetrics: null,
  dgmMetrics: null,
  swarmMetrics: null,
  events: [],
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<EvolutionStatus>('get_evolution_status');
      set({
        status,
        events: status.recent_events,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchAdasMetrics: async () => {
    try {
      const adasMetrics = await invoke<AdasMetrics>('get_adas_metrics');
      set({ adasMetrics });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchDgmMetrics: async () => {
    try {
      const dgmMetrics = await invoke<DgmMetrics>('get_dgm_metrics');
      set({ dgmMetrics });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchSwarmMetrics: async () => {
    try {
      const swarmMetrics = await invoke<SwarmMetrics>('get_swarm_metrics');
      set({ swarmMetrics });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchEvents: async (limit = 10) => {
    try {
      const events = await invoke<EvolutionEvent[]>('get_evolution_events', { limit });
      set({ events });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  simulateStep: async () => {
    try {
      await invoke('simulate_evolution_step');
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
