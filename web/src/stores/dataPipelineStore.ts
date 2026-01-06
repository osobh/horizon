import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export type StageStatus = 'idle' | 'processing' | 'waiting' | 'backpressure';
export type TransferDirection = 'upload' | 'download';
export type TransferOperation = 'reading' | 'encrypting' | 'hashing' | 'transmitting' | 'decrypting' | 'writing';

export interface PipelineStats {
  bytes_processed: number;
  encryption_throughput_gbps: number;
  hashing_throughput_gbps: number;
  pipeline_utilization_pct: number;
  gpu_memory_used_bytes: number;
  active_streams: number;
  pinned_memory_bytes: number;
  backend: string;
}

export interface StageStats {
  stage_name: string;
  bytes_processed: number;
  throughput_gbps: number;
  latency_ms: number;
  buffer_fill_pct: number;
  status: StageStatus;
}

export interface TransferJob {
  id: string;
  source_name: string;
  total_bytes: number;
  processed_bytes: number;
  direction: TransferDirection;
  operation: TransferOperation;
  current_throughput_gbps: number;
  eta_seconds: number;
  gpu_accelerated: boolean;
}

export interface DataPipelineStatus {
  stats: PipelineStats;
  stages: StageStats[];
  active_jobs: TransferJob[];
  capacity_gbps: number;
  encryption_algo: string;
  hash_algo: string;
}

interface DataPipelineState {
  status: DataPipelineStatus | null;
  stats: PipelineStats | null;
  stages: StageStats[];
  activeJobs: TransferJob[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  fetchStats: () => Promise<void>;
  fetchStages: () => Promise<void>;
  fetchJobs: () => Promise<void>;
  simulateActivity: () => Promise<void>;
}

export const useDataPipelineStore = create<DataPipelineState>((set) => ({
  status: null,
  stats: null,
  stages: [],
  activeJobs: [],
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<DataPipelineStatus>('get_data_pipeline_status');
      set({
        status,
        stats: status.stats,
        stages: status.stages,
        activeJobs: status.active_jobs,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchStats: async () => {
    try {
      const stats = await invoke<PipelineStats>('get_pipeline_stats');
      set({ stats });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchStages: async () => {
    try {
      const stages = await invoke<StageStats[]>('get_pipeline_stages');
      set({ stages });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchJobs: async () => {
    try {
      const activeJobs = await invoke<TransferJob[]>('get_pipeline_jobs');
      set({ activeJobs });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  simulateActivity: async () => {
    try {
      await invoke('simulate_pipeline_activity');
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
