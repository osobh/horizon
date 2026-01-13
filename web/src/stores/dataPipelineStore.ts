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
  // Extended properties for the view
  active_pipelines: number;
  records_processed: number;
  throughput_records_per_sec: number;
  gpu_acceleration_enabled: boolean;
  average_latency_ms: number;
  gpu_speedup_factor: number;
}

// Pipeline stage type for the view
export interface PipelineStage {
  id: string;
  name: string;
  stage_type: string;
  health: 'healthy' | 'degraded' | 'unhealthy';
  records_in: number;
  records_out: number;
  latency_ms: number;
  errors: number;
}

// Pipeline job type for the view
export interface PipelineJob {
  id: string;
  name: string;
  pipeline_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  records_processed: number;
  started_at: string;
  duration_seconds: number | null;
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

export interface DataPipelineState {
  status: DataPipelineStatus | null;
  stats: PipelineStats | null;
  stages: PipelineStage[];
  jobs: PipelineJob[];
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

// Helper to convert StageStats to PipelineStage
function convertToStage(stage: StageStats, index: number): PipelineStage {
  return {
    id: `stage-${index}`,
    name: stage.stage_name,
    stage_type: stage.status,
    health: stage.status === 'backpressure' ? 'degraded' : 'healthy',
    records_in: Math.floor(stage.bytes_processed / 100),
    records_out: Math.floor(stage.bytes_processed / 100),
    latency_ms: stage.latency_ms,
    errors: 0,
  };
}

// Helper to convert TransferJob to PipelineJob
function convertToJob(job: TransferJob): PipelineJob {
  const progress = job.total_bytes > 0 ? (job.processed_bytes / job.total_bytes) * 100 : 0;
  return {
    id: job.id,
    name: job.source_name,
    pipeline_id: `pipeline-${job.direction}`,
    status: progress >= 100 ? 'completed' : 'running',
    progress,
    records_processed: Math.floor(job.processed_bytes / 100),
    started_at: new Date().toISOString(),
    duration_seconds: job.eta_seconds > 0 ? Math.floor(job.eta_seconds) : null,
  };
}

// Helper to extend PipelineStats with view-required properties
function extendStats(stats: Partial<PipelineStats>): PipelineStats {
  return {
    bytes_processed: stats.bytes_processed ?? 0,
    encryption_throughput_gbps: stats.encryption_throughput_gbps ?? 0,
    hashing_throughput_gbps: stats.hashing_throughput_gbps ?? 0,
    pipeline_utilization_pct: stats.pipeline_utilization_pct ?? 0,
    gpu_memory_used_bytes: stats.gpu_memory_used_bytes ?? 0,
    active_streams: stats.active_streams ?? 0,
    pinned_memory_bytes: stats.pinned_memory_bytes ?? 0,
    backend: stats.backend ?? 'cpu',
    active_pipelines: stats.active_streams ?? 0,
    records_processed: Math.floor((stats.bytes_processed ?? 0) / 100),
    throughput_records_per_sec: Math.floor(((stats.encryption_throughput_gbps ?? 0) * 1e9) / 100),
    gpu_acceleration_enabled: stats.backend === 'cuda' || stats.backend === 'metal',
    average_latency_ms: 5.0,
    gpu_speedup_factor: stats.backend === 'cuda' || stats.backend === 'metal' ? 8.5 : 1.0,
  };
}

export const useDataPipelineStore = create<DataPipelineState>((set) => ({
  status: null,
  stats: null,
  stages: [],
  jobs: [],
  activeJobs: [],
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      set({ loading: true, error: null });
      const status = await invoke<DataPipelineStatus>('get_data_pipeline_status');
      set({
        status,
        stats: extendStats(status.stats),
        stages: status.stages.map(convertToStage),
        jobs: status.active_jobs.map(convertToJob),
        activeJobs: status.active_jobs,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchStats: async () => {
    try {
      const rawStats = await invoke<PipelineStats>('get_pipeline_stats');
      set({ stats: extendStats(rawStats) });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchStages: async () => {
    try {
      const rawStages = await invoke<StageStats[]>('get_pipeline_stages');
      set({ stages: rawStages.map(convertToStage) });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchJobs: async () => {
    try {
      const activeJobs = await invoke<TransferJob[]>('get_pipeline_jobs');
      set({
        activeJobs,
        jobs: activeJobs.map(convertToJob),
      });
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
