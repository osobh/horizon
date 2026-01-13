/**
 * RustySpark Store
 *
 * Zustand store for RustySpark data processing job state management.
 */

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export interface RustySparkStatus {
  connected: boolean;
  server_url: string;
  version: string | null;
  last_error: string | null;
}

export type SparkJobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface SparkJob {
  id: string;
  name: string;
  status: SparkJobStatus;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  num_stages: number;
  num_tasks: number;
  completed_stages: number;
  completed_tasks: number;
  input_bytes: number;
  output_bytes: number;
  shuffle_read_bytes: number;
  shuffle_write_bytes: number;
}

export type SparkStageStatus = 'pending' | 'active' | 'complete' | 'failed' | 'skipped';

export interface SparkStage {
  stage_id: number;
  job_id: string;
  name: string;
  status: SparkStageStatus;
  num_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  input_bytes: number;
  output_bytes: number;
  shuffle_read_bytes: number;
  shuffle_write_bytes: number;
  executor_run_time_ms: number;
  executor_cpu_time_ms: number;
}

export type SparkTaskStatus = 'running' | 'success' | 'failed' | 'killed';

export interface SparkTask {
  task_id: number;
  stage_id: number;
  executor_id: string;
  host: string;
  status: SparkTaskStatus;
  launch_time: string;
  finish_time: string | null;
  duration_ms: number | null;
  input_bytes: number;
  output_bytes: number;
  shuffle_read_bytes: number;
  shuffle_write_bytes: number;
  error_message: string | null;
}

export interface SparkSummary {
  total_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  total_stages: number;
  active_stages: number;
  total_tasks: number;
  active_tasks: number;
  total_input_bytes: number;
  total_output_bytes: number;
  total_shuffle_bytes: number;
}

interface RustySparkState {
  // Connection state
  status: RustySparkStatus;
  isLoading: boolean;
  error: string | null;

  // Data
  summary: SparkSummary | null;
  jobs: SparkJob[];
  selectedJob: SparkJob | null;
  stages: SparkStage[];
  tasks: SparkTask[];

  // Filters
  jobStatusFilter: SparkJobStatus | null;
  jobLimit: number;

  // Actions
  fetchStatus: () => Promise<void>;
  setServerUrl: (url: string) => Promise<void>;
  fetchSummary: () => Promise<void>;
  fetchJobs: (status?: SparkJobStatus | null, limit?: number) => Promise<void>;
  fetchJob: (id: string) => Promise<void>;
  fetchJobStages: (jobId: string) => Promise<void>;
  fetchStageTasks: (stageId: number) => Promise<void>;
  cancelJob: (id: string) => Promise<void>;
  selectJob: (job: SparkJob | null) => void;
  setJobStatusFilter: (status: SparkJobStatus | null) => void;
  setJobLimit: (limit: number) => void;
  clearError: () => void;
}

export const useRustySparkStore = create<RustySparkState>((set, get) => ({
  // Initial state
  status: {
    connected: false,
    server_url: 'http://localhost:4040',
    version: null,
    last_error: null,
  },
  isLoading: false,
  error: null,
  summary: null,
  jobs: [],
  selectedJob: null,
  stages: [],
  tasks: [],
  jobStatusFilter: null,
  jobLimit: 50,

  // Actions
  fetchStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      const status = await invoke<RustySparkStatus>('get_rustyspark_status');
      set({ status, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  setServerUrl: async (url: string) => {
    set({ isLoading: true, error: null });
    try {
      await invoke('set_rustyspark_server_url', { url });
      await get().fetchStatus();
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchSummary: async () => {
    set({ isLoading: true, error: null });
    try {
      const summary = await invoke<SparkSummary>('get_rustyspark_summary');
      set({ summary, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchJobs: async (status?: SparkJobStatus | null, limit?: number) => {
    set({ isLoading: true, error: null });
    try {
      const { jobStatusFilter, jobLimit } = get();
      const jobs = await invoke<SparkJob[]>('list_spark_jobs', {
        status: status !== undefined ? status : jobStatusFilter,
        limit: limit !== undefined ? limit : jobLimit,
      });
      set({ jobs, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchJob: async (id: string) => {
    set({ isLoading: true, error: null });
    try {
      const job = await invoke<SparkJob>('get_spark_job', { id });
      set({ selectedJob: job, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchJobStages: async (jobId: string) => {
    set({ isLoading: true, error: null });
    try {
      const stages = await invoke<SparkStage[]>('get_spark_job_stages', { jobId });
      set({ stages, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  fetchStageTasks: async (stageId: number) => {
    set({ isLoading: true, error: null });
    try {
      const tasks = await invoke<SparkTask[]>('get_spark_stage_tasks', { stageId });
      set({ tasks, isLoading: false });
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  cancelJob: async (id: string) => {
    set({ isLoading: true, error: null });
    try {
      await invoke('cancel_spark_job', { id });
      // Refresh jobs list
      await get().fetchJobs();
    } catch (err) {
      set({ error: String(err), isLoading: false });
    }
  },

  selectJob: (job: SparkJob | null) => {
    set({ selectedJob: job, stages: [], tasks: [] });
    if (job) {
      get().fetchJobStages(job.id);
    }
  },

  setJobStatusFilter: (status: SparkJobStatus | null) => {
    set({ jobStatusFilter: status });
    get().fetchJobs(status);
  },

  setJobLimit: (limit: number) => {
    set({ jobLimit: limit });
    get().fetchJobs(undefined, limit);
  },

  clearError: () => set({ error: null }),
}));
