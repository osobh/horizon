import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export interface GpuInfo {
  index: number;
  name: string;
  vendor: string;
  memory_total: number;
  memory_available: number;
  compute_units: number;
  is_primary: boolean;
}

export interface SchedulerStats {
  total_gpus: number;
  available_gpus: number;
  queued_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  tenant_count: number;
}

export interface FairShareInfo {
  tenant_name: string;
  priority_weight: number;
  max_gpus: number;
  current_gpus: number;
  queued_jobs: number;
  running_jobs: number;
}

export interface TenantInfo {
  id: string;
  name: string;
  priority_weight: number;
  max_gpus: number;
  max_concurrent_jobs: number;
  current_gpus: number;
  status: string;
}

export interface JobInfo {
  id: string;
  name: string;
  tenant_id: string;
  status: string;
  priority: string;
  gpus_requested: number;
  submitted_at: number;
  started_at: number | null;
  assigned_gpus: number[];
}

export interface JobList {
  queued: JobInfo[];
  running: JobInfo[];
  completed: JobInfo[];
}

export interface SubmitJobInput {
  name: string;
  tenant_id: string;
  gpus_requested: number;
  priority?: string;
  estimated_duration_secs?: number;
}

interface SchedulerState {
  stats: SchedulerStats | null;
  gpus: GpuInfo[];
  jobs: JobList | null;
  tenants: TenantInfo[];
  fairShare: FairShareInfo[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchStats: () => Promise<void>;
  fetchGpus: () => Promise<void>;
  fetchJobs: () => Promise<void>;
  fetchTenants: () => Promise<void>;
  fetchFairShare: () => Promise<void>;
  submitJob: (input: SubmitJobInput) => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;
}

export const useSchedulerStore = create<SchedulerState>((set, get) => ({
  stats: null,
  gpus: [],
  jobs: null,
  tenants: [],
  fairShare: [],
  loading: false,
  error: null,

  fetchStats: async () => {
    try {
      set({ loading: true, error: null });
      const stats = await invoke<SchedulerStats>('get_scheduler_summary');
      set({ stats, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchGpus: async () => {
    try {
      const gpus = await invoke<GpuInfo[]>('scheduler_list_gpus');
      set({ gpus });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchJobs: async () => {
    try {
      const allJobs = await invoke<JobInfo[]>('list_scheduler_jobs');
      const jobs: JobList = {
        queued: allJobs.filter(j => j.status === 'queued'),
        running: allJobs.filter(j => j.status === 'running'),
        completed: allJobs.filter(j => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled'),
      };
      set({ jobs });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchTenants: async () => {
    try {
      const tenants = await invoke<TenantInfo[]>('list_tenants');
      set({ tenants });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchFairShare: async () => {
    try {
      const fairShare = await invoke<FairShareInfo[]>('get_fair_share');
      set({ fairShare });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  submitJob: async (input: SubmitJobInput) => {
    try {
      await invoke('submit_scheduler_job', { input });
      await get().fetchJobs();
      await get().fetchStats();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  cancelJob: async (jobId: string) => {
    try {
      await invoke('cancel_scheduler_job', { jobId });
      await get().fetchJobs();
      await get().fetchStats();
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));
