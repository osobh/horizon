import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

export interface TrainingMetricPoint {
  epoch: number;
  train_loss: number;
  val_loss?: number;
}

export interface TrainingJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  epoch: number;
  total_epochs: number;
  loss: number;
  metrics: TrainingMetricPoint[];
  // Extended for view
  model_type: string;
  current_epoch: number;
  gpus_allocated: number;
  dataset: string;
  batch_size: number;
  learning_rate: number;
  current_loss?: number;
  best_loss?: number;
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  learning_rate: number;
  timestamp: string;
}

export interface TrainingSummary {
  active_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  gpus_used: number;
  gpus_total: number;
  total_epochs_completed: number;
}

export interface TrainingConfig {
  name: string;
  model: string;
  dataset: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  device: string;
}

export interface TrainingState {
  jobs: TrainingJob[];
  summary: TrainingSummary | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchJobs: () => Promise<void>;
  fetchSummary: () => Promise<void>;
  startTraining: (config: TrainingConfig) => Promise<string>;
  pauseTraining: (jobId: string) => Promise<void>;
  resumeTraining: (jobId: string) => Promise<void>;
  cancelTraining: (jobId: string) => Promise<void>;
  // Aliases for view compatibility
  pauseJob: (jobId: string) => Promise<void>;
  resumeJob: (jobId: string) => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  jobs: [],
  summary: null,
  loading: false,
  error: null,

  fetchJobs: async () => {
    try {
      set({ loading: true, error: null });
      const jobs = await invoke<TrainingJob[]>('list_training_jobs');
      set({ jobs, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchSummary: async () => {
    try {
      const summary = await invoke<TrainingSummary>('get_training_summary');
      set({ summary });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  startTraining: async (config: TrainingConfig) => {
    try {
      set({ loading: true, error: null });
      const jobId = await invoke<string>('start_training', { config });
      // Refresh jobs list
      const jobs = await invoke<TrainingJob[]>('list_training_jobs');
      set({ jobs, loading: false });
      return jobId;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  pauseTraining: async (jobId: string) => {
    try {
      await invoke('pause_training', { jobId });
      const jobs = await invoke<TrainingJob[]>('list_training_jobs');
      set({ jobs });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  resumeTraining: async (jobId: string) => {
    try {
      await invoke('resume_training', { jobId });
      const jobs = await invoke<TrainingJob[]>('list_training_jobs');
      set({ jobs });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  cancelTraining: async (jobId: string) => {
    try {
      await invoke('cancel_training', { jobId });
      const jobs = await invoke<TrainingJob[]>('list_training_jobs');
      set({ jobs });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  // Aliases for view compatibility
  pauseJob: async (jobId: string) => {
    await get().pauseTraining(jobId);
  },

  resumeJob: async (jobId: string) => {
    await get().resumeTraining(jobId);
  },

  cancelJob: async (jobId: string) => {
    await get().cancelTraining(jobId);
  },
}));
