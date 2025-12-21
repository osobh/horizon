import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

interface TrainingJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  epoch: number;
  total_epochs: number;
  loss: number;
  metrics: Record<string, number>;
}

interface TrainingConfig {
  name: string;
  model: string;
  dataset: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  device: string;
}

interface TrainingState {
  jobs: TrainingJob[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchJobs: () => Promise<void>;
  startTraining: (config: TrainingConfig) => Promise<string>;
  pauseTraining: (jobId: string) => Promise<void>;
  resumeTraining: (jobId: string) => Promise<void>;
  cancelTraining: (jobId: string) => Promise<void>;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  jobs: [],
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
}));
