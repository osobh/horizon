/**
 * useMetrics - React hook for real-time metrics updates from Tauri backend
 *
 * Subscribes to Tauri events for system metrics, training progress, and cluster status.
 */

import { useEffect, useState, useCallback } from 'react';
import { listen, UnlistenFn } from '@tauri-apps/api/event';

// Event channel names (must match Rust backend)
const METRICS_UPDATE = 'metrics:update';
const TRAINING_PROGRESS = 'training:progress';
const CLUSTER_STATUS = 'cluster:status';
const SYSTEM_ALERT = 'system:alert';

// Type definitions
export interface MetricsUpdate {
  cpu_usage: number;
  memory_usage: number;
  gpu_usage: number[];
  network_bytes_per_sec: number;
  timestamp: number;
}

export interface TrainingProgressEvent {
  job_id: string;
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  loss: number;
  learning_rate: number;
  samples_per_second: number;
  eta_seconds: number;
}

export interface ClusterStatusEvent {
  connected: boolean;
  node_count: number;
  healthy_nodes: number;
  total_gpus: number;
  active_jobs: number;
}

export interface SystemAlert {
  message: string;
  severity: 'info' | 'warning' | 'error';
  timestamp: number;
}

// History buffer size for charts
const HISTORY_SIZE = 60;

/**
 * Hook for subscribing to system metrics updates.
 */
export function useMetrics() {
  const [metrics, setMetrics] = useState<MetricsUpdate | null>(null);
  const [history, setHistory] = useState<MetricsUpdate[]>([]);

  useEffect(() => {
    let unlisten: UnlistenFn | null = null;

    const setupListener = async () => {
      try {
        unlisten = await listen<MetricsUpdate>(METRICS_UPDATE, (event) => {
          const newMetrics = event.payload;
          setMetrics(newMetrics);
          setHistory((prev) => {
            const updated = [...prev, newMetrics];
            // Keep only the last HISTORY_SIZE entries
            return updated.slice(-HISTORY_SIZE);
          });
        });
      } catch (error) {
        console.error('Failed to subscribe to metrics:', error);
      }
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  return { metrics, history };
}

/**
 * Hook for subscribing to training progress updates.
 */
export function useTrainingProgress() {
  const [progress, setProgress] = useState<Map<string, TrainingProgressEvent>>(new Map());

  useEffect(() => {
    let unlisten: UnlistenFn | null = null;

    const setupListener = async () => {
      try {
        unlisten = await listen<TrainingProgressEvent>(TRAINING_PROGRESS, (event) => {
          setProgress((prev) => {
            const updated = new Map(prev);
            updated.set(event.payload.job_id, event.payload);
            return updated;
          });
        });
      } catch (error) {
        console.error('Failed to subscribe to training progress:', error);
      }
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  const getProgress = useCallback((jobId: string) => progress.get(jobId), [progress]);

  return { progress, getProgress };
}

/**
 * Hook for subscribing to cluster status updates.
 */
export function useClusterStatus() {
  const [status, setStatus] = useState<ClusterStatusEvent | null>(null);

  useEffect(() => {
    let unlisten: UnlistenFn | null = null;

    const setupListener = async () => {
      try {
        unlisten = await listen<ClusterStatusEvent>(CLUSTER_STATUS, (event) => {
          setStatus(event.payload);
        });
      } catch (error) {
        console.error('Failed to subscribe to cluster status:', error);
      }
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  return status;
}

/**
 * Hook for subscribing to system alerts.
 */
export function useSystemAlerts() {
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);

  useEffect(() => {
    let unlisten: UnlistenFn | null = null;

    const setupListener = async () => {
      try {
        unlisten = await listen<SystemAlert>(SYSTEM_ALERT, (event) => {
          setAlerts((prev) => [...prev, event.payload]);
        });
      } catch (error) {
        console.error('Failed to subscribe to system alerts:', error);
      }
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  const dismissAlert = useCallback((index: number) => {
    setAlerts((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  return { alerts, dismissAlert, clearAlerts };
}
