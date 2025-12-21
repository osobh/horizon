/**
 * DataPipeline - GPU-Accelerated Data Transfer Visualization (Synergy 4)
 *
 * Visualizes WARP's GPU-accelerated encryption and transfer pipeline:
 * - ChaCha20-Poly1305 GPU encryption at 20+ GB/s
 * - Blake3 GPU hashing at 15-20 GB/s
 * - Triple-buffer streaming pipeline
 * - Pinned memory for zero-copy DMA
 */

import { useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Shield,
  RefreshCw,
  Activity,
  ArrowRight,
  Zap,
  HardDrive,
  Lock,
  Hash,
  Upload,
  Download,
} from 'lucide-react';

interface PipelineStats {
  bytes_processed: number;
  encryption_throughput_gbps: number;
  hashing_throughput_gbps: number;
  pipeline_utilization_pct: number;
  gpu_memory_used_bytes: number;
  active_streams: number;
  pinned_memory_bytes: number;
  backend: string;
}

interface StageStats {
  stage_name: string;
  bytes_processed: number;
  throughput_gbps: number;
  latency_ms: number;
  buffer_fill_pct: number;
  status: 'idle' | 'processing' | 'waiting' | 'backpressure';
}

interface TransferJob {
  id: string;
  source_name: string;
  total_bytes: number;
  processed_bytes: number;
  direction: 'upload' | 'download';
  operation: 'reading' | 'encrypting' | 'hashing' | 'transmitting' | 'decrypting' | 'writing';
  current_throughput_gbps: number;
  eta_seconds: number;
  gpu_accelerated: boolean;
}

interface DataPipelineStatus {
  stats: PipelineStats;
  stages: StageStats[];
  active_jobs: TransferJob[];
  capacity_gbps: number;
  encryption_algo: string;
  hash_algo: string;
}

const STATUS_COLORS = {
  idle: 'bg-slate-500',
  processing: 'bg-green-500',
  waiting: 'bg-yellow-500',
  backpressure: 'bg-red-500',
};

const OPERATION_ICONS = {
  reading: HardDrive,
  encrypting: Lock,
  hashing: Hash,
  transmitting: Zap,
  decrypting: Lock,
  writing: HardDrive,
};

interface DataPipelineProps {
  compact?: boolean;
}

export default function DataPipeline({ compact = false }: DataPipelineProps) {
  const [status, setStatus] = useState<DataPipelineStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const data = await invoke<DataPipelineStatus>('get_data_pipeline_status');
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const simulateActivity = async () => {
    try {
      await invoke('simulate_pipeline_activity');
      await fetchData();
    } catch (err) {
      console.error('Failed to simulate activity:', err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin text-slate-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-900/20 border border-red-700/50 rounded-lg">
        <p className="text-sm text-red-400">Failed to load data pipeline: {error}</p>
        <button
          onClick={fetchData}
          className="mt-2 text-xs text-red-300 hover:text-red-200"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!status) {
    return null;
  }

  const formatBytes = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }
    return `${value.toFixed(1)} ${units[unitIndex]}`;
  };

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Shield className="w-4 h-4 text-emerald-400" />
          <span className="text-slate-300">GPU Pipeline</span>
          <span className="text-xs bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded">
            {status.stats.encryption_throughput_gbps.toFixed(1)} GB/s
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded bg-emerald-900/30 border border-emerald-700/50">
            <div className="text-emerald-400">Encryption</div>
            <div className="font-mono">{status.stats.encryption_throughput_gbps.toFixed(1)} GB/s</div>
          </div>
          <div className="p-2 rounded bg-blue-900/30 border border-blue-700/50">
            <div className="text-blue-400">Hashing</div>
            <div className="font-mono">{status.stats.hashing_throughput_gbps.toFixed(1)} GB/s</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-emerald-400" />
          <h3 className="font-medium">GPU Data Pipeline</h3>
          <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded">
            {status.encryption_algo}
          </span>
          <span className="text-xs text-slate-400">
            {status.stats.backend}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={simulateActivity}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-emerald-600 hover:bg-emerald-500 rounded transition-colors"
          >
            <Activity className="w-3 h-3" />
            Simulate
          </button>
          <button
            onClick={fetchData}
            className="p-1.5 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-300"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Throughput Stats */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-emerald-900/20 rounded-lg border border-emerald-700/50 p-3 text-center">
          <div className="text-xs text-emerald-400 mb-1">Encryption</div>
          <div className="text-2xl font-mono font-bold text-emerald-400">
            {status.stats.encryption_throughput_gbps.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500">GB/s</div>
        </div>
        <div className="bg-blue-900/20 rounded-lg border border-blue-700/50 p-3 text-center">
          <div className="text-xs text-blue-400 mb-1">Hashing</div>
          <div className="text-2xl font-mono font-bold text-blue-400">
            {status.stats.hashing_throughput_gbps.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500">GB/s</div>
        </div>
        <div className="bg-purple-900/20 rounded-lg border border-purple-700/50 p-3 text-center">
          <div className="text-xs text-purple-400 mb-1">Utilization</div>
          <div className="text-2xl font-mono font-bold text-purple-400">
            {status.stats.pipeline_utilization_pct.toFixed(0)}%
          </div>
          <div className="text-xs text-slate-500">pipeline</div>
        </div>
        <div className="bg-cyan-900/20 rounded-lg border border-cyan-700/50 p-3 text-center">
          <div className="text-xs text-cyan-400 mb-1">GPU Memory</div>
          <div className="text-lg font-mono font-bold text-cyan-400">
            {formatBytes(status.stats.gpu_memory_used_bytes)}
          </div>
          <div className="text-xs text-slate-500">{status.stats.active_streams} streams</div>
        </div>
      </div>

      {/* Triple-Buffer Pipeline Visualization */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
          <Zap className="w-4 h-4" />
          Triple-Buffer Pipeline
        </div>
        <div className="flex items-center justify-between gap-2">
          {status.stages.map((stage, i) => (
            <div key={stage.stage_name} className="flex items-center flex-1">
              <div className="flex-1 bg-slate-700/50 rounded-lg p-3 border border-slate-600">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium truncate">{stage.stage_name}</span>
                  <div className={`w-2 h-2 rounded-full ${STATUS_COLORS[stage.status]}`} />
                </div>
                <div className="text-lg font-mono font-bold text-white">
                  {stage.throughput_gbps.toFixed(1)} GB/s
                </div>
                <div className="text-xs text-slate-400">{stage.latency_ms.toFixed(2)} ms</div>
                {/* Buffer fill bar */}
                <div className="mt-2 h-1.5 bg-slate-600 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${stage.status === 'backpressure' ? 'bg-red-500' : 'bg-emerald-500'}`}
                    style={{ width: `${stage.buffer_fill_pct}%` }}
                  />
                </div>
                <div className="text-xs text-slate-500 mt-1">{stage.buffer_fill_pct.toFixed(0)}% buffer</div>
              </div>
              {i < status.stages.length - 1 && (
                <ArrowRight className="w-4 h-4 text-slate-500 mx-1 flex-shrink-0" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Active Transfer Jobs */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
          <Activity className="w-4 h-4" />
          Active Transfers ({status.active_jobs.length})
        </div>
        <div className="space-y-2">
          {status.active_jobs.map((job) => {
            const progress = (job.processed_bytes / job.total_bytes) * 100;
            const OpIcon = OPERATION_ICONS[job.operation];
            return (
              <div key={job.id} className="bg-slate-700/50 rounded p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {job.direction === 'upload' ? (
                      <Upload className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Download className="w-4 h-4 text-blue-400" />
                    )}
                    <span className="text-sm truncate max-w-[200px]">{job.source_name}</span>
                    <span className="text-xs bg-slate-600 px-1.5 py-0.5 rounded flex items-center gap-1">
                      <OpIcon className="w-3 h-3" />
                      {job.operation}
                    </span>
                    {job.gpu_accelerated && (
                      <span className="text-xs bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded">
                        GPU
                      </span>
                    )}
                  </div>
                  <span className="text-sm font-mono text-slate-300">
                    {job.current_throughput_gbps.toFixed(1)} GB/s
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-slate-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-blue-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-400 w-24 text-right">
                    {formatBytes(job.processed_bytes)} / {formatBytes(job.total_bytes)}
                  </span>
                </div>
                {job.eta_seconds > 0 && (
                  <div className="text-xs text-slate-500 mt-1">
                    ETA: {job.eta_seconds.toFixed(1)}s
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Pipeline Info */}
      <div className="flex items-center gap-4 text-xs text-slate-400">
        <div className="flex items-center gap-1">
          <Lock className="w-3 h-3" />
          <span>{status.encryption_algo}</span>
        </div>
        <div className="flex items-center gap-1">
          <Hash className="w-3 h-3" />
          <span>{status.hash_algo}</span>
        </div>
        <div className="flex items-center gap-1">
          <span>Pinned: {formatBytes(status.stats.pinned_memory_bytes)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span>Capacity: {status.capacity_gbps} GB/s</span>
        </div>
      </div>
    </div>
  );
}
