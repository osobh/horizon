/**
 * SystemInfoPanel - Displays local system information including GPU detection
 *
 * Uses Tauri IPC to fetch system info from the Rust backend.
 */

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Cpu,
  HardDrive,
  Monitor,
  Zap,
  RefreshCw,
  Server,
} from 'lucide-react';

interface DetectedGpu {
  id: number;
  name: string;
  vendor: string;
  memory_gb: number | null;
  driver_version: string | null;
  gpu_type: 'nvidia' | 'amd' | 'intel' | 'apple' | 'unknown';
}

interface SystemInfo {
  hostname: string;
  os_name: string;
  os_version: string;
  cpu_model: string;
  cpu_cores: number;
  total_memory_gb: number;
  available_memory_gb: number;
  gpus: DetectedGpu[];
}

const GPU_TYPE_COLORS = {
  nvidia: 'text-green-400 bg-green-900/30 border-green-700/50',
  amd: 'text-red-400 bg-red-900/30 border-red-700/50',
  intel: 'text-blue-400 bg-blue-900/30 border-blue-700/50',
  apple: 'text-slate-300 bg-slate-700/50 border-slate-600',
  unknown: 'text-slate-400 bg-slate-800 border-slate-700',
};

const GPU_TYPE_LABELS = {
  nvidia: 'NVIDIA CUDA',
  amd: 'AMD ROCm',
  intel: 'Intel Arc',
  apple: 'Apple Metal',
  unknown: 'Unknown',
};

interface SystemInfoPanelProps {
  compact?: boolean;
}

export default function SystemInfoPanel({ compact = false }: SystemInfoPanelProps) {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSystemInfo = async () => {
    setLoading(true);
    setError(null);
    try {
      const info = await invoke<SystemInfo>('get_system_info');
      setSystemInfo(info);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemInfo();
  }, []);

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
        <p className="text-sm text-red-400">Failed to load system info: {error}</p>
        <button
          onClick={fetchSystemInfo}
          className="mt-2 text-xs text-red-300 hover:text-red-200"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!systemInfo) {
    return null;
  }

  const memoryUsedPercent = ((systemInfo.total_memory_gb - systemInfo.available_memory_gb) / systemInfo.total_memory_gb) * 100;

  if (compact) {
    return (
      <div className="space-y-2">
        {/* Compact System Summary */}
        <div className="flex items-center gap-2 text-sm">
          <Monitor className="w-4 h-4 text-slate-400" />
          <span className="text-slate-300">{systemInfo.hostname}</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">{systemInfo.os_name}</span>
        </div>

        {/* GPU Count Badge */}
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-yellow-400" />
          <span className="text-sm">
            {systemInfo.gpus.length} GPU{systemInfo.gpus.length !== 1 ? 's' : ''}
          </span>
          {systemInfo.gpus.length > 0 && (
            <span className={`text-xs px-1.5 py-0.5 rounded border ${GPU_TYPE_COLORS[systemInfo.gpus[0].gpu_type]}`}>
              {GPU_TYPE_LABELS[systemInfo.gpus[0].gpu_type]}
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Server className="w-5 h-5 text-slate-400" />
          <h3 className="font-medium">Local System</h3>
        </div>
        <button
          onClick={fetchSystemInfo}
          className="p-1.5 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-300"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
            <Monitor className="w-3 h-3" />
            Hostname
          </div>
          <div className="font-medium truncate">{systemInfo.hostname}</div>
          <div className="text-xs text-slate-500">
            {systemInfo.os_name} {systemInfo.os_version}
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
            <Cpu className="w-3 h-3" />
            CPU
          </div>
          <div className="font-medium truncate text-sm">{systemInfo.cpu_model}</div>
          <div className="text-xs text-slate-500">{systemInfo.cpu_cores} cores</div>
        </div>
      </div>

      {/* Memory */}
      <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2 text-slate-400 text-xs">
            <HardDrive className="w-3 h-3" />
            Memory
          </div>
          <div className="text-xs text-slate-400">
            {systemInfo.available_memory_gb.toFixed(1)} / {systemInfo.total_memory_gb.toFixed(1)} GB available
          </div>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all"
            style={{ width: `${memoryUsedPercent}%` }}
          />
        </div>
      </div>

      {/* GPUs */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-slate-400 text-xs">
          <Zap className="w-3 h-3" />
          GPUs ({systemInfo.gpus.length})
        </div>

        {systemInfo.gpus.length === 0 ? (
          <div className="bg-slate-800 rounded-lg p-3 border border-slate-700 text-center">
            <p className="text-sm text-slate-500">No GPUs detected</p>
          </div>
        ) : (
          <div className="space-y-2">
            {systemInfo.gpus.map((gpu) => (
              <div
                key={gpu.id}
                className={`rounded-lg p-3 border ${GPU_TYPE_COLORS[gpu.gpu_type]}`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="font-medium text-sm">{gpu.name}</div>
                    <div className="text-xs opacity-80">{gpu.vendor}</div>
                  </div>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-black/20">
                    {GPU_TYPE_LABELS[gpu.gpu_type]}
                  </span>
                </div>

                <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                  {gpu.memory_gb !== null && (
                    <div>
                      <span className="opacity-70">Memory: </span>
                      <span className="font-mono">{gpu.memory_gb.toFixed(1)} GB</span>
                    </div>
                  )}
                  {gpu.driver_version && (
                    <div>
                      <span className="opacity-70">Driver: </span>
                      <span className="font-mono">{gpu.driver_version}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
