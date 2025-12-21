/**
 * TensorMesh - Displays distributed GPU-to-GPU tensor operations
 *
 * Combines RMPI type-safe collectives with nebula RDMA transport for:
 * - 400+ Gbps GPU-direct transfers
 * - Type-safe collective operations (all-reduce, broadcast, etc.)
 * - Real-time tensor transfer monitoring
 * - Multi-node training coordination
 */

import { useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Cpu,
  RefreshCw,
  Activity,
  ArrowRightLeft,
  Layers,
  Zap,
  Server,
} from 'lucide-react';

interface TensorNode {
  id: string;
  hostname: string;
  gpu_index: number;
  gpu_model: string;
  gpu_memory_gb: number;
  rdma_enabled: boolean;
  tensor_memory_gb: number;
  role: 'coordinator' | 'worker' | 'parameter_server';
  status: 'ready' | 'computing' | 'transferring' | 'synchronizing' | 'idle' | 'error';
}

interface TensorConnection {
  source: string;
  target: string;
  transport: 'rdma_gpu_direct' | 'roce' | 'nvlink' | 'tcp';
  max_bandwidth_gbps: number;
  current_bandwidth_gbps: number;
  latency_us: number;
  active_transfers: number;
}

interface CollectiveStats {
  all_reduce_ops: number;
  broadcast_ops: number;
  scatter_ops: number;
  gather_ops: number;
  reduce_scatter_ops: number;
  all_gather_ops: number;
  avg_all_reduce_ms: number;
  collective_bytes: number;
  ops_per_second: number;
}

interface TensorTransfer {
  id: string;
  source: string;
  target: string;
  tensor_name: string;
  shape: number[];
  dtype: string;
  total_bytes: number;
  transferred_bytes: number;
  transfer_type: 'gradient_sync' | 'parameter_broadcast' | 'activation_transfer' | 'tensor_shard' | 'all_gather';
  bandwidth_gbps: number;
}

interface TensorMeshStatus {
  nodes: TensorNode[];
  connections: TensorConnection[];
  collective_stats: CollectiveStats;
  active_transfers: TensorTransfer[];
  total_bandwidth_gbps: number;
  utilization_pct: number;
  backend: string;
  parallelism_mode: 'data_parallel' | 'tensor_parallel' | 'pipeline_parallel' | 'fsdp' | 'hybrid';
}

const STATUS_COLORS = {
  ready: 'bg-green-500',
  computing: 'bg-blue-500',
  transferring: 'bg-purple-500',
  synchronizing: 'bg-yellow-500',
  idle: 'bg-slate-500',
  error: 'bg-red-500',
};

const TRANSPORT_COLORS = {
  rdma_gpu_direct: { bg: 'bg-purple-500', text: 'text-purple-400', label: 'RDMA' },
  roce: { bg: 'bg-blue-500', text: 'text-blue-400', label: 'RoCE' },
  nvlink: { bg: 'bg-green-500', text: 'text-green-400', label: 'NVLink' },
  tcp: { bg: 'bg-yellow-500', text: 'text-yellow-400', label: 'TCP' },
};

const TRANSFER_TYPE_LABELS = {
  gradient_sync: 'Gradient Sync',
  parameter_broadcast: 'Param Broadcast',
  activation_transfer: 'Activation',
  tensor_shard: 'Shard',
  all_gather: 'All-Gather',
};

interface TensorMeshProps {
  compact?: boolean;
}

export default function TensorMesh({ compact = false }: TensorMeshProps) {
  const [status, setStatus] = useState<TensorMeshStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const data = await invoke<TensorMeshStatus>('get_tensor_mesh_status');
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
      await invoke('simulate_tensor_mesh_activity');
      await fetchData();
    } catch (err) {
      console.error('Failed to simulate activity:', err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
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
        <p className="text-sm text-red-400">Failed to load tensor mesh: {error}</p>
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

  const formatNumber = (num: number) => {
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
    return num.toString();
  };

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-300">Tensor Mesh</span>
          <span className="text-xs bg-cyan-500/20 text-cyan-400 px-1.5 py-0.5 rounded">
            {status.nodes.length} GPUs
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded bg-purple-900/30 border border-purple-700/50">
            <div className="text-purple-400">All-Reduce</div>
            <div className="font-mono">{formatNumber(status.collective_stats.all_reduce_ops)}</div>
          </div>
          <div className="p-2 rounded bg-cyan-900/30 border border-cyan-700/50">
            <div className="text-cyan-400">Bandwidth</div>
            <div className="font-mono">{status.utilization_pct.toFixed(0)}%</div>
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
          <Cpu className="w-5 h-5 text-cyan-400" />
          <h3 className="font-medium">Tensor Mesh</h3>
          <span className="text-xs bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded">
            RMPI + RDMA
          </span>
          <span className="text-xs text-slate-400">
            {status.parallelism_mode.replace('_', ' ').toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={simulateActivity}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-cyan-600 hover:bg-cyan-500 rounded transition-colors"
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

      {/* GPU Nodes Grid */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
          <Server className="w-4 h-4" />
          GPU Nodes ({status.nodes.length})
        </div>
        <div className="grid grid-cols-4 gap-2">
          {status.nodes.map((node) => (
            <div
              key={node.id}
              className="bg-slate-700/50 rounded-lg p-2 border border-slate-600"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium truncate">{node.id}</span>
                <div className={`w-2 h-2 rounded-full ${STATUS_COLORS[node.status]}`} />
              </div>
              <div className="text-xs text-slate-400 truncate">{node.gpu_model.split(' ').slice(1).join(' ')}</div>
              <div className="flex items-center justify-between mt-1 text-xs">
                <span className="text-cyan-400">{node.tensor_memory_gb.toFixed(1)}GB</span>
                <span className="text-slate-500">/ {node.gpu_memory_gb}GB</span>
              </div>
              {/* Memory usage bar */}
              <div className="mt-1 h-1 bg-slate-600 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500"
                  style={{ width: `${(node.tensor_memory_gb / node.gpu_memory_gb) * 100}%` }}
                />
              </div>
              <div className="flex items-center gap-1 mt-1">
                {node.rdma_enabled && (
                  <span className="text-xs bg-purple-500/20 text-purple-400 px-1 rounded">RDMA</span>
                )}
                <span className={`text-xs ${node.role === 'coordinator' ? 'text-yellow-400' : 'text-slate-400'}`}>
                  {node.role === 'coordinator' ? 'COORD' : 'WORKER'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Collective Stats + Connections Row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Collective Operations */}
        <div className="bg-purple-900/20 rounded-lg border border-purple-700/50 p-3">
          <div className="flex items-center gap-2 text-purple-400 text-sm mb-3">
            <Layers className="w-4 h-4" />
            Collective Operations
            <span className="text-xs text-slate-400">
              {status.collective_stats.ops_per_second.toFixed(0)} ops/s
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-slate-800/50 rounded p-2 text-center">
              <div className="text-slate-400">All-Reduce</div>
              <div className="font-mono text-lg">{formatNumber(status.collective_stats.all_reduce_ops)}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2 text-center">
              <div className="text-slate-400">Broadcast</div>
              <div className="font-mono text-lg">{formatNumber(status.collective_stats.broadcast_ops)}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2 text-center">
              <div className="text-slate-400">All-Gather</div>
              <div className="font-mono text-lg">{formatNumber(status.collective_stats.all_gather_ops)}</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Avg All-Reduce</div>
              <div className="font-mono">{status.collective_stats.avg_all_reduce_ms.toFixed(2)} ms</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Total Transferred</div>
              <div className="font-mono">{formatBytes(status.collective_stats.collective_bytes)}</div>
            </div>
          </div>
        </div>

        {/* Mesh Bandwidth */}
        <div className="bg-cyan-900/20 rounded-lg border border-cyan-700/50 p-3">
          <div className="flex items-center gap-2 text-cyan-400 text-sm mb-3">
            <Zap className="w-4 h-4" />
            Mesh Bandwidth
            <span className="text-xs text-slate-400">
              {status.utilization_pct.toFixed(0)}% utilized
            </span>
          </div>
          <div className="space-y-2">
            {status.connections.slice(0, 4).map((conn, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-16 truncate text-slate-300">{conn.source.split('-').pop()}</span>
                <ArrowRightLeft className={`w-3 h-3 ${TRANSPORT_COLORS[conn.transport].text}`} />
                <span className="w-16 truncate text-slate-300">{conn.target.split('-').pop()}</span>
                <span className={`${TRANSPORT_COLORS[conn.transport].bg} text-white px-1 rounded text-xs`}>
                  {TRANSPORT_COLORS[conn.transport].label}
                </span>
                <div className="flex-1 h-1.5 bg-slate-600 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-500"
                    style={{ width: `${(conn.current_bandwidth_gbps / conn.max_bandwidth_gbps) * 100}%` }}
                  />
                </div>
                <span className="text-slate-400 w-16 text-right">
                  {conn.current_bandwidth_gbps.toFixed(0)} Gbps
                </span>
              </div>
            ))}
          </div>
          <div className="mt-2 pt-2 border-t border-slate-700 text-xs text-slate-400">
            Total: {status.total_bandwidth_gbps.toFixed(0)} Gbps capacity
          </div>
        </div>
      </div>

      {/* Active Transfers */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
          <ArrowRightLeft className="w-4 h-4" />
          Active Tensor Transfers ({status.active_transfers.length})
        </div>
        <div className="space-y-2">
          {status.active_transfers.map((transfer) => {
            const progress = (transfer.transferred_bytes / transfer.total_bytes) * 100;
            return (
              <div key={transfer.id} className="bg-slate-700/50 rounded p-2">
                <div className="flex items-center justify-between text-xs mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400">{transfer.source.split('-').pop()}</span>
                    <span className="text-slate-500">→</span>
                    <span className="text-cyan-400">{transfer.target.split('-').pop()}</span>
                    <span className="bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded">
                      {TRANSFER_TYPE_LABELS[transfer.transfer_type]}
                    </span>
                  </div>
                  <span className="text-slate-400">{transfer.bandwidth_gbps.toFixed(0)} Gbps</span>
                </div>
                <div className="text-xs text-slate-400 truncate mb-1">
                  {transfer.tensor_name} [{transfer.shape.join('×')}] {transfer.dtype}
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-1.5 bg-slate-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-cyan-500"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-400 w-20 text-right">
                    {formatBytes(transfer.transferred_bytes)} / {formatBytes(transfer.total_bytes)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Transport Legend */}
      <div className="flex items-center gap-4 text-xs text-slate-400">
        {Object.entries(TRANSPORT_COLORS).map(([key, { bg, label }]) => (
          <div key={key} className="flex items-center gap-1">
            <div className={`w-2 h-2 rounded-full ${bg}`} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
