/**
 * NetworkTopology - Displays RDMA/ZK mesh network visualization
 *
 * Shows:
 * - Mesh node topology with connections
 * - RDMA transport statistics (400+ Gbps)
 * - Zero-Knowledge proof generation metrics
 * - Connection utilization and latency
 */

import { useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Network,
  RefreshCw,
  Zap,
  Shield,
  Server,
  Laptop,
  Radio,
  Globe,
  Activity,
  ArrowRight,
} from 'lucide-react';

interface RdmaStats {
  peak_bandwidth_gbps: number;
  current_bandwidth_gbps: number;
  avg_latency_ns: number;
  min_latency_ns: number;
  bytes_transferred: number;
  operations_completed: number;
  active_connections: number;
  gpu_direct_enabled: boolean;
  queue_pairs: number;
}

interface ZkStats {
  proofs_generated: number;
  proofs_per_second: number;
  gpu_utilization: number;
  cpu_utilization: number;
  gpu_proof_ratio: number;
  avg_proof_time_ms: number;
  total_verification_time_ms: number;
  distributed_enabled: boolean;
  coordinators: number;
  proof_types: string[];
}

interface MeshNode {
  id: string;
  hostname: string;
  ip_address: string;
  node_type: 'gpu_server' | 'workstation' | 'edge' | 'relay';
  gpu_count: number;
  rdma_capable: boolean;
  zk_enabled: boolean;
  load: number;
  peer_count: number;
}

interface MeshConnection {
  source: string;
  target: string;
  connection_type: 'rdma' | 'roce' | 'tcp' | 'local';
  bandwidth_gbps: number;
  latency_us: number;
  utilization: number;
}

interface MeshTopology {
  nodes: MeshNode[];
  connections: MeshConnection[];
  total_bandwidth_gbps: number;
  active_transfers: number;
}

interface NebulaStatus {
  rdma: RdmaStats;
  zk: ZkStats;
  topology: MeshTopology;
  health: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
}

const NODE_ICONS = {
  gpu_server: Server,
  workstation: Laptop,
  edge: Globe,
  relay: Radio,
};

const CONNECTION_COLORS = {
  rdma: 'text-purple-400',
  roce: 'text-blue-400',
  tcp: 'text-yellow-400',
  local: 'text-green-400',
};

const CONNECTION_BG = {
  rdma: 'bg-purple-500',
  roce: 'bg-blue-500',
  tcp: 'bg-yellow-500',
  local: 'bg-green-500',
};

interface NetworkTopologyProps {
  compact?: boolean;
}

export default function NetworkTopology({ compact = false }: NetworkTopologyProps) {
  const [status, setStatus] = useState<NebulaStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const data = await invoke<NebulaStatus>('get_nebula_status');
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
      await invoke('simulate_nebula_activity');
      await fetchData();
    } catch (err) {
      console.error('Failed to simulate activity:', err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
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
        <p className="text-sm text-red-400">Failed to load nebula status: {error}</p>
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

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Network className="w-4 h-4 text-purple-400" />
          <span className="text-slate-300">Nebula Mesh</span>
          <span className={`text-xs px-1.5 py-0.5 rounded ${
            status.health === 'healthy'
              ? 'bg-green-500/20 text-green-400'
              : status.health === 'degraded'
              ? 'bg-yellow-500/20 text-yellow-400'
              : 'bg-red-500/20 text-red-400'
          }`}>
            {status.health}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded bg-purple-900/30 border border-purple-700/50">
            <div className="text-purple-400">RDMA</div>
            <div className="font-mono">{status.rdma.current_bandwidth_gbps.toFixed(1)} Gbps</div>
          </div>
          <div className="p-2 rounded bg-blue-900/30 border border-blue-700/50">
            <div className="text-blue-400">ZK Proofs</div>
            <div className="font-mono">{status.zk.proofs_per_second.toFixed(1)}/s</div>
          </div>
        </div>
      </div>
    );
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

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-5 h-5 text-purple-400" />
          <h3 className="font-medium">Nebula Mesh Network</h3>
          <span className={`text-xs px-2 py-0.5 rounded ${
            status.health === 'healthy'
              ? 'bg-green-500/20 text-green-400'
              : status.health === 'degraded'
              ? 'bg-yellow-500/20 text-yellow-400'
              : 'bg-red-500/20 text-red-400'
          }`}>
            {status.health}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={simulateActivity}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-purple-600 hover:bg-purple-500 rounded transition-colors"
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

      {/* RDMA + ZK Stats Row */}
      <div className="grid grid-cols-2 gap-4">
        {/* RDMA Stats */}
        <div className="bg-purple-900/20 rounded-lg border border-purple-700/50 p-3">
          <div className="flex items-center gap-2 text-purple-400 text-sm mb-3">
            <Zap className="w-4 h-4" />
            RDMA Transport
            {status.rdma.gpu_direct_enabled && (
              <span className="text-xs bg-purple-500/30 px-1.5 py-0.5 rounded">GPU-Direct</span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Bandwidth</div>
              <div className="font-mono text-lg">{status.rdma.current_bandwidth_gbps.toFixed(1)}</div>
              <div className="text-slate-500">/ {status.rdma.peak_bandwidth_gbps} Gbps</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Latency</div>
              <div className="font-mono text-lg">{status.rdma.avg_latency_ns}</div>
              <div className="text-slate-500">ns avg</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Transferred</div>
              <div className="font-mono">{formatBytes(status.rdma.bytes_transferred)}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Connections</div>
              <div className="font-mono">{status.rdma.active_connections} active</div>
            </div>
          </div>
        </div>

        {/* ZK Stats */}
        <div className="bg-blue-900/20 rounded-lg border border-blue-700/50 p-3">
          <div className="flex items-center gap-2 text-blue-400 text-sm mb-3">
            <Shield className="w-4 h-4" />
            ZK Proofs
            {status.zk.distributed_enabled && (
              <span className="text-xs bg-blue-500/30 px-1.5 py-0.5 rounded">Distributed</span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Throughput</div>
              <div className="font-mono text-lg">{status.zk.proofs_per_second.toFixed(1)}</div>
              <div className="text-slate-500">proofs/sec</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">GPU Utilization</div>
              <div className="font-mono text-lg">{status.zk.gpu_utilization.toFixed(0)}%</div>
              <div className="text-slate-500">of GPU proofs</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Total Proofs</div>
              <div className="font-mono">{status.zk.proofs_generated.toLocaleString()}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Proof Types</div>
              <div className="font-mono truncate">{status.zk.proof_types.join(', ')}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Mesh Topology */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center justify-between text-sm mb-3">
          <span className="text-slate-400">Mesh Topology</span>
          <span className="text-xs text-slate-500">
            {status.topology.nodes.length} nodes, {status.topology.connections.length} connections
          </span>
        </div>

        {/* Nodes */}
        <div className="grid grid-cols-5 gap-2 mb-4">
          {status.topology.nodes.map((node) => {
            const Icon = NODE_ICONS[node.node_type];
            return (
              <div
                key={node.id}
                className="bg-slate-700/50 rounded-lg p-2 text-center border border-slate-600"
              >
                <Icon className={`w-5 h-5 mx-auto mb-1 ${
                  node.rdma_capable ? 'text-purple-400' : 'text-slate-400'
                }`} />
                <div className="text-xs font-medium truncate">{node.id}</div>
                <div className="text-xs text-slate-400">{node.hostname.split('.')[0]}</div>
                <div className="flex items-center justify-center gap-1 mt-1">
                  {node.gpu_count > 0 && (
                    <span className="text-xs bg-green-500/20 text-green-400 px-1 rounded">
                      {node.gpu_count} GPU
                    </span>
                  )}
                  {node.zk_enabled && (
                    <Shield className="w-3 h-3 text-blue-400" />
                  )}
                </div>
                {/* Load bar */}
                <div className="mt-1 h-1 bg-slate-600 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      node.load > 80 ? 'bg-red-500' : node.load > 50 ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${node.load}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Connections */}
        <div className="space-y-1">
          <div className="text-xs text-slate-400 mb-2">Active Connections</div>
          {status.topology.connections.map((conn, i) => (
            <div
              key={i}
              className="flex items-center gap-2 text-xs bg-slate-700/30 rounded px-2 py-1"
            >
              <span className="font-mono text-slate-300 w-24 truncate">{conn.source}</span>
              <ArrowRight className={`w-3 h-3 ${CONNECTION_COLORS[conn.connection_type]}`} />
              <span className="font-mono text-slate-300 w-24 truncate">{conn.target}</span>
              <span className={`${CONNECTION_BG[conn.connection_type]} text-white text-xs px-1.5 py-0.5 rounded uppercase`}>
                {conn.connection_type}
              </span>
              <span className="text-slate-400">{conn.bandwidth_gbps} Gbps</span>
              <span className="text-slate-400">{conn.latency_us.toFixed(1)} us</span>
              {/* Utilization bar */}
              <div className="flex-1 h-1.5 bg-slate-600 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    conn.utilization > 80 ? 'bg-red-500' : conn.utilization > 50 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${conn.utilization}%` }}
                />
              </div>
              <span className="text-slate-400 w-8 text-right">{conn.utilization.toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Connection Type Legend */}
      <div className="flex items-center gap-4 text-xs text-slate-400">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-purple-500" />
          <span>RDMA (InfiniBand)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-blue-500" />
          <span>RoCE (Ethernet)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-yellow-500" />
          <span>TCP/IP</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span>Local</span>
        </div>
      </div>
    </div>
  );
}
