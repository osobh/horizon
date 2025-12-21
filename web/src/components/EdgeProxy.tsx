/**
 * EdgeProxy - Vortex + SLAI Intelligent Routing Visualization (Synergy 5)
 *
 * Visualizes:
 * - Protocol transmutation: HTTP -> RMPI, WARP, TCP
 * - SLAI brain-driven routing decisions
 * - GPU failure predictions with recommended actions
 * - Backend health monitoring
 */

import { useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Globe,
  RefreshCw,
  Activity,
  ArrowRight,
  Brain,
  AlertTriangle,
  Zap,
  Server,
  Cpu,
  Shield,
} from 'lucide-react';

interface ProtocolStats {
  source_protocol: string;
  target_protocol: string;
  request_count: number;
  avg_latency_ms: number;
  success_rate_pct: number;
}

interface RoutingDecision {
  request_id: string;
  path: string;
  source_protocol: string;
  target_protocol: string;
  target_node: string;
  reason: string;
  decision_latency_ms: number;
  timestamp: number;
}

interface BackendHealth {
  node_id: string;
  hostname: string;
  status: 'healthy' | 'degraded' | 'at_risk' | 'failed' | 'unknown';
  gpu_utilization_pct: number;
  memory_utilization_pct: number;
  cpu_utilization_pct: number;
  active_jobs: number;
  failure_probability: number;
  last_heartbeat_secs: number;
}

interface EdgeProxyStatus {
  active_connections: number;
  requests_per_second: number;
  protocols: ProtocolStats[];
  routing_decisions: RoutingDecision[];
  backend_health: BackendHealth[];
  uptime_seconds: number;
  total_requests: number;
}

interface BrainStatus {
  registered_nodes: number;
  healthy_gpus: number;
  at_risk_gpus: number;
  failed_gpus: number;
  predictions_made: number;
  migrations_triggered: number;
  jobs_saved: number;
  model_accuracy_pct: number;
  active_monitors: number;
}

interface FailurePrediction {
  gpu_id: string;
  node_id: string;
  probability: number;
  estimated_ttf_secs: number | null;
  primary_factor: string;
  recommended_action: string;
  jobs_at_risk: number;
}

interface EdgeProxyBrainStatus {
  proxy: EdgeProxyStatus;
  brain: BrainStatus;
  predictions: FailurePrediction[];
}

const STATUS_COLORS = {
  healthy: 'bg-green-500',
  degraded: 'bg-yellow-500',
  at_risk: 'bg-orange-500',
  failed: 'bg-red-500',
  unknown: 'bg-slate-500',
};

const REASON_LABELS: Record<string, string> = {
  best_availability: 'Best Availability',
  data_affinity: 'Data Affinity',
  lowest_utilization: 'Low Utilization',
  closest_proximity: 'Proximity',
  label_match: 'Label Match',
  low_carbon: 'Low Carbon',
  load_balance: 'Load Balance',
};

interface EdgeProxyProps {
  compact?: boolean;
}

export default function EdgeProxy({ compact = false }: EdgeProxyProps) {
  const [status, setStatus] = useState<EdgeProxyBrainStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const data = await invoke<EdgeProxyBrainStatus>('get_edge_proxy_status');
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
      await invoke('simulate_edge_proxy_activity');
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
        <p className="text-sm text-red-400">Failed to load edge proxy: {error}</p>
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

  const formatNumber = (num: number) => {
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
    return num.toString();
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    return `${days}d ${hours}h`;
  };

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Globe className="w-4 h-4 text-indigo-400" />
          <span className="text-slate-300">Edge Proxy</span>
          <span className="text-xs bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded">
            {status.proxy.requests_per_second.toFixed(0)} rps
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded bg-indigo-900/30 border border-indigo-700/50">
            <div className="text-indigo-400">Connections</div>
            <div className="font-mono">{formatNumber(status.proxy.active_connections)}</div>
          </div>
          <div className="p-2 rounded bg-purple-900/30 border border-purple-700/50">
            <div className="text-purple-400">SLAI Brain</div>
            <div className="font-mono">{status.brain.healthy_gpus} GPUs</div>
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
          <Globe className="w-5 h-5 text-indigo-400" />
          <h3 className="font-medium">Edge Proxy</h3>
          <span className="text-xs bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded">
            Vortex + SLAI
          </span>
          <span className="text-xs text-slate-400">
            Uptime: {formatUptime(status.proxy.uptime_seconds)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={simulateActivity}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-indigo-600 hover:bg-indigo-500 rounded transition-colors"
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

      {/* Stats Overview */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-indigo-900/20 rounded-lg border border-indigo-700/50 p-3 text-center">
          <div className="text-xs text-indigo-400 mb-1">Requests/sec</div>
          <div className="text-2xl font-mono font-bold text-indigo-400">
            {formatNumber(status.proxy.requests_per_second)}
          </div>
        </div>
        <div className="bg-purple-900/20 rounded-lg border border-purple-700/50 p-3 text-center">
          <div className="text-xs text-purple-400 mb-1">Connections</div>
          <div className="text-2xl font-mono font-bold text-purple-400">
            {formatNumber(status.proxy.active_connections)}
          </div>
        </div>
        <div className="bg-emerald-900/20 rounded-lg border border-emerald-700/50 p-3 text-center">
          <div className="text-xs text-emerald-400 mb-1">Jobs Saved</div>
          <div className="text-2xl font-mono font-bold text-emerald-400">
            {status.brain.jobs_saved}
          </div>
        </div>
        <div className="bg-cyan-900/20 rounded-lg border border-cyan-700/50 p-3 text-center">
          <div className="text-xs text-cyan-400 mb-1">Model Accuracy</div>
          <div className="text-2xl font-mono font-bold text-cyan-400">
            {status.brain.model_accuracy_pct.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Protocol Transmutation + SLAI Brain Row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Protocol Transmutation */}
        <div className="bg-indigo-900/20 rounded-lg border border-indigo-700/50 p-3">
          <div className="flex items-center gap-2 text-indigo-400 text-sm mb-3">
            <Zap className="w-4 h-4" />
            Protocol Transmutation
          </div>
          <div className="space-y-2">
            {status.proxy.protocols.map((proto, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-14 bg-slate-700 px-1.5 py-0.5 rounded text-center">
                  {proto.source_protocol}
                </span>
                <ArrowRight className="w-3 h-3 text-indigo-400" />
                <span className="w-14 bg-indigo-600 px-1.5 py-0.5 rounded text-center">
                  {proto.target_protocol}
                </span>
                <div className="flex-1 h-1.5 bg-slate-600 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500"
                    style={{ width: `${proto.success_rate_pct}%` }}
                  />
                </div>
                <span className="text-slate-400 w-16 text-right">
                  {formatNumber(proto.request_count)}
                </span>
                <span className="text-slate-500 w-12 text-right">
                  {proto.avg_latency_ms.toFixed(2)}ms
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* SLAI Brain */}
        <div className="bg-purple-900/20 rounded-lg border border-purple-700/50 p-3">
          <div className="flex items-center gap-2 text-purple-400 text-sm mb-3">
            <Brain className="w-4 h-4" />
            SLAI Brain
            <span className="text-xs text-slate-400">
              {status.brain.active_monitors} monitors
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Healthy GPUs</div>
              <div className="text-lg font-mono text-green-400">{status.brain.healthy_gpus}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">At Risk</div>
              <div className="text-lg font-mono text-orange-400">{status.brain.at_risk_gpus}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Predictions</div>
              <div className="font-mono">{formatNumber(status.brain.predictions_made)}</div>
            </div>
            <div className="bg-slate-800/50 rounded p-2">
              <div className="text-slate-400">Migrations</div>
              <div className="font-mono">{status.brain.migrations_triggered}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Backend Health */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
          <Server className="w-4 h-4" />
          Backend Health ({status.proxy.backend_health.length} nodes)
        </div>
        <div className="grid grid-cols-4 gap-2">
          {status.proxy.backend_health.map((backend) => (
            <div
              key={backend.node_id}
              className={`bg-slate-700/50 rounded-lg p-2 border ${
                backend.status === 'at_risk' ? 'border-orange-500/50' :
                backend.status === 'failed' ? 'border-red-500/50' :
                'border-slate-600'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium truncate">{backend.hostname}</span>
                <div className={`w-2 h-2 rounded-full ${STATUS_COLORS[backend.status]}`} />
              </div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <div>
                  <span className="text-slate-500">GPU</span>
                  <div className="font-mono">{backend.gpu_utilization_pct.toFixed(0)}%</div>
                </div>
                <div>
                  <span className="text-slate-500">Mem</span>
                  <div className="font-mono">{backend.memory_utilization_pct.toFixed(0)}%</div>
                </div>
              </div>
              {backend.failure_probability > 0.1 && (
                <div className="mt-1 text-xs text-orange-400">
                  Risk: {(backend.failure_probability * 100).toFixed(0)}%
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Failure Predictions */}
      {status.predictions.length > 0 && (
        <div className="bg-orange-900/20 rounded-lg border border-orange-700/50 p-3">
          <div className="flex items-center gap-2 text-orange-400 text-sm mb-3">
            <AlertTriangle className="w-4 h-4" />
            GPU Failure Predictions ({status.predictions.length})
          </div>
          <div className="space-y-2">
            {status.predictions.map((pred) => (
              <div key={pred.gpu_id} className="bg-slate-800/50 rounded p-2">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-orange-400" />
                    <span className="text-sm font-medium">{pred.gpu_id}</span>
                    <span className="text-xs text-slate-500">({pred.node_id})</span>
                  </div>
                  <span className="text-sm font-mono text-orange-400">
                    {(pred.probability * 100).toFixed(0)}% risk
                  </span>
                </div>
                <div className="text-xs text-slate-400 mb-1">{pred.primary_factor}</div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-emerald-400">{pred.recommended_action}</span>
                  <span className="text-slate-500">{pred.jobs_at_risk} jobs at risk</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Routing Decisions */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
          <Shield className="w-4 h-4" />
          Recent Routing Decisions
        </div>
        <div className="space-y-1">
          {status.proxy.routing_decisions.slice(0, 5).map((decision) => (
            <div key={decision.request_id} className="flex items-center gap-2 text-xs">
              <span className="w-24 truncate text-slate-400">{decision.path}</span>
              <span className="w-12 bg-slate-700 px-1 py-0.5 rounded text-center">
                {decision.source_protocol}
              </span>
              <ArrowRight className="w-3 h-3 text-indigo-400" />
              <span className="w-12 bg-indigo-600 px-1 py-0.5 rounded text-center">
                {decision.target_protocol}
              </span>
              <ArrowRight className="w-3 h-3 text-slate-500" />
              <span className="flex-1 truncate">{decision.target_node}</span>
              <span className="bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded">
                {REASON_LABELS[decision.reason] || decision.reason}
              </span>
              <span className="text-slate-500 w-12 text-right">
                {decision.decision_latency_ms.toFixed(2)}ms
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
