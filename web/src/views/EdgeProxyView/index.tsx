import { useEffect, useState } from 'react';
import {
  Shield,
  Route,
  Brain,
  Zap,
  AlertTriangle,
  RefreshCw,
  Activity,
  Server,
  CheckCircle,
  XCircle,
  Clock,
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { useEdgeProxyStore, FailurePrediction, BackendHealth, RoutingDecision } from '../../stores/edgeProxyStore';

export default function EdgeProxyView() {
  const {
    proxyStatus,
    brainStatus,
    failurePredictions,
    backendHealth,
    routingDecisions,
    loading,
    error,
    fetchProxyStatus,
    fetchBrainStatus,
    fetchFailurePredictions,
    fetchBackendHealth,
    fetchRoutingDecisions,
    simulateActivity,
  } = useEdgeProxyStore();

  const [selectedTab, setSelectedTab] = useState<'overview' | 'routing' | 'brain' | 'backends'>('overview');

  useEffect(() => {
    fetchProxyStatus();
    fetchBrainStatus();
    fetchFailurePredictions();
    fetchBackendHealth();
    fetchRoutingDecisions();
    const interval = setInterval(() => {
      fetchProxyStatus();
      fetchBackendHealth();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <Shield className="w-6 h-6 text-red-400 mr-3" />
        <h1 className="text-xl font-semibold">Edge Proxy (Vortex)</h1>
        <div className="flex-1" />
        <button
          onClick={simulateActivity}
          className="mr-2 px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded-lg text-sm transition-colors"
        >
          Simulate Traffic
        </button>
        <button
          onClick={() => { fetchProxyStatus(); fetchBrainStatus(); }}
          className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Status Bar */}
      {proxyStatus && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${proxyStatus.healthy ? 'bg-green-400' : 'bg-red-400'}`} />
            <span className="text-slate-400">Status:</span>
            <span className={proxyStatus.healthy ? 'text-green-400' : 'text-red-400'}>
              {proxyStatus.healthy ? 'Healthy' : 'Degraded'}
            </span>
          </div>
          <div><span className="text-slate-400">Active Connections:</span> {proxyStatus.active_connections.toLocaleString()}</div>
          <div><span className="text-slate-400">Requests/sec:</span> {proxyStatus.requests_per_second.toLocaleString()}</div>
          <div><span className="text-slate-400">Avg Latency:</span> {proxyStatus.average_latency_ms.toFixed(1)}ms</div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          {['overview', 'routing', 'brain', 'backends'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-4 border-b-2 transition-colors capitalize ${
                selectedTab === tab ? 'border-red-400 text-red-400' : 'border-transparent text-slate-400 hover:text-white'
              }`}
            >
              {tab}
              {tab === 'backends' && backendHealth.filter(b => !b.healthy).length > 0 && (
                <span className="ml-2 bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
                  {backendHealth.filter(b => !b.healthy).length}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">{error}</div>}

        {selectedTab === 'overview' && proxyStatus && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Traffic Chart */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Traffic by Protocol</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={[
                  { protocol: 'HTTP', requests: 45000 },
                  { protocol: 'HTTPS', requests: 120000 },
                  { protocol: 'gRPC', requests: 35000 },
                  { protocol: 'WebSocket', requests: 8000 },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="protocol" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v/1000}k`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Bar dataKey="requests" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Failure Predictions */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-400" />
                SLAI Failure Predictions
              </h3>
              <div className="space-y-3">
                {failurePredictions.slice(0, 5).map((pred) => (
                  <PredictionCard key={pred.backend_id} prediction={pred} />
                ))}
                {failurePredictions.length === 0 && (
                  <div className="text-slate-500 text-center py-4">No active predictions</div>
                )}
              </div>
            </div>

            {/* Stats Grid */}
            <div className="lg:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                icon={Activity}
                label="Total Requests"
                value={proxyStatus.total_requests.toLocaleString()}
                color="text-blue-400"
              />
              <StatCard
                icon={Server}
                label="Backends"
                value={`${backendHealth.filter(b => b.healthy).length}/${backendHealth.length}`}
                color="text-green-400"
              />
              <StatCard
                icon={Zap}
                label="Cache Hit Rate"
                value={`${(proxyStatus.cache_hit_rate * 100).toFixed(1)}%`}
                color="text-amber-400"
              />
              <StatCard
                icon={Shield}
                label="DDoS Blocked"
                value={proxyStatus.ddos_blocked.toLocaleString()}
                color="text-red-400"
              />
            </div>
          </div>
        )}

        {selectedTab === 'routing' && (
          <div className="space-y-4">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Recent Routing Decisions</h3>
              <div className="space-y-2">
                {routingDecisions.map((decision) => (
                  <RoutingDecisionRow key={decision.request_id} decision={decision} />
                ))}
                {routingDecisions.length === 0 && (
                  <div className="text-slate-500 text-center py-8">No recent routing decisions</div>
                )}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'brain' && brainStatus && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                SLAI Brain Status
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Model Version</span>
                  <span className="font-mono">{brainStatus.model_version}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Last Trained</span>
                  <span>{new Date(brainStatus.last_trained).toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Prediction Accuracy</span>
                  <span className="text-green-400">{(brainStatus.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Predictions Made</span>
                  <span>{brainStatus.predictions_made.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Failures Prevented</span>
                  <span className="text-emerald-400">{brainStatus.failures_prevented.toLocaleString()}</span>
                </div>
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Prediction Confidence Over Time</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={[
                  { time: '00:00', confidence: 0.92 },
                  { time: '04:00', confidence: 0.94 },
                  { time: '08:00', confidence: 0.91 },
                  { time: '12:00', confidence: 0.95 },
                  { time: '16:00', confidence: 0.93 },
                  { time: '20:00', confidence: 0.96 },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0.8, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Confidence']} />
                  <Line type="monotone" dataKey="confidence" stroke="#a855f7" strokeWidth={2} dot={{ fill: '#a855f7' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {selectedTab === 'backends' && (
          <div className="space-y-4">
            {backendHealth.map((backend) => (
              <BackendCard key={backend.id} backend={backend} />
            ))}
            {backendHealth.length === 0 && (
              <div className="p-8 text-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                No backends configured
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`w-5 h-5 ${color}`} />
        <span className="text-slate-400 text-sm">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function PredictionCard({ prediction }: { prediction: FailurePrediction }) {
  const riskColors = {
    low: 'text-green-400 bg-green-500/10',
    medium: 'text-amber-400 bg-amber-500/10',
    high: 'text-red-400 bg-red-500/10',
  };

  return (
    <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
      <div>
        <div className="font-medium">{prediction.backend_id}</div>
        <div className="text-sm text-slate-400">{prediction.predicted_issue}</div>
      </div>
      <div className="text-right">
        <span className={`px-2 py-1 rounded text-xs ${riskColors[prediction.risk_level]}`}>
          {prediction.risk_level.toUpperCase()}
        </span>
        <div className="text-sm text-slate-400 mt-1">{(prediction.probability * 100).toFixed(0)}% likely</div>
      </div>
    </div>
  );
}

function RoutingDecisionRow({ decision }: { decision: RoutingDecision }) {
  return (
    <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
      <div className="flex items-center gap-3">
        <Route className="w-4 h-4 text-slate-400" />
        <div>
          <div className="font-mono text-sm">{decision.request_id.slice(0, 8)}...</div>
          <div className="text-xs text-slate-400">{decision.source_ip}</div>
        </div>
      </div>
      <div className="text-center">
        <div className="text-sm">{decision.selected_backend}</div>
        <div className="text-xs text-slate-400">{decision.reason}</div>
      </div>
      <div className="text-right">
        <div className="text-sm">{decision.latency_ms.toFixed(1)}ms</div>
        <div className="text-xs text-slate-400">{new Date(decision.timestamp).toLocaleTimeString()}</div>
      </div>
    </div>
  );
}

function BackendCard({ backend }: { backend: BackendHealth }) {
  return (
    <div className={`bg-slate-800 rounded-lg border p-4 ${backend.healthy ? 'border-slate-700' : 'border-red-500'}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {backend.healthy ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <XCircle className="w-5 h-5 text-red-400" />
          )}
          <div>
            <div className="font-medium">{backend.name}</div>
            <div className="text-sm text-slate-400">{backend.address}</div>
          </div>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm ${
          backend.healthy ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
        }`}>
          {backend.healthy ? 'Healthy' : 'Unhealthy'}
        </span>
      </div>
      <div className="grid grid-cols-4 gap-4 text-sm">
        <div>
          <span className="text-slate-400">Response Time</span>
          <div className="font-medium">{backend.response_time_ms.toFixed(1)}ms</div>
        </div>
        <div>
          <span className="text-slate-400">Success Rate</span>
          <div className="font-medium text-green-400">{(backend.success_rate * 100).toFixed(1)}%</div>
        </div>
        <div>
          <span className="text-slate-400">Active Conns</span>
          <div className="font-medium">{backend.active_connections}</div>
        </div>
        <div>
          <span className="text-slate-400">Last Check</span>
          <div className="font-medium">{new Date(backend.last_check).toLocaleTimeString()}</div>
        </div>
      </div>
    </div>
  );
}
