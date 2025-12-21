import { useEffect, useState } from 'react';
import {
  Server,
  Cpu,
  HardDrive,
  Wifi,
  WifiOff,
  RefreshCw,
  Plus,
  Monitor,
  Laptop,
  Box,
  Database,
  FileCode,
  LayoutGrid,
} from 'lucide-react';
import { useClusterStore } from '../../stores/clusterStore';
import SwarmConfigEditor from '../../components/SwarmConfigEditor';

const NODE_TYPE_ICONS = {
  datacenter: Server,
  workstation: Monitor,
  laptop: Laptop,
  edge: Box,
  storage: Database,
};

const STATUS_COLORS = {
  online: 'bg-green-500',
  offline: 'bg-red-500',
  degraded: 'bg-yellow-500',
  starting: 'bg-blue-500',
};

type TabId = 'nodes' | 'config';

export default function ClusterView() {
  const {
    connected,
    endpoint,
    nodeCount,
    totalGpus,
    totalMemoryGb,
    healthyNodes,
    nodes,
    loading,
    error,
    fetchStatus,
    fetchNodes,
    connect,
    disconnect,
  } = useClusterStore();

  const [connectEndpoint, setConnectEndpoint] = useState('localhost:9000');
  const [activeTab, setActiveTab] = useState<TabId>('nodes');

  useEffect(() => {
    fetchStatus();
    if (connected) {
      fetchNodes();
    }
  }, [fetchStatus, fetchNodes, connected]);

  const handleConnect = async () => {
    await connect(connectEndpoint);
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-semibold">Cluster Management</h1>

          {/* Tabs */}
          <div className="flex items-center gap-1 bg-slate-700/50 rounded-lg p-1">
            <button
              onClick={() => setActiveTab('nodes')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-colors ${
                activeTab === 'nodes'
                  ? 'bg-slate-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <LayoutGrid className="w-4 h-4" />
              Nodes
            </button>
            <button
              onClick={() => setActiveTab('config')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-colors ${
                activeTab === 'config'
                  ? 'bg-slate-600 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <FileCode className="w-4 h-4" />
              .swarm Config
            </button>
          </div>

          <div className="flex items-center gap-2">
            {connected ? (
              <>
                <Wifi className="w-4 h-4 text-green-400" />
                <span className="text-sm text-green-400">Connected to {endpoint}</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-400">Disconnected</span>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {!connected ? (
            <>
              <input
                type="text"
                value={connectEndpoint}
                onChange={(e) => setConnectEndpoint(e.target.value)}
                placeholder="Cluster endpoint"
                className="px-3 py-1.5 bg-slate-700 border border-slate-600 rounded text-sm w-48"
              />
              <button
                onClick={handleConnect}
                disabled={loading}
                className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded text-sm"
              >
                <Plus className="w-4 h-4" />
                Connect
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => fetchNodes()}
                disabled={loading}
                className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-sm"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              <button
                onClick={disconnect}
                className="flex items-center gap-1 px-3 py-1.5 bg-red-600/80 hover:bg-red-600 rounded text-sm"
              >
                Disconnect
              </button>
            </>
          )}
        </div>
      </div>

      {/* Stats Bar */}
      {connected && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-4 grid grid-cols-4 gap-4">
          <StatCard
            icon={Server}
            label="Nodes"
            value={nodeCount}
            subtext={`${healthyNodes} healthy`}
          />
          <StatCard
            icon={Cpu}
            label="Total GPUs"
            value={totalGpus}
            subtext="CUDA/Metal"
          />
          <StatCard
            icon={HardDrive}
            label="Total Memory"
            value={`${totalMemoryGb} GB`}
            subtext="Cluster RAM"
          />
          <StatCard
            icon={Wifi}
            label="Status"
            value={healthyNodes === nodeCount ? 'Healthy' : 'Degraded'}
            subtext={healthyNodes === nodeCount ? 'All nodes online' : `${nodeCount - healthyNodes} issues`}
            valueColor={healthyNodes === nodeCount ? 'text-green-400' : 'text-yellow-400'}
          />
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mx-6 mt-4 p-3 bg-red-900/30 border border-red-700 rounded text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Content Area - Conditional based on active tab */}
      {activeTab === 'nodes' ? (
        /* Nodes Grid */
        connected ? (
          <div className="flex-1 overflow-auto p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {nodes.map((node) => {
                const Icon = NODE_TYPE_ICONS[node.node_type] || Server;
                return (
                  <div
                    key={node.id}
                    className="bg-slate-800 rounded-lg border border-slate-700 p-4 hover:border-slate-600 transition-colors"
                  >
                    {/* Node Header */}
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center">
                        <Icon className="w-5 h-5 text-slate-300" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">{node.hostname}</div>
                        <div className="text-xs text-slate-400 capitalize">
                          {node.node_type}
                        </div>
                      </div>
                      <div
                        className={`w-2 h-2 rounded-full ${STATUS_COLORS[node.status]}`}
                        title={node.status}
                      />
                    </div>

                    {/* Node Stats */}
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <div className="text-slate-400">GPUs</div>
                        <div className="font-mono">
                          {node.gpu_count}x ({node.gpu_memory_gb}GB)
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-400">CPU Cores</div>
                        <div className="font-mono">{node.cpu_cores}</div>
                      </div>
                      <div>
                        <div className="text-slate-400">Memory</div>
                        <div className="font-mono">{node.memory_gb} GB</div>
                      </div>
                      <div>
                        <div className="text-slate-400">Status</div>
                        <div className="capitalize">{node.status}</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center text-slate-400">
              <Server className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg mb-2">No cluster connected</p>
              <p className="text-sm">Enter a StratoSwarm endpoint to connect</p>
            </div>
          </div>
        )
      ) : (
        /* Swarm Config Editor */
        <SwarmConfigEditor
          onSave={(config) => {
            console.log('Saving .swarm config:', config);
            // TODO: Persist to file or backend
          }}
          onDeploy={(config) => {
            console.log('Deploying .swarm config:', config);
            // TODO: Send to stratoswarm for deployment
          }}
        />
      )}
    </div>
  );
}

interface StatCardProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
  subtext: string;
  valueColor?: string;
}

function StatCard({ icon: Icon, label, value, subtext, valueColor = 'text-white' }: StatCardProps) {
  return (
    <div className="bg-slate-700/50 rounded-lg p-4">
      <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
        <Icon className="w-4 h-4" />
        {label}
      </div>
      <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
      <div className="text-xs text-slate-500">{subtext}</div>
    </div>
  );
}
