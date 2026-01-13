import { useEffect } from 'react';
import {
  Server,
  Cpu,
  Power,
  PowerOff,
  Pause,
  Settings,
} from 'lucide-react';
import { useHpcCiStore, AgentStatus, AgentSummary } from '../../stores/hpcciStore';

const statusConfig: Record<AgentStatus, { color: string; bg: string; label: string }> = {
  online: { color: 'text-green-400', bg: 'bg-green-400', label: 'Online' },
  offline: { color: 'text-slate-500', bg: 'bg-slate-500', label: 'Offline' },
  draining: { color: 'text-yellow-400', bg: 'bg-yellow-400', label: 'Draining' },
  maintenance: { color: 'text-orange-400', bg: 'bg-orange-400', label: 'Maintenance' },
};

export function AgentList() {
  const { agents, fetchAgents, drainAgent, enableAgent } = useHpcCiStore();

  useEffect(() => {
    fetchAgents();
    const interval = setInterval(fetchAgents, 15000);
    return () => clearInterval(interval);
  }, [fetchAgents]);

  const onlineCount = agents.filter((a) => a.status === 'online').length;
  const totalCapacity = agents.reduce((sum, a) => sum + a.max_jobs, 0);
  const currentLoad = agents.reduce((sum, a) => sum + a.current_jobs, 0);

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Online Agents" value={`${onlineCount}/${agents.length}`} />
        <StatCard label="Total Capacity" value={`${totalCapacity} jobs`} />
        <StatCard label="Current Load" value={`${currentLoad} jobs`} />
        <StatCard
          label="Utilization"
          value={`${totalCapacity > 0 ? Math.round((currentLoad / totalCapacity) * 100) : 0}%`}
        />
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map((agent) => (
          <AgentCard
            key={agent.id}
            agent={agent}
            onDrain={() => drainAgent(agent.id)}
            onEnable={() => enableAgent(agent.id)}
          />
        ))}
      </div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string;
}

function StatCard({ label, value }: StatCardProps) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
      <div className="text-sm text-slate-400">{label}</div>
      <div className="text-xl font-semibold text-white mt-1">{value}</div>
    </div>
  );
}

interface AgentCardProps {
  agent: AgentSummary;
  onDrain: () => void;
  onEnable: () => void;
}

function AgentCard({ agent, onDrain, onEnable }: AgentCardProps) {
  const status = statusConfig[agent.status];
  const utilization =
    agent.max_jobs > 0 ? Math.round((agent.current_jobs / agent.max_jobs) * 100) : 0;

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center">
            <Server className="w-5 h-5 text-slate-300" />
          </div>
          <div>
            <div className="text-white font-medium">{agent.id}</div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${status.bg}`} />
              <span className={`text-sm ${status.color}`}>{status.label}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="p-4 space-y-3">
        {/* Jobs */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-slate-400">Jobs</span>
            <span className="text-white">
              {agent.current_jobs} / {agent.max_jobs}
            </span>
          </div>
          <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${utilization}%` }}
            />
          </div>
        </div>

        {/* GPU */}
        {agent.gpu_count !== null && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <Cpu className="w-4 h-4" />
              GPUs
            </div>
            <span className="text-sm text-white">{agent.gpu_count}</span>
          </div>
        )}

        {/* Capabilities */}
        <div className="flex flex-wrap gap-1">
          {agent.capabilities.map((cap) => (
            <span
              key={cap}
              className="px-2 py-0.5 text-xs bg-slate-700 text-slate-300 rounded"
            >
              {cap}
            </span>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex border-t border-slate-700">
        {agent.status === 'online' ? (
          <button
            onClick={onDrain}
            className="flex-1 flex items-center justify-center gap-2 py-2 text-sm text-yellow-400 hover:bg-slate-700/50 transition-colors"
          >
            <Pause className="w-4 h-4" />
            Drain
          </button>
        ) : agent.status === 'draining' || agent.status === 'maintenance' ? (
          <button
            onClick={onEnable}
            className="flex-1 flex items-center justify-center gap-2 py-2 text-sm text-green-400 hover:bg-slate-700/50 transition-colors"
          >
            <Power className="w-4 h-4" />
            Enable
          </button>
        ) : (
          <button
            disabled
            className="flex-1 flex items-center justify-center gap-2 py-2 text-sm text-slate-500 cursor-not-allowed"
          >
            <PowerOff className="w-4 h-4" />
            Offline
          </button>
        )}
        <div className="w-px bg-slate-700" />
        <button className="flex items-center justify-center gap-2 px-4 py-2 text-sm text-slate-400 hover:bg-slate-700/50 transition-colors">
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
