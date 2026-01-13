/**
 * Swarm Stats Component
 *
 * Displays swarm-wide statistics and tier distribution.
 */

import React from 'react';
import {
  Users,
  Activity,
  Clock,
  Zap,
  WifiOff,
  Trophy,
  CheckCircle,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { useStratoSwarmStore, getTierColor } from '../../stores/stratoswarmStore';

interface StatCardProps {
  icon: React.ReactNode;
  iconColor: string;
  iconBg: string;
  label: string;
  value: string | number;
  subValue?: string;
}

function StatCard({ icon, iconColor, iconBg, label, value, subValue }: StatCardProps) {
  return (
    <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-4">
      <div className="flex items-start justify-between">
        <div className={`p-2 rounded-lg ${iconBg}`}>
          <div className={iconColor}>{icon}</div>
        </div>
      </div>
      <div className="mt-3">
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-sm text-gray-400">{label}</div>
        {subValue && <div className="text-xs text-gray-500 mt-1">{subValue}</div>}
      </div>
    </div>
  );
}

export function SwarmStats() {
  const { stats, status, isLoading, fetchStats, fetchStatus, simulateActivity } = useStratoSwarmStore();

  React.useEffect(() => {
    fetchStatus();
    fetchStats();
    const interval = setInterval(() => {
      fetchStats();
    }, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus, fetchStats]);

  const handleRefresh = () => {
    fetchStatus();
    fetchStats();
  };

  if (isLoading && !stats) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-6 w-6 animate-spin text-[#00d4ff]" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Status Banner */}
      <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${status.connected ? 'bg-green-400' : 'bg-red-400'}`} />
          <span className="text-sm text-gray-300">
            {status.connected ? `Connected to ${status.cluster_name || 'StratoSwarm'}` : 'Disconnected'}
          </span>
          {status.version && (
            <span className="text-xs text-gray-500">v{status.version}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={simulateActivity}
            disabled={isLoading}
            className="px-3 py-1 text-xs bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 rounded transition-colors disabled:opacity-50"
          >
            Simulate Activity
          </button>
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-[#1e293b] rounded transition-colors disabled:opacity-50"
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      {stats && (
        <>
          <div className="grid grid-cols-5 gap-4">
            <StatCard
              icon={<Users size={20} />}
              iconColor="text-blue-400"
              iconBg="bg-blue-400/10"
              label="Total Agents"
              value={stats.total_agents}
            />
            <StatCard
              icon={<Activity size={20} />}
              iconColor="text-green-400"
              iconBg="bg-green-400/10"
              label="Active"
              value={stats.active_agents}
            />
            <StatCard
              icon={<Clock size={20} />}
              iconColor="text-yellow-400"
              iconBg="bg-yellow-400/10"
              label="Idle"
              value={stats.idle_agents}
            />
            <StatCard
              icon={<Zap size={20} />}
              iconColor="text-purple-400"
              iconBg="bg-purple-400/10"
              label="Evolving"
              value={stats.evolving_agents}
            />
            <StatCard
              icon={<WifiOff size={20} />}
              iconColor="text-gray-400"
              iconBg="bg-gray-400/10"
              label="Offline"
              value={stats.offline_agents}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <StatCard
              icon={<Trophy size={20} />}
              iconColor="text-cyan-400"
              iconBg="bg-cyan-400/10"
              label="Total XP"
              value={stats.total_xp.toLocaleString()}
            />
            <StatCard
              icon={<CheckCircle size={20} />}
              iconColor="text-emerald-400"
              iconBg="bg-emerald-400/10"
              label="Tasks Today"
              value={stats.tasks_completed_today.toLocaleString()}
              subValue={`${stats.tasks_in_progress} in progress`}
            />
            <StatCard
              icon={<Activity size={20} />}
              iconColor="text-orange-400"
              iconBg="bg-orange-400/10"
              label="Success Rate"
              value={`${(stats.average_success_rate * 100).toFixed(1)}%`}
            />
          </div>

          {/* Tier Distribution */}
          <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-4">
            <h3 className="text-sm font-medium text-white mb-4">Tier Distribution</h3>
            <div className="flex items-end gap-4 h-24">
              {['bronze', 'silver', 'gold', 'platinum', 'diamond'].map((tier) => {
                const count = stats.tier_distribution[tier] || 0;
                const maxCount = Math.max(...Object.values(stats.tier_distribution), 1);
                const height = (count / maxCount) * 100;
                const color = getTierColor(tier as any);

                return (
                  <div key={tier} className="flex-1 flex flex-col items-center gap-2">
                    <div className="w-full flex flex-col items-center justify-end h-20">
                      <span className="text-xs text-gray-400 mb-1">{count}</span>
                      <div
                        className="w-full rounded-t transition-all duration-500"
                        style={{
                          height: `${height}%`,
                          backgroundColor: color,
                          minHeight: count > 0 ? '8px' : '0',
                        }}
                      />
                    </div>
                    <span
                      className="text-xs font-medium uppercase"
                      style={{ color }}
                    >
                      {tier}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
